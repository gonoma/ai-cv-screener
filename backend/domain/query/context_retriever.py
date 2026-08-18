import os

import psycopg

from ...data.models import CandidateRow, ChunkRow, NameMatchRow, QueryRoute, RetrievedContext
from ...providers.embeddings import EmbeddingModel


class ContextRetriever:
    # The semantic route's whole cost, and the one number to turn down first
    # when a token budget bites: each chunk is up to CHUNK_CHARS (currently 900)
    # of prompt, so this is roughly 1.8k tokens[1] of context on every open-ended question,
    # whether or not the answer needed all eight.
    #
    # [1] 8 chunks × 900 chars = 7,200 chars, and at the ~4 chars/token rule of
    # thumb that's ≈1.8k tokens.
    _TOP_K: int = int(os.environ.get("SEMANTIC_SEARCH_TOP_K", "8"))

    CHUNK_COLUMNS: str = "source_file, name, section, content"

    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection
        self._embedding_model = EmbeddingModel()

    def retrieve_context(self, question: str, route: QueryRoute) -> RetrievedContext:
        if route.route == "structured":
            return self._retrieve_all_candidates_matching_filters(route)
        if route.route == "profile":
            return self._retrieve_chunks_for_named_candidate(question, route)
        return self._retrieve_chunks_by_similarity(question)

    def _retrieve_all_candidates_matching_filters(self, route: QueryRoute) -> RetrievedContext:
        """Return *every* candidate matching the filters, carrying only the asked-about fields.

        This is the route that exists because similarity search cannot count.
        Asked "who has experience with Python" a top-k of 8 returns eight names
        whether eight or eighteen match, and nothing in the answer reveals the
        truncation. A SQL predicate over the skills array returns all of them.
        """
        columns = self._columns_worth_sending(route)
        conditions, parameters = self._build_sql_conditions(route)
        sql = f"SELECT source_file, name, {', '.join(columns)} FROM candidates"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY name"

        rows: list[CandidateRow] = self.connection.execute(sql, parameters).fetchall()
        if not rows:
            return RetrievedContext(
                text="No candidate in the corpus matches that filter.", source_files=[]
            )
        return RetrievedContext(
            text=f"All {len(rows)} matching candidates:\n"
            + self._format_candidate_rows(
                rows=rows,
                columns=columns,
            ),
            source_files=[row[0] for row in rows],
        )

    @staticmethod
    def _columns_worth_sending(route: QueryRoute) -> list[str]:
        columns = []
        if route.skills:
            columns.append("skills")
        if route.institution:
            columns.append("institutions")
        if route.minimum_years_experience is not None:
            columns.append("years_experience")
        return columns or ["role", "current_company", "years_experience", "skills", "institutions"]

    def _retrieve_chunks_for_named_candidate(
        self, question: str, route: QueryRoute
    ) -> RetrievedContext:
        if not route.candidate_name:
            return self._retrieve_chunks_by_similarity(question)

        matches: list[NameMatchRow] = self.connection.execute(
            "SELECT id, name, source_file, role, current_company "
            "FROM candidates WHERE lower(name) LIKE lower(%s) ORDER BY id",
            [f"%{route.candidate_name}%"],
        ).fetchall()

        if not matches:
            return RetrievedContext(
                text=f"No candidate named {route.candidate_name} is in the corpus.",
                source_files=[],
            )

        rows: list[ChunkRow] = self.connection.execute(
            f"SELECT {self.CHUNK_COLUMNS} FROM chunks "
            "WHERE candidate_id = ANY(%s) ORDER BY candidate_id, id",
            [[match[0] for match in matches]],
        ).fetchall()
        return RetrievedContext(
            text=self._format_chunk_rows(self._drop_repeated_overlap(rows)),
            source_files=sorted({row[0] for row in rows}),
            disambiguation_note=self._build_ambiguous_name_note(
                candidate_name=route.candidate_name,
                matches=matches,
            ),
        )

    @staticmethod
    def _drop_repeated_overlap(rows: list[ChunkRow]) -> list[ChunkRow]:
        """
        Un-overlap consecutive chunks, which this route reads back in order.

        Chunks deliberately repeat their edges so a fact on a boundary survives
        whole in one of them — that is a retrieval property, and it is worth
        paying for at search time.
        """
        trimmed: list[ChunkRow] = []
        for row in rows:
            source_file, name, section, content = row
            if trimmed:
                previous = trimmed[-1]
                same_document_and_section = (previous[0], previous[2]) == (source_file, section)
                if same_document_and_section:
                    overlap = ContextRetriever._shared_run(
                        before=previous[3],
                        after=content,
                    )
                    content = content[overlap:].lstrip()
                    if not content:
                        continue
            trimmed.append((source_file, name, section, content))
        return trimmed

    @staticmethod
    def _shared_run(before: str, after: str) -> int:
        """Length of the longest suffix of `before` that starts `after`."""
        for length in range(min(len(before), len(after)), 0, -1):
            if before.endswith(after[:length]):
                return length
        return 0

    def _retrieve_chunks_by_similarity(self, question: str) -> RetrievedContext:
        question_vector = self._embedding_model.embed_single_text(question)
        rows: list[ChunkRow] = self.connection.execute(
            f"SELECT {self.CHUNK_COLUMNS} FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            [str(question_vector), self._TOP_K],
        ).fetchall()
        return RetrievedContext(
            text=self._format_chunk_rows(rows),
            source_files=sorted({row[0] for row in rows}),
        )

    @staticmethod
    def _build_sql_conditions(route: QueryRoute) -> tuple[list[str], list]:
        conditions: list[str] = []
        parameters: list = []
        for skill in route.skills:
            conditions.append("EXISTS (SELECT 1 FROM unnest(skills) s WHERE lower(s) = lower(%s))")
            parameters.append(skill)
        if route.institution:
            conditions.append(
                "EXISTS (SELECT 1 FROM unnest(institutions) i "
                "WHERE lower(i) LIKE lower(%s) OR lower(%s) LIKE lower(i))"
            )
            parameters += [f"%{route.institution}%", f"%{route.institution}%"]
        if route.minimum_years_experience is not None:
            conditions.append("years_experience >= %s")
            parameters.append(route.minimum_years_experience)
        return conditions, parameters

    @staticmethod
    def _format_candidate_rows(rows: list[CandidateRow], columns: list[str]) -> str:
        """
        One line per candidate, carrying whichever columns were selected.

        Labels are dropped for the fields the model can identify unaided, a
        list of technologies does not need to be introduced as skills, because
        a label repeated across thirty rows is thirty times its own length.
        """

        def render(value) -> str:
            return ", ".join(value) if isinstance(value, list) else str(value)

        return "\n".join(
            f"- {name} ({source_file}): "
            + "; ".join(render(value) for value in values if value not in (None, [], ""))
            for source_file, name, *values in rows
        )

    @staticmethod
    def _format_chunk_rows(rows: list[ChunkRow]) -> str:
        return "\n\n".join(
            f"[{source_file} — {name} — {section or 'CV'}]\n{content}"
            for source_file, name, section, content in rows
        )

    @staticmethod
    def _build_ambiguous_name_note(candidate_name: str, matches: list[NameMatchRow]) -> str | None:
        if len(matches) < 2:
            return None
        described = "; ".join(
            f"{candidate_name} ({source_file}) — {role} at {company}"
            for _candidate_id, candidate_name, source_file, role, company in matches
        )
        return (
            f"{len(matches)} different candidates share the name {candidate_name}. "
            "Describe each one separately and tell them apart by employer and role. "
            f"Do not merge them: {described}"
        )
