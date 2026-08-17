import os

import psycopg

from ...data.models import CandidateRow, ChunkRow, NameMatchRow, QueryRoute, RetrievedContext
from ...providers.embeddings import EmbeddingModel


class ContextRetriever:
    _TOP_K: int = int(os.environ.get("SEMANTIC_SEARCH_TOP_K", "8"))

    CANDIDATE_COLUMNS: str = (
        "source_file, name, role, current_company, years_experience, skills, institutions"
    )
    CHUNK_COLUMNS: str = "source_file, name, section, content"

    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection
        self._embedding_model = EmbeddingModel()

    def retrieve_context(self, question: str, route: QueryRoute) -> RetrievedContext:
        if route.route == "structured":
            return self.retrieve_all_candidates_matching_filters(route)
        if route.route == "profile":
            return self.retrieve_chunks_for_named_candidate(route)
        return self.retrieve_chunks_by_similarity(question)

    def retrieve_all_candidates_matching_filters(self, route: QueryRoute) -> RetrievedContext:
        """Return *every* candidate matching the filters, not the nearest few.

        This is the route that exists because similarity search cannot count.
        Asked "who has experience with Python" a top-k of 8 returns eight names
        whether eight or eighteen match, and nothing in the answer reveals the
        truncation. A SQL predicate over the skills array returns all of them.
        """
        conditions, parameters = self._build_sql_conditions(route)
        sql = f"SELECT {self.CANDIDATE_COLUMNS} FROM candidates"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY name"

        rows: list[CandidateRow] = self.connection.execute(sql, parameters).fetchall()
        if not rows:
            return RetrievedContext(
                text="No candidate in the corpus matches that filter.", source_files=[]
            )
        return RetrievedContext(
            text=f"All {len(rows)} matching candidates:\n" + self._format_candidate_rows(rows),
            source_files=[row[0] for row in rows],
        )

    def retrieve_chunks_for_named_candidate(self, route: QueryRoute) -> RetrievedContext:
        """Resolve the name to candidate ids first, then fetch chunks by id.

        The extra hop exists because names are not identifiers: the corpus
        deliberately contains two different people with the same name. Filtering
        chunks on the name directly would interleave two careers into a single
        incoherent profile, and taking the first match would silently answer
        about the wrong person. Matching on ids keeps them apart, and
        `_build_ambiguous_name_note` tells the model to describe both.
        """
        if not route.candidate_name:
            return self.retrieve_chunks_by_similarity("")

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
            text=self._format_chunk_rows(rows),
            source_files=sorted({row[0] for row in rows}),
            disambiguation_note=self._build_ambiguous_name_note(route.candidate_name, matches),
        )

    def retrieve_chunks_by_similarity(self, question: str) -> RetrievedContext:
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
    def _format_candidate_rows(rows: list[CandidateRow]) -> str:
        return "\n".join(
            f"- {name} ({source_file}): {role} at {company}, {years} years experience. "
            f"Skills: {', '.join(skills)}. Education: {', '.join(institutions)}."
            for source_file, name, role, company, years, skills, institutions in rows
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
