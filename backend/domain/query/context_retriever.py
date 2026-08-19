import os

import psycopg

from ...data.models import CandidateRow, ChunkRow, NameMatchRow, QueryRoute, RetrievedContext
from ...providers.embeddings import EmbeddingModel
from .. import candidate_facts


class ContextRetriever:
    # The semantic route's whole cost, and the one number to turn down first
    # when a token budget bites: each chunk is up to CHUNK_CHARS (currently 900)
    # of prompt, so this is roughly 1.8k tokens[1] of context on every open-ended question,
    # whether or not the answer needed all eight.
    #
    # [1] 8 chunks × 900 chars = 7,200 chars, and at the ~4 chars/token rule of
    # thumb that's ≈1.8k tokens.
    _TOP_K: int = int(os.environ.get("SEMANTIC_SEARCH_TOP_K", "8"))

    # No CV may occupy more than this many of the _TOP_K slots. Two leaves room
    # for a profile line and its supporting evidence without letting one CV
    # crowd out the people it is being compared against.
    _MAX_CHUNKS_PER_CANDIDATE: int = 2

    # Rows ranked in Postgres per slot sent to the model, so the cap above still
    # has _TOP_K chunks to fill after the surplus from any one CV is dropped.
    _OVER_FETCH: int = 4

    CHUNK_COLUMNS: str = "source_file, name, section, content"

    # Which column settles which superlative.
    _RANKING_COLUMNS: dict[str, str] = {
        "experience": "years_experience",
        "tenure": "longest_tenure_years",
    }

    # Rows sent for a ranking question. The leader answers it; the runners-up
    # are what "and why" is argued against. Everything past them is ordering
    # nobody asked to see, charged by the token.
    _RANKED_ROWS: int = 8

    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection
        self._embedding_model = EmbeddingModel()

    def retrieve_context(self, question: str, route: QueryRoute) -> RetrievedContext:
        if route.route == "structured":
            if route.ranking:
                return self._retrieve_ranked_candidates(route)
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

    def _retrieve_ranked_candidates(self, route: QueryRoute) -> RetrievedContext:
        """Order every matching candidate on the number the question asks for, and send the top.

        A superlative is the one shape neither other route can answer. Similarity
        search sees eight chunks and cannot know it is missing the ninth, and an
        unordered filter hands the model thirty rows and asks it to do the
        comparing — which is how "who has the most experience" comes back as
        whichever CV listed the most achievements.

        Only the head of the ordering is sent: past the leaders the rows cannot
        change the answer, and the header says how many were ranked so the
        truncation is visible rather than implied.
        """
        column = self._RANKING_COLUMNS[route.ranking]
        conditions, parameters = self._build_sql_conditions(route)

        sql = (
            f"SELECT source_file, name, {column}, role, current_company, positions, "
            "count(*) OVER () FROM candidates"
        )
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        # Nulls last so a candidate whose dates could not be read sinks rather
        # than tops the ranking on Postgres's default ordering.
        sql += f" ORDER BY {column} DESC NULLS LAST, name LIMIT %s"

        rows = self.connection.execute(sql, [*parameters, self._RANKED_ROWS]).fetchall()
        if not rows:
            return RetrievedContext(
                text="No candidate in the corpus matches that filter.", source_files=[]
            )
        return RetrievedContext(
            text=self._format_ranked_rows(rows, route),
            source_files=[row[0] for row in rows],
        )

    def _format_ranked_rows(self, rows: list, route: QueryRoute) -> str:
        matched = rows[0][6]
        shown = len(rows)
        filtered = self._describe_filter(route)

        if route.ranking == "tenure":
            header = (
                f"The {shown} longest single positions of the {matched} candidates{filtered}, "
                "longest first. A position's length is its own start-to-end, not a career total:"
            )
            lines = [
                f"- {name} ({source_file}): {self._longest_position_line(positions, years)}"
                for source_file, name, years, _role, _company, positions, _matched in rows
            ]
        else:
            header = (
                f"The {shown} longest careers of the {matched} candidates{filtered}, longest "
                "first. Years run from the first role's start to the latest role's end. The CVs "
                "do not date individual skills, so this is career length, not years spent using "
                "any one of them:"
            )
            lines = [
                f"- {name} ({source_file}): {years}y, {role or 'role not stated'}"
                + (f" at {company}" if company else "")
                for source_file, name, years, role, company, _positions, _matched in rows
            ]
        return header + "\n" + "\n".join(lines) + self._ranking_caveat(route)

    @staticmethod
    def _ranking_caveat(route: QueryRoute) -> str:
        """Repeat, under the rows, what the number does not measure.

        Stated only in the header it is read before the evidence and forgotten
        after it: the model ranked Python users by career length and reported it
        as "longest experience in Python". The last line before the question is
        the one that survives.
        """
        if route.ranking == "experience" and route.skills:
            listed = " and ".join(route.skills)
            return (
                f"\n\nThese years are whole careers. Nothing above says how long anyone has used "
                f"{listed}, so say that the ranking is by total experience among the candidates "
                f"who list {listed}."
            )
        return ""

    @staticmethod
    def _describe_filter(route: QueryRoute) -> str:
        """Say what the rows have in common, not merely that something was filtered.

        The rows carry dates and titles and nothing else, so a header reading
        "matching the filter" leaves the model with no grounds to connect them to
        the skill that was asked about — and, correctly by its own rules, it
        refuses to answer. Naming the filter is what makes the list evidence.
        """
        clauses = []
        if route.skills:
            clauses.append(f"whose CVs list {' and '.join(route.skills)}")
        if route.institution:
            clauses.append(f"who studied at {route.institution}")
        if route.minimum_years_experience is not None:
            clauses.append(f"with at least {route.minimum_years_experience} years of experience")
        return f" {' and '.join(clauses)}" if clauses else ""

    @staticmethod
    def _longest_position_line(positions: list, years: int | None) -> str:
        """Name the job that produced the number, since the number alone is not a reason."""
        longest = candidate_facts.longest_tenure(positions or [])
        if longest is None:
            return f"{years}y"
        role = longest.get("role") or "role not stated"
        company = longest.get("company") or "company not stated"
        start, end = longest.get("start_year"), longest.get("end_year") or "present"
        return f"{years}y as {role} at {company} ({start}-{end})"

    @staticmethod
    def _columns_worth_sending(route: QueryRoute) -> list[str]:
        # A breakdown is thirty rows at once, so every column is paid for thirty
        # times: what each person does is the answer, and where they studied is
        # thirty lines of noise.
        if route.breakdown:
            return ["role", "skills"]
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
        """Nearest chunks, spread across candidates rather than concentrated on one.

        Asked who is best for a role, the model answers from what it can see, and
        an unspread top-k lets one CV occupy half the context: whoever writes the
        most quantified bullets wins on volume of evidence rather than on being
        the better candidate. Capping the chunks any one CV contributes costs
        nothing — the extra rows are ranked in Postgres and only the kept ones
        are ever sent.
        """
        question_vector = self._embedding_model.embed_single_text(question)
        ranked: list[ChunkRow] = self.connection.execute(
            f"SELECT {self.CHUNK_COLUMNS} FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            [str(question_vector), self._TOP_K * self._OVER_FETCH],
        ).fetchall()

        rows = self._spread_across_candidates(ranked)
        return RetrievedContext(
            text=self._roster(rows) + self._format_chunk_rows(rows),
            source_files=sorted({row[0] for row in rows}),
        )

    def _spread_across_candidates(self, ranked: list[ChunkRow]) -> list[ChunkRow]:
        """Keep the best _TOP_K chunks, letting no CV contribute more than its share."""
        kept: list[ChunkRow] = []
        per_candidate: dict[str, int] = {}
        for row in ranked:
            source_file = row[0]
            if per_candidate.get(source_file, 0) >= self._MAX_CHUNKS_PER_CANDIDATE:
                continue
            per_candidate[source_file] = per_candidate.get(source_file, 0) + 1
            kept.append(row)
            if len(kept) == self._TOP_K:
                break
        return kept

    def _roster(self, rows: list[ChunkRow]) -> str:
        """Name, role and years for each candidate the context mentions.

        A chunk carries what a CV says, not what it amounts to: seniority lives
        in a field, and a section of achievements reads impressively whether it
        belongs to a staff engineer of seven years or a contractor of four.
        Comparative questions turn on exactly that field, so it is worth ~10
        tokens a head to state it rather than let the model infer it from prose.
        """
        source_files = sorted({row[0] for row in rows})
        candidates = self.connection.execute(
            "SELECT name, source_file, role, years_experience FROM candidates "
            "WHERE source_file = ANY(%s) ORDER BY name",
            [source_files],
        ).fetchall()
        if not candidates:
            return ""
        # A field the extraction left empty is skipped rather than rendered as an
        # empty slot: ", 3y experience" reads as a missing role to a person and as
        # nothing at all to a model.
        lines = "\n".join(
            f"- {name} ({source_file}): "
            + ", ".join(part for part in (role, f"{years}y experience" if years else "") if part)
            for name, source_file, role, years in candidates
        )
        return f"Candidates appearing below:\n{lines}\n\n"

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
