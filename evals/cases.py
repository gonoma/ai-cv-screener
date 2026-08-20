"""Builds the list of questions the eval suite asks, and how to mark each answer.

Where the questions come from: `data/ground_truth.json`, the answer key that the
corpus generator writes at the same time as it writes the fake CVs. It records,
for every candidate, what is actually true of them — their skills, their
employers, how many years they have worked.

Why generate the questions from that file instead of typing out a list by hand?
Because the corpus can be rebuilt at any time (`make generate`), and then it
contains different invented people. A hardcoded question like "does María López
know Python?" would quietly become a question about somebody who no longer
exists, and the suite would report failures that are really just a stale test.
Deriving both the question and its expected answer from the key keeps the two in
step automatically.

One `Case` below = one question, plus the rules for deciding whether the answer
that came back was right.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from backend.domain import candidate_facts

from . import REPO_ROOT


@dataclass
class Case:
    """A single question and everything needed to mark the answer to it."""

    # Which shape of question this is ("aggregation", "lookup", ...). Printed in
    # the results table and used to check no shape is missing. Each value is
    # produced by one method of CaseBuilder, grouped under a matching heading.
    kind: str
    # The exact text sent to the backend's /chat endpoint.
    question: str
    # Which of the backend's three strategies should have handled this question:
    # "structured" (query the database), "profile" (one named person) or
    # "semantic" (vector search over CV text). None means we do not check it.
    expected_route: str | None = None

    # --- the ways an answer can be marked; a case uses whichever fit it -------

    # The strictest check: the answer must name exactly these people — everyone
    # in the set, and nobody outside it. None means this case is not marked on
    # names at all.
    expected_names: set[str] | None = field(default=None)
    # Looser: these people must be named, but naming others is fine. Used where
    # a good answer legitimately mentions people the question is not about — a
    # "who has worked somewhere longest?" answer names the winner and usually
    # the runner-up it beat, and the strict check above would call that
    # runner-up a wrong extra name.
    must_name: set[str] = field(default_factory=set)
    # Minimums, for questions where no exact set of people is "the" right
    # answer. A summary of the whole corpus is fine as long as it covers most of
    # it, and minimum_sources catches an answer that was built from too few CVs
    # — that is, one where retrieval quietly narrowed the question.
    minimum_names: int = 0
    minimum_sources: int = 0
    # A regular expression the answer text has to match, for when the right
    # answer is a *shape* rather than a set of names: "give me percentages"
    # is satisfied by any categories, as long as there are percentages.
    must_match: str = ""
    # True when the honest answer is "the CVs do not say". The system is
    # supposed to admit that instead of inventing somebody.
    must_decline: bool = False
    # Strings that must appear in the answer word for word. Used to check that
    # two same-named candidates were kept apart: each of their employers has to
    # show up, or the answer merged them into one person.
    must_distinguish: list[str] = field(default_factory=list)
    # Free text printed next to the result, so a failure in the table explains
    # what the case was checking without anyone opening this file.
    note: str = ""


class AnswerKey:
    """Reads `data/ground_truth.json` and answers questions about the corpus.

    This is the ground truth: the facts the corpus generator invented, before any
    CV was rendered to a PDF and long before anything was parsed back out of one.
    Both halves of the suite need it — the questions below are built from it, and
    `extraction.py` marks the parsed CVs against it — so it is loaded once and
    then asked for what each caller needs.
    """

    _PATH: Path = REPO_ROOT / "data" / "ground_truth.json"

    def __init__(self) -> None:
        if not self._PATH.exists():
            raise FileNotFoundError(f"{self._PATH} missing — generate the corpus first")
        self._ground_truth: dict = json.loads(self._PATH.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def candidates(self) -> list[dict]:
        return self._ground_truth["candidates"]

    @property
    def corpus_year(self) -> int:
        """The year the CVs were written in.

        Needed because CVs say "2021 - present", and turning that into a number
        of years requires knowing what year "present" is. It is the corpus's own
        year, not today's, so the numbers stay stable as time passes.
        """
        return self._ground_truth["corpus_year"]

    @property
    def candidate_names(self) -> set[str]:
        return {candidate["name"] for candidate in self.candidates}

    def candidates_by_source_file(self) -> dict[str, dict]:
        """The same candidates, keyed by CV filename instead of listed in order.

        A parsed extraction on disk knows which PDF it came from, so this is how
        `extraction.py` finds the truth to compare each one against.
        """
        return {candidate["source_file"]: candidate for candidate in self.candidates}

    def candidates_with_skill(self, skill: str) -> set[str]:
        """Everyone the key says has this skill, however their CV wrote it down.

        The key stores each skills line exactly as the CV prints it, and CVs are
        inconsistent: "Python" can appear on its own, inside "Programming (Java,
        Python)", or after "Technical background:". So we search inside each
        line rather than comparing whole strings — otherwise the expected answer
        would depend on CV formatting, and a perfectly correct answer from the
        system would be marked wrong.

        The (?<!\\w) and (?!\\w) around the skill mean "not preceded/followed by
        another word character", i.e. match the whole word. Without them, asking
        for "R" would match every candidate with "Ruby" on their CV.
        """
        pattern = re.compile(rf"(?<!\w){re.escape(skill.lower())}(?!\w)")
        return {
            candidate["name"]
            for candidate in self.candidates
            if any(pattern.search(skills_line.lower()) for skills_line in candidate["skills"])
        }


class CaseBuilder:
    """Turns the answer key into one question of each shape.

    One question per shape, not several. Every case costs two calls to the
    language model (one to pick a route, one to write the answer), and the free
    tiers this project runs on are measured in tens of calls per day. Asking the
    same shape twice with a different skill exercises exactly the same code, so
    it would spend that budget without testing anything new.
    """

    # The shapes the exercise brief (section 6) asks for. If the corpus cannot
    # produce one of them, the run says so out loud at the end rather than
    # silently running a shorter suite and reporting all green.
    _REQUIRED_KINDS: tuple[str, ...] = (
        "aggregation",
        "lookup",
        "profile",
        "ambiguous",
        "multi_hop",
        "qualitative",
        "unanswerable",
        # Shapes added later, as the backend grew a route for each. Every one of
        # these was a question the system got wrong at some point, so each is
        # here to catch that same mistake coming back.
        "ranking_tenure",
        "ranking_skill",
        "breakdown",
        "mention_only",
    )

    # Skills to ask about when we need a question nobody can answer. Only one
    # that no CV mentions is ever used: "unanswerable" has to be genuinely
    # unanswerable for the corpus in front of us, or the case tests nothing.
    _CANDIDATE_ABSENT_SKILLS: tuple[str, ...] = (
        "COBOL",
        "Fortran",
        "Erlang",
        "LabVIEW",
        "SAP ABAP",
        "Verilog",
    )

    # How many CVs must list a skill before we will ask "who knows it?". With
    # only one holder the answer is a single person, which is the lookup case
    # again rather than a test of finding everybody.
    _SHARED_SKILL_MINIMUM_HOLDERS: int = 2

    def __init__(self, answer_key: AnswerKey) -> None:
        self._answer_key = answer_key
        # Counted once here because several cases need them: how often each name
        # occurs (to find the deliberate duplicate) and how many CVs list each
        # skill (to find one that plenty of people share).
        self._name_counts = Counter(candidate["name"] for candidate in answer_key.candidates)
        self._skill_counts = Counter(
            skill for candidate in answer_key.candidates for skill in candidate["skills"]
        )
        self._shared_skills = [
            skill
            for skill, holder_count in self._skill_counts.most_common()
            if holder_count >= self._SHARED_SKILL_MINIMUM_HOLDERS
        ]

    def build(self) -> list[Case]:
        """Every case this corpus can support, in the order they will be asked.

        Each builder below returns None when this particular corpus cannot
        support that shape — no two people share a name, say — and those are
        dropped here.
        """
        built_cases = [
            self._skill_aggregation_case(),
            self._institution_lookup_case(),
            self._unique_name_profile_case(),
            self._ambiguous_name_profile_case(),
            self._skill_and_experience_multi_hop_case(),
            self._longest_tenure_ranking_case(),
            self._longest_career_in_skill_case(),
            self._corpus_breakdown_case(),
            self._skill_named_as_example_case(),
            self._qualitative_opinion_case(),
            *self._unanswerable_question_cases(),
        ]
        return [case for case in built_cases if case is not None]

    def missing_kinds(self, cases: list[Case]) -> list[str]:
        """Required shapes this corpus could not produce a question for."""
        produced_kinds = {case.kind for case in cases}
        return [kind for kind in self._REQUIRED_KINDS if kind not in produced_kinds]

    @property
    def _most_shared_skill(self) -> str | None:
        """The skill the most CVs list, or None if no skill is shared at all."""
        return self._shared_skills[0] if self._shared_skills else None

    # --- kind "aggregation" --------------------------------------------------

    def _skill_aggregation_case(self) -> Case | None:
        """ "Who has experience with X?", asked about the most widely held skill.

        The answer has to be a complete list, which is exactly what plain vector
        search is bad at: it returns the handful of CV chunks that look most
        similar to the question, so with twenty holders and eight slots it always
        leaves people out. That is why this one is marked strictly — every
        holder, and nobody else.
        """
        skill = self._most_shared_skill
        if skill is None:
            return None
        holder_names = self._answer_key.candidates_with_skill(skill)
        return Case(
            kind="aggregation",
            question=f"Who has experience with {skill}?",
            expected_route="structured",
            expected_names=holder_names,
            note=f"{len(holder_names)} candidates hold {skill}",
        )

    # --- kind "lookup" -------------------------------------------------------

    def _institution_lookup_case(self) -> Case | None:
        """ "Which candidate graduated from <university>?" — one exact entity.

        A university rather than an employer or a job title because the backend
        can filter on exactly three things — a name, a skill and an institution —
        and the other two already have cases of their own (profile and
        aggregation). So this is the one remaining exact-match filter to test.

        It prefers a university several candidates attended and, between equally
        popular ones, the one with the shortest name.

        A short name usually means an abbreviation, and abbreviations are the
        hard case for vector search: "UPC" and "UPM" are near-identical strings,
        so they end up close together in the model's vector space and one is
        easily returned in place of the other.
        """
        institutions = Counter(
            institution
            for candidate in self._answer_key.candidates
            for institution in candidate["institutions"]
        )
        if not institutions:
            return None
        institution = min(institutions, key=lambda name: (-institutions[name], len(name)))
        expected_names = {
            candidate["name"]
            for candidate in self._answer_key.candidates
            if institution in candidate["institutions"]
        }
        return Case(
            kind="lookup",
            question=f"Which candidate graduated from {institution}?",
            expected_route="structured",
            expected_names=expected_names,
            note=f"{institution}: {len(expected_names)} candidate(s)",
        )

    # --- kind "profile" ------------------------------------------------------

    def _unique_name_profile_case(self) -> Case | None:
        """ "Summarise the profile of <person>" for a name only one candidate has.

        With one possible subject, anything the answer gets wrong is a reading
        mistake rather than a mix-up between two people. Shared names are a
        different problem, tested by the next case.
        """
        uniquely_named = [
            candidate
            for candidate in self._answer_key.candidates
            if self._name_counts[candidate["name"]] == 1
        ]
        if not uniquely_named:
            return None
        person = uniquely_named[0]
        return Case(
            kind="profile",
            question=f"Summarise the profile of {person['name']}.",
            expected_route="profile",
            expected_names={person["name"]},
            must_distinguish=[person["current_company"]],
            note=person["current_role"],
        )

    # --- kind "ambiguous" ----------------------------------------------------

    def _ambiguous_name_profile_case(self) -> Case | None:
        """The same "summarise <person>" question, for a name two candidates share.

        The corpus gives two different people the same name on purpose. The system
        has to notice there are two of them and keep them apart, rather than
        blending both careers into one person who never existed.
        """
        repeated_names = [name for name, count in self._name_counts.items() if count > 1]
        if not repeated_names:
            return None
        name = repeated_names[0]
        namesakes = [
            candidate for candidate in self._answer_key.candidates if candidate["name"] == name
        ]
        return Case(
            kind="ambiguous",
            question=f"Summarise the profile of {name}.",
            expected_route="profile",
            expected_names={name},
            # Every employer has to be named. If one is missing, the answer
            # either dropped a person or merged the two into a single career.
            must_distinguish=[namesake["current_company"] for namesake in namesakes],
            note=f"{len(namesakes)} candidates share this name",
        )

    # --- kind "multi_hop" ----------------------------------------------------

    def _skill_and_experience_multi_hop_case(self) -> Case | None:
        """ "Who has skill X *and* at least N years of experience?" — the answer needs
        two facts about each person, not one.

        The years threshold is picked so that it actually rules somebody out
        (it is the middle value among the holders). If every holder passed it,
        the answer would be identical to the aggregation case's and the extra
        condition would prove nothing.
        """
        for skill in self._shared_skills:
            skill_holders = [
                candidate
                for candidate in self._answer_key.candidates
                if any(skills_line.lower() == skill.lower() for skills_line in candidate["skills"])
            ]
            years = sorted(candidate["years_experience"] for candidate in skill_holders)
            if len(years) < 2 or years[0] == years[-1]:
                continue
            threshold = years[len(years) // 2]
            expected_names = {
                candidate["name"]
                for candidate in skill_holders
                if candidate["years_experience"] >= threshold
            }
            if 0 < len(expected_names) < len(skill_holders):
                return Case(
                    kind="multi_hop",
                    question=f"Who has both {skill} and at least {threshold} years of experience?",
                    expected_route="structured",
                    expected_names=expected_names,
                    note=(
                        f"{skill} + >={threshold}y excludes "
                        f"{len(skill_holders) - len(expected_names)}"
                    ),
                )
        return None

    # --- kinds "ranking_tenure" and "ranking_skill" --------------------------
    # "Who has the most ...?" A filter cannot answer these: everybody qualifies,
    # and what decides the answer is the ordering, so the system has to sort
    # rather than select.

    def _longest_tenure_ranking_case(self) -> Case | None:
        """Who stayed in one job the longest.

        Marked with must_name rather than expected_names — i.e. "the winner must
        appear", not "only the winner may appear" — because a good answer says
        who came second and by how much, and we do not want to fail it for that.
        """
        corpus_year = self._answer_key.corpus_year
        tenures = {
            candidate["name"]: candidate_facts.tenure_years(longest_position, as_of=corpus_year)
            for candidate in self._answer_key.candidates
            if (
                longest_position := candidate_facts.longest_tenure(
                    candidate["experience"], as_of=corpus_year
                )
            )
        }
        if not tenures:
            return None
        longest_tenure_years = max(tenures.values())
        return Case(
            kind="ranking_tenure",
            question="Which candidate has worked for the longest period in a single job?",
            expected_route="structured",
            must_name={name for name, years in tenures.items() if years == longest_tenure_years},
            note=f"{longest_tenure_years}y is the longest single position in the corpus",
        )

    def _longest_career_in_skill_case(self) -> Case | None:
        """The same ranking, but only among the people who have a given skill — so
        the system has to filter and sort in one go, which is the harder version.

        Ties are handled by expecting anyone on the top score, since several
        candidates can share the longest career.
        """
        skill = self._most_shared_skill
        if skill is None:
            return None
        holder_names = self._answer_key.candidates_with_skill(skill)
        skill_holders = [
            candidate
            for candidate in self._answer_key.candidates
            if candidate["name"] in holder_names
        ]
        most_years = max(candidate["years_experience"] for candidate in skill_holders)
        return Case(
            kind="ranking_skill",
            question=f"Which candidate has the longest experience in {skill}, and why?",
            expected_route="structured",
            must_name={
                candidate["name"]
                for candidate in skill_holders
                if candidate["years_experience"] == most_years
            },
            note=f"{most_years}y is the longest career among {len(skill_holders)} {skill} holders",
        )

    # --- kind "breakdown" ----------------------------------------------------

    def _corpus_breakdown_case(self) -> Case:
        """ "Divide the candidates into groups" — about the corpus as a whole, not
        about any subset of it.

        There is no single right set of groups (is "ML engineer" its own category
        or part of "data"?), so instead of marking the categories we check two
        things that are not judgement calls: that the answer was built from every
        CV, and that it is written as proportions rather than as a list of thirty
        names.
        """
        return Case(
            kind="breakdown",
            question="Divide the candidates by roughly what they do, as percentages.",
            expected_route="structured",
            # "sources" are the CVs the backend actually retrieved to answer
            # with. Requiring all of them is how we catch an answer that
            # summarised a third of the corpus and presented it as the whole.
            minimum_sources=len(self._answer_key),
            must_match=r"\d+\s*%|\d+\s+candidates?",
            note=f"must see all {len(self._answer_key)} candidates and answer in proportions",
        )

    # --- kind "mention_only": an example is not a filter ---------------------

    def _skill_named_as_example_case(self) -> Case | None:
        """The same breakdown question, with a skill named only as an illustration
        ("40% might do backend work with Python").

        This is a real bug the suite caught: the system treated that example as a
        filter, retrieved only the Python holders, and then described them as if
        they were the entire corpus. So the check here is on how many CVs were
        retrieved — more than the number of holders means the example was read as
        an example.
        """
        skill = self._most_shared_skill
        if skill is None:
            return None
        holder_names = self._answer_key.candidates_with_skill(skill)
        return Case(
            kind="mention_only",
            question=(
                "Divide all the candidates by roughly what they do — for example, "
                f"40% might do backend work with {skill}."
            ),
            expected_route="structured",
            minimum_sources=len(holder_names) + 1,
            note=(
                f"filtering on {skill} would retrieve only "
                f"{len(holder_names)} of {len(self._answer_key)}"
            ),
        )

    # --- kind "qualitative": a matter of opinion -----------------------------

    def _qualitative_opinion_case(self) -> Case:
        """ "Who seems strongest at ...?" has no single correct answer, so this one is
        marked loosely: the system has to give a real answer and point at the CV it
        got the impression from, whoever it picks.
        """
        return Case(
            kind="qualitative",
            question="Who seems strongest at leading teams?",
            expected_route="semantic",
            note="loose: must answer and cite something",
        )

    # --- kind "unanswerable": the corpus simply does not say -----------------

    def _unanswerable_question_cases(self) -> list[Case]:
        """Questions whose honest answer is "the CVs do not contain that".

        This is the check against hallucination — inventing a plausible-sounding
        candidate is the failure mode here, and saying "nobody" is the pass.
        """
        cases = []
        skills_present = {skill.lower() for skill in self._skill_counts}
        skills_absent = [
            skill for skill in self._CANDIDATE_ABSENT_SKILLS if skill.lower() not in skills_present
        ]
        if skills_absent:
            cases.append(
                Case(
                    kind="unanswerable",
                    question=f"Which candidate has experience with {skills_absent[0]}?",
                    must_decline=True,
                    note=f"{skills_absent[0]} appears in no CV",
                )
            )
        # A second one that is not about skills at all: the corpus has no field
        # for licences, so no amount of searching can turn one up.
        cases.append(
            Case(
                kind="unanswerable",
                question="Which candidate holds a commercial pilot's licence?",
                must_decline=True,
                note="not a field the corpus contains",
            )
        )
        return cases
