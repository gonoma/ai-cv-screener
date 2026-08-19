from collections.abc import Iterator

from ...data.models import RetrievedContext
from ...providers.llm_provider import LlmProvider


class AnswerGenerator:
    SYSTEM_PROMPT: str = (
        "You answer a recruiter's questions about a corpus of CVs.\n"
        "\n"
        "The context below is all you have. If it does not contain the answer, say so and\n"
        "stop — no general knowledge, no guessing at what a CV probably said, no\n"
        "plausible filler.\n"
        "\n"
        "A list of candidates matching a filter is complete: name all of them.\n"
        "{citation_rule}\n"
        "Be brief: no preamble, no restating the question, no closing summary. Brief is\n"
        "not the same as partial — if the question asks which candidates, or why, or to\n"
        "compare, give the reason for each one. A ranking with no reasons does not\n"
        "answer the question that was asked.\n"
        "\n"
        "Lay it out: the verdict on its own line, then a blank line, then one short\n"
        "paragraph per reason or per candidate. Never one unbroken block. Name a person\n"
        "once, then carry on without repeating the full name — the same name at the\n"
        "head of every paragraph reads like a form letter. Repeat it only where the\n"
        "answer moves between several people and the reader could lose track.\n"
        "\n"
        "Context that arrives already ordered by a ranking has done the comparing for\n"
        "you: the first row is the answer. Name it, say what the number is, and name\n"
        "the runner-up it beat — do not recite the rest of the list.\n"
        "\n"
        "Asked who is best, rank on the roster at the top of the context first: title\n"
        "and years are what seniority means, and the longer list of achievements is\n"
        "usually the CV that itemises more, not the stronger candidate. Then say who\n"
        "else was close and why they lost.\n"
        "\n"
        "Plain text only — the reply is not rendered as markdown, so asterisks and\n"
        "hashes reach the reader as literal characters."
    )

    CITE_EVERY_CLAIM: str = (
        "\nCite the source file in brackets after each claim, like [ana-silva.pdf].\n"
    )

    CITE_NOTHING: str = (
        "\nEvery fact below comes from the same CV, which is already shown to the\n"
        "reader, so do not cite it. No filenames, no bracketed references.\n"
    )

    def __init__(self) -> None:
        self._llm = LlmProvider()

    def stream_grounded_answer(self, question: str, context: RetrievedContext) -> Iterator[str]:
        user_prompt = ""
        if context.disambiguation_note:
            user_prompt += f"IMPORTANT: {context.disambiguation_note}\n\n"
        user_prompt += f"Context:\n{context.text}\n\nQuestion: {question}"
        yield from self._llm.stream_text_tokens(
            system_prompt=self._system_prompt(context),
            user_prompt=user_prompt,
        )

    def _system_prompt(self, context: RetrievedContext) -> str:
        cites_are_useful = len(set(context.source_files)) > 1
        return self.SYSTEM_PROMPT.format(
            citation_rule=self.CITE_EVERY_CLAIM if cites_are_useful else self.CITE_NOTHING
        )
