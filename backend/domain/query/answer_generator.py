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
        "Answer in as few words as the question allows. No preamble, no restating the\n"
        "question, no closing summary."
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
