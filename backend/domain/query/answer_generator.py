from collections.abc import Iterator

from ...data.models import RetrievedContext
from ...providers.llm_provider import LlmProvider


class AnswerGenerator:
    SYSTEM_PROMPT: str = """You answer a recruiter's questions about a corpus of CVs.

The context below is the only information you have. It is not a starting point
and not a hint: if it does not contain the answer, say so plainly and stop. Do
not use general knowledge about the world, do not guess at what a CV probably
said, and do not fill a gap with something plausible.

When the context lists all candidates matching a filter, that list is complete —
name all of them, not a selection.

Cite the source file in brackets after each claim, like [ana-silva.pdf].
Be brief. No preamble."""

    def __init__(self) -> None:
        self._llm = LlmProvider()

    def stream_grounded_answer(self, question: str, context: RetrievedContext) -> Iterator[str]:
        user_prompt = ""
        if context.disambiguation_note:
            user_prompt += f"IMPORTANT: {context.disambiguation_note}\n\n"
        user_prompt += f"Context:\n{context.text}\n\nQuestion: {question}"
        yield from self._llm.stream_text_tokens(self.SYSTEM_PROMPT, user_prompt)
