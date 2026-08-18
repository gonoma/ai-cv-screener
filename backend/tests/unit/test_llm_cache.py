import pytest

from backend.providers.llm_provider import LlmProvider

SCHEMA = {"type": "object", "properties": {"route": {"type": "string"}}}


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setattr(LlmProvider, "_CACHE_DIRECTORY", tmp_path / "llm_cache")
    return LlmProvider()


def count_calls(monkeypatch, provider: LlmProvider) -> list[str]:
    """Replace only the uncached path, so the cache itself is what is exercised."""
    seen: list[str] = []

    def fake(self, prompt, json_schema):
        seen.append(prompt)
        return {"route": "structured"}

    monkeypatch.setattr(LlmProvider, "_generate_json_object_uncached", fake)
    return seen


def test_the_same_prompt_is_answered_once(monkeypatch, provider):
    seen = count_calls(
        monkeypatch=monkeypatch,
        provider=provider,
    )

    first = provider.generate_json_object(
        prompt="who knows Python?",
        json_schema=SCHEMA,
    )
    second = provider.generate_json_object(
        prompt="who knows Python?",
        json_schema=SCHEMA,
    )

    assert len(seen) == 1
    assert first == second == {"route": "structured"}


def test_a_changed_prompt_is_not_served_from_cache(monkeypatch, provider):
    """The property that makes this safe to leave on while tuning.

    The prompts carry the system prompt and the retrieved context, so editing
    either has to miss. A cache keyed on the bare question would serve the old
    answer and quietly hide the change being tuned.
    """
    seen = count_calls(
        monkeypatch=monkeypatch,
        provider=provider,
    )

    provider.generate_json_object(
        prompt="who knows Python?",
        json_schema=SCHEMA,
    )
    provider.generate_json_object(
        prompt="who knows Python? Answer concisely.",
        json_schema=SCHEMA,
    )

    assert len(seen) == 2


def test_nothing_is_cached_when_the_flag_is_unset(monkeypatch, provider):
    monkeypatch.delenv("LLM_CACHE")
    seen = count_calls(
        monkeypatch=monkeypatch,
        provider=provider,
    )

    provider.generate_json_object(
        prompt="who knows Python?",
        json_schema=SCHEMA,
    )
    provider.generate_json_object(
        prompt="who knows Python?",
        json_schema=SCHEMA,
    )

    assert len(seen) == 2


def test_an_abandoned_stream_is_not_cached(monkeypatch, provider):
    """A partial answer replayed as a whole one would be worse than no cache."""
    monkeypatch.setattr(
        LlmProvider,
        "_stream_text_tokens_uncached",
        lambda self, system_prompt, user_prompt: iter(["one ", "two ", "three"]),
    )

    stream = provider.stream_text_tokens(
        system_prompt="system",
        user_prompt="user",
    )
    assert next(stream) == "one "
    stream.close()

    replayed = "".join(
        provider.stream_text_tokens(
            system_prompt="system",
            user_prompt="user",
        )
    )
    assert replayed == "one two three"
