import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .. import REPO_ROOT


class LlmProvider:
    _GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
    _OPENROUTER_MODEL_NAME: str = os.environ.get(
        "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
    )
    _OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    _DEFAULT_PROVIDER: str = "openrouter"

    # The openai SDK defaults to a 600s timeout and 2 retries, so one stalled
    # extraction call can hold an ingestion worker for half an hour, and a
    # rate-limited one silently becomes three requests against a 50-per-day
    # free-tier budget. Measured latency is ~48s, so 120s is a wide margin.
    _REQUEST_TIMEOUT_SECONDS: float = 120.0
    _MAX_RETRIES: int = 1

    # Off unless LLM_CACHE is set. Replaying a stored answer is right while
    # tuning and wrong when serving: two users asking the same question would
    # get byte-identical responses forever, and a provider outage would look
    # like a working system.
    _CACHE_DIRECTORY: Path = REPO_ROOT / "data" / "llm_cache"

    def active_provider(self) -> str:
        return os.environ.get("TEXT_PROVIDER", self._DEFAULT_PROVIDER).strip().lower()

    def generate_json_object(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        cache_key = self._cache_key("json", prompt, json.dumps(json_schema, sort_keys=True))
        replayed = self._read_cache(cache_key)
        if replayed is not None:
            return json.loads(replayed)

        result = self._generate_json_object_uncached(prompt, json_schema)
        self._write_cache(cache_key, json.dumps(result))
        return result

    def stream_text_tokens(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield answer text as it arrives — except on Gemini, which yields once.

        The Gemini branch deliberately returns the whole answer as a single
        chunk rather than guessing at a streaming shape I have not verified.
        Callers cannot tell the difference: both paths are generators feeding
        the same SSE contract, so the only visible effect is that Gemini's
        tokens arrive together. A replayed answer behaves the same way.
        """
        cache_key = self._cache_key("text", system_prompt, user_prompt)
        replayed = self._read_cache(cache_key)
        if replayed is not None:
            yield replayed
            return

        spoken = []
        for token in self._stream_text_tokens_uncached(system_prompt, user_prompt):
            spoken.append(token)
            yield token

        # Only after the generator runs to completion. A consumer that abandons
        # the stream never reaches this line, so a half-written answer is not
        # stored and replayed as though it were the whole thing.
        self._write_cache(cache_key, "".join(spoken))

    def _generate_json_object_uncached(
        self, prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        provider = self.active_provider()
        if provider == "openrouter":
            completion = self._openrouter_client().chat.completions.create(
                model=self._OPENROUTER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "result", "strict": True, "schema": json_schema},
                },
            )
            return json.loads(completion.choices[0].message.content)

        if provider == "gemini":
            return json.loads(
                self._gemini_interaction(
                    prompt,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": json_schema,
                    },
                ).output_text
            )

        raise RuntimeError(self._unknown_provider_message(provider))

    def _stream_text_tokens_uncached(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        provider = self.active_provider()
        if provider == "openrouter":
            stream = self._openrouter_client().chat.completions.create(
                model=self._OPENROUTER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            for stream_chunk in stream:
                token = stream_chunk.choices[0].delta.content
                if token:
                    yield token
            return

        if provider == "gemini":
            yield self._gemini_interaction(f"{system_prompt}\n\n{user_prompt}").output_text
            return

        raise RuntimeError(self._unknown_provider_message(provider))

    def _cache_key(self, kind: str, *material: str) -> str:
        """Hash everything that determines the response, not just the question.

        The prompts carry the retrieved context and the system prompt, so a
        tuned prompt or a changed retrieval result produces a different key and
        misses — which is the only way a cache is safe to keep on while tuning
        the thing it is caching. The model name is in there for the same reason.
        """
        digest = hashlib.sha256("\x00".join([kind, self._model_name(), *material]).encode())
        return f"{kind}-{digest.hexdigest()[:32]}"

    def _model_name(self) -> str:
        return (
            self._GEMINI_MODEL_NAME
            if self.active_provider() == "gemini"
            else self._OPENROUTER_MODEL_NAME
        )

    def _read_cache(self, cache_key: str) -> str | None:
        if not self._caching_enabled():
            return None
        cache_file = self._CACHE_DIRECTORY / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        return json.loads(cache_file.read_text(encoding="utf-8"))["response"]

    def _write_cache(self, cache_key: str, response: str) -> None:
        if not self._caching_enabled():
            return
        self._CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        (self._CACHE_DIRECTORY / f"{cache_key}.json").write_text(
            json.dumps({"response": response}), encoding="utf-8"
        )

    @staticmethod
    def _caching_enabled() -> bool:
        return os.environ.get("LLM_CACHE", "").strip().lower() in {"1", "true", "yes"}

    def _openrouter_client(self):
        import openai

        return openai.OpenAI(
            base_url=self._OPENROUTER_BASE_URL,
            api_key=self._required_key("OPENROUTER_API_KEY"),
            timeout=self._REQUEST_TIMEOUT_SECONDS,
            max_retries=self._MAX_RETRIES,
        )

    def _gemini_interaction(self, prompt: str, **options):
        """Bind the client to a name — `genai.Client().interactions.create(...)` fails.

        `.interactions` does not keep its parent alive, so in a one-liner the
        Client is the last reference to itself: refcounting frees it the moment
        the attribute chain moves past it, closing the httpx pool it owns, and
        the request that follows dies on `Cannot send a request, as the client
        has been closed`. The local binding is the whole fix.
        """
        self._required_key("GEMINI_API_KEY")
        from google import genai

        client = genai.Client()
        return client.interactions.create(model=self._GEMINI_MODEL_NAME, input=prompt, **options)

    @staticmethod
    def _required_key(variable_name: str) -> str:
        value = os.environ.get(variable_name)
        if not value:
            raise RuntimeError(f"{variable_name} is not set. Add it to .env.")
        return value

    @staticmethod
    def _unknown_provider_message(provider: str) -> str:
        return f"TEXT_PROVIDER must be 'gemini' or 'openrouter', got {provider!r}"
