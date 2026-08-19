import json
import os
import re
from collections.abc import Iterator
from typing import Any


class BaseLlmClient:
    """One provider's wire protocol, and nothing else.

    A client knows how to ask its own API for JSON and for streamed prose. It
    does not cache, does not parse, and does not know another provider exists —
    everything shared lives above it in `LlmProvider`.
    """

    MODEL_NAME: str

    # A max-tokens cap on the answer, because output tokens are the priciest and the only
    # unbounded part of a request — the prompt is capped by the retrieved chunks, but a
    # rambling model isn't capped by anything else.
    # A truncation guard, not a budget: only emitted tokens are billed, so the cap
    # costs nothing until it bites. Gemini draws thinking from this same pool, so a
    # tight cap spends the budget reasoning and cuts the answer mid-sentence.
    _MAX_OUTPUT_TOKENS: int = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "2000"))

    def create_json_response(self, prompt: str, json_schema: dict[str, Any]) -> str | None:
        """The model's raw reply to a schema-constrained prompt, still unparsed."""
        raise NotImplementedError

    def stream_text_tokens(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Answer text as this provider hands it over."""
        raise NotImplementedError

    @staticmethod
    def _required_key(variable_name: str) -> str:
        value = os.environ.get(variable_name)
        if not value:
            raise RuntimeError(f"{variable_name} is not set. Add it to .env.")
        return value


class GeminiClient(BaseLlmClient):
    MODEL_NAME: str = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

    # `low`, not `minimal`: the SDK's type accepts minimal but gemini-3.7-flash
    # rejects it outright with a 400, so it is not a safe default to ship.
    _THINKING_LEVEL: str = os.environ.get("GEMINI_THINKING_LEVEL", "low")

    def create_json_response(self, prompt: str, json_schema: dict[str, Any]) -> str | None:
        return self._interaction(
            prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": json_schema,
            },
            generation_config=self._generation_config(),
        ).output_text

    def stream_text_tokens(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        yield self._interaction(
            prompt=user_prompt,
            system_instruction=system_prompt,
            generation_config=self._generation_config(self._MAX_OUTPUT_TOKENS),
        ).output_text

    def _generation_config(self, max_output_tokens: int | None = None) -> dict[str, Any]:
        config: dict[str, Any] = {"thinking_level": self._THINKING_LEVEL}
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens
        return config

    def _interaction(self, prompt: str, **options):
        """Bind the client to a name — `genai.Client().interactions.create(...)` fails."""
        self._required_key("GEMINI_API_KEY")
        from google import genai

        client = genai.Client()
        return client.interactions.create(model=self.MODEL_NAME, input=prompt, **options)


class OpenRouterClient(BaseLlmClient):
    MODEL_NAME: str = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    _BASE_URL: str = "https://openrouter.ai/api/v1"

    # The openai SDK defaults to a 600s timeout and 2 retries, so a stuck call
    # can block a worker for 30 minutes and a failing one quietly bills you three
    # times for the same batch, so retries are capped at one.
    _MAX_RETRIES: int = 1

    # The timeout had to go above 120s because reasoning models spend a long time
    # thinking before emitting anything — real batches take ~170s, so 120s made every
    # batch time out and look like a hang.
    _REQUEST_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))

    # Off by default, and not merely low. Neither job here is a reasoning
    # problem — one transcribes fields off a CV, the other summarises retrieved
    # text — and this model spends the budget anyway: a single-CV extraction at
    # `low` came back with finish_reason='error' having burnt 1,534 tokens, all
    # of them reasoning, and no content at all. Set it to low/medium/high to
    # turn thinking back on for a model that needs it.
    _REASONING_EFFORT: str = os.environ.get("OPENROUTER_REASONING_EFFORT", "off")

    _REASONING_OFF: frozenset[str] = frozenset({"off", "none", "no", "0", "false", ""})

    def create_json_response(self, prompt: str, json_schema: dict[str, Any]) -> str | None:
        completion = self._client().chat.completions.create(
            model=self.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": json_schema},
            },
            extra_body={"reasoning": self._reasoning_option()},
        )
        return self._get_required_content(completion)

    def stream_text_tokens(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        stream = self._client().chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._MAX_OUTPUT_TOKENS,
            stream=True,
            # Left on, this model streams its deliberation as ordinary content:
            # the reader watches it talk itself through the answer and the output
            # budget is gone before the answer starts. `exclude` is not enough —
            # it only hides reasoning that arrives in its own field. Measured on
            # one question: off returns 1.6k characters of answer where low
            # returned 9k characters of thinking.
            extra_body={"reasoning": self._reasoning_option()},
        )
        for stream_chunk in stream:
            token = stream_chunk.choices[0].delta.content
            if token:
                yield token

    @classmethod
    def _reasoning_option(cls) -> dict[str, Any]:
        """Disabled, or an effort level with the thinking kept out of the content."""
        if cls._REASONING_EFFORT.strip().lower() in cls._REASONING_OFF:
            return {"enabled": False}
        return {"effort": cls._REASONING_EFFORT, "exclude": True}

    def _client(self):
        import openai

        return openai.OpenAI(
            base_url=self._BASE_URL,
            api_key=self._required_key("OPENROUTER_API_KEY"),
            timeout=self._REQUEST_TIMEOUT_SECONDS,
            max_retries=self._MAX_RETRIES,
        )

    @staticmethod
    def _get_required_content(completion) -> str:
        choice = completion.choices[0]
        if choice.message.content:
            return choice.message.content

        usage = getattr(completion, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        raise RuntimeError(
            f"{completion.model} returned no content "
            f"(finish_reason={choice.finish_reason!r}, "
            f"output_tokens={getattr(usage, 'completion_tokens', '?')}, "
            f"reasoning_tokens={getattr(details, 'reasoning_tokens', '?')}). "
            "Usually the output budget went on reasoning — lower BATCH_SIZE or "
            "pick a model with more headroom."
        )


class LlmProvider:
    _CLIENTS: dict[str, type[BaseLlmClient]] = {
        "gemini": GeminiClient,
        "openrouter": OpenRouterClient,
    }

    _DEFAULT_PROVIDER: str = "openrouter"

    def _active_provider(self) -> str:
        return os.environ.get("TEXT_PROVIDER", self._DEFAULT_PROVIDER).strip().lower()

    def _client(self) -> BaseLlmClient:
        provider = self._active_provider()
        client_class = self._CLIENTS.get(provider)
        if client_class is None:
            raise RuntimeError(self._unknown_provider_message(provider))
        return client_class()

    def generate_json_object(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        client = self._client()
        return self._parse_json(
            text=client.create_json_response(
                prompt=prompt,
                json_schema=json_schema,
            ),
            model_name=client.MODEL_NAME,
        )

    def stream_text_tokens(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        yield from self._client().stream_text_tokens(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    # ```json … ``` — a model asked for JSON will sometimes hand back a Markdown
    # code block instead, because that is how JSON appears in its training data.
    _FENCED_JSON = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

    @classmethod
    def _parse_json(cls, text: str | None, model_name: str) -> dict[str, Any]:
        if text is None or not text.strip():
            raise RuntimeError(f"{model_name} returned an empty response where JSON was required.")

        fenced = cls._FENCED_JSON.match(text)
        candidate = fenced.group(1) if fenced else text

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as invalid:
            raise RuntimeError(
                f"{model_name} returned text that is not JSON ({invalid.msg}). "
                f"It began: {candidate.strip()[:200]!r}"
            ) from invalid

    @staticmethod
    def _unknown_provider_message(provider: str) -> str:
        return f"TEXT_PROVIDER must be 'gemini' or 'openrouter', got {provider!r}"
