import json

from fastapi.testclient import TestClient

from backend.data.models import QueryRoute, RetrievedContext
from backend.endpoints import chat as chat_endpoint
from backend.main import app


class FakeResult:
    def fetchone(self) -> tuple[int, int]:
        return (3, 12)  # candidates, chunks


class FakeConnection:
    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def execute(self, *args, **kwargs) -> FakeResult:
        return FakeResult()


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_chat_streams_meta_then_tokens_then_done(monkeypatch) -> None:
    monkeypatch.setattr(chat_endpoint.database, "open_connection", lambda: FakeConnection())
    # Patched on the class, not on a module-level instance: the router is built
    # per request now, because it reads the corpus vocabulary off the request's
    # own connection to route most questions without calling a model.
    monkeypatch.setattr(
        chat_endpoint.QueryRouter,
        "classify_question",
        lambda self, question: QueryRoute(route="structured"),
    )
    monkeypatch.setattr(
        chat_endpoint.ContextRetriever,
        "retrieve_context",
        lambda self, question, route: RetrievedContext(text="ctx", source_files=["ana-silva.pdf"]),
    )
    monkeypatch.setattr(
        chat_endpoint.answer_generator,
        "stream_grounded_answer",
        lambda question, context: iter(["Ana ", "Silva."]),
    )

    response = TestClient(app).post("/chat", json={"question": "who knows Python?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    assert events[0] == {
        "type": "meta",
        "route": "structured",
        "forced": False,
        "sources": ["ana-silva.pdf"],
    }
    assert [e["text"] for e in events if e["type"] == "token"] == ["Ana ", "Silva."]
    assert events[-1] == {"type": "done"}


def test_route_override_forces_the_naive_path(monkeypatch) -> None:

    def must_not_run(self, question: str) -> QueryRoute:
        raise AssertionError("classifier must not run when a route is forced")

    monkeypatch.setattr(chat_endpoint.database, "open_connection", lambda: FakeConnection())
    monkeypatch.setattr(chat_endpoint.QueryRouter, "classify_question", must_not_run)
    monkeypatch.setattr(
        chat_endpoint.ContextRetriever,
        "retrieve_context",
        lambda self, question, route: RetrievedContext(text="ctx", source_files=[]),
    )
    monkeypatch.setattr(
        chat_endpoint.answer_generator,
        "stream_grounded_answer",
        lambda question, context: iter(["x"]),
    )

    events = _events(
        TestClient(app).post("/chat", json={"question": "q", "route": "semantic"}).text
    )
    assert events[0]["route"] == "semantic"
    assert events[0]["forced"] is True


def test_all_three_endpoints_are_registered() -> None:
    spec = TestClient(app).get("/openapi.json").json()["paths"]
    assert set(spec) == {"/health", "/ingest", "/chat"}


def test_a_provider_failure_mid_stream_is_reported_in_band(monkeypatch) -> None:
    """After the first byte there is no status code left to fail with.

    The 200 and the headers are already sent, so an exception escaping the
    generator merely closes the connection: fetch() resolved ok, nothing throws,
    and the UI shows a route label above a blank answer. A rate limit is the
    likeliest cause and the least deserving of looking like silence.
    """
    monkeypatch.setattr(chat_endpoint.database, "open_connection", lambda: FakeConnection())
    monkeypatch.setattr(
        chat_endpoint.QueryRouter,
        "classify_question",
        lambda self, question: QueryRoute(route="structured"),
    )
    monkeypatch.setattr(
        chat_endpoint.ContextRetriever,
        "retrieve_context",
        lambda self, question, route: RetrievedContext(text="ctx", source_files=[]),
    )

    def rate_limited(question, context):
        raise RuntimeError("Error code: 429 quota exceeded. Please retry in 21.5s.")
        yield  # pragma: no cover - generator, never reached

    monkeypatch.setattr(chat_endpoint.answer_generator, "stream_grounded_answer", rate_limited)

    events = _events(TestClient(app).post("/chat", json={"question": "q"}).text)
    assert [e["type"] for e in events] == ["meta", "error", "done"]
    assert "rate limiting" in events[1]["text"]
    assert "22s" in events[1]["text"]
