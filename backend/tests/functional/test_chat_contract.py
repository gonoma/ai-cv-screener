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
    monkeypatch.setattr(
        chat_endpoint.query_router, "classify_question", lambda q: QueryRoute(route="structured")
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

    def must_not_run(question: str) -> QueryRoute:
        raise AssertionError("classifier must not run when a route is forced")

    monkeypatch.setattr(chat_endpoint.database, "open_connection", lambda: FakeConnection())
    monkeypatch.setattr(chat_endpoint.query_router, "classify_question", must_not_run)
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
