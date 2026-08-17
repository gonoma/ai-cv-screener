import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..data.database import Database
from ..data.models import ROUTES, ChatRequest, QueryRoute
from ..domain.query import AnswerGenerator, ContextRetriever, QueryRouter

router = APIRouter()
database = Database()

query_router = QueryRouter()
answer_generator = AnswerGenerator()


def _server_sent_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        with database.open_connection() as connection:
            if database.count_ingested_rows(connection)["candidates"] == 0:
                yield _server_sent_event(
                    {
                        "type": "token",
                        "text": "No CVs have been ingested yet. Run `make ingest`.",
                    }
                )
                yield _server_sent_event({"type": "done"})
                return

            route_was_forced = request.route in ROUTES
            route = (
                QueryRoute(route=request.route)
                if route_was_forced
                else query_router.classify_question(request.question)
            )
            context = ContextRetriever(connection).retrieve_context(request.question, route)

            # Meta first, so the UI can label the answer before a token arrives.
            yield _server_sent_event(
                {
                    "type": "meta",
                    "route": route.route,
                    "forced": route_was_forced,
                    "sources": context.source_files,
                }
            )
            for token in answer_generator.stream_grounded_answer(request.question, context):
                yield _server_sent_event({"type": "token", "text": token})
            yield _server_sent_event({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
