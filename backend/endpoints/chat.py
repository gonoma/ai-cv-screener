import json
import re
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..data.database import Database
from ..data.models import ROUTES, ChatRequest
from ..domain.query import AnswerGenerator, ContextRetriever, QueryRouter

router = APIRouter()
database = Database()

# QueryRouter is built per request, not here: it reads the corpus vocabulary to
# route most questions without spending a call, so it needs the request's
# connection and should not outlive an ingest that changes what it caches.
answer_generator = AnswerGenerator()


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

            query_router = QueryRouter(connection)
            route_was_forced = request.route in ROUTES
            route = (
                query_router.force_route(
                    route_name=request.route,
                    question=request.question,
                )
                if route_was_forced
                else query_router.classify_question(request.question)
            )
            context = ContextRetriever(connection).retrieve_context(
                question=request.question,
                route=route,
            )

            # Meta first, so the UI can label the answer before a token arrives.
            yield _server_sent_event(
                {
                    "type": "meta",
                    "route": route.route,
                    "forced": route_was_forced,
                    "sources": context.source_files,
                }
            )
            # Anything raised from here on has to be reported *inside* the
            # stream.
            # The 200 and the headers are already on the wire, so an
            # exception escaping this generator just closes the connection: the
            # browser sees a stream that ended, `response.ok` was true, no error
            # is thrown anywhere, and the user gets a blank answer under a route
            # label with nothing to explain it. A rate limit is the likeliest
            # cause and the least deserving of looking like a silent failure.
            try:
                for token in answer_generator.stream_grounded_answer(
                    question=request.question,
                    context=context,
                ):
                    yield _server_sent_event({"type": "token", "text": token})
            except Exception as failure:
                yield _server_sent_event({"type": "error", "text": _readable(failure)})
            yield _server_sent_event({"type": "done"})

    # Server Sent Event (SSE), not JSON: the answer leaves as `data: {...}` frames over one open
    # response, so the user reads the first words while the model is still
    # writing the rest.
    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _readable(failure: Exception) -> str:
    """Say what went wrong in terms the person asking the question can act on."""
    text = str(failure)
    lowered = text.lower()

    if "429" in text or "resource_exhausted" in lowered or "too_many_requests" in lowered:
        seconds = re.search(r"retry in ([\d.]+)s", text, re.IGNORECASE)
        wait = f" Try again in about {round(float(seconds.group(1)))}s." if seconds else ""
        return f"Free-tier request limit reached — the provider is rate limiting this key.{wait}"

    if any(clue in text for clue in ("401", "403")) or "api key" in lowered:
        return "The provider rejected the API key. Check GEMINI_API_KEY or OPENROUTER_API_KEY."

    if "timeout" in lowered or "timed out" in lowered:
        return "The model took too long to answer and the request timed out."

    if "connect" in lowered or "network" in lowered or "unreachable" in lowered:
        return "Could not reach the model provider — check the network and try again."

    if "no content" in lowered:
        return "The model returned an empty answer. Lower BATCH_SIZE or try a different model."

    first_line = text.strip().splitlines()[0] if text.strip() else "no detail"
    return f"Unknown error: {type(failure).__name__}: {first_line[:300]}"


def _server_sent_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
