"""Asks the running backend every question in the suite and prints a results table.

This is the half of the evaluation that costs money: each case sends a real
question to a real language model. Start the system first, then run it:

    make db && make api && make ingest     # then
    make evals

Two things worth knowing before reading the code.

*It talks to the backend over HTTP*, instead of importing the modules and
calling them directly. That is on purpose: the things being measured — picking a
route, retrieving CVs, building the prompt, streaming the reply — all happen
inside the /chat endpoint, and a test that imported the pieces would skip the
endpoint that actually serves users.

*There are three exit codes, not two.* 0 = everything passed, 1 = the system
answered something wrongly, 2 = the cases could not be run at all (backend down,
API key out of quota). The third one matters because "we could not test it" is
not the same as "it is broken", and CI should not page anyone for a quota reset.
"""

import json
import os
import sys
import urllib.error
import urllib.request

from .cases import AnswerKey, Case, CaseBuilder
from .grade import Grader


class BackendClient:
    """Talks to the running backend — the system under test.

    The address comes from the BACKEND_API environment variable, the same one
    the Makefile uses to start the server and the frontend uses to proxy to it,
    so pointing the suite at a different machine means editing .env and nothing
    else.
    """

    _BASE_URL: str = os.environ.get("BACKEND_API", "http://localhost:8000").rstrip("/")

    # Answering a question means calling a language model, which can take a
    # couple of minutes on a free tier. Checking /health only opens a socket, so
    # a long timeout there would just delay the message telling you it is down.
    _ANSWER_TIMEOUT_SECONDS: float = 180.0
    _HEALTH_TIMEOUT_SECONDS: float = 10.0

    # The backend streams its reply as Server-Sent Events: a series of lines that
    # each look like `data: {...json...}`. This is that prefix.
    _EVENT_PREFIX: str = "data: "

    @property
    def base_url(self) -> str:
        return self._BASE_URL

    def unreachable_reason(self) -> str | None:
        """Returns None if the backend answers, or the error text if it does not."""
        try:
            urllib.request.urlopen(
                f"{self._BASE_URL}/health", timeout=self._HEALTH_TIMEOUT_SECONDS
            ).read()
        except (urllib.error.URLError, TimeoutError) as error:
            return str(error)
        return None

    def ask(self, question: str) -> tuple[str | None, list[str], str]:
        """Send one question to /chat and read the streamed reply to the end.

        Returns three things:
          - route:   which strategy the backend used ("structured", "profile",
                     "semantic")
          - sources: the CV filenames it used as evidence
          - answer:  the reply text, reassembled from the stream

        The reply does not arrive in one piece. It comes as a "meta" event first
        (route and sources), then one "token" event per fragment of text, which
        is what makes the UI type the answer out word by word. Here we simply
        collect them all and join them.
        """
        request = urllib.request.Request(
            f"{self._BASE_URL}/chat",
            data=json.dumps({"question": question}).encode(),
            headers={"content-type": "application/json"},
        )
        route, sources, answer_parts = None, [], []
        with urllib.request.urlopen(request, timeout=self._ANSWER_TIMEOUT_SECONDS) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith(self._EVENT_PREFIX):
                    continue
                event = json.loads(line[len(self._EVENT_PREFIX) :])
                if event["type"] == "meta":
                    route, sources = event["route"], event["sources"]
                elif event["type"] == "token":
                    answer_parts.append(event["text"])
                elif event["type"] == "error":
                    # Errors arrive inside the stream, not as an HTTP error code:
                    # the 200 OK was already sent before anything went wrong. If
                    # we ignored these events, a failed answer would look like a
                    # very short one, and the case would fail for the wrong
                    # reason.
                    raise RuntimeError(event["text"])
        return route, sources, "".join(answer_parts)


class QuestionSuite:
    """Runs every case: asks the question, marks the answer, prints the summary."""

    _EXIT_ALL_PASSED: int = 0
    _EXIT_ANSWERED_WRONGLY: int = 1
    _EXIT_COULD_NOT_RUN: int = 2

    # Width of the ---- separator lines in the printed table.
    _RULE_WIDTH: int = 96

    def __init__(self) -> None:
        self._answer_key = AnswerKey()
        self._case_builder = CaseBuilder(self._answer_key)
        self._cases: list[Case] = self._case_builder.build()
        self._grader = Grader(self._answer_key.candidate_names)
        self._backend_client = BackendClient()
        # Counted separately all the way through: a failure is a wrong answer,
        # an error is a question we never got an answer to.
        self._failures = 0
        self._errors = 0

    def run(self) -> int:
        # Worked out before the first request so that a gap in the corpus is
        # still reported at the end, even if the run went badly for other
        # reasons.
        missing_kinds = self._case_builder.missing_kinds(self._cases)

        unreachable_reason = self._backend_client.unreachable_reason()
        if unreachable_reason:
            print(f"backend not reachable at {self._backend_client.base_url}: {unreachable_reason}")
            print("start it with `make db && make api && make ingest`")
            return self._EXIT_COULD_NOT_RUN

        print(f"{len(self._cases)} cases over {len(self._answer_key)} candidates\n")
        print(f"{'kind':14} {'route':11} {'result':7} detail")
        print("-" * self._RULE_WIDTH)

        for case in self._cases:
            self._ask_and_grade(case)

        print("-" * self._RULE_WIDTH)
        self._report(missing_kinds)
        return self._exit_code()

    def _ask_and_grade(self, case: Case) -> None:
        """Ask one question, mark the answer, print its row of the table."""
        try:
            route, sources, answer = self._backend_client.ask(case.question)
        except Exception as error:
            # One dead request should not take down the whole run: the remaining
            # cases still have something to tell us, so this is recorded as an
            # error and the loop carries on.
            print(f"{case.kind:14} {'-':11} {'ERROR':7} {type(error).__name__}: {error}")
            self._errors += 1
            return

        # Two independent judgements: was the answer right, and was it produced
        # the right way. A case passes only if both are.
        answer_ok, detail = self._grader.grade(case, answer, sources)
        route_ok, route_detail = self._grader.grade_route(case, route)
        passed = answer_ok and route_ok
        self._failures += not passed

        note = detail
        if not route_ok:
            # Keep both explanations when both went wrong, but do not print a
            # cheerful "ok" next to a routing failure.
            note = f"{note}; {route_detail}" if note != "ok" else route_detail
        print(f"{case.kind:14} {str(route):11} {'PASS' if passed else 'FAIL':7} {note}")

    def _report(self, missing_kinds: list[str]) -> None:
        passed_cases = len(self._cases) - self._failures - self._errors
        print(f"{passed_cases}/{len(self._cases)} passed")

        if self._errors:
            # Reported apart from failures on purpose. A rate-limited API key or
            # a provider outage tells you nothing about whether the system
            # answers correctly, and calling it a regression sends someone
            # hunting for a bug that tomorrow's quota reset would have "fixed".
            print(f"{self._errors} case(s) could not run — the request never completed.")

        if missing_kinds:
            # This is a gap in the corpus, not a fault in the system, so it is
            # printed loudly but does not fail the run — a suite that always
            # exits non-zero is a suite people stop reading.
            print()
            for kind in missing_kinds:
                print(f"NOT COVERED: no `{kind}` case exists in this corpus.")
            if "ambiguous" in missing_kinds:
                print("  No two candidates share a name, so the disambiguation path is untested.")

    def _exit_code(self) -> int:
        """Wrong answers beat unrun cases: a real failure is the more useful signal."""
        if self._failures:
            return self._EXIT_ANSWERED_WRONGLY
        return self._EXIT_COULD_NOT_RUN if self._errors else self._EXIT_ALL_PASSED


def main() -> int:
    return QuestionSuite().run()


if __name__ == "__main__":
    sys.exit(main())
