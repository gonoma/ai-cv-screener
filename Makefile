.PHONY: help install check-tools guard-venv generate regenerate db api ui ingest down evals eval-extraction test test-fast lint format

COUNT ?= 30

# The interpreter that builds the venv. Everything afterwards runs the venv's own
# .venv/bin/* binaries, never this one.
PYTHON ?= python3
VENV := $(abspath .venv)

# The backend URL lives in .env, which make does not read on its own. An
# environment variable still wins over the file, and the fallback matches
# .env.example.
BACKEND_API ?= $(shell sed -n 's/^BACKEND_API=//p' .env 2>/dev/null | tail -1)
BACKEND_API := $(or $(BACKEND_API),http://localhost:8000)
# uvicorn binds a port, not a URL: the last :-separated field of it.
BACKEND_PORT := $(lastword $(subst :, ,$(BACKEND_API)))
.DEFAULT_GOAL := help

help:      ## This list
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /|/' | column -t -s '|'

# A recipe cannot activate anything in the shell that called make: every line runs
# in its own subshell, and it dies with the line. So install ends by handing back an
# interactive shell that *is* activated — `exit` returns to the one you started in.
install:   ## Fresh clone to ready: tool checks, venv, deps, .env, then a venv shell
	@$(MAKE) --no-print-directory check-tools
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	cd frontend && npm install
	@test -f .env || (cp .env.example .env && echo "wrote .env from .env.example")
	@echo
	@grep -qE '^(GEMINI|OPENROUTER)_API_KEY=.+' .env \
		|| echo "NEXT: open .env and paste a GEMINI_API_KEY (or OPENROUTER_API_KEY)"
	@echo "THEN: make db && make api        # terminal 1"
	@echo "      make ingest && make ui     # terminal 2, then open http://localhost:5173"
	@echo
	@if [ -t 0 ]; then \
		echo "Dropping you into the venv. Type 'exit' to leave it."; \
		echo; \
		. .venv/bin/activate && exec $$SHELL; \
	else \
		echo "Not a terminal, so nothing to drop into. Activate with: source .venv/bin/activate"; \
	fi

# Fails early and says what is missing, rather than halfway through pip.
check-tools:
	@$(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3, 12))' \
		|| { echo "need Python 3.12+ (found: $$($(PYTHON) -V 2>&1)) — set PYTHON=/path/to/python3.12"; exit 1; }
	@command -v node >/dev/null \
		|| { echo "need Node 20+ — https://nodejs.org"; exit 1; }
	@node -e 'process.exit(+process.versions.node.split(".")[0] >= 20 ? 0 : 1)' \
		|| { echo "need Node 20+ (found: $$(node -v))"; exit 1; }
	@docker info >/dev/null 2>&1 \
		|| echo "note: Docker is not running — start it before \`make db\`"

# Every target that runs Python refuses to start unless this repo's venv is the
# active one. The recipes call .venv/bin/* directly, so they would technically
# work without it — the guard is here so a shell that has not activated is told
# so, instead of a developer half-working outside the environment they think
# they are in. db, down and ui are exempt: Docker and npm, no Python involved.
guard-venv:
	@[ "$(VIRTUAL_ENV)" = "$(VENV)" ] || { \
		echo "not inside this repo's venv (VIRTUAL_ENV=$${VIRTUAL_ENV:-unset})."; \
		echo "  activate it:  source .venv/bin/activate"; \
		echo "  or set it up: make install"; \
		exit 1; \
	}

generate:  guard-venv ## Rebuilds the corpus in data/: COUNT=30 CVs, needs a key in .env
	.venv/bin/python -m data_generation.run --count $(COUNT)

regenerate: guard-venv ## Rebuilds one CV from scratch, ignoring the cache; replaces data/
	.venv/bin/python -m data_generation.run --count 1 --force

db:        ## Postgres + pgvector in Docker, waits until it accepts connections
	docker compose up -d --wait

api:       guard-venv ## FastAPI on BACKEND_API's port with reload; needs `make db` first
	.venv/bin/uvicorn backend.main:app --reload --port $(BACKEND_PORT)

ui:        ## Vite on :5173, proxying /api to the backend; needs `make api`
	cd frontend && npm run dev

ingest:    guard-venv ## Reads data/cvs into the database; needs `make api` running
	curl -sS -X POST $(BACKEND_API)/ingest | .venv/bin/python -m json.tool

down:      ## Stops Postgres. `docker compose down -v` also drops the data
	docker compose down

evals:     guard-venv ## Question suite against a running backend; costs LLM calls
	.venv/bin/python -m evals.run

eval-extraction: guard-venv ## Scores ingestion against the answer key; no API, no calls
	.venv/bin/python -m evals.extraction

test:      guard-venv ## Generation and backend suites
	.venv/bin/pytest

test-fast: guard-venv ## Skips the tests that read the generated corpus in data/
	.venv/bin/pytest -m "not integration"

lint:      guard-venv ## ruff check
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format:    guard-venv ## ruff fix + format
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .
