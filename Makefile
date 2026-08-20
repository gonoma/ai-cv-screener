.PHONY: help install generate db api ui ingest down evals eval-extraction test test-fast lint format

COUNT ?= 30

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

install:   ## venv + Python deps, frontend deps, and .env from the template
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	cd frontend && npm install
	test -f .env || cp .env.example .env

generate:  ## Rebuilds the corpus in data/: COUNT=30 CVs, needs a key in .env
	.venv/bin/python -m data_generation.run --count $(COUNT)

db:        ## Postgres + pgvector in Docker, waits until it accepts connections
	docker compose up -d --wait

api:       ## FastAPI on BACKEND_API's port with reload; needs `make db` first
	.venv/bin/uvicorn backend.main:app --reload --port $(BACKEND_PORT)

ui:        ## Vite on :5173, proxying /api to the backend; needs `make api`
	cd frontend && npm run dev

ingest:    ## Reads data/cvs into the database; needs `make api` running
	curl -sS -X POST $(BACKEND_API)/ingest | python3 -m json.tool

down:      ## Stops Postgres. `docker compose down -v` also drops the data
	docker compose down

evals:     ## Question suite against a running backend; costs LLM calls
	.venv/bin/python -m evals.run

eval-extraction: ## Scores ingestion against the answer key; no API, no calls
	.venv/bin/python -m evals.extraction

test:      ## Generation and backend suites
	.venv/bin/pytest

test-fast: ## Skips the tests that read the generated corpus in data/
	.venv/bin/pytest -m "not integration"

lint:      ## ruff check
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format:    ## ruff fix + format
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .
