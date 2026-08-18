.PHONY: help install generate db api ui ingest down test test-fast lint format

COUNT ?= 30
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

api:       ## FastAPI on :8000 with reload; needs `make db` first
	.venv/bin/uvicorn backend.main:app --reload --port 8000

ui:        ## Vite on :5173, proxying /api to the backend; needs `make api`
	cd frontend && npm run dev

ingest:    ## Reads data/cvs into the database; needs `make api` running
	curl -sS -X POST http://localhost:8000/ingest | python3 -m json.tool

down:      ## Stops Postgres. `docker compose down -v` also drops the data
	docker compose down

test:      ## Generation and backend suites
	.venv/bin/pytest

test-fast: ## Skips the tests that read the generated corpus in data/
	.venv/bin/pytest -m "not integration"

lint:      ## ruff check
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format:    ## ruff fix + format
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .
