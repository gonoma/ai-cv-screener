.PHONY: db api ingest test test-fast lint format generate

COUNT ?= 30

db:      ## Postgres + pgvector in Docker, waits until it accepts connections
	docker compose up -d --wait

api:     ## FastAPI on :8000 with reload; needs `make db` first
	.venv/bin/uvicorn backend.main:app --reload --port 8000

ingest:  ## Reads data/cvs into the database; needs `make api` running
	curl -sS -X POST http://localhost:8000/ingest | python3 -m json.tool

test:
	pytest

test-fast: ## Skips the tests that read the generated corpus in data/
	pytest -m "not integration"

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

generate:
	python -m data_generation.run --count $(COUNT)
