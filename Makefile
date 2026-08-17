.PHONY: db api ingest test lint format generate

COUNT ?= 30

db:      ## Postgres + pgvector in Docker, waits until it accepts connections
	docker compose up -d --wait

api:
	.venv/bin/uvicorn backend.main:app --reload --port 8000

ingest:
	curl -sS -X POST http://localhost:8000/ingest | python3 -m json.tool

test:
	pytest data_generation

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

generate:
	python -m data_generation.run --count $(COUNT)
