.PHONY: test lint format

test:
	pytest data_generation

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .