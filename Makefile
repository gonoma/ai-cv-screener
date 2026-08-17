.PHONY: test lint format generate

COUNT ?= 30

test:
	pytest data_generation

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

generate:
	python -m data_generation.run --count $(COUNT)
