# Makefile

.PHONY: install run debug clean lint
 
install:
	uv sync

run:
	@uv run python -m src

debug:
	@uv run python -m pdb -m src

clean: 
	rm -rf __pycache__ .mypy_cache src/__pycache__ .pytest_cache
	find . -name '*.pyc' -delete

lint: 
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs