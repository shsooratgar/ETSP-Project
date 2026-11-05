.PHONY: install test lint data pilot experiments analyze clean

install:
	pip install -r requirements.txt
	pip install --no-deps summac==0.0.4
	pip install -e .

test:
	pytest

lint:
	ruff check src scripts tests

data:
	python scripts/download_data.py --cut val
	python scripts/download_data.py --cut test

# Smoke test on 40 examples with the smallest model.
pilot:
	python scripts/run_inference.py --model Qwen/Qwen2.5-0.5B-Instruct --cut val --limit 40 --tag pilot

# Every model, both cuts, plus prompt variants. Hours on a laptop; resumable.
experiments:
	PYTHON=python scripts/run_all.sh

analyze:
	python scripts/analyze.py

clean:
	rm -rf .pytest_cache **/__pycache__
