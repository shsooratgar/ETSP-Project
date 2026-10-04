.PHONY: install test lint data pilot analyze clean

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest

lint:
	ruff check src scripts tests

data:
	python scripts/download_data.py --cut val
	python scripts/download_data.py --cut test

# Smoke test on 20 examples with the smallest model before booking GPU time.
pilot:
	python scripts/run_inference.py --model Qwen/Qwen2.5-0.5B-Instruct --cut val --limit 20

analyze:
	python scripts/analyze.py --cut test

clean:
	rm -rf .pytest_cache **/__pycache__
