.PHONY: setup run evaluate bertscore lint format clean

setup:
	uv sync

run: setup
	uv run python dataset.py
	uv run python models.py
	uv run python main.py
	uv run python evaluate.py
	uv run python bertscore.py

evaluate:
	uv run python evaluate.py

bertscore:
	uv run python bertscore.py

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

clean:
	rm -rf outputs/*.jsonl outputs/*.csv data/final_benchmark_50.csv
