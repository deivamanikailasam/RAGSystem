.PHONY: install run test eval lint docker fmt clean

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

eval:
	python eval/run_eval.py

docker:
	docker compose up --build

clean:
	rm -rf data __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
