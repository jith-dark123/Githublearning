.PHONY: install run test eval
install:
	pip install -r requirements.txt
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
test:
	pytest -q
eval:
	python -m eval.evaluate
