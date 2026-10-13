.PHONY: install train run test
install:
	pip install -r requirements.txt
train:
	python -m ml.train
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
test:
	pytest -q
