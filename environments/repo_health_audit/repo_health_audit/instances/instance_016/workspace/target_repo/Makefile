.PHONY: test-all coverage lint

test-all:
	python -m pytest services/ -v

coverage:
	python -m pytest services/ --cov=services --cov=libs --cov-report=term

lint:
	ruff check services/ libs/

install:
	pip install -r requirements.txt
