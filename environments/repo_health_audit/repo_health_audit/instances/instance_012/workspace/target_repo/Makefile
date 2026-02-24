test:
	python -m pytest specs/ -v

coverage:
	python -m pytest specs/ --cov=taskrunner --cov-report=term-missing

lint:
	ruff check taskrunner/ specs/
