.PHONY: test coverage lint migrate

test:
	pytest services/ --tb=short

coverage:
	pytest services/ --cov=services --cov=lib --cov-report=term-missing

lint:
	ruff check services/ lib/

migrate:
	@for f in migrations/*.sql; do \
		echo "Applying $$f ..."; \
	done

etl-test:
	pytest services/etl/tests/ --tb=short

api-test:
	pytest services/api/tests/ --tb=short
