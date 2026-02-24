.PHONY: test coverage build clean

test:
	pytest orchestrator/tests/ -v
	cd engine && cargo test

coverage:
	pytest orchestrator/tests/ --cov=orchestrator --cov-report=term
	cd engine && cargo tarpaulin

build:
	cd engine && cargo build --release

clean:
	rm -rf engine/target
	find . -type d -name __pycache__ -exec rm -rf {} +
