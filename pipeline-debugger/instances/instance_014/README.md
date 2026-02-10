# Pipeline Debugger Instance

This instance contains a small ETL pipeline:

1. Extract raw orders and customers from `data/`
2. Transform orders into a cleaned analytics table
3. Load outputs into `outputs/`

Expected behavior:
- `python run_pipeline.py` exits with status 0
- `outputs/fact_orders.csv` exists and matches `expected_schema.json`
- malformed rows are captured in `outputs/rejected_rows.csv` instead of crashing the pipeline
- pipeline output is deterministic across repeated runs
