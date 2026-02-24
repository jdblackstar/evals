from __future__ import annotations

import os

import pandas as pd
import yaml

from .io import read_input, write_output
from .transforms import normalize_columns, filter_rows

CONFIG_PATH = os.getenv("PROCESSOR_CONFIG", "pipeline_config.yaml")


def load_config() -> dict:
    """Load pipeline configuration from YAML, with env var overrides."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    if os.getenv("PROCESSOR_INPUT_PATH"):
        cfg["input_path"] = os.getenv("PROCESSOR_INPUT_PATH")
    if os.getenv("PROCESSOR_OUTPUT_PATH"):
        cfg["output_path"] = os.getenv("PROCESSOR_OUTPUT_PATH")
    return cfg


def run_pipeline(input_path: str, output_path: str) -> int:
    """Run the full data processing pipeline. Returns row count."""
    df = read_input(input_path)
    df = normalize_columns(df)
    df = filter_rows(df)
    write_output(df, output_path)
    return len(df)
