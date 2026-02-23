from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Mutation:
    name: str
    category: str
    difficulty: str
    file_path: str
    find_text: str
    replace_text: str
    description: str


MUTATIONS: dict[str, Mutation] = {
    "schema_drift": Mutation(
        name="schema_drift",
        category="Schema drift",
        difficulty="Easy",
        file_path="pipeline/transform.py",
        find_text='JOIN_KEY = "customer_id"',
        replace_text='JOIN_KEY = "cust_id"',
        description="Join key uses outdated upstream column name.",
    ),
    "type_coercion": Mutation(
        name="type_coercion",
        category="Type coercion",
        difficulty="Easy",
        file_path="pipeline/transform.py",
        find_text='amount = float(raw["amount"])',
        replace_text='amount = int(float(raw["amount"]))',
        description="Amount is truncated to integer, corrupting numeric precision.",
    ),
    "join_logic": Mutation(
        name="join_logic",
        category="Join logic",
        difficulty="Medium",
        file_path="pipeline/transform.py",
        find_text="merged = cleaned_df.merge(customers_df, how=JOIN_HOW, on=JOIN_KEY, sort=False)",
        replace_text='merged = cleaned_df.merge(customers_df, how=JOIN_HOW, left_on="order_id", right_on="customer_id", sort=False)',
        description="Rows are joined on the wrong key pair.",
    ),
    "aggregation": Mutation(
        name="aggregation",
        category="Aggregation",
        difficulty="Medium",
        file_path="pipeline/transform.py",
        find_text='merged["discount"] = (merged["amount"] * discount_rates).round(2)',
        replace_text='merged["discount"] = (merged["amount"] * 0.01).round(2)',
        description="Discount ignores segment and applies a flat 1% rate.",
    ),
    "ordering_dependency": Mutation(
        name="ordering_dependency",
        category="Ordering dependency",
        difficulty="Medium",
        file_path="run_pipeline.py",
        find_text="EXECUTE_TRANSFORM_BEFORE_LOAD = True",
        replace_text="EXECUTE_TRANSFORM_BEFORE_LOAD = False",
        description="Load step runs before transform output is materialized.",
    ),
    "silent_data_loss": Mutation(
        name="silent_data_loss",
        category="Silent data loss",
        difficulty="Hard",
        file_path="pipeline/transform.py",
        find_text="DROP_VALID_ROWS_BELOW = None",
        replace_text="DROP_VALID_ROWS_BELOW = 300.0",
        description="Valid rows below threshold are silently dropped.",
    ),
    "config_env": Mutation(
        name="config_env",
        category="Config/env",
        difficulty="Easy-Medium",
        file_path="pipeline/extract.py",
        find_text='ORDERS_DELIMITER = ","',
        replace_text='ORDERS_DELIMITER = ";"',
        description="Reader delimiter is configured incorrectly.",
    ),
    "error_handling": Mutation(
        name="error_handling",
        category="Error handling",
        difficulty="Medium",
        file_path="pipeline/transform.py",
        find_text="CRASH_ON_BAD_ROW = False",
        replace_text="CRASH_ON_BAD_ROW = True",
        description="Malformed row now crashes the entire pipeline.",
    ),
    "wrong_output_path": Mutation(
        name="wrong_output_path",
        category="Config/env",
        difficulty="Easy",
        file_path="pipeline/load.py",
        find_text='OUTPUT_FILENAME = "fact_orders.csv"',
        replace_text='OUTPUT_FILENAME = "fact_orders_final.csv"',
        description="Load step writes output to an unexpected filename.",
    ),
}


def apply_mutations(
    instance_dir: Path, mutation_names: Iterable[str]
) -> list[Mutation]:
    """Apply selected mutations to copied template files."""
    applied: list[Mutation] = []
    for mutation_name in mutation_names:
        mutation = MUTATIONS[mutation_name]
        target = instance_dir / mutation.file_path
        text = target.read_text()
        if mutation.find_text not in text:
            raise ValueError(
                f"Could not apply mutation '{mutation.name}' to {mutation.file_path}."
            )
        target.write_text(text.replace(mutation.find_text, mutation.replace_text, 1))
        applied.append(mutation)
    return applied
