"""
Persistence and export utilities for experiment runs.

Handles saving/loading runs and generating human-friendly reports.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.logging import ExperimentRun

_CSV_RESERVED_FIELDS: set[str] = {"index", "prompt", "response", "label"}


def _flatten_result_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a single result dict into a CSV row dict.

    Notes:
        Some sweep dimensions may use names like "index" which would otherwise collide
        with reserved export fields (e.g. the result index). To prevent silent
        corruption, colliding variable keys are written with a "var_" prefix.
    """
    row: dict[str, Any] = {}

    variables = result.get("variables", {})
    if isinstance(variables, dict):
        for key, value in variables.items():
            if key in _CSV_RESERVED_FIELDS:
                row[f"var_{key}"] = value
            else:
                row[key] = value

    # Reserved export fields: set these last so they always reflect the true result.
    row["index"] = result.get("index")
    row["prompt"] = result.get("prompt", "")[:200]  # Truncate
    row["response"] = result.get("response", "")[:500]  # Truncate
    row["label"] = result.get("judgment", {}).get("label", "")
    return row


def save_run(
    run: ExperimentRun,
    output_dir: Path | str,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """
    Save an experiment run to disk.

    Args:
        run: The experiment run to save.
        output_dir: Directory to save to.
        formats: Output formats ('json', 'csv', 'html'). Defaults to ['json'].

    Returns:
        Mapping of format to output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = formats or ["json"]
    outputs: dict[str, Path] = {}

    # Always save JSON (primary format)
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(run.to_dict(), f, indent=2, default=str)
    outputs["json"] = json_path

    # Save config separately for easy reference
    config_path = output_dir / "config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(run.config, f, default_flow_style=False)
    outputs["config"] = config_path

    # CSV export
    if "csv" in formats and run.results:
        import csv

        csv_path = output_dir / "results.csv"

        # Flatten results for CSV
        fieldnames = set()
        rows = []
        for result in run.results:
            row = _flatten_result_for_csv(result)

            fieldnames.update(row.keys())
            rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

        outputs["csv"] = csv_path

    # HTML report
    if "html" in formats:
        html_path = output_dir / "report.html"
        _generate_html_report(run, html_path)
        outputs["html"] = html_path

    return outputs


def load_run(path: Path | str) -> ExperimentRun:
    """
    Load an experiment run from disk.

    Args:
        path: Path to results directory or JSON file.

    Returns:
        ExperimentRun object.
    """
    path = Path(path)

    if path.is_file():
        json_path = path
    else:
        json_path = path / "results.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    return ExperimentRun.from_dict(data)


def _generate_html_report(run: ExperimentRun, output_path: Path) -> None:
    """Generate an HTML report for the experiment."""
    import html as _html
    import re

    def _esc(value: Any) -> str:
        """HTML-escape arbitrary values for safe embedding into HTML text/attrs."""
        return _html.escape("" if value is None else str(value), quote=True)

    def _css_token(value: Any) -> str:
        """Convert arbitrary values into a safe, limited CSS token."""
        s = "" if value is None else str(value)
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9_-]+", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s

    # Count labels
    labels = [r.get("judgment", {}).get("label", "unknown") for r in run.results]
    label_counts = Counter(labels)

    run_name = _esc(run.name)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{run_name} - Experiment Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f5f5f5;
        }}
        h1 {{ color: #1a1a1a; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }}
        .stat {{
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2563eb;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.875rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; }}
        .label {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.875rem;
        }}
        .label-compliance {{ background: #dcfce7; color: #166534; }}
        .label-refusal {{ background: #fee2e2; color: #991b1b; }}
        .label-partial {{ background: #fef3c7; color: #92400e; }}
        .label-evasion {{ background: #ede9fe; color: #5b21b6; }}
        pre {{
            background: #1a1a1a;
            color: #e5e5e5;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>{run_name}</h1>

    <div class="card">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(run.results)}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(label_counts)}</div>
                <div class="stat-label">Unique Labels</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(run.errors)}</div>
                <div class="stat-label">Errors</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Label Distribution</h2>
        <table>
            <thead>
                <tr><th>Label</th><th>Count</th><th>Percentage</th></tr>
            </thead>
            <tbody>
"""

    total = len(labels)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        label_text = _esc(label)
        label_class = (
            f"label-{_css_token(label)}"
            if label in ("compliance", "refusal", "partial", "evasion")
            else ""
        )
        html += f"""                <tr>
                    <td><span class="label {label_class}">{label_text}</span></td>
                    <td>{count}</td>
                    <td>{pct:.1f}%</td>
                </tr>
"""

    html += """            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Metadata</h2>
        <table>
            <tr><th>Started</th><td>{started}</td></tr>
            <tr><th>Completed</th><td>{completed}</td></tr>
            <tr><th>Git Hash</th><td><code>{git_hash}</code></td></tr>
            <tr><th>Git Branch</th><td>{git_branch}</td></tr>
            <tr><th>Uncommitted Changes</th><td>{git_dirty}</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Configuration</h2>
        <pre>{config}</pre>
    </div>
</body>
</html>
""".format(
        started=_esc(run.started_at),
        completed=_esc(run.completed_at or "In progress"),
        git_hash=_esc(run.git_hash or "N/A"),
        git_branch=_esc(run.git_branch or "N/A"),
        git_dirty=_esc("Yes" if run.git_dirty else "No"),
        config=_esc(json.dumps(run.config, indent=2)),
    )

    with open(output_path, "w") as f:
        f.write(html)
