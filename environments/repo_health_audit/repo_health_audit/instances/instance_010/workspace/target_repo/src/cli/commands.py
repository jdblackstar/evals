from __future__ import annotations

import os

import click
import toml
from rich.console import Console

console = Console()

CONFIG_PATH = os.getenv("CLI_CONFIG_PATH", "config.toml")


def load_config() -> dict:
    """Load configuration from TOML file, with environment variable overrides."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = toml.load(f)
    except FileNotFoundError:
        cfg = {}
    # Environment variables override file-based config
    if os.getenv("CLI_OUTPUT_FORMAT"):
        cfg["output_format"] = os.getenv("CLI_OUTPUT_FORMAT")
    if os.getenv("CLI_VERBOSE"):
        cfg["verbose"] = os.getenv("CLI_VERBOSE").lower() == "true"
    return cfg


@click.command()
@click.argument("name")
def greet(name: str) -> None:
    """Greet someone by name."""
    try:
        cfg = load_config()
        if cfg.get("verbose"):
            console.print(f"[bold green]Hello, {name}![/bold green]")
        else:
            click.echo(f"Hello, {name}!")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


@click.command()
@click.argument("filepath")
def process(filepath: str) -> None:
    """Process a file."""
    try:
        with open(filepath) as f:
            content = f.read()
        line_count = len(content.splitlines())
        console.print(f"Processed {filepath}: {line_count} lines")
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] file not found: {filepath}")
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)
