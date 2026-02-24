from __future__ import annotations

import click

from .commands import greet, process


@click.group()
def cli() -> None:
    """A simple CLI tool."""
    pass


cli.add_command(greet)
cli.add_command(process)


if __name__ == "__main__":
    cli()
