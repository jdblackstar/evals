"""Nox session definitions for the data platform."""

import nox

nox.options.sessions = ["test"]


@nox.session(python=["3.11"])
def test(session):
    """Run the test suite."""
    session.install("-e", ".[dev]")
    session.run("python", "-m", "pytest", "pipelines/", "-v")


@nox.session(python=["3.11"])
def coverage(session):
    """Run tests with coverage reporting."""
    session.install("-e", ".[dev]")
    session.run(
        "python", "-m", "pytest", "pipelines/",
        "--cov=pipelines", "--cov=shared",
        "--cov-report=term",
    )


@nox.session(python=["3.11"])
def lint(session):
    """Run linting checks."""
    session.install("ruff")
    session.run("ruff", "check", "pipelines/", "shared/")
