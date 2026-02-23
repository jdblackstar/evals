"""Module entrypoint expected by verifiers (`env_id = pipeline-debugger`)."""

from environments.pipeline_debugger import load_environment

__all__ = ["load_environment"]
