"""
Experiment logging and reproducibility.

Captures full experiment state including config, seeds, model versions,
and results for reproducible research.
"""

import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExperimentRun:
    """
    Complete record of an experiment run.

    Captures everything needed to reproduce the experiment.
    """

    name: str
    config: dict[str, Any]
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    git_hash: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    python_version: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Capture environment info on creation."""
        if self.git_hash is None:
            self._capture_git_info()
        if self.python_version is None:
            self._capture_python_info()

    def _capture_git_info(self) -> None:
        """Capture current git state."""
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.git_hash = result.stdout.strip()

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.git_branch = result.stdout.strip()

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.git_dirty = bool(result.stdout.strip())

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _capture_python_info(self) -> None:
        """Capture Python environment info."""
        import sys

        self.python_version = sys.version

        # Try to get key package versions
        packages = ["openai", "pydantic", "numpy", "pandas", "jinja2"]
        for pkg in packages:
            try:
                import importlib.metadata

                self.package_versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass

    def mark_completed(self) -> None:
        """Mark the run as completed."""
        self.completed_at = datetime.now().isoformat()

    def add_result(self, result: dict[str, Any]) -> None:
        """Add a result to the run."""
        self.results.append(result)

    def add_error(self, error: dict[str, Any]) -> None:
        """Record an error."""
        self.errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                **error,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary."""
        return cls(**data)
