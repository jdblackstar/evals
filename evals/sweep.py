"""
Sweep engine for parameter exploration.

Handles dimension expansion, Cartesian products, and prompt templating.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

from jinja2 import BaseLoader, Environment, TemplateSyntaxError

from evals.config import DimensionConfig, SweepConfig


@dataclass
class SweepPoint:
    """A single point in the sweep grid."""

    variables: dict[str, Any]
    prompt: str
    index: int

    def __repr__(self) -> str:
        var_str = ", ".join(f"{k}={v}" for k, v in self.variables.items())
        return f"SweepPoint({var_str})"


@dataclass
class SweepGrid:
    """
    A grid of all points to sweep over.

    Represents the Cartesian product of all dimension values.
    """

    dimensions: dict[str, list[Any]]
    points: list[SweepPoint] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Total number of points in the grid."""
        return len(self.points)

    @property
    def dimension_names(self) -> list[str]:
        """Names of all dimensions."""
        return list(self.dimensions.keys())

    def get_dimension_values(self, name: str) -> list[Any]:
        """Get all values for a specific dimension."""
        return self.dimensions.get(name, [])


class SweepEngine:
    """
    Engine for generating sweep grids and expanding prompts.

    Handles:
    - Discrete value lists
    - Numeric ranges
    - LLM-generated expansions
    - Jinja2 prompt templating
    """

    def __init__(self) -> None:
        self._jinja_env = Environment(loader=BaseLoader())

    def expand_dimension(
        self,
        dim_config: DimensionConfig,
        llm_expander: Any | None = None,
    ) -> list[Any]:
        """
        Expand a dimension configuration into concrete values.

        Args:
            dim_config: The dimension configuration.
            llm_expander: Optional callable for LLM-based expansion.

        Returns:
            List of concrete values for this dimension.
        """
        if dim_config.type == "llm_expand":
            if llm_expander is None:
                raise ValueError(
                    f"Dimension '{dim_config.name}' requires LLM expansion but no expander provided"
                )
            return self._expand_with_llm(dim_config, llm_expander)

        return dim_config.get_values()

    def _expand_with_llm(
        self,
        dim_config: DimensionConfig,
        llm_expander: Any,
    ) -> list[Any]:
        """
        Use an LLM to expand seed values into more variations.

        Args:
            dim_config: Dimension config with expansion settings.
            llm_expander: Callable that takes a prompt and returns expanded values.

        Returns:
            List of expanded values.
        """
        seed_values = dim_config.seed_values or []
        expand_prompt = dim_config.expand_prompt or (
            f"Generate {dim_config.expand_count} variations of these values: {seed_values}"
        )

        expanded = llm_expander(expand_prompt, count=dim_config.expand_count)

        # Combine seed values with expanded ones, deduplicating
        all_values = list(seed_values)
        for val in expanded:
            if val not in all_values:
                all_values.append(val)

        return all_values

    def build_grid(
        self,
        sweep_config: SweepConfig,
        prompt_template: str,
        llm_expander: Any | None = None,
    ) -> SweepGrid:
        """
        Build a complete sweep grid from configuration.

        Args:
            sweep_config: Sweep configuration with dimensions.
            prompt_template: Jinja2 template for prompts.
            llm_expander: Optional callable for LLM-based expansion.

        Returns:
            SweepGrid with all points and rendered prompts.
        """
        # Expand all dimensions
        dimensions: dict[str, list[Any]] = {}
        for dim in sweep_config.dimensions:
            dimensions[dim.name] = self.expand_dimension(dim, llm_expander)

        # Generate Cartesian product
        dim_names = list(dimensions.keys())
        dim_values = [dimensions[name] for name in dim_names]

        points: list[SweepPoint] = []
        for idx, combo in enumerate(product(*dim_values)):
            variables = dict(zip(dim_names, combo))
            prompt = self.render_prompt(prompt_template, variables)
            points.append(SweepPoint(variables=variables, prompt=prompt, index=idx))

        return SweepGrid(dimensions=dimensions, points=points)

    def render_prompt(self, template: str, variables: dict[str, Any]) -> str:
        """
        Render a prompt template with given variables.

        Args:
            template: Jinja2 template string.
            variables: Variables to inject into the template.

        Returns:
            Rendered prompt string.

        Raises:
            ValueError: If template rendering fails.
        """
        try:
            jinja_template = self._jinja_env.from_string(template)
            return jinja_template.render(**variables)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid template syntax: {e}") from e
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}") from e

    def validate_template(self, template: str, dimension_names: list[str]) -> list[str]:
        """
        Validate a template and return any warnings.

        Args:
            template: Jinja2 template to validate.
            dimension_names: Expected variable names.

        Returns:
            List of warning messages (empty if all good).
        """
        warnings: list[str] = []

        try:
            ast = self._jinja_env.parse(template)
        except TemplateSyntaxError as e:
            return [f"Template syntax error: {e}"]

        # Find all variables used in template
        from jinja2 import meta

        used_vars = meta.find_undeclared_variables(ast)

        # Check for unused dimensions
        for dim_name in dimension_names:
            if dim_name not in used_vars:
                warnings.append(f"Dimension '{dim_name}' is not used in template")

        # Check for undefined variables
        for var in used_vars:
            if var not in dimension_names:
                warnings.append(f"Template uses undefined variable '{var}'")

        return warnings


def _create_default_engine() -> SweepEngine:
    """Create a default sweep engine instance."""
    return SweepEngine()


# Module-level convenience functions
_default_engine: SweepEngine | None = None


def get_engine() -> SweepEngine:
    """Get the default sweep engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = _create_default_engine()
    return _default_engine


def build_grid(
    sweep_config: SweepConfig,
    prompt_template: str,
    llm_expander: Any | None = None,
) -> SweepGrid:
    """
    Build a sweep grid using the default engine.

    Args:
        sweep_config: Sweep configuration with dimensions.
        prompt_template: Jinja2 template for prompts.
        llm_expander: Optional callable for LLM-based expansion.

    Returns:
        SweepGrid with all points and rendered prompts.
    """
    return get_engine().build_grid(sweep_config, prompt_template, llm_expander)
