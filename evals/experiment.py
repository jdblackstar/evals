"""
Experiment orchestration.

Ties together sweep generation, model execution, judgment, and logging
into a complete experiment pipeline. Supports both single-turn and
multi-turn sequence tasks with behavior embeddings and metacognition probes.
"""

import asyncio
from collections import defaultdict, deque
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from evals.analysis import ExperimentResult, ExperimentResults
from evals.config import ExperimentConfig
from evals.logging import ExperimentRun
from evals.outputs import save_run
from evals.probes.behavior_vectorizer import compute_behavior_embedding
from evals.probes.judge import BaseJudge, create_judge
from evals.probes.metaprobes import run_all_metaprobes
from evals.runner import Completion, ModelRunner
from evals.sequences import SequenceTask, TurnTemplate, compute_hysteresis
from evals.sweep import SweepGrid, SweepPoint, build_grid

console = Console()


def _build_sequence_task(config: ExperimentConfig) -> SequenceTask:
    """Build a SequenceTask from experiment configuration."""
    if config.task.sequence is None:
        raise ValueError("Sequence config is required for sequence tasks")

    turns = []
    for i, turn_config in enumerate(config.task.sequence.turns):
        turns.append(
            TurnTemplate(
                role=turn_config.role,
                content_template=turn_config.content_template,
                turn_index=i,
                metadata=turn_config.metadata or {},
            )
        )

    return SequenceTask(
        name=config.name,
        turns=turns,
        system_prompt=config.task.system_prompt,
        description=config.description or "",
    )


def _match_completions_to_samples(
    batch: list[tuple[SweepPoint, int]],
    completions: list[Completion],
) -> list[tuple[SweepPoint, int, Completion]]:
    """
    Align completions back to their submission slot.

    Uses the request_index (set by ModelRunner.batch_complete) so that duplicate
    prompts remain associated with their original sample index even when earlier
    batch items fail. Falls back to prompt-based ordering if the index is missing.
    """
    index_lookup = {i: pair for i, pair in enumerate(batch)}

    prompt_to_batch: dict[str, deque[tuple[SweepPoint, int]]] = defaultdict(deque)
    for point, sample_idx in batch:
        prompt_to_batch[point.prompt].append((point, sample_idx))

    matched: list[tuple[SweepPoint, int, Completion]] = []
    for completion in completions:
        submission_index = completion.request_index
        if submission_index is None:
            submission_index = completion.metadata.get("request_index")

        if submission_index is not None and submission_index in index_lookup:
            point, sample_idx = index_lookup.pop(submission_index)
        else:
            if not prompt_to_batch[completion.prompt]:
                raise ValueError(
                    "Batch completion prompt did not match any pending sample. "
                    "This can happen if the batch result order is inconsistent with "
                    "the input prompts, or if a completion returned an unexpected prompt."
                )
            point, sample_idx = prompt_to_batch[completion.prompt].popleft()

        matched.append((point, sample_idx, completion))

    return matched


async def _run_single_turn_experiment(
    config: ExperimentConfig,
    runner: ModelRunner,
    judge: BaseJudge,
    grid: SweepGrid,
    run: ExperimentRun,
    use_cache: bool = True,
) -> list[ExperimentResult]:
    """Run a single-turn experiment."""
    # Expand for samples per point
    all_points: list[tuple[SweepPoint, int]] = []
    for point in grid.points:
        for sample_idx in range(config.task.samples_per_point):
            all_points.append((point, sample_idx))

    console.print(f"[green]Total samples: {len(all_points)}[/]")

    results: list[ExperimentResult] = []

    with Progress(console=console) as progress:
        task = progress.add_task(
            "[cyan]Running experiment...",
            total=len(all_points),
        )

        batch_size = 10
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]

            # Get completions
            prompts = [point.prompt for point, _ in batch]
            batch_result = await runner.batch_complete(
                prompts=prompts,
                system_prompt=config.task.system_prompt,
                use_cache=use_cache,
            )

            matched_completions = _match_completions_to_samples(
                batch, batch_result.completions
            )

            for point, sample_idx, completion in matched_completions:
                # Judge the response
                try:
                    judgment = await judge.evaluate(
                        prompt=completion.prompt,
                        response=completion.content,
                    )
                    judgment_dict = {
                        "label": judgment.label,
                        "confidence": judgment.confidence,
                        "reasoning": judgment.reasoning,
                    }
                except Exception as e:
                    judgment_dict = {
                        "label": "judge_error",
                        "error": str(e),
                    }
                    run.add_error(
                        {
                            "type": "judgment_error",
                            "index": point.index,
                            "sample": sample_idx,
                            "error": str(e),
                        }
                    )

                # Compute behavior embedding if enabled
                behavior_emb = None
                if config.task.behavior_embedding.enabled:
                    emb = compute_behavior_embedding(completion.content)
                    behavior_emb = emb.to_list()

                # Run metaprobes if enabled
                meta_probe = None
                if config.task.metaprobes.enabled:
                    try:
                        probe_result = await run_all_metaprobes(
                            runner=runner,
                            context_prompt=completion.prompt,
                            context_response=completion.content,
                            system_prompt=config.task.system_prompt,
                            probes=list(config.task.metaprobes.probes),
                        )
                        meta_probe = probe_result.to_dict()
                    except Exception as e:
                        run.add_error(
                            {
                                "type": "metaprobe_error",
                                "index": point.index,
                                "sample": sample_idx,
                                "error": str(e),
                            }
                        )

                result = ExperimentResult(
                    index=point.index * config.task.samples_per_point + sample_idx,
                    variables=point.variables,
                    prompt=completion.prompt,
                    response=completion.content,
                    judgment=judgment_dict,
                    behavior_embedding=behavior_emb,
                    meta_probe=meta_probe,
                    metadata={
                        "sweep_index": point.index,
                        "sample_index": sample_idx,
                        "cached": completion.cached,
                        "tokens": completion.tokens_used,
                    },
                )
                results.append(result)

                run.add_result(
                    {
                        "index": result.index,
                        "variables": result.variables,
                        "prompt": result.prompt,
                        "response": result.response,
                        "judgment": result.judgment,
                        "behavior_embedding": result.behavior_embedding,
                        "meta_probe": result.meta_probe,
                        "metadata": result.metadata,
                    }
                )

            for error in batch_result.errors:
                run.add_error(
                    {
                        "type": "completion_error",
                        **error,
                    }
                )

            progress.update(task, advance=len(batch))

    return results


async def _run_sequence_experiment(
    config: ExperimentConfig,
    runner: ModelRunner,
    judge: BaseJudge,
    grid: SweepGrid,
    run: ExperimentRun,
    use_cache: bool = True,
) -> list[ExperimentResult]:
    """Run a sequence (multi-turn) experiment."""
    sequence_task = _build_sequence_task(config)

    # Get turn overrides from config
    turn_overrides = None
    if config.task.sequence and config.task.sequence.turn_overrides:
        turn_overrides = config.task.sequence.turn_overrides

    # Expand for samples per point
    all_points: list[tuple[SweepPoint, int]] = []
    for point in grid.points:
        for sample_idx in range(config.task.samples_per_point):
            all_points.append((point, sample_idx))

    console.print(f"[green]Total samples: {len(all_points)}[/]")

    results: list[ExperimentResult] = []

    with Progress(console=console) as progress:
        task = progress.add_task(
            "[cyan]Running sequence experiment...",
            total=len(all_points),
        )

        for point, sample_idx in all_points:
            # Build conversation from sequence task
            turns = sequence_task.build_conversation(
                variables=point.variables,
                turn_overrides=turn_overrides,
            )

            # Run forward conversation
            try:
                conv_result = await runner.complete_conversation(
                    turns=turns,
                    system_prompt=config.task.system_prompt,
                    use_cache=use_cache,
                )
                final_response = conv_result.last_response
                all_responses = conv_result.all_responses

            except Exception as e:
                run.add_error(
                    {
                        "type": "sequence_error",
                        "index": point.index,
                        "sample": sample_idx,
                        "error": str(e),
                    }
                )
                progress.update(task, advance=1)
                continue

            # Judge the final response
            try:
                judgment = await judge.evaluate(
                    prompt=turns[-1].content if turns else "",
                    response=final_response,
                )
                judgment_dict = {
                    "label": judgment.label,
                    "confidence": judgment.confidence,
                    "reasoning": judgment.reasoning,
                }
            except Exception as e:
                judgment_dict = {
                    "label": "judge_error",
                    "error": str(e),
                }
                run.add_error(
                    {
                        "type": "judgment_error",
                        "index": point.index,
                        "sample": sample_idx,
                        "error": str(e),
                    }
                )

            # Compute behavior embedding
            behavior_emb = None
            if config.task.behavior_embedding.enabled:
                emb = compute_behavior_embedding(final_response)
                behavior_emb = emb.to_list()

            # Run metaprobes
            meta_probe = None
            if config.task.metaprobes.enabled:
                try:
                    probe_result = await run_all_metaprobes(
                        runner=runner,
                        context_prompt=turns[-1].content if turns else "",
                        context_response=final_response,
                        system_prompt=config.task.system_prompt,
                        probes=list(config.task.metaprobes.probes),
                    )
                    meta_probe = probe_result.to_dict()
                except Exception as e:
                    run.add_error(
                        {
                            "type": "metaprobe_error",
                            "index": point.index,
                            "error": str(e),
                        }
                    )

            # Run hysteresis if enabled
            hysteresis_data = None
            if config.task.sequence and config.task.sequence.enable_hysteresis:
                reversed_task = sequence_task.reversed()
                reversed_turns = reversed_task.build_conversation(
                    variables=point.variables,
                    turn_overrides=turn_overrides,
                )

                try:
                    reversed_result = await runner.complete_conversation(
                        turns=reversed_turns,
                        system_prompt=config.task.system_prompt,
                        use_cache=use_cache,
                    )
                    reversed_responses = reversed_result.all_responses

                    hyst = compute_hysteresis(all_responses, reversed_responses)
                    hysteresis_data = {
                        "score": hyst.hysteresis_score,
                        "has_hysteresis": hyst.has_hysteresis,
                        "per_turn": hyst.per_turn_comparison,
                    }
                except Exception as e:
                    run.add_error(
                        {
                            "type": "hysteresis_error",
                            "index": point.index,
                            "error": str(e),
                        }
                    )

            result = ExperimentResult(
                index=point.index * config.task.samples_per_point + sample_idx,
                variables=point.variables,
                prompt=str([t.content for t in turns]),  # All turn contents
                response=final_response,
                judgment=judgment_dict,
                behavior_embedding=behavior_emb,
                meta_probe=meta_probe,
                turns=[{"role": t.role, "content": t.content} for t in turns],
                metadata={
                    "sweep_index": point.index,
                    "sample_index": sample_idx,
                    "num_turns": len(turns),
                    "all_responses": all_responses,
                    "tokens": conv_result.total_tokens,
                    "hysteresis": hysteresis_data,
                },
            )
            results.append(result)

            run.add_result(
                {
                    "index": result.index,
                    "variables": result.variables,
                    "prompt": result.prompt,
                    "response": result.response,
                    "judgment": result.judgment,
                    "behavior_embedding": result.behavior_embedding,
                    "meta_probe": result.meta_probe,
                    "turns": result.turns,
                    "metadata": result.metadata,
                }
            )

            progress.update(task, advance=1)

    return results


async def _run_experiment_async(
    config: ExperimentConfig,
    use_cache: bool = True,
) -> ExperimentResults:
    """
    Run an experiment asynchronously.

    Handles both single-turn and sequence (multi-turn) tasks,
    with optional behavior embeddings and metacognition probes.

    Args:
        config: Experiment configuration.
        use_cache: Whether to cache API responses.

    Returns:
        ExperimentResults with all data points.
    """
    # Initialize experiment run for logging
    run = ExperimentRun(
        name=config.name,
        config=config.model_dump(),
    )

    # Determine task type and get prompt template for grid building
    is_sequence = config.task.is_sequence

    if is_sequence:
        # For sequences, build a "dummy" template from all turn templates
        # This allows the sweep engine to validate dimension usage
        seq_config = config.task.sequence
        combined_template = "\n".join(
            t.content_template
            for t in seq_config.turns  # type: ignore
        )
        prompt_template = combined_template
    else:
        prompt_template = config.task.prompt_template or ""

    # Build sweep grid
    console.print("[dim]Building sweep grid...[/]")
    grid = build_grid(
        sweep_config=config.sweep,
        prompt_template=prompt_template,
    )
    console.print(f"[green]Generated {grid.size} sweep points[/]")

    # Initialize runner and judge
    runner = ModelRunner(config=config.model)
    judge = create_judge(config.judge)

    # Run appropriate experiment type
    if is_sequence:
        console.print("[cyan]Running sequence (multi-turn) experiment...[/]")
        results = await _run_sequence_experiment(
            config, runner, judge, grid, run, use_cache
        )
    else:
        console.print("[cyan]Running single-turn experiment...[/]")
        results = await _run_single_turn_experiment(
            config, runner, judge, grid, run, use_cache
        )

    # Mark complete
    run.mark_completed()

    # Save results
    output_dir = Path(config.output.dir) / config.name
    console.print(f"[dim]Saving results to {output_dir}...[/]")
    outputs = save_run(run, output_dir, formats=config.output.formats)

    for fmt, path in outputs.items():
        console.print(f"  [green]✓[/] {fmt}: {path}")

    # Build and return results object
    return ExperimentResults(
        name=config.name,
        results=results,
        config=config.model_dump(),
        metadata={
            "output_dir": str(output_dir),
            "total_samples": len(results),
            "cache_hits": sum(1 for r in results if r.metadata.get("cached")),
            "errors": len(run.errors),
            "is_sequence": is_sequence,
            "behavior_embedding_enabled": config.task.behavior_embedding.enabled,
            "metaprobes_enabled": config.task.metaprobes.enabled,
        },
    )


def run_experiment(
    config: ExperimentConfig,
    use_cache: bool = True,
) -> ExperimentResults:
    """
    Run an experiment synchronously.

    This is the main entry point for running experiments.

    Args:
        config: Experiment configuration.
        use_cache: Whether to cache API responses.

    Returns:
        ExperimentResults with all data points.
    """
    return asyncio.run(_run_experiment_async(config, use_cache))


async def run_experiment_batch(
    configs: list[ExperimentConfig],
    use_cache: bool = True,
) -> list[ExperimentResults]:
    """
    Run multiple experiments.

    Args:
        configs: List of experiment configurations.
        use_cache: Whether to cache API responses.

    Returns:
        List of ExperimentResults.
    """
    results = []
    for config in configs:
        console.print(f"\n[bold blue]Running: {config.name}[/]")
        result = await _run_experiment_async(config, use_cache)
        results.append(result)

    return results
