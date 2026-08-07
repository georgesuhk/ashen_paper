"""Wetted fraction vs. eta, overlaying the normal and rho19 datasets.

Reuses ashen's diagnostics layer and its per-series drawer
(``draw_wetted_fraction_vs_x``), the way ``ashen.cli.plot.
_compare_wetted_fraction``'s ``datasets`` branch does -- only the style and
save step differ, since that CLI path is PNG/no-LaTeX.
"""

from __future__ import annotations

from pathlib import Path

from ashen.cases import Case
from ashen.comparisons import Comparison
from ashen.diagnostics.poincare_cache import read_step
from ashen.diagnostics.theta_histogram import (
    pooled_crossing_angles,
    theta_histogram,
    wetted_fraction,
)
from ashen.paths import RunPaths, read_float
from ashen.plotting.colors import DISCRETE_PALETTE
from ashen.plotting.wetted_fraction import draw_wetted_fraction_vs_x

from paper_toolkit.save import save_pdf
from paper_toolkit.style import journal_style

__all__ = ["make"]

_DEFAULT_COMPARISON_NAME = "wetted_vs_eta"


def make(
    cases: dict[str, Case],
    comparisons: dict[str, Comparison],
    runs_root: Path,
    out_dir: Path,
    *,
    ashen_repo: Path | None = None,
    paper_repo: Path | None = None,
    dataset_names: list[str] | None = None,
    comparison_name: str | None = None,
) -> Path:
    """`dataset_names` restricts which of the comparison's datasets are
    drawn (default: all) -- e.g. to render "normal" and "rho19" as separate
    figures instead of one overlay. `comparison_name` selects which
    `[comparisons.*]` block to draw (default: `_DEFAULT_COMPARISON_NAME`)."""
    name = comparison_name or _DEFAULT_COMPARISON_NAME
    if name not in comparisons:
        raise ValueError(f"no comparison {name!r}; known: {list(comparisons)}")
    comparison = comparisons[name]

    datasets = comparison.datasets
    if dataset_names is not None:
        unknown = [n for n in dataset_names if n not in datasets]
        if unknown:
            raise ValueError(f"no dataset(s) {unknown}; known: {list(datasets)}")
        datasets = {n: datasets[n] for n in dataset_names}
    print(f"wetted_fraction_vs_eta: comparison {comparison.name!r}, dataset(s) {list(datasets)}")

    series: list[tuple[str, list[float], list[float]]] = []
    colors: list[str] = []
    used_steps: dict[str, dict[str, list[int]]] = {}
    for idx, (ds_name, dataset) in enumerate(datasets.items()):
        x_values = dataset.x_values or comparison.x_values
        if x_values is None:
            raise ValueError(
                f"dataset {ds_name!r} of comparison {comparison.name!r} has no "
                "x_values (and the comparison sets none to fall back on)"
            )
        x_by_case = dict(zip(dataset.cases, x_values))
        print(f"  dataset {ds_name!r}: {len(dataset.cases)} case(s)")

        xs: list[float] = []
        ys: list[float] = []
        case_steps: dict[str, list[int]] = {}
        for case_name in dataset.cases:
            case = cases[case_name]
            paths = RunPaths.detect(runs_root / case_name)
            real_psi_edge = read_float(paths.real_psi_edge)

            steps = case.steps_for("theta_hist")
            case_steps[case_name] = steps
            print(f"    {case_name}: reading {len(steps)} step(s) from cache")
            records_by_step = {step: read_step(paths, step) for step in steps}

            target = comparison.theta_target_psi or case.theta_target_psi
            n_bins = comparison.theta_bins or case.theta_bins
            theta_range = comparison.theta_psi_n_range or case.theta_psi_n_range
            result = pooled_crossing_angles(
                records_by_step, steps,
                target_psi=target, real_psi_edge=real_psi_edge,
                psi_n_range=tuple(theta_range) if theta_range else None,
            )
            counts, _ = theta_histogram(result.angles, bins=n_bins)
            threshold = (
                comparison.theta_wetted_threshold
                or case.theta_wetted_threshold
                or 1.0 / n_bins
            )
            fraction = wetted_fraction(counts, threshold=threshold)
            print(f"    {case_name}: wetted fraction = {fraction:.3g}")
            xs.append(x_by_case[case_name])
            ys.append(fraction)

        used_steps[ds_name] = case_steps
        series.append((dataset.series_label, xs, ys))
        colors.append(dataset.color or DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)])

    print("wetted_fraction_vs_eta: drawing and saving")
    out = _draw_and_save(
        series, colors=colors, xlabel=comparison.x_label, out_dir=out_dir,
        ashen_repo=ashen_repo, paper_repo=paper_repo,
        comparison=comparison.name, datasets={n: ds.cases for n, ds in datasets.items()},
        steps=used_steps,
    )
    return out


def _draw_and_save(
    series: list[tuple[str, list[float], list[float]]],
    *,
    colors: list[str],
    xlabel: str,
    out_dir: Path,
    ashen_repo: Path | None,
    paper_repo: Path | None,
    **provenance,
) -> Path:
    import matplotlib.pyplot as plt

    from paper_toolkit.layout import PRL_ONE_COLUMN_WIDTH

    with journal_style():
        fig, ax = plt.subplots(
            figsize=(PRL_ONE_COLUMN_WIDTH * 1.6, PRL_ONE_COLUMN_WIDTH), layout="constrained",
        )
        for (label, x, y), color in zip(series, colors):
            draw_wetted_fraction_vs_x(
                ax, x, y, xlabel=xlabel, ylabel="Wetted fraction",
                label=label, color=color,
            )
        ax.legend()

        out = save_pdf(
            fig, out_dir / "wetted_fraction_vs_eta.pdf",
            maker="figs.wetted_fraction_vs_eta.make",
            ashen_repo=ashen_repo, paper_repo=paper_repo,
            **provenance,
        )
    plt.close(fig)
    return out
