"""Wetted fraction vs. eta, overlaying the normal and rho19 datasets.

Reuses ashen's diagnostics layer and its per-series drawer
(``draw_wetted_fraction_vs_x``), the way ``ashen.cli.plot.
_compare_wetted_fraction``'s ``datasets`` branch does -- only the style and
save step differ, since that CLI path is PNG/no-LaTeX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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

# --- Overlays, ported from prod_plots_draft0.ipynb's eta_plot ------------------
# ashen's own wetted_fraction module deliberately doesn't carry any of these
# (see its docstring: "not ported -- add them back here if a future comparison
# actually needs one, not speculatively"). They're annotations about *this*
# paper's argument, not general scan-plot furniture, so they live here.
# Values are the ones the notebook's wetted_area.pdf call actually used.

#: Vertical dotted line at the experimentally-observed benign/non-benign
#: resistivity threshold, in Ohm-m. None to omit the line and its annotations.
THRESHOLD_ETA: float | None = 2e-3
#: Which end of the axes the two annotations sit at: "bottom" or "top".
THRESHOLD_ANNOTATION_POS = "bottom"
#: Text either side of the threshold. The notebook took a `vline_label`
#: argument ("exp. threshold") but never drew it -- these hardcoded strings
#: are what actually appeared, so they're what's reproduced.
THRESHOLD_LABEL_LEFT = "Non-benign"
THRESHOLD_LABEL_RIGHT = "Benign"

#: Green Gaussian-faded band marking the experimentally-relevant window.
#: Centre in Ohm-m, width as a standard deviation in decades. None to omit.
BAND_CENTRE: float | None = 0.85e-2
BAND_WIDTH_DECADES = 0.4
#: Peak opacity at the band's centre -- the notebook's comment notes this is
#: held below 1 deliberately so the grid stays visible through it.
BAND_MAX_ALPHA = 0.5
#: Strips the Gaussian is discretised into, over +-3 sigma in log-space.
_BAND_STRIPS = 100

#: Points to ring in black, as (dataset name, index into that dataset's
#: cases). The notebook indexed datasets positionally ([0, 4] and [1, 2]);
#: naming them instead survives reordering the datasets in cases.toml.
HIGHLIGHT_POINTS: list[tuple[str, int]] = [("normal", 4), ("rho19", 2)]
HIGHLIGHT_COLOR = "black"

#: Right-hand x-limit (Ohm-m), so the band isn't clipped mid-fade. None to
#: let matplotlib choose.
X_MAX: float | None = 1.2e-2


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
    # Keyed by dataset *name* (not legend label) so HIGHLIGHT_POINTS can name
    # the dataset it means rather than depend on ordering.
    series_by_key: dict[str, tuple[list[float], list[float]]] = {}
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
            percent = fraction * 100
            print(f"    {case_name}: wetted fraction = {percent:.3g}%")
            xs.append(x_by_case[case_name])
            ys.append(percent)

        used_steps[ds_name] = case_steps
        series.append((dataset.series_label, xs, ys))
        series_by_key[ds_name] = (xs, ys)
        colors.append(dataset.color or DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)])

    print("wetted_fraction_vs_eta: drawing and saving")
    out = _draw_and_save(
        series, colors=colors, xlabel=comparison.x_label, out_dir=out_dir,
        series_by_key=series_by_key,
        ashen_repo=ashen_repo, paper_repo=paper_repo,
        comparison=comparison.name, datasets={n: ds.cases for n, ds in datasets.items()},
        steps=used_steps,
    )
    return out


def _draw_fading_band(ax) -> None:
    """A green band centred on BAND_CENTRE, fading out as a Gaussian in
    log-space. Drawn as many thin axvspans rather than one gradient image so
    it composites correctly on a log x-axis and in vector PDF output."""
    if BAND_CENTRE is None:
        return
    log_centre = np.log10(BAND_CENTRE)
    edges = np.linspace(
        log_centre - 3 * BAND_WIDTH_DECADES,
        log_centre + 3 * BAND_WIDTH_DECADES,
        _BAND_STRIPS + 1,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = np.exp(-0.5 * ((centres - log_centre) / BAND_WIDTH_DECADES) ** 2)
    for i, weight in enumerate(weights):
        ax.axvspan(
            10 ** edges[i], 10 ** edges[i + 1],
            color="green", alpha=BAND_MAX_ALPHA * float(weight), linewidth=0, zorder=0,
        )


def _draw_threshold(ax) -> None:
    """The benign/non-benign threshold line, with a label either side.

    The labels are positioned in blended coordinates -- x in data (so they
    stay pinned either side of the line) and y in axes fraction (so they hold
    their height whatever the y-range turns out to be).
    """
    if THRESHOLD_ETA is None:
        return
    from matplotlib.transforms import blended_transform_factory

    ax.axvline(THRESHOLD_ETA, lw=1.5, color="black", ls=":", alpha=0.8, zorder=1)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    at_bottom = THRESHOLD_ANNOTATION_POS == "bottom"
    y, va = (0.05, "bottom") if at_bottom else (0.95, "top")
    ax.text(
        THRESHOLD_ETA * 0.85, y, THRESHOLD_LABEL_LEFT, transform=trans,
        ha="right", va=va, fontsize=14, fontweight="bold",
    )
    ax.text(
        THRESHOLD_ETA * 1.15, y, THRESHOLD_LABEL_RIGHT, transform=trans,
        ha="left", va=va, fontsize=14, fontweight="bold",
    )


def _draw_highlights(ax, series_by_key: dict[str, tuple[list[float], list[float]]]) -> None:
    """Ring the called-out points in an unfilled circle, over the data."""
    for ds_name, idx in HIGHLIGHT_POINTS:
        if ds_name not in series_by_key:
            continue  # dataset not drawn this time (e.g. --dataset normal)
        xs, ys = series_by_key[ds_name]
        if not -len(xs) <= idx < len(xs):
            print(
                f"  note: highlight ({ds_name!r}, {idx}) is out of range for "
                f"{len(xs)} point(s), skipped"
            )
            continue
        ax.scatter(
            xs[idx], ys[idx], s=180, facecolors="none",
            edgecolors=HIGHLIGHT_COLOR, linewidths=2.5, zorder=10,
        )


def _draw_and_save(
    series: list[tuple[str, list[float], list[float]]],
    *,
    colors: list[str],
    xlabel: str,
    out_dir: Path,
    series_by_key: dict[str, tuple[list[float], list[float]]] | None = None,
    ashen_repo: Path | None,
    paper_repo: Path | None,
    **provenance,
) -> Path:
    import matplotlib.pyplot as plt

    # Matches prod_plots_draft0.ipynb's own eta_plot(..., fig_height=2.5) call
    # that produced "wetted_area.pdf" -- figsize=(6, fig_height) there, not a
    # PRL_ONE_COLUMN_WIDTH-derived size -- so a re-upload needn't be reformatted.
    with journal_style():
        fig, ax = plt.subplots(figsize=(6, 2.5), layout="constrained")
        # Band first: it is zorder=0 and must sit under both the grid that
        # draw_wetted_fraction_vs_x turns on and the data lines themselves.
        _draw_fading_band(ax)
        for (label, x, y), color in zip(series, colors):
            # "%" is a literal LaTeX comment character under journal_style()'s
            # usetex=True (unlike ashen.plotting's own style, which has usetex
            # off) -- must be escaped or LaTeX silently eats the rest of the
            # label. Matches prod_plots_draft0.ipynb's own r"Wetted fraction
            # [\%]" call for this exact figure.
            draw_wetted_fraction_vs_x(
                ax, x, y, xlabel=xlabel, ylabel=r"Wetted fraction [\%]",
                label=label, color=color,
            )

        _draw_threshold(ax)
        if series_by_key:
            _draw_highlights(ax, series_by_key)
        if X_MAX is not None:
            ax.set_xlim(right=X_MAX)
        ax.legend(loc="best", frameon=False, fontsize=14)

        out = save_pdf(
            fig, out_dir / "wetted_fraction_vs_eta.pdf",
            maker="figs.wetted_fraction_vs_eta.make",
            ashen_repo=ashen_repo, paper_repo=paper_repo,
            **provenance,
        )
    plt.close(fig)
    return out
