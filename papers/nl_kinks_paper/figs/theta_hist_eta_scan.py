"""Eta-scan theta-crossing histogram: one panel per resistivity, journal style.

Reuses ashen's diagnostics layer (``pooled_crossing_angles``) and its
per-panel drawer (``draw_theta_histogram``) exactly as
``ashen.cli.plot._compare_theta_hist`` does -- only the grid assembly, style,
and save step differ, since that CLI path is hardcoded to ashen's own
PNG/dpi=200/no-LaTeX convention (``ashen.plotting.theta_histogram.
plot_theta_histogram_grid``) and can't be reused as-is for journal PDF
output. If a second paper needs this same grid, that's the trigger to add an
optional style/save hook to ashen's own grid function -- not worth doing
speculatively for one paper.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ashen.cases import Case
from ashen.comparisons import Comparison
from ashen.diagnostics.poincare_cache import read_step
from ashen.diagnostics.theta_histogram import pooled_crossing_angles
from ashen.paths import RunPaths, read_float
from ashen.plotting.theta_histogram import draw_theta_histogram

from paper_toolkit.layout import PRL_TWO_COLUMN_WIDTH
from paper_toolkit.save import save_pdf
from paper_toolkit.style import journal_style

__all__ = ["make"]

_DEFAULT_COMPARISON_NAME = "eta_scan"

#: Inches of figure height that are *not* panel: the shared x-label, the
#: x-tick labels, and the per-panel title. Without this the figure is sized
#: as if the panels were the whole thing, so at 5 columns the entire figure
#: is one panel-width tall (~1.4in) and the y-label is clipped off the edge.
_CHROME_HEIGHT = 1.15

#: Draw a dashed line at the wetted-fraction threshold -- the bin height a
#: theta bin must exceed to count as "wetted" in
#: `figs/wetted_fraction_vs_eta.py`. Makes the two figures legible together:
#: this one shows which bins clear the bar, that one plots how many do.
#: ashen's own grid deliberately doesn't port this (see
#: ashen/src/ashen/plotting/theta_histogram.py's docstring) -- it's a
#: paper-figure choice, so it lives here.
SHOW_WETTED_THRESHOLD = True


def _wetted_threshold(comparison: Comparison, case: Case, bins: int) -> float:
    """The bin height this case's field lines must clear to count as wetted.

    Deliberately the same three-tier fallback
    ``figs/wetted_fraction_vs_eta.py`` resolves for the identical quantity --
    comparison, then case, then ``1/bins``. Skipping the case tier here (as
    an earlier version did) draws the line at ``1/bins`` while the companion
    figure computes its fraction against the case's own value, so the two
    figures silently disagree about where the bar is.
    """
    return (
        comparison.theta_wetted_threshold
        or case.theta_wetted_threshold
        or 1.0 / bins
    )


def make(
    cases: dict[str, Case],
    comparisons: dict[str, Comparison],
    runs_root: Path,
    out_dir: Path,
    *,
    ashen_repo: Path | None = None,
    paper_repo: Path | None = None,
    comparison_name: str | None = None,
) -> Path:
    name = comparison_name or _DEFAULT_COMPARISON_NAME
    if name not in comparisons:
        raise ValueError(f"no comparison {name!r}; known: {list(comparisons)}")
    comparison = comparisons[name]
    print(f"theta_hist_eta_scan: comparison {comparison.name!r}, {len(comparison.cases)} case(s)")

    panels: list[tuple[str, np.ndarray]] = []
    used_steps: dict[str, list[int]] = {}
    for label, case_name in comparison.labelled_cases():
        case = cases[case_name]
        paths = RunPaths.detect(runs_root / case_name)
        real_psi_edge = read_float(paths.real_psi_edge)

        steps = case.steps_for("theta_hist")
        used_steps[case_name] = steps
        print(f"  {case_name}: reading {len(steps)} step(s) from cache")
        records_by_step = {step: read_step(paths, step) for step in steps}

        target = comparison.theta_target_psi or case.theta_target_psi
        theta_range = comparison.theta_psi_n_range or case.theta_psi_n_range
        result = pooled_crossing_angles(
            records_by_step, steps,
            target_psi=target, real_psi_edge=real_psi_edge,
            psi_n_range=tuple(theta_range) if theta_range else None,
        )
        print(f"  {case_name}: {result.angles.size} crossing(s) pooled")
        panels.append((label, result.angles))

    bins = comparison.theta_bins or 500
    # Per panel, not one shared value: the threshold can be set per case, and
    # a panel has to show the bar its own case is actually measured against.
    # Identical values collapse to a level line across the row anyway.
    thresholds = [
        _wetted_threshold(comparison, cases[case_name], bins)
        for _, case_name in comparison.labelled_cases()
    ]
    if len(set(thresholds)) > 1:
        print(f"  note: per-case wetted thresholds differ across panels: {thresholds}")

    print("theta_hist_eta_scan: drawing and saving")
    out = _draw_and_save(
        panels, bins=bins, n_cols=comparison.n_cols, out_dir=out_dir,
        thresholds=thresholds if SHOW_WETTED_THRESHOLD else None,
        ashen_repo=ashen_repo, paper_repo=paper_repo,
        comparison=comparison.name, cases=comparison.cases, steps=used_steps,
        theta_target_psi=comparison.theta_target_psi, theta_bins=bins,
        theta_wetted_thresholds=dict(zip(comparison.cases, thresholds)),
    )
    return out


def _draw_and_save(
    panels: list[tuple[str, np.ndarray]],
    *,
    bins: int,
    n_cols: int,
    out_dir: Path,
    thresholds: list[float] | None = None,
    ashen_repo: Path | None,
    paper_repo: Path | None,
    **provenance,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n_panels = len(panels)
    n_rows = math.ceil(n_panels / n_cols) if n_panels else 1
    panel_width = PRL_TWO_COLUMN_WIDTH / n_cols

    # Bottom-of-column axis for column c: the last visible panel with
    # idx % n_cols == c. sharex/sharey are off and limits set by hand for the
    # same reason ashen.plotting.theta_histogram.plot_theta_histogram_grid
    # does -- a partially-filled last row leaves some columns' true bottom
    # axis one row up, and a true shared axis suppresses labels by grid
    # position rather than "last panel actually present in this column".
    bottom_of_column = {idx % n_cols: idx for idx in range(n_panels)}

    with journal_style():
        # Height is panel area *plus* chrome -- see _CHROME_HEIGHT. wspace/
        # hspace go to the layout engine, not gridspec_kw: constrained layout
        # computes its own spacing and silently ignores the gridspec values.
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(PRL_TWO_COLUMN_WIDTH, panel_width * n_rows + _CHROME_HEIGHT),
            layout="constrained", squeeze=False,
        )
        # wspace has to clear the +-pi tick labels, which sit hard against
        # each panel's left/right edge (set_ha below) -- too tight and the
        # neighbouring "pi" and "-pi" run together into one glyph soup.
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.10, hspace=0.06)
        axes = axes.flatten()

        all_counts = []
        for idx, ax in enumerate(axes):
            if idx >= n_panels:
                ax.set_visible(False)
                continue
            label, angles = panels[idx]
            counts = draw_theta_histogram(ax, angles, bins=bins)
            all_counts.append(counts)

            if thresholds is not None and idx < len(thresholds):
                ax.axhline(
                    thresholds[idx], color="black", linestyle="--", linewidth=1.5, zorder=4,
                )

            # Centred, not left-aligned: at 5 narrow columns a left-aligned
            # title runs out over its neighbour instead of naming its own panel.
            ax.set_title(label, loc="center", pad=4)
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, 0, np.pi])

            if bottom_of_column.get(idx % n_cols) == idx:
                labels = ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
                labels[0].set_ha("left")
                labels[-1].set_ha("right")
            else:
                ax.tick_params(labelbottom=False)

            if idx % n_cols == 0:
                # "#" is a literal LaTeX macro-parameter character under
                # journal_style()'s usetex=True (unlike ashen.plotting's own
                # style, which has usetex off) -- must be escaped or LaTeX
                # rejects the whole label.
                ax.set_ylabel(r"\# field-lines [a.u.]", labelpad=4)
            else:
                ax.tick_params(labelleft=False)

        limit = max((float(np.max(c)) for c in all_counts if c.size), default=0.0)
        limit = limit * 1.1 if limit > 0 else 1.0
        for ax in axes:
            if ax.get_visible():
                ax.set_ylim(0, limit)
                # Cap the tick count explicitly: on panels this short
                # matplotlib's default locator settles for two labels
                # (0.00/0.05), too coarse to read a distribution against.
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))

        fig.supxlabel(r"$\theta$ [rad]")

        out = save_pdf(
            fig, out_dir / "theta_hist_eta_scan.pdf",
            maker="figs.theta_hist_eta_scan.make",
            ashen_repo=ashen_repo, paper_repo=paper_repo,
            **provenance,
        )
    plt.close(fig)
    return out
