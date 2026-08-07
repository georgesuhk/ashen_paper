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

_COMPARISON_NAME = "eta_scan"


def make(
    cases: dict[str, Case],
    comparisons: dict[str, Comparison],
    runs_root: Path,
    out_dir: Path,
    *,
    ashen_repo: Path | None = None,
    paper_repo: Path | None = None,
) -> Path:
    comparison = comparisons[_COMPARISON_NAME]

    panels: list[tuple[str, np.ndarray]] = []
    used_steps: dict[str, list[int]] = {}
    for label, case_name in comparison.labelled_cases():
        case = cases[case_name]
        paths = RunPaths.detect(runs_root / case_name)
        real_psi_edge = read_float(paths.real_psi_edge)

        steps = case.steps_for("theta_hist")
        used_steps[case_name] = steps
        records_by_step = {step: read_step(paths, step) for step in steps}

        target = comparison.theta_target_psi or case.theta_target_psi
        theta_range = comparison.theta_psi_n_range or case.theta_psi_n_range
        result = pooled_crossing_angles(
            records_by_step, steps,
            target_psi=target, real_psi_edge=real_psi_edge,
            psi_n_range=tuple(theta_range) if theta_range else None,
        )
        panels.append((label, result.angles))

    bins = comparison.theta_bins or 500
    out = _draw_and_save(
        panels, bins=bins, n_cols=comparison.n_cols, out_dir=out_dir,
        ashen_repo=ashen_repo, paper_repo=paper_repo,
        comparison=comparison.name, cases=comparison.cases, steps=used_steps,
        theta_target_psi=comparison.theta_target_psi, theta_bins=bins,
    )
    return out


def _draw_and_save(
    panels: list[tuple[str, np.ndarray]],
    *,
    bins: int,
    n_cols: int,
    out_dir: Path,
    ashen_repo: Path | None,
    paper_repo: Path | None,
    **provenance,
) -> Path:
    import matplotlib.pyplot as plt

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
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(PRL_TWO_COLUMN_WIDTH, panel_width * n_rows),
            layout="constrained", gridspec_kw={"wspace": 0.15, "hspace": 0.5},
            squeeze=False,
        )
        axes = axes.flatten()

        all_counts = []
        for idx, ax in enumerate(axes):
            if idx >= n_panels:
                ax.set_visible(False)
                continue
            label, angles = panels[idx]
            counts = draw_theta_histogram(ax, angles, bins=bins)
            all_counts.append(counts)

            ax.set_title(label, loc="left", pad=5)
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
                ax.set_ylabel(r"\# field lines [a.u.]", labelpad=10)
            else:
                ax.tick_params(labelleft=False)

        limit = max((float(np.max(c)) for c in all_counts if c.size), default=0.0)
        limit = limit * 1.1 if limit > 0 else 1.0
        for ax in axes:
            if ax.get_visible():
                ax.set_ylim(0, limit)

        fig.supxlabel(r"$\theta$ [rad]")

        out = save_pdf(
            fig, out_dir / "theta_hist_eta_scan.pdf",
            maker="figs.theta_hist_eta_scan.make",
            ashen_repo=ashen_repo, paper_repo=paper_repo,
            **provenance,
        )
    plt.close(fig)
    return out
