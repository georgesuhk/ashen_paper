"""Composite LCTT figure: two eta values, an overview strip each plus a
zoomed side-by-side pair with rational-surface and event-time overlays.

Reuses ashen's connection-length diagnostics/plotting as-is
(``connection_length_matrix``, ``draw_connection_length_map``) the same way
``ashen.cli.plot._plot_connection_length`` does -- only composition (which
axes, which zoom window, which overlays) is new, and it's paper-specific by
nature (hand-picked panels/markers), so it lives here rather than in ashen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ashen.cases import Case
from ashen.comparisons import Comparison
from ashen.diagnostics.connection_length import connection_length_matrix
from ashen.diagnostics.poincare_cache import read_step
from ashen.diagnostics.qprofile import find_rational_surfaces, read_qprofile
from ashen.logfile import r_axis
from ashen.paths import RunPaths, read_float
from ashen.plotting.connection_length import draw_connection_length_map
from ashen.postproc import read_zeroD

from paper_toolkit.save import save_pdf
from paper_toolkit.style import journal_style

__all__ = ["make", "LcttPanel", "RationalSurfaceLine"]


@dataclass
class RationalSurfaceLine:
    label: str
    color: str
    #: Exactly one of these two must be set: a hardcoded psi_n, or an (m, n)
    #: mode whose q = m/n surface is looked up from `qprofile_step`'s cache.
    psi_n: float | None = None
    mode: tuple[int, int] | None = None
    linestyle: str = "--"


@dataclass
class LcttPanel:
    case_name: str
    #: True-time window (µs) the zoomed map is cropped to.
    zoom_xlim: tuple[float, float]
    #: True-time window (µs) the overview strip is cropped to; None = full range.
    overview_xlim: tuple[float, float] | None = None
    #: True-time (µs) of the vertical dashed marker; None = no marker.
    marker_time: float | None = None
    rational_surfaces: list[RationalSurfaceLine] = field(default_factory=list)
    #: Which step's qprofile cache backs any `mode`-based rational_surfaces.
    qprofile_step: int | None = None
    eta_label: str = ""


# PLACEHOLDER: real case names/windows/markers -- see cases.toml's own
# PLACEHOLDER notes for the run data these currently point at.
PANELS: dict[str, LcttPanel] = {
    "qa2.1_g2.3/eta1e-4_RE": LcttPanel(
        case_name="qa2.1_g2.3/eta1e-4_RE",
        zoom_xlim=(15.0, 22.0),
        marker_time=18.0,
        eta_label=r"$\eta = 10^{-4}\ \Omega\mathrm{m}$",
        rational_surfaces=[
            RationalSurfaceLine("2/1 TM", "tab:green", mode=(2, 1), linestyle="-"),
            RationalSurfaceLine("3/2 TM", "cyan", mode=(3, 2), linestyle=":"),
        ],
        qprofile_step=18000,
    ),
    "qa2.1_g2.3/eta1e-2_RE": LcttPanel(
        case_name="qa2.1_g2.3/eta1e-2_RE",
        zoom_xlim=(3.5, 5.5),
        marker_time=4.6,
        eta_label=r"$\eta = 10^{-2}\ \Omega\mathrm{m}$",
        rational_surfaces=[
            RationalSurfaceLine("2/1 TM", "tab:green", mode=(2, 1), linestyle="-"),
            RationalSurfaceLine("3/2 TM", "cyan", mode=(3, 2), linestyle=":"),
        ],
        qprofile_step=4600,
    ),
}
OVERVIEW_ORDER = ["qa2.1_g2.3/eta1e-4_RE", "qa2.1_g2.3/eta1e-2_RE"]  # (a), (b)
ZOOM_ORDER = ["qa2.1_g2.3/eta1e-2_RE", "qa2.1_g2.3/eta1e-4_RE"]  # (c), (d)


def _matrix_and_times(case: Case, paths: RunPaths) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Same data prep as ``ashen.cli.plot._plot_connection_length``, using
    only public functions -- reads what's already cached, never gathers."""
    real_psi_edge = read_float(paths.real_psi_edge)
    R0 = r_axis(paths.log)

    psi_n_in = case.lc_psi_n_in if case.lc_psi_n_in is not None else case.psi_n_in
    psi_n_targets = [p * real_psi_edge for p in psi_n_in]

    steps = case.steps_for("connection_length")
    print(f"    {case.name}: reading {len(steps)} step(s) from cache")
    records_by_step = {step: read_step(paths, step) for step in steps}
    matrix = connection_length_matrix(
        records_by_step, steps, psi_n_targets, real_psi_edge=real_psi_edge, R0=R0
    )

    print(f"    {case.name}: reading true-time for {len(steps)} step(s)")
    true_times = [read_zeroD(paths.zero_d(step))["Time"] for step in steps]
    x = np.asarray(true_times) * 1e6  # µs
    return matrix, x, psi_n_in


def _rational_surface_psi_n(line: RationalSurfaceLine, paths: RunPaths, panel: LcttPanel) -> float:
    if line.psi_n is not None:
        return line.psi_n
    m, n = line.mode
    psi_n_q, q = read_qprofile(paths.qprofile(panel.qprofile_step))
    crossings = find_rational_surfaces(psi_n_q, q, m / n)
    if not crossings:
        raise ValueError(
            f"no q={m}/{n} rational surface found in {panel.case_name}'s "
            f"qprofile at step {panel.qprofile_step}"
        )
    return crossings[0]


def _draw_zoom_overlays(ax, panel: LcttPanel, paths: RunPaths) -> None:
    # Label offset inward from the left spine (1% of the zoom window) so
    # text doesn't sit flush on/outside the axis edge.
    x0, x1 = panel.zoom_xlim
    label_x = x0 + 0.01 * (x1 - x0)
    for line in panel.rational_surfaces:
        psi_n = _rational_surface_psi_n(line, paths, panel)
        ax.axhline(psi_n, color=line.color, linestyle=line.linestyle, linewidth=1.5)
        ax.text(
            label_x, psi_n, line.label, color=line.color,
            va="bottom", ha="left", fontsize=9,
        )
    if panel.marker_time is not None:
        ax.axvline(panel.marker_time, color="blue", linestyle="--", linewidth=1.5)


def make(
    cases: dict[str, Case],
    comparisons: dict[str, Comparison],
    runs_root: Path,
    out_dir: Path,
    *,
    ashen_repo: Path | None = None,
    paper_repo: Path | None = None,
) -> Path:
    data: dict[str, tuple[np.ndarray, np.ndarray, list[float], RunPaths]] = {}
    for case_name in PANELS:
        print(f"lctt_composite: {case_name}")
        case = cases[case_name]
        paths = RunPaths.detect(runs_root / case_name)
        matrix, x, psi_n_in = _matrix_and_times(case, paths)
        data[case_name] = (matrix, x, psi_n_in, paths)

    import string

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print("lctt_composite: drawing and saving")
    with journal_style():
        fig = plt.figure(figsize=(7.1, 8.0), layout="constrained")
        gs = GridSpec(4, 2, figure=fig, height_ratios=[0.6, 0.6, 3.5, 0.25])

        letters = iter(string.ascii_lowercase)
        for row, case_name in enumerate(OVERVIEW_ORDER):
            panel = PANELS[case_name]
            matrix, x, psi_n_in, paths = data[case_name]
            ax = fig.add_subplot(gs[row, :])
            draw_connection_length_map(ax, matrix, x, psi_n_in, log=True)
            if panel.overview_xlim is not None:
                ax.set_xlim(*panel.overview_xlim)
            ax.set_title(f"({next(letters)}) {panel.eta_label}", loc="left", fontsize=10)
            ax.set_yticks([])
            if row < len(OVERVIEW_ORDER) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"t [$\mu s$]")

        pcm = None
        for col, case_name in enumerate(ZOOM_ORDER):
            panel = PANELS[case_name]
            matrix, x, psi_n_in, paths = data[case_name]
            ax = fig.add_subplot(gs[2, col])
            pcm = draw_connection_length_map(ax, matrix, x, psi_n_in, log=True, xlabel=r"t [$\mu s$]")
            ax.set_xlim(*panel.zoom_xlim)
            _draw_zoom_overlays(ax, panel, paths)
            ax.set_title(f"({next(letters)}) {panel.eta_label}", loc="left", fontsize=10)

        cbar_ax = fig.add_subplot(gs[3, :])
        fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal", label="$L_c$ [m]")
        # "infinity" swatch: field lines that never left, drawn as a small
        # black patch beside the colorbar -- draw_connection_length_map
        # already masks these black on the map itself; this is the legend
        # entry for that colour, which matplotlib's colorbar has no built-in
        # slot for.
        inf_ax = fig.add_axes([0.92, cbar_ax.get_position().y0, 0.03, cbar_ax.get_position().height])
        inf_ax.set_facecolor("black")
        inf_ax.set_xticks([])
        inf_ax.set_yticks([])
        inf_ax.text(0.5, -1.4, r"$\infty$", ha="center", va="top", transform=inf_ax.transAxes)

        out = save_pdf(
            fig, out_dir / "lctt_composite.pdf",
            maker="figs.lctt_composite.make",
            ashen_repo=ashen_repo, paper_repo=paper_repo,
            cases=list(PANELS),
        )
    plt.close(fig)
    return out
