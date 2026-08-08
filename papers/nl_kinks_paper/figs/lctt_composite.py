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
    #: prod_plots2.ipynb's plot_side_by_side_color_con_lengths used 2.5/4.5
    #: (not matplotlib's thin default) so the lines read at figure scale.
    linewidth: float = 2.0
    #: Label placement, matching the notebook's `ax.get_yaxis_transform()`
    #: call: x is an axes fraction, y a psi_n offset from the line itself
    #: (it put "3/2 TM" at 0.73 for a line at 0.75).
    label_x: float = 0.2
    label_offset: float = -0.02


@dataclass
class LcttPanel:
    case_name: str
    #: True-time window (µs) the zoomed map is cropped to; None = this case's
    #: own full range, i.e. the true times of the steps cases.toml configures
    #: for it (its `[cases."<name>".connection_length]` steps, else its plain
    #: `steps`). Narrow the window by narrowing those steps rather than by
    #: restating a µs range here, so the two can't drift apart.
    zoom_xlim: tuple[float, float] | None = None
    #: True-time window (µs) the overview strip is cropped to; None = shared
    #: full range across every case in OVERVIEW_ORDER (see `make`), matching
    #: prod_plots2.ipynb's explicit shared `xlims` on the stacked plot.
    overview_xlim: tuple[float, float] | None = None
    #: True-time (µs) of the vertical dashed marker; None = no marker.
    marker_time: float | None = None
    rational_surfaces: list[RationalSurfaceLine] = field(default_factory=list)
    #: Which step's qprofile cache backs any `mode`-based rational_surfaces.
    qprofile_step: int | None = None
    eta_label: str = ""
    #: Only the leftmost ZOOM_ORDER panel needs its labels -- both zoom
    #: panels typically mark the same physical surfaces (data_jorek.py's
    #: side-by-side plotter only ever labelled `idx == 0`); set True on more
    #: than one panel if their surfaces are at genuinely different psi_n.
    show_rational_labels: bool = True
    #: 3-point boxcar smoothing across steps, ignoring inf (confined) lines --
    #: ports the `smooth=` flag both prod_plots2.ipynb plotters took.
    smooth: bool = False


# PLACEHOLDER: real case names/windows/markers -- see cases.toml's own
# PLACEHOLDER notes for the run data these currently point at.
PANELS: dict[str, LcttPanel] = {
    # PLACEHOLDER: rational-surface psi_n hardcoded to prod_plots2.ipynb's
    # plot_side_by_side_color_con_lengths values (0.75, 0.95) for both panels,
    # same as the original notebook -- swap to `mode=(m, n)` per panel once
    # each case's own qprofile is available to compute a real crossing.
    "qa2.1_g2.3/eta1e-3_RE": LcttPanel(
        case_name="qa2.1_g2.3/eta1e-3_RE",
        marker_time=4.6,
        eta_label=r"$\eta = 10^{-2}\ \Omega\mathrm{m}$",
        rational_surfaces=[
            RationalSurfaceLine("2/1 TM", "lime", psi_n=0.95, linestyle="-", linewidth=4.5),
            RationalSurfaceLine("3/2 TM", "aqua", psi_n=0.75, linestyle=":", linewidth=2.5),
        ],
        qprofile_step=4600,
    ),
    "qa2.1_g2.3/eta1e-5_RE": LcttPanel(
        case_name="qa2.1_g2.3/eta1e-5_RE",
        marker_time=18.0,
        eta_label=r"$\eta = 10^{-4}\ \Omega\mathrm{m}$",
        rational_surfaces=[
            RationalSurfaceLine("2/1 TM", "lime", psi_n=0.95, linestyle="-", linewidth=4.5),
            RationalSurfaceLine("3/2 TM", "aqua", psi_n=0.75, linestyle=":", linewidth=2.5),
        ],
        qprofile_step=18000,
        show_rational_labels=False,
    ),
}
OVERVIEW_ORDER = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-5_RE"]  # (a), (b)
ZOOM_ORDER = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-5_RE"]  # (c), (d)

#: Explicit y-ticks for the (shared-y) zoom row; None = matplotlib's own.
#: prod_plots2.ipynb hardcoded these to keep a tick from colliding with the
#: 2/1 TM label -- only meaningful once the real psi_n window is known.
ZOOM_YTICKS: list[float] | None = None

# --- Figure geometry, straight from prod_plots2.ipynb's final gridspec cell ---
_FIGSIZE = (7.1, 6.0)
_OUTER_HEIGHT_RATIOS = [0.2, 0.8]
_OUTER_HSPACE = 0.5
_TOP_HSPACE = 1.1
_BOTTOM_WSPACE = 0.06
#: Figure margins. The notebook reserved its footer with
#: ``tight_layout(rect=[0, 0.1, 1, 1])``, but tight_layout warns
#: ("Axes that are not compatible ... results might be incorrect") and lays
#: out only approximately on the nested subgridspecs used here. Setting the
#: outer gridspec's margins directly is the same reservation, stated once and
#: honoured exactly -- and it makes the colourbar arithmetic below reliable,
#: since it reads the zoom axes' coordinates back.
_MARGINS = {"left": 0.14, "right": 0.97, "top": 0.95, "bottom": 0.22}
#: Footer colourbar, in figure coords. The bar spans the zoom row's width
#: minus a gap and the square "infinity" swatch, rather than the whole figure.
_CBAR_HEIGHT = 0.04
_CBAR_GAP = 0.03
_CBAR_DROP = 0.15  # below the zoom axes' bottom


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
    for line in panel.rational_surfaces:
        psi_n = _rational_surface_psi_n(line, paths, panel)
        ax.axhline(psi_n, color=line.color, linestyle=line.linestyle, linewidth=line.linewidth, zorder=5)
        if panel.show_rational_labels:
            # x in axes fraction, y in data -- prod_plots2.ipynb placed these
            # with ax.get_yaxis_transform() rather than in data coords, so the
            # label keeps its inset regardless of the zoom window's width.
            ax.text(
                line.label_x, psi_n + line.label_offset, line.label,
                color=line.color, transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=16, fontweight="bold", zorder=6,
            )
    if panel.marker_time is not None:
        # lw=3, alpha=0.7 -- prod_plots2.ipynb's "Overlap Visualizers" crosshair.
        ax.axvline(panel.marker_time, color="blue", linestyle="--", linewidth=3, alpha=0.7, zorder=5)


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

    # Shared x-limits across every OVERVIEW_ORDER panel that doesn't set its
    # own -- ports prod_plots2.ipynb's explicit shared `xlims` on the stacked
    # plot, so both strips line up even though the runs cover different spans.
    overview_x_all = np.concatenate([data[c][1] for c in OVERVIEW_ORDER])
    shared_overview_xlim = (float(overview_x_all.min()), float(overview_x_all.max()))

    import string

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    print("lctt_composite: drawing and saving")
    # prod_plots2.ipynb sets plt.rcParams per sub-plotter, and both stacked
    # and side-by-side ran in the same combined-figure cell -- by the time
    # everything actually rendered, plot_side_by_side_color_con_lengths's
    # values (the one called last) were the active rcParams. Matched exactly
    # here rather than journal_style()'s own (smaller) defaults.
    with journal_style(overrides={
        "font.size": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }):
        # Nested gridspec straight from prod_plots2.ipynb's final combination
        # cell: a 0.2/0.8 height split between the stacked overview strips and
        # the side-by-side zoom maps. No constrained/tight layout manager on
        # the figure itself -- the colourbar below is positioned by hand from
        # the zoom axes' final coordinates, which a layout manager would then
        # invalidate.
        fig = plt.figure(figsize=_FIGSIZE)
        gs = GridSpec(
            2, 1, figure=fig, height_ratios=_OUTER_HEIGHT_RATIOS, hspace=_OUTER_HSPACE,
            **_MARGINS,
        )
        gs_top = gs[0].subgridspec(len(OVERVIEW_ORDER), 1, hspace=_TOP_HSPACE)
        gs_bottom = gs[1].subgridspec(1, len(ZOOM_ORDER), wspace=_BOTTOM_WSPACE)

        letters = iter(string.ascii_lowercase)
        overview_axes = []
        for row, case_name in enumerate(OVERVIEW_ORDER):
            panel = PANELS[case_name]
            matrix, x, psi_n_in, paths = data[case_name]
            ax = fig.add_subplot(gs_top[row])
            overview_axes.append(ax)
            draw_connection_length_map(ax, matrix, x, psi_n_in, log=True, smooth=panel.smooth)
            ax.set_xlim(*(panel.overview_xlim or shared_overview_xlim))
            ax.set_title(
                f"({next(letters)}) {panel.eta_label}",
                loc="left", fontsize=14, fontweight="bold", pad=-1.5,
            )
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True)
            if row < len(OVERVIEW_ORDER) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"t [$\mu s$]")

        pcm = None
        zoom_axes = []
        for col, case_name in enumerate(ZOOM_ORDER):
            panel = PANELS[case_name]
            matrix, x, psi_n_in, paths = data[case_name]
            ax = fig.add_subplot(gs_bottom[col])
            zoom_axes.append(ax)
            pcm = draw_connection_length_map(
                ax, matrix, x, psi_n_in, log=True, xlabel=r"t [$\mu s$]", smooth=panel.smooth
            )
            # Default to this case's own configured time span (cases.toml's
            # steps), not a restated µs window -- see LcttPanel.zoom_xlim.
            ax.set_xlim(*(panel.zoom_xlim or (float(x.min()), float(x.max()))))
            ax.set_ylim(float(np.min(psi_n_in)), float(np.max(psi_n_in)))
            _draw_zoom_overlays(ax, panel, paths)
            ax.set_title(f"({next(letters)}) {panel.eta_label}", loc="left", fontsize=14, fontweight="bold")
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True)
            if col == 0:
                ax.set_ylabel(r"$\Psi_N$", fontsize=16)
                if ZOOM_YTICKS is not None:
                    ax.set_yticks(ZOOM_YTICKS)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

        # Freeze the layout -- every position below is read off these final
        # axes coordinates, so nothing may move after this point.
        fig.canvas.draw()

        # One shared "$\Psi_N$" label spanning the overview strips, instead
        # of a per-panel ylabel -- prod_plots2.ipynb's plot_stacked_color_
        # con_lengths positions this the same way, via fig.text between the
        # top and bottom overview axes rather than on either axes itself.
        # Its x is taken from the zoom row's real ylabel rather than a fixed
        # offset, so the two read as one aligned column down the page (the
        # notebook's hardcoded -0.098 only lined up for its own tick widths).
        top_pos = overview_axes[0].get_position()
        bottom_pos = overview_axes[-1].get_position()
        zoom_label_bbox = zoom_axes[0].yaxis.label.get_window_extent()
        label_x = fig.transFigure.inverted().transform(
            (zoom_label_bbox.x0 + zoom_label_bbox.width / 2, 0.0)
        )[0]
        fig.text(
            label_x, (top_pos.y1 + bottom_pos.y0) / 2, r"$\Psi_N$",
            va="center", ha="center", rotation="vertical", fontsize=16,
        )

        # Footer colourbar, sized to the zoom row rather than the whole
        # figure, with a square black "infinity" swatch beside it: field lines
        # that never left. draw_connection_length_map already masks those
        # black on the map itself; this is the legend entry for that colour,
        # which matplotlib's colorbar has no built-in slot for.
        x0 = zoom_axes[0].get_position().x0
        x1 = zoom_axes[-1].get_position().x1
        y_pos = zoom_axes[0].get_position().y0 - _CBAR_DROP
        square = _CBAR_HEIGHT
        cbar_width = (x1 - x0) - _CBAR_GAP - square

        cax = fig.add_axes([x0, y_pos, cbar_width, _CBAR_HEIGHT])
        cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal")
        cbar.set_label("$L_c$ [m]", fontsize=16)

        rect_x = x0 + cbar_width + _CBAR_GAP
        fig.patches.append(
            Rectangle(
                (rect_x, y_pos), square, square,
                facecolor="black", transform=fig.transFigure, clip_on=False,
            )
        )
        fig.text(
            rect_x + square / 2, y_pos - 0.02, r"$\infty$",
            transform=fig.transFigure, ha="center", va="top", fontsize=18,
        )

        out = save_pdf(
            fig, out_dir / "lctt_composite.pdf",
            maker="figs.lctt_composite.make",
            ashen_repo=ashen_repo, paper_repo=paper_repo,
            cases=list(PANELS),
            # The colourbar/swatch live outside the axes area; without this
            # they are cropped off the saved page.
            savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.02},
        )
    plt.close(fig)
    return out
