"""Journal figure styling.

Structurally mirrors :func:`ashen.plotting.style` (same ``rc_context``
pattern) so it drops in around the same ``draw_*`` calls ashen's own figures
use -- only the dict differs. Nothing is mutated at import time; apply this
only around the plotting calls that want it:

    with journal_style():
        fig, ax = plt.subplots()
        draw_poincare(ax, records, ...)

Unlike :data:`ashen.plotting.STYLE`, this turns ``text.usetex`` on. Ashen's
own style deliberately keeps it off so figures render without a LaTeX install
on the HPC or the Windows dev clone; a paper figure is instead assumed to be
built somewhere a LaTeX install is available (or the caller passes
``overrides={"text.usetex": False}`` to fall back to mathtext).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

__all__ = ["JOURNAL_STYLE", "journal_style"]

#: A PRL/journal-style preset -- larger fonts and real LaTeX rendering,
#: unlike ashen's exploratory STYLE (ashen/src/ashen/plotting/__init__.py).
JOURNAL_STYLE: dict = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    # Type42 embedded fonts, not Type3 -- keeps PDF text selectable/editable
    # in Overleaf and journal production pipelines instead of rasterizing it.
    "pdf.fonttype": 42,
}


@contextmanager
def journal_style(overrides: dict | None = None) -> Iterator[None]:
    """``rc_context`` scoped to :data:`JOURNAL_STYLE`, with optional
    per-figure overrides (e.g. ``{"font.size": 12}`` for a denser panel)."""
    import matplotlib.pyplot as plt

    with plt.rc_context({**JOURNAL_STYLE, **(overrides or {})}):
        yield
