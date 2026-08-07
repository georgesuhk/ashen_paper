"""Journal figure-width constants.

Widths match ``Columbia/NL_kinks/prod_plots_draft0.ipynb``'s own convention
(``figsize=(7.1, ...)`` for a two-column figure) rather than a generic PRL
default, so a ported figure keeps the same physical size it had in the
notebook. Height is each figure's own choice -- only width is standardized
here, since that's what has to match the manuscript's column.
"""

from __future__ import annotations

__all__ = ["PRL_ONE_COLUMN_WIDTH", "PRL_TWO_COLUMN_WIDTH"]

#: Inches. A single PRL/APS column.
PRL_ONE_COLUMN_WIDTH = 3.375

#: Inches. Full text width -- the notebook's own figures used 7.1.
PRL_TWO_COLUMN_WIDTH = 7.1
