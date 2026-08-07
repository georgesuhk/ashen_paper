"""Shared, physics-free helpers for building paper figures on top of ashen.

Deliberately small and separate from ``ashen``: journal styling, PDF saving
with provenance, and journal figsize constants are genuinely reusable across
any future paper, but which cases go in which panel, which steps to pool, and
how a composite figure is laid out is paper-specific and belongs under
``papers/<slug>/`` (a sibling of this package, both under
``Columbia/ashen_paper/``), not here.

Not an installable package -- imported the same way ``ashen`` itself is, via
a ``sys.path.insert`` in each paper's entry-point script only (see
``papers/*/make_figures.py``). Library modules in this package must never
touch ``sys.path`` themselves, mirroring ``ashen``'s own rule.
"""

from __future__ import annotations
