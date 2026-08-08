"""Saving a finished figure as a provenanced, journal-ready PDF.

Every figure this package saves gets two files: the PDF itself, and a
``<name>.pdf.json`` sidecar recording what produced it. The sidecar, not the
PDF's own ``metadata=``, is the primary provenance record -- it is greppable
and diffable in git, and survives a journal's or Overleaf's own PDF
post-processing, which can silently strip or rewrite embedded metadata. The
PDF's ``metadata=`` is kept too, as a redundant convenience for anyone who
opens the file standalone with no access to the sidecar.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["save_pdf"]


def _git_info(repo_dir: Path | str | None) -> dict[str, Any] | None:
    """``{"commit": <sha>, "dirty": <bool>}`` for the repo containing
    ``repo_dir``, or ``None`` if ``repo_dir`` is ``None`` or not inside a git
    repo (e.g. ``Columbia/`` itself, which has no ``.git`` yet)."""
    if repo_dir is None:
        return None
    repo_dir = Path(repo_dir)
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def save_pdf(
    fig,
    out_path: Path | str,
    *,
    maker: str,
    ashen_repo: Path | str | None = None,
    paper_repo: Path | str | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    **provenance: Any,
) -> Path:
    """Save ``fig`` as a vector PDF at ``out_path`` plus a provenance sidecar.

    ``maker`` identifies the ``figs/*.py`` function that built this figure
    (e.g. ``"figs.theta_hist_eta_scan.make"``), for a reader who only has the
    output and wants to find the code that produced it. ``ashen_repo``/
    ``paper_repo`` are the repo roots to record git provenance for -- pass
    ``None`` for either if it isn't (yet) a git repo. ``savefig_kwargs`` is
    forwarded to ``fig.savefig`` -- needed by figures that place artists
    outside the axes area (a manually positioned colourbar, say), which are
    clipped without ``{"bbox_inches": "tight"}``. Remaining keyword arguments
    (case names, steps, thresholds used, ...) are recorded verbatim in the
    sidecar under their own key.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    metadata = {
        "Title": out_path.stem,
        "Creator": f"paper_toolkit via {maker}",
        "CreationDate": now,  # matplotlib's pdf backend requires a real datetime, not a string
    }
    fig.savefig(out_path, metadata=metadata, **(savefig_kwargs or {}))

    import matplotlib

    sidecar = {
        "figure": str(out_path),
        "maker": maker,
        "created": now.isoformat(),
        "matplotlib_version": matplotlib.__version__,
        "ashen": _git_info(ashen_repo),
        "paper": _git_info(paper_repo),
        **provenance,
    }
    sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")

    return out_path
