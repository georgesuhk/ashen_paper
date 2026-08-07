#!/usr/bin/env python3
"""Build this paper's figures.

    python make_figures.py                       # every registered figure
    python make_figures.py theta_hist_eta_scan    # just this one
    python make_figures.py --list

Same bootstrap convention as ashen/bin/plot -- sys.path is set up here, in
the entry-point script only; nothing under figs/ or paper_toolkit/ touches
sys.path itself.

Layout this assumes:

    Columbia/ashen_paper/               <- its own git repo (paper_toolkit +
      paper_toolkit/                       every paper's figures are small,
      papers/<slug>/make_figures.py        text+PDF, and want real history --
                                            unlike Columbia/NL_kinks's run data,
                                            which this never touches beyond
                                            reading already-cached results)
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# ashen_paper/papers/<slug> -> papers -> ashen_paper -> Columbia -> repo root
_ASHEN_PAPER = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[3]

sys.path.insert(0, str(_REPO_ROOT / "ashen" / "src"))
sys.path.insert(0, str(_ASHEN_PAPER))  # for paper_toolkit

from ashen.cases import load_cases  # noqa: E402  (import follows bootstrap)
from ashen.comparisons import load_comparisons  # noqa: E402

from figs import lctt_composite, theta_hist_eta_scan, wetted_fraction_vs_eta  # noqa: E402

REGISTRY = {
    "theta_hist_eta_scan": theta_hist_eta_scan.make,
    "wetted_fraction_vs_eta": wetted_fraction_vs_eta.make,
    "lctt_composite": lctt_composite.make,
}

_ASHEN_REPO = _REPO_ROOT / "ashen"
_PAPER_REPO = _ASHEN_PAPER  # save_pdf tolerates this not (yet) being a git repo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures", nargs="*", default=list(REGISTRY),
        help="Which registered figures to build (default: all).",
    )
    parser.add_argument("--list", action="store_true", help="List registered figures and exit.")
    parser.add_argument(
        "--runs-root", type=Path, default=_REPO_ROOT / "Columbia" / "NL_kinks",
        help="Campaign folder run names in cases.toml resolve against.",
    )
    parser.add_argument(
        "--cases-toml", type=Path, default=_HERE / "cases.toml",
        help="This paper's frozen case/comparison config.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_HERE / "figures",
        help="Where PDFs (and their .pdf.json sidecars) are written.",
    )
    parser.add_argument(
        "--dataset", action="append", dest="datasets_selected",
        help="Which dataset(s) to draw, for a figure whose comparison uses "
        "datasets (repeatable; default: all; ignored by figures that don't "
        "take this option).",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in REGISTRY:
            print(name)
        return 0

    unknown = [name for name in args.figures if name not in REGISTRY]
    if unknown:
        print(f"unknown figure(s): {unknown}; known: {list(REGISTRY)}", file=sys.stderr)
        return 1

    cases = load_cases(args.cases_toml)
    comparisons = load_comparisons(args.cases_toml, cases)

    for name in args.figures:
        fn = REGISTRY[name]
        kwargs = {"ashen_repo": _ASHEN_REPO, "paper_repo": _PAPER_REPO}
        if args.datasets_selected and "dataset_names" in inspect.signature(fn).parameters:
            kwargs["dataset_names"] = args.datasets_selected
        out = fn(cases, comparisons, args.runs_root, args.out_dir, **kwargs)
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
