# nl_kinks_paper figures

Production figures for this paper, built from `ashen`'s existing
diagnostics/plotting layer plus `paper_toolkit`'s journal styling and PDF
saving. Not part of `ashen` itself -- see
`Columbia/paper_workspace/paper_toolkit/__init__.py` for why.

This paper folder lives under `Columbia/paper_workspace/`, a sibling of
(not nested inside) `Columbia/NL_kinks/`'s run data -- see
`paper_workspace`'s own layout note at the top of `make_figures.py`: it's
meant to be its own git repo, kept small and separate from multi-GB
simulation output.

## Layout

- `cases.toml` -- a **frozen** subset of case/comparison definitions this
  paper cites, copied by hand from (not read live from) `NL_kinks/`'s own
  config, so a figure's meaning doesn't drift as exploratory work continues
  there. Same TOML dialect as `ashen/cases.example.toml`.
- `figs/<name>.py` -- one module per figure, each `make(cases, comparisons,
  runs_root, out_dir, *, ashen_repo, paper_repo) -> Path`. Ordinary Python,
  not a declarative config -- composite/multi-panel figures are inherently
  imperative.
- `figures/` -- output: `<name>.pdf` + `<name>.pdf.json` (provenance: which
  cases/steps/thresholds, matplotlib version, `ashen`/paper git commit +
  dirty flag). Meant to be git-tracked once `paper_workspace/` is under
  version control.
- `make_figures.py` -- driver. Run from anywhere:

```bash
python Columbia/paper_workspace/papers/nl_kinks_paper/make_figures.py                    # every figure
python Columbia/paper_workspace/papers/nl_kinks_paper/make_figures.py theta_hist_eta_scan
python Columbia/paper_workspace/papers/nl_kinks_paper/make_figures.py --list
```

`--runs-root` (default `Columbia/NL_kinks`) is where case names in
`cases.toml` resolve against; override it if a run lives somewhere else.

## Adding a new figure

1. Write `figs/<name>.py` with a `make(...)` function, reusing
   `ashen.diagnostics.*` for data reduction and `ashen.plotting.*`'s
   `draw_*` functions for the actual drawing wherever they already exist --
   never re-derive physics that ashen already computes correctly.
2. Wrap the figure-owning code in `paper_toolkit.style.journal_style()` and
   save with `paper_toolkit.save.save_pdf(...)`, not `fig.savefig` directly,
   so the provenance sidecar is written.
3. Register it in `make_figures.py`'s `REGISTRY`.

If a `draw_*` function doesn't exist yet for what you need (e.g. it's only
ever been in the notebook as a `plot_*` that also owns styling/saving), port
just the pure-drawing half yourself here first -- don't add it to `ashen`
unless a second paper will also need it.

## Overleaf sync

Not automated. Once this paper is committed somewhere, copy `figures/*.pdf`
into the Overleaf project's own git-bridge clone and commit/push there --
Overleaf's git bridge is one repo per project and doesn't reliably support a
submodule pointing outside itself, so two repos + a manual copy step is the
standard approach:

```bash
cp figures/*.pdf /path/to/overleaf-clone/figures/
cd /path/to/overleaf-clone && git add figures && git commit -m "update figures" && git push
```

## Provenance

Every `<name>.pdf.json` records which git commit of `ashen` and of this
paper folder produced the figure (and whether either was dirty), plus the
case/step/threshold values used. If a figure looks wrong, check the sidecar
before re-deriving anything by hand.
