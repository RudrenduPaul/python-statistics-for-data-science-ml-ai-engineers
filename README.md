# Applied Statistics for Data Science, Machine Learning, and AI Engineers

Interactive rewrite of the original manuscript
(`../Applied Statistics for Data Science_ A Handbook for Data Scientists and Engineers.docx`,
left untouched as the source draft). This project does not modify or replace that file.

## Structure

Each chapter is a pair of files in `chapters/`:

- `chapter-0N-<slug>.md`: the prose, in Quarto-flavored Markdown. Every interactive figure is
  a `::: {#fig-slug} ... :::` Quarto figure div wrapping a raw-HTML `<iframe>` that points at
  a self-contained page in `_generated/`, so the book never depends on a live Python kernel to
  render.
- `chapter-0N-<slug>-plots.py`: the plotting code for that chapter, standalone and runnable on
  its own (e.g. `python chapters/chapter-01-descriptive-statistics-plots.py`). Each function
  builds one Plotly figure and writes it to `_generated/` as a self-contained interactive HTML
  page that the matching `.md` file embeds. This is the source of truth for regenerating figures.
- `chapter-0N-<slug>-plots.ipynb`: an executed, rendered preview of the same code for browsing
  on GitHub, with each figure's static output baked in as an image (GitHub's notebook viewer
  cannot run the interactive JS the live book uses). Not used by the build; regenerate it by
  rerunning the notebook-export step, not by hand-editing it.

Chapter order:

**Part 1: Foundations of Applied Statistics**
1. `chapter-01-descriptive-statistics`
2. `chapter-02-hypothesis-testing`
3. `chapter-03-probability-and-distributions`

**Part 2: Regression and Predictive Modeling**
4. `chapter-04-regression-modeling` (OLS, logistic regression, regularization, plus a short
   Bayesian preview)
5. `chapter-05-cross-validation`
6. `chapter-06-regression-splines`
7. `chapter-07-decision-trees-random-forests`
8. `chapter-08-gradient-boosting` (XGBoost and LightGBM)

Chapters 4 through 8 each close with a short "A Bayesian perspective" preview of the chapter's
classical method; Part 3 gives each of those a full standalone chapter.

**Part 3: Bayesian Methods**
9. `chapter-09-bayesian-regression` (Bayesian linear/logistic regression, Ridge=Gaussian prior,
   Lasso=Laplace prior, horseshoe prior, credible vs. confidence intervals, PyMC/ArviZ)
10. `chapter-10-bayesian-model-selection` (PSIS-LOO, WAIC, `arviz.compare()`)
11. `chapter-11-bayesian-gaussian-processes` (GPs as the Bayesian counterpart to smoothing
    splines, scikit-learn's `GaussianProcessRegressor`)
12. `chapter-12-bayesian-bart` (Bayesian Additive Regression Trees, `pymc-bart`)
13. `chapter-13-bayesian-boosting` (Bayesian hyperparameter optimization via Optuna, NGBoost,
    honestly scoped as the thinnest Bayesian treatment in the book)
14. `chapter-14-bayesian-ab-testing` (Beta-Binomial conjugacy, win probability, expected loss,
    the closing case for the whole book)

15. `chapter-15-conclusion`

## Building

```bash
pip install -r requirements.txt
python chapters/chapter-01-descriptive-statistics-plots.py   # regenerate figures for one chapter
quarto render                                                  # build the book locally
```

Every chapter's `-plots.py` file needs to be run once (or whenever the figure code changes) to
populate `_generated/` before `quarto render`. Requires the Quarto CLI
(`brew install --cask quarto`).

## License

Book text is licensed under CC BY 4.0 (see `LICENSE`). Plotting and figure-generation code
is licensed under MIT (see `LICENSE-CODE`).
