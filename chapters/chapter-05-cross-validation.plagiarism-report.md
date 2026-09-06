# Plagiarism & Attribution Audit Report

**Document**: chapter-05-cross-validation.md (renamed from chapter-cv-cross-validation.md during
the book-wide renumbering pass)
**Audit date**: 2026-08-31
**Audited by**: /plagiarism-check

**Addendum, 2026-09-01 (plain-language layer pass)**: two short intuition paragraphs added
(training-vs-test-error framed as studying practice questions vs. sitting the test itself,
PSIS-LOO framed as estimating a modified-test score without retaking it), both generic teaching
analogies with no direct quotes, statistics, or named frameworks introduced. No web-search sweep
was warranted. Classified CLEAR by inspection. Verdict unchanged: PUBLICATION-READY.

---

## Executive summary

This chapter is new content (no raw docx source), built around the Chapter 4 latency/regularization
running example and standard, public-domain cross-validation theory. Both named citations
(Stone 1974 for cross-validation, Vehtari, Gelman, and Gabry 2017 for PSIS-LOO/WAIC) were
verified live via WebSearch and WebFetch during drafting, matching the title, journal, volume,
issue, and page range reported by independent sources. No direct quotes, no uncited statistics
making an external factual claim, and no distinctive phrasing traced to an outside source were
found.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~2,350 (pre-2026-09-05 addition; see addendum below for the added section) |
| Sections analyzed | 8 (plus references) |
| Paragraphs | ~24 |
| Phrases extracted for verification | 2 (the two named citations) |
| Web searches and fetches executed | 4 (during drafting, this session) |
| Critical issues | 0 |
| Warning issues | 0 |
| Note flags | 0 |
| Verified clear | 2 citations, all other content |
| Existing citations in document | 2 (Stone 1974, Vehtari/Gelman/Gabry 2017) |

---

## Verified passages

- **Stone (1974) citation**: confirmed via WebFetch against the Oxford Academic listing for
  *Journal of the Royal Statistical Society: Series B*, volume 36, issue 2, pages 111-133,
  matching the in-text description of the paper's contribution.
- **Vehtari, Gelman, and Gabry (2017) citation**: confirmed via WebFetch against the arXiv
  preprint (1507.04544), matching the published version in *Statistics and Computing*
  27(5):1413-1432, and the description of PSIS as a regularization procedure for importance
  weights.
- **All formulas, the bias-variance trade-off argument, and the k=5/k=10 heuristic**: standard,
  public-domain statistical methodology with no single attributable origin, consistent with
  how the chapter's model text (book-writing-style-islr) treats well-established results.
- **The entire worked example** (the latency model's lambda selection, the validation-strategy
  variance comparison, the ELPD-by-complexity figure): newly authored for this chapter, built
  on the Chapter 4 running example, no source match expected or found.

---

## Citation health by section

| Section | Paragraphs | Citations | Status |
|---------|-----------|-----------|--------|
| Leave-one-out cross-validation | 2 | 1 | OK |
| A Bayesian perspective | 4 | 1 | OK |
| All other sections | ~18 | 0 | OK, standard methodology and original worked example |

No run of 3+ paragraphs containing an unattributed claim that required a citation.

---

## Recommended action sequence

None. No fixes required.

---

## Addendum: what/why/how and rigor pass (2026-09-01)

One targeted edit, audited fresh: the PSIS-LOO paragraph previously stated the method "stays
reliable even when a handful of observations wield an unusually large influence on the fit,"
an overclaim given the method's own reliability limits. Verified via WebSearch against the
`loo` package documentation and the Vehtari, Gelman, and Gabry (2017) paper cited in this
chapter: reliability holds only when the Pareto shape parameter k stays below a threshold
(roughly k < 0.7); above it, the estimate gets flagged as unreliable and needs a direct refit.
Rewrote the sentence to state this correctly, still citing `[@vehtarigelmangabry2017]`, and
adjusted the later WAIC-comparison sentence to avoid repeating the same diagnostic explanation
twice. No new citation added. No direct quotes, no fabricated URLs.

Verdict for this addition: **CLEAR**. No critical or warning findings.

---

## Addendum: sentence-level clarity pass (2026-09-01)

A fourth-pass clarity edit reworded roughly 17 sentences throughout the chapter: splitting
overlong sentences (the training-vs-test-error closing paragraph, the validation-set noise
paragraph, the LOOCV computational-cost sentence, the bias-variance trade-off paragraph, the
lambda-selection closing sentence, the data-leakage and time-series paragraphs, the Pareto-k
reliability sentence, and the closing WAIC/Bayesian-route paragraph), removing one filler opener
("Note that"), and rewriting one ambiguous "None of this" pronoun reference to name what it
refers to. No formula, citation, figure reference, fact, or number was changed. No new claim,
statistic, quote, or named framework was introduced, so no web search was warranted for this
pass. Spot-checked the edited passages against the citations verified above (Stone 1974,
Vehtari/Gelman/Gabry 2017): each citation's placement and the claim it supports are unchanged.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: second clarity pass (2026-09-01)

Three edits, audited fresh:

- **Terminology fix**: "noise feature" in the opening paragraph (referring back to Chapter 4's
  regularization example) changed to "noise predictor" to match that chapter's own dominant term
  for the same variable, corrected there in the same pass.
- **Precision fix, no new claim**: the PSIS-LOO paragraph said it was "the Bayesian analog most
  directly comparable to k-fold CV." PSIS-LOO is itself a leave-one-out method, and this chapter
  treats LOOCV as the special case of k-fold CV where k=n, so "classical LOOCV" is the closer
  match and reads consistently with the rest of the paragraph, which compares PSIS-LOO's
  refitting cost directly against LOOCV's. No citation or fact changed.
- **Missing citation added**: the WAIC sentence introduced the Watanabe-Akaike information
  criterion with no citation of its own, even though the paper cited two sentences earlier in the
  same paragraph, Vehtari, Gelman, and Gabry (2017), covers WAIC directly (the paper's own title
  is "Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC," per
  `references.bib`). Added `[@vehtarigelmangabry2017]` to the WAIC sentence. No new source
  introduced; this is the same citation verified in the original audit above.

Ran WebSearch checks against five distinctive phrases from this chapter and chapters 4 and 6
(the Runge-phenomenon description, the leave-one-out "removes the arbitrariness" phrasing, the
Gaussian-process function-space framing, the cubic-spline derivative-matching description, and the
Bayesian-regression opening framing) as part of this pass's audit; no close or near-close source
match was found for any of them, consistent with original, chapter-authored prose.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: clarity and jargon-grounding pass (2026-09-01)

Four sentence-level edits, audited fresh, none introducing a new fact, statistic, quote, or named
framework requiring a web-search sweep:

- **LOOCV shortcut, named concretely.** "Certain linear models have a shortcut" became "certain
  linear models, ordinary least squares among them, have a closed-form shortcut," followed by a
  plain-language description of the mechanism (recovering each leave-one-out error from the
  single full-data fit rather than refitting $n$ times). This states standard, public-domain
  linear-regression theory with no single attributable origin, the same category the original
  audit above classified as CLEAR for this chapter's formulas and heuristics.
- **Cross-chapter Recall bridge added.** A sentence pointing the Bayesian-perspective section back
  to Chapter 4's own "A Bayesian perspective" section (prior-narrows-to-posterior analogy). Pure
  cross-reference, no new claim.
- **"Posterior draws" defined on first use**, tied to the Markov chain Monte Carlo mention present
  in Chapter 4. States a standard definition of an MCMC draw; no citation needed beyond the
  general statistical-methodology treatment this chapter relies on for cross-validation and
  Bayesian model-comparison theory.
- **"Pareto-k" identified as a Pareto shape-parameter estimate.** This restates a fact verified in
  this report's prior addendum (2026-09-01, what/why/how pass), which confirmed against `loo`
  package documentation and Vehtari, Gelman, and Gabry (2017) that reliability holds only when the
  Pareto shape parameter k stays below roughly 0.7. No new source consulted; this is the same
  citation and the same verified fact, restated with the parameter's identity made explicit.

No direct quotes, no fabricated URLs, no attribution gap introduced. Spot-checked both citations
in this chapter (Stone 1974, Vehtari/Gelman/Gabry 2017) against the edited passages: placement and
supported claims are unchanged.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: figure-first restructure and callout pass (2026-09-01)

Every figure div was moved to appear immediately after the section heading, or after the
minimal definitional setup a chart's axis labels require (the k-fold definition ahead of the
validation-variance comparison, the ELPD definition ahead of the ELPD-by-complexity chart), with
all interpretive prose, including the sentence originally naming the figure, moved to follow the
image. All four captions were rewritten to state the chart's takeaway. Roughly a dozen overlong
paragraphs were split to stay within 3-4 lines each. Three new callout boxes (`.callout-note` on
training error only ever falling or holding flat as flexibility rises, `.callout-tip` on the
OLS leave-one-out shortcut, `.callout-warning` on preprocessing-driven data leakage) were added.

Audited against Categories A-E: no direct quotes, no new statistics, no new named framework or
methodology, and no attributed claim lacking a source. All three callouts restate a claim this
report verified as CLEAR in an earlier pass (the training-versus-test-error distinction from the
opening section; the closed-form LOOCV shortcut named in the "clarity and jargon-grounding pass"
addendum above; the data-leakage mechanism from the "what cross-validation does not fix"
section) in new, shorter phrasing. No formula, citation, fact, or number was changed. Spot-checked
both citations in this chapter (Stone 1974, Vehtari/Gelman/Gabry 2017) against the reordered
passages: placement and supported claims are unchanged.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: paragraph-split, callout, and placement re-check (2026-09-01)

Re-verified all 4 figure placements and all 4 captions against their PNGs: no change needed,
both hold up as accurate. Split two paragraphs at existing sentence boundaries (no wording
changed). Added three new callout boxes, each restating a claim present and verified in this
chapter's own preceding prose: preferring k-fold over a single split below a few hundred
observations, the k=5/k=10 default as a quick-reference rule, and the time-series shuffling
pitfall from non-independent data. No direct quotes, no new statistics, no new named framework,
no citation touched.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: from-scratch k-fold grid-search section added (2026-09-05)

**Scope of this pass**: audits only the new section, "Building k-fold cross-validation from
scratch: tuning two knobs at once," inserted between "Cross-validation for model selection:
picking lambda" and "What cross-validation does not fix." Everything above this addendum was
unchanged by this pass and remains as previously audited.

**Source and originality check**: this section was written to mirror a *structural* pattern
found in gedeck's "Statistical Machine Learning" notebook (from-scratch fold assignment, a
nested hyperparameter grid-search loop, a pivot-table summary), per the research notes at
`task-todo/research-gedeck-ch5-7.md`. Per that research file's own instruction, no gedeck
wording, variable names, dataset, or numbers were reused. Concretely:

- **Domain**: gedeck's example is consumer lending (`loan_data.csv.gz`, learning rate and max
  tree depth tuned for XGBoost). This section uses a different domain (mobile app-crash
  telemetry), a different model class (k-nearest-neighbor regression, not boosting), and
  different hyperparameters (neighbor count and weighting scheme, not learning rate and depth).
- **Variable and dataset names**: `heap_pressure`, `hours_since_restart`, `fold_id`,
  `neighbor_grid`, `weight_grid`, `cv_error`, `knn_predict`, `knn_grid_search_cv`,
  `make_crash_dataset`, `true_crash_rate`. None overlap with gedeck's `loan200`, `loan3000`,
  `borrower_score`, `payment_inc_ratio`, or any lending-domain identifier.
- **Numbers**: every CV-error value in the pivot table (16.46 through 10.05) comes from this
  chapter's own simulated crash-telemetry dataset (seeded, `np.random.default_rng`), generated
  and computed fresh for this section; none match or were derived from gedeck's reported figures.
- **Prose**: the "why it matters" argument (a one-hyperparameter-at-a-time search can miss the
  joint optimum a full grid search finds) is a general, well-established property of grid search
  discussed across many independent sources (see web search below), not unique phrasing traced to
  gedeck or any other single source.

**Web search sweep** (Category D, distinctive phrases): ran two WebSearch queries against the
most distinctive phrases in the new section, `"distance weighting only pays off once" k-nearest
neighbors` and `"a search that tunes one at a time, holding others at a default" hyperparameters`.
Neither returned a matching or closely similar result. The second query surfaced general
discussion of one-at-a-time hyperparameter tuning as a known, named alternative to grid search
(e.g., "Importance of Tuning Hyperparameters of Machine Learning Algorithms," arXiv:2007.07588),
which independently confirms this is standard, publicly documented methodology with no single
attributable origin, the same category this report applies elsewhere to the chapter's other
heuristics (k=5/k=10 default, the bias-variance trade-off argument).

**Categories A-E review**:
- A (direct quotes): none in the new section.
- B (statistics): all numeric values (10.05, 10.57, 10.62, "roughly 5 percent," 240 observations,
  70 fits) are outputs of this chapter's own code, computed and verified against
  `chapter-05-cross-validation-plots.py`'s `knn_grid_search_cv()` function, not external claims
  requiring citation.
- C (named frameworks): `GridSearchCV`, `cross_val_score`, `KNeighborsRegressor`,
  `StandardScaler` are scikit-learn API names, referenced the same uncited way this book
  references `LogisticRegression`, `RandomForestClassifier`, and `cross_val_score` elsewhere
  (see chapters 10 and 13's existing, unflagged usage of `cross_val_score`).
- D (distinctive phrases): checked above, no match found.
- E (attributed claims): none; no organization or researcher named as a source in the new text.

**Verification against the chapter's own figure code**: read `knn_grid_search_cv()` and
`fig_knn_grid_search()` in `chapter-05-cross-validation-plots.py` directly and confirmed every
number quoted in prose (10.57 at k=5/uniform, 10.62 at k=5/distance, 10.05 at k=25/distance, and
the resulting roughly 5 percent gap) matches the function's computed output, re-run and printed
during drafting.

Verdict for this addition: **CLEAR**. No critical or warning findings.

Verdict for the chapter overall remains: **PUBLICATION-READY** (local review only, not for public
release per the standing "do not publish" instruction on this project).

---

## Addendum: post-humanize re-check on the new section (2026-09-05)

The humanize pass (see `.humanize-log-chapter-05-cross-validation.md`) reworded five sentences in
the new section to remove negative-parallelism constructions and standardized five variant
phrasings of "one-at-a-time search" into one consistent term. No fact, number, citation, or
formula was changed by that pass; only sentence structure and terminology were rewritten.
Re-checked the reworded sentences against Categories A-E: no new quote, statistic, named
framework, or attributed claim was introduced by the rewording. Every number quoted in the new
section (10.57, 10.62, 10.05, roughly 5 percent) matches the version verified above.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.
