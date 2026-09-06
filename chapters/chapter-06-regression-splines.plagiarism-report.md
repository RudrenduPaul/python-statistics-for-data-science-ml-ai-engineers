# Plagiarism & Attribution Audit Report

**Document**: chapter-06-regression-splines.md
**Audit date**: 2026-09-05
**Audited by**: /plagiarism-check

---

## Executive summary

This audit covers the full chapter, with focused verification on the three sections added in
this pass: local regression (LOESS), generalized additive models (GAM), and the partial residual
plot technique. The new content builds on the book's own established checkout-latency dataset
(concurrent load, payload size) and introduces a new two-predictor dataset
(`simulated_load_payload_latency`) with original variable names throughout. No dataset names,
variable names, worked examples, or specific numbers from the gedeck
`practical-statistics-for-data-scientists` reference repository were reused, consistent with the
constraint set in `task-todo/research-gedeck-ch5-7.md`.

Nine targeted web searches on the most distinctive multi-word phrases in the new sections
(the street-map analogy for local regression, the backfitting procedure description, the
partial-residual-plot framing, and several supporting sentences) returned no identical or
near-identical matches against any existing source, including Wikipedia's own "Partial residual
plot" article, whose formula and phrasing differ from this chapter's GAM-specific notation and
prose. The backfitting algorithm description matches the general, unattributable structure of
the technique as covered across many independent sources (ScienceDirect, arXiv, textbook lecture
notes); no single source's wording was mirrored.

Two new citations were added to `references.bib` and used correctly: Cleveland (1979) for LOESS,
the original source for the technique, and Hastie & Tibshirani (1990) for generalized additive
models, the original source for that technique and for backfitting. Both are verifiable Tier 1
academic sources appropriate for a book chapter citing named statistical techniques.

**Verdict: PUBLICATION-READY.**

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~5,000 |
| Sections analyzed (H2) | 10 |
| New sections added this pass | 3 (Local regression, GAMs, partial residual plot as an H3 under GAMs) |
| Phrases extracted for verification | 9 |
| Web searches executed | 9 |
| Critical issues | 0 |
| Warning issues | 0 |
| Note flags | 1 |
| Verified clear | 9 |
| Existing citations in document (before this pass) | 4 |
| Citations added this pass | 2 (`cleveland1979`, `hastietibshirani1990`) |

---

## Critical issues

None found.

---

## Warning issues

None found.

---

## Note flags

**Generalized additive models section, backfitting algorithm description.**
The four-step backfitting procedure (start terms at zero, update one term against the residual
of the others, center, repeat) describes a standard, well-documented statistical algorithm
covered nearly identically in structure (not wording) across many independent textbooks and
papers (Hastie and Tibshirani 1990, ScienceDirect Topics, university lecture notes, arXiv
papers). This is expected: an algorithm's steps are not protectable expression, and the
sentence-level wording here is original. Flagged as a note only because the general shape of a
numbered algorithm description will always show structural similarity to other correct
descriptions of the same algorithm. No fix needed; a citation to Hastie and Tibshirani (1990)
covers this material.

---

## Verified passages

- **LOESS / local regression section**: span, tricube weight formula, and the four-step fitting
  procedure. Cross-checked against Cleveland (1979), the algorithm's original source, cited
  inline as `[@cleveland1979]`. Prose (street-map analogy, production-cost framing, "cutting the
  top of the curve off" description of oversmoothing) returned no web matches; original.
- **GAM section**: the additive-model formula, the backfitting procedure, and the worked
  numeric example (payload coefficient recovered at 8.90 ms/KB against a true value of 9.0
  ms/KB) are drawn from this chapter's own new code
  (`fit_backfit_gam` in `chapter-06-regression-splines-plots.py`), run and its output quoted
  directly rather than invented. Cited to Hastie and Tibshirani (1990) as
  `[@hastietibshirani1990]`.
- **Partial residual plot subsection**: definition and two-step computation checked against
  Wikipedia's "Partial residual plot" article. The underlying algebraic identity is the same
  well-known identity (necessarily, since both describe the same statistical quantity), but the
  notation (GAM term functions $f_j$, not linear coefficients $\hat\beta_i X_i$), the worked
  example, and all prose are original to this chapter. Explicitly framed in-text as "a paraphrase
  of a general diagnostic," not attributed to any single named source, consistent with the task
  instruction to build a device structurally inspired by, and not copied from, any one reference
  implementation.
- **New dataset and code**: `simulated_load_payload_latency` and `fit_backfit_gam` in
  `chapter-06-regression-splines-plots.py` use this book's own established variable names
  (`rho`, `payload_kb`, `latency`) and reuse Chapter 4's own previously-established payload
  coefficient (9 ms/KB) rather than any gedeck dataset, variable name, or number
  (`loan3000`, `borrower_score`, `payment_inc_ratio`, and similar gedeck identifiers do not
  appear anywhere in this chapter).
- **Chapter 4 cross-references**: the claims made about what Chapter 4 established (payload size
  fit close to linear, two-predictor OLS model) are consistent with Chapter 4's own text and are
  not new factual claims requiring separate citation.

---

## Citation health by section

| Section | Paragraphs | Citations | Density | Status |
|---------|-----------|-----------|---------|--------|
| Why a straight line breaks down | 5 | 0 | n/a | OK, queueing-theory formula, no external claim requiring citation |
| Polynomial regression and instability | 4 | 1 (`runge1901`) | 1 per 4 | OK |
| Basis functions | 5 | 0 | n/a | OK, definitional, no external claim |
| Regression splines | 5 | 0 | n/a | OK |
| Natural cubic splines | 3 | 0 | n/a | OK |
| Smoothing splines | 4 | 0 | n/a | OK |
| Local regression (LOESS), new | 8 | 1 (`cleveland1979`) | 1 per 8 | OK |
| Generalized additive models, new | 9 | 1 (`hastietibshirani1990`) | 1 per 9 | OK |
| Partial residual plot, new subsection of GAM | included above | included above | n/a | OK |
| A Bayesian perspective | 6 | 1 (`rasmussenwilliams2006`) | 1 per 6 | OK |

**Citation gaps**: None. This is a technical book chapter working from first principles and
original simulated data; most paragraphs derive results algebraically or from the chapter's own
code rather than citing external claims, consistent with the citation density established in the
pre-existing sections of this chapter.

---

## Recommended action sequence

No action required. Chapter is publication-ready as of this audit.
