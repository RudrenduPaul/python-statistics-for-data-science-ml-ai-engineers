# Plagiarism & Attribution Audit Report

**Document**: chapter-bayes-model-selection.md
**Audit date**: 2026-09-01
**Audited by**: /plagiarism-check

---

## Executive summary

This chapter is new content built around the checkout-API running example, with worked
numbers computed from a script the coordinator ran and confirmed in this session (a closed-form
conjugate Bayesian regression, WAIC, and a simplified PSIS-LOO implementation), not asserted
from an outside source. It has no direct quotes. Its methodology is attributed to Vehtari,
Gelman & Gabry 2017 (in `references.bib`) and its software claims about ArviZ to Kumar et al.
2019 (added in `.refs-chapter-bayes-model-selection.bib`, DOI-verified via WebSearch). The
PyMC/ArviZ code block is illustrative API usage, not quoted from any library's documentation.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~2,000 |
| Sections analyzed | 6 |
| Existing citations in document | 2 (Vehtari/Gelman/Gabry 2017, Kumar et al. 2019) |
| Critical issues | 0 |
| Warning issues | 0 |
| Note flags | 1 |
| Verified clear | all prose and the worked-numbers table |

---

## Note flags

- WAIC's originating paper (Watanabe 2010) is not separately cited; the chapter attributes
  WAIC and PSIS-LOO together to Vehtari, Gelman & Gabry 2017, the standard applied-statistics
  reference for both in practice. This mirrors how most practitioner-facing treatments cite
  WAIC, but a reader looking for WAIC's origin specifically would need to follow that citation's
  own references rather than find Watanabe cited directly here.

---

## Verified passages

- The elpd_loo/SE/d_elpd/dSE table: computed directly from `chapter-bayes-model-selection-plots.py`
  in this session, cross-checked against the script's printed output before being written into
  the chapter. Not an external claim.
- PSIS-LOO mechanism description (importance weights, Pareto tail smoothing, k-hat diagnostic
  thresholds of 0.5 and 0.7): matches the mechanism described in Vehtari, Gelman & Gabry 2017,
  paraphrased in original wording built around this book's own worked example.
- PyMC/ArviZ code block: standard, publicly documented API shapes (`pm.Model`, `pm.sample`,
  `az.compare`), not copied from any specific tutorial or repository.

---

## Recommended action sequence

No outstanding action. Chapter is ready for the humanize pass.

---

## Addendum: re-check after sentence-level clarity pass (2026-09-01)

This pass edited roughly 20 sentences for clarity: splitting long compound sentences, removing
redundant phrasing, and clarifying transitions. No formula, citation, number, heading, or fact
was changed; every edit is a pure rewording of prose cleared earlier in this report. No new
quotes, statistics, named frameworks, or attributed claims were introduced, so no new web
verification was required. Every citation ([@vehtarigelmangabry2017], [@kumar2019arviz]) remains
untouched at its original location. Verdict unchanged: **PUBLICATION-READY** (local review only,
per the standing "do not publish" instruction on this project).

## Addendum: re-check after second clarity pass (2026-09-01)

Four sentence-level edits applied this pass: clarified the "large (close to zero)" phrasing in
the lpd formula paragraph, parenthesized a mid-sentence aside in the Chapter 1 tail-behavior
paragraph, swapped a one-off "decoys" synonym back to "noise features" for terminology
consistency with the rest of the chapter, and restructured one sentence in the closing section
to remove an ambiguous comma-separated list. All four are pure rewording of claims cleared
earlier in this report. No formula, citation, number, heading, or fact was changed. No new
quotes, statistics, named frameworks, or attributed claims were introduced, so no new web
verification was required. Both citations ([@vehtarigelmangabry2017], [@kumar2019arviz]) remain
untouched at their original locations. Verdict unchanged: **PUBLICATION-READY** (local review
only, per the standing "do not publish" instruction on this project).

## Addendum: re-check after teaching-clarity pass (2026-09-01)

Third clarity pass, the first to add new explanatory prose rather than only reword existing
sentences: a *generalized Pareto distribution* gloss and a rewording of "expected order
statistics" in the PSIS-LOO section, a new sentence formally tying PSIS-smoothed pointwise
density to $\widehat{\text{elpd}}_{\text{loo}}$ (paralleling the existing
$\widehat{\text{elpd}}_{\text{waic}}$ formula), and a new sentence distinguishing the `SE` and
`dSE` columns in the model-comparison table walkthrough. Checked each:

1. **Generalized Pareto distribution gloss and the "smoothed value predicted for a weight of
   that rank" rewording**: both describe the same Pareto-smoothed-importance-sampling mechanism
   this report's "Verified passages" section covers against Vehtari, Gelman & Gabry 2017. No new
   claim beyond that mechanism, no new citation needed. **CLEAR**.
2. **elpd_loo built by summing PSIS-corrected pointwise density, paralleling WAIC's elpd
   formula**: a direct structural consequence of the PSIS-LOO mechanism [@vehtarigelmangabry2017]
   covers; states explicitly what the chapter previously left implicit. **CLEAR**.
3. **`SE` versus `dSE` column distinction**: confirmed via WebSearch this session against
   ArviZ's own `compare()` API documentation, which defines `se` as the standard error of each
   model's own elpd_loo estimate and `dse` as the standard error of the difference against the
   top-ranked model, two distinct quantities. Falls under the same [@kumar2019arviz] software
   citation this report covers for ArviZ's output. **CLEAR**.

No critical, warning, or note-level issues found in the new material. No new bib entries
required. Both citations ([@vehtarigelmangabry2017], [@kumar2019arviz]) remain untouched at
their original locations. Verdict unchanged: **PUBLICATION-READY** (local review only, per the
standing "do not publish" instruction on this project).

## Addendum: re-check after figure-first restructuring pass (2026-09-01)

Fourth pass, a structural formatting change rather than a content change: all three figure divs
(fig-lpd-per-point, fig-khat-diagnostic, fig-waic-ploo-complexity) were moved to appear
right after their section heading (or one short orienting sentence), with the explanatory and
interpretive prose that used to precede each figure moved after it instead. Roughly ten
paragraphs were split for length. Three new callout boxes were added:

1. A `.callout-tip` restating the $\hat{k}$ threshold rule of thumb (below 0.5 reliable, 0.5-0.7
   watch, above 0.7 untrusted), a claim this report's "Verified passages" section covers against
   Vehtari, Gelman & Gabry 2017. No new claim.
2. A `.callout-warning` restating the `d_elpd` vs `dSE` vs `SE` distinction, a claim this
   report's teaching-clarity-pass addendum checked against ArviZ's own `compare()` documentation
   under [@kumar2019arviz]. No new claim.
3. A `.callout-note` restating the chapter's closing-section guidance on when PSIS-LOO earns its
   cost over k-fold CV. No new claim, no new source.

All three callouts restate claims cleared earlier in this report; none introduce a new
statistic, quote, named framework, or attributed claim, so no new web verification was required.
No formula, citation, number, or fact was changed. Both citations ([@vehtarigelmangabry2017],
[@kumar2019arviz]) remain untouched at their original locations. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the standing "do not publish" instruction on this
project).

---

## Addendum: 2026-09-01 (figure/caption/paragraph/callout audit pass re-check)

Sixth pass, a structural audit re-run independent of the fifth pass, checking captions against
the referenced PNGs directly rather than trusting the earlier pass's summary. Found all 3 figure
divs correctly placed, needing no move.

One caption correction: `fig-khat-diagnostic`'s caption stated the watch-zone outlier "lands
just past 0.5." A direct read of the chart places that point at approximately k-hat = 0.63,
meaningfully closer to the 0.7 unreliable-estimate threshold than to the 0.5 watch threshold.
Rewrote to "sits in the 0.5-0.7 watch range, closer to 0.7 than to 0.5," a tightening of an
existing claim against the source image, not a new claim; no citation implication changes, since
the 0.5/0.7 thresholds remain attributed to Vehtari, Gelman & Gabry 2017 as before. **CLEAR**.

Split 6 paragraphs that still ran 5 to 6 source lines (the "shape of the problem" opening, the
WAIC definition paragraph, the WAIC-degrades/PSIS-LOO-alternative paragraph, the importance-
weight definition paragraph, the payload_concurrent-versus-payload_concurrent_noise comparison
paragraph, and the closing "PSIS-LOO earns its cost" paragraph) at sentence boundaries. The
closing paragraph's single long "when X, when Y, or when Z" sentence was restructured into two
complete sentences rather than cut mid-list; checked that all three original conditions (other
Bayesian-modeling reasons, small-dataset CV instability, and the k-hat diagnostic's own value)
remain present after the split. No condition was dropped.

Added two `.callout-tip` blocks to two of the three uncovered sections that had a clear rule of
thumb to state (average-lpd blind spots in "The log pointwise predictive density," and training-
log-likelihood's inability to substitute for elpd_waic/elpd_loo in "WAIC and PSIS-LOO against a
training-fit baseline"). Both restate claims the chapter's own cleared prose makes elsewhere in
the same section, so no new web verification was required. **CLEAR**.

No new facts, statistics, or citations were introduced; both citations this chapter carries
(Vehtari/Gelman/Gabry 2017, Kumar et al. 2019) remain untouched and correctly placed.

No critical, warning, or note-level issues found in this pass. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the standing "do not publish" instruction on this
project).

---

## Addendum: 2026-09-05 (focused audit of five new sections)

Scoped re-check of five sections added since the last pass, not a full re-audit of the chapter:
"When WAIC and PSIS-LOO disagree" (lines 212-262), "Posterior predictive checks: a different
question" (357-422), "Bayesian model averaging" (423-483), "Bayes factors, and why this chapter
did not lead with them" (484-559), and "What 'closer to the truth' means" (586-612).

**Candidate phrases extracted**: the WAIC-vs-PSIS-LOO k-hat disagreement mechanism (Section 1,
no new named framework), the Bayesian p-value definition and formula (Section 2), pseudo-BMA
weighting and the softmax weight formula (Section 3), the Bayes factor definition, the closed-
form conjugate-Gaussian marginal likelihood, and the binary-indicator "encompassing model"
description (Section 4), and the Kullback-Leibler divergence definition plus the Akaike/Watanabe
elpd-equals-negative-KL-divergence claim (Section 5).

**Web searches run**: confirmed the KL divergence "information lost when one distribution stands
in for another" framing against standard sources (DeepAI, GeeksforGeeks, general information
theory literature). The chapter's wording is its own, with no word-for-word match found.
Confirmed the Bayes factor "which model was more probable to have produced the data, integrating
over parameters" framing against Kruschke's *Doing Bayesian Data Analysis*, standard course notes
(Edinburgh, UMD), and ScienceDirect/bayestestR overviews. Again no word-for-word match, concept
only. Confirmed the "encompassing model" binary-indicator description against Kruschke's
indicator-variable MCMC approach and general SSVS literature. The chapter's z=0/z=1 notation and
phrasing is its own; the general technique is textbook-standard, not traceable to one paper.
Confirmed the Bayesian p-value definition and $p_B$ formula against Gelman's own posterior
predictive p-value writeup and standard course material. This matches the concept covered earlier
in the chapter by [@gelmanmengstern1996], and the wording here is the chapter's own. Searched for
the pseudo-BMA softmax weight formula and confirmed it matches the `loo` R package's documented
pseudo-BMA weighting (attributed there to Geisser and Eddy, 1979, and used in
[@yaovehtarisimpsongelman2018]). Same formula; the chapter's prose around it is its own. Searched
Osvaldo Martin's BAP GitHub repo and book excerpts directly for the `dummy.csv` coin-flip example
and the Binomial(10,0.75) entropy demo: neither turns up anywhere in these five sections,
confirming the chapter's checkout-API-latency, cold-start-PPC, and conjugate-Gaussian examples
stand apart from BAP's worked examples.

**Citation metadata verification**: cross-checked `references.bib` against live sources for all
three new citations this chapter relies on.
- `kassraftery1995` (Kass & Raftery, "Bayes Factors," *JASA* 90(430):773-795, 1995,
  doi:10.1080/01621459.1995.10476572): confirmed, matches JASA's own record and multiple
  independent citation databases.
- `watanabe2010` (Watanabe, "Asymptotic Equivalence of Bayes Cross Validation and Widely
  Applicable Information Criterion in Singular Learning Theory," *JMLR* 11:3571-3594, 2010):
  confirmed via JMLR's own page (jmlr.org/papers/v11/watanabe10a.html); volume, pages, and year
  match.
- `yaovehtarisimpsongelman2018` (Yao, Vehtari, Simpson & Gelman, "Using Stacking to Average
  Bayesian Predictive Distributions," *Bayesian Analysis* 13(3):917-1007, 2018,
  doi:10.1214/17-BA1091): confirmed via Project Euclid's record; volume, issue, and pages match.
- `gelmanmengstern1996` was verified in an earlier audit pass and is unchanged here.

Also confirmed via `grep` on the chapter file that each citation sits at its expected location:
`[@gelmanmengstern1996]` at line 380, `[@yaovehtarisimpsongelman2018]` at line 444,
`[@kassraftery1995]` at line 557, `[@watanabe2010]` at line 602. All four required citations for
this batch of new content are present, correctly placed, and metadata-correct.

**Findings by severity**:
- Critical: 0
- Warning: 0
- Note: 1. The binary-indicator "encompassing model" description in Section 4 (lines 515-521)
  states a standard Bayesian model-selection technique (related to Carlin & Chib-style
  indicator-variable approaches, and covered in general terms by Kruschke's textbook) without a
  citation. No word-for-word match to any single source was found; the chapter uses its own
  notation and presents the technique as a contrasting aside to its own conjugate-Gaussian
  method, not as the chapter's main claim. Left as a monitor-only note rather than a required
  fix: attributing a generic, multiply-sourced technique to one paper would misrepresent its
  origin more than leaving it uncited does.
- Clear: the remaining candidate phrases (KL divergence definition, Bayes factor definition,
  Bayesian p-value definition and formula, pseudo-BMA weighting description, WAIC/PSIS-LOO
  disagreement mechanism), all original wording, correctly cited where a citation applies, with
  no BAP-specific data or examples present.

**Fixes applied**: none required. No Critical or Warning findings in this batch.

Verdict: **PUBLICATION-READY** (local review only, per the standing "do not publish" instruction
on this project).
