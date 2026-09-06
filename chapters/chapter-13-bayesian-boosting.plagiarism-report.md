# Plagiarism & Attribution Audit Report

**Document**: chapter-13-bayesian-boosting.md
**Audit date**: 2026-09-05
**Audited by**: /plagiarism-check (scoped to the newly added "Quantile regression" and
"Conformal prediction" sections; the rest of the chapter carries its own prior audit history,
documented in this chapter's `.humanize-log-chapter-13-bayesian-boosting.md`)

---

## Executive summary

This pass audits the two sections added to extend the chapter's coverage from Bayesian
hyperparameter optimization and NGBoost into quantile regression and conformal prediction:
"Quantile regression: predicting the tail, not the average" and "Conformal prediction: a
coverage guarantee without a posterior," plus the short bridging paragraphs added at the end of
the NGBoost section and before "Where the field stands, honestly."

Three new citations were added to `references.bib` in this pass: Koenker and Bassett (1978) for
the origin of quantile regression, Vovk, Gammerman, and Shafer (2005) for the origin of conformal
prediction, and Romano, Patterson, and Candès (2019) for conformalized quantile regression (CQR),
the technique the "Conformal prediction" section builds. All three were verified live via web
search against author names, title, venue, and year before being added; none were fabricated.

The worked example (a checkout-latency capacity-planning scenario, its synthetic dataset, and
every number quoted in the prose) is generated from a new simulation written for this pass
(`simulate_checkout_latency` in `chapter-13-bayesian-boosting-plots.py`, seeded independently of
the chapter's existing figures) and is not drawn from, or modeled closely on, any external
dataset, textbook example, or published tutorial. No direct quotes appear in either new section.
No statistic in the new prose originates outside this chapter's own simulation. Verdict:
**PUBLICATION-READY**.

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total chapter word count | ~4,240 |
| New sections audited | 2 (plus 2 short bridging paragraphs) |
| New paragraphs audited | ~24 |
| New citations added | 3 |
| Web searches executed | 3 |
| 🔴 Critical issues | 0 |
| 🟡 Warning issues | 0 |
| 🔵 Note flags | 1 |
| ✅ Verified clear | 3 citations, 2 code examples, 1 worked example/dataset |
| Existing citations in document (pre-pass) | 11 |

---

## 🔴 Critical issues

None found.

---

## 🟡 Warning issues

None found.

---

## 🔵 Note flags

**["Conformal prediction" section, paragraph on the conformal-prediction family]**
Severity: 🔵
Phrase: "one member of the conformal-prediction family Vovk, Gammerman, and Shafer built into a
general framework for distribution-free prediction"
Reason: names a well-established field-origin claim (conformal prediction as a research program
originating with Vovk, Gammerman, and Shafer). The claim itself is verified accurate (see
Verified passages below), and the sentence is original phrasing, not a copied description from
any single source. Flagged only because origin-attribution sentences are worth a second human
read before publication, per this audit's own convention for Category C findings. No fix
required; monitor on the next full-chapter pass.

---

## ✅ Verified passages

1. **Koenker, R., and Bassett, G., Jr. (1978). "Regression Quantiles." Econometrica, 46(1),
   33-50.** Confirmed via live web search (Econometric Society, RePEc, JSTOR stable ID 1913643):
   author names, title, journal, volume, issue, and page range all match the citation added to
   `references.bib` as `koenkerbassett1978`. [Econometric Society](https://www.econometricsociety.org/publications/econometrica/1978/01/01/regression-quantiles)

2. **Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic Learning in a Random World.
   Springer.** Confirmed via live web search (Royal Holloway Research Portal, Amazon, ALRW.net):
   author names, title, publisher, and year all match the citation added to `references.bib` as
   `vovkgammermanshafer2005`. [Royal Holloway](https://pure.royalholloway.ac.uk/en/publications/algorithmic-learning-in-a-random-world(55595013-e01f-4d0a-bd12-f66422d4289e).html)

3. **Romano, Y., Patterson, E., and Candès, E. J. (2019). "Conformalized Quantile Regression."
   NeurIPS 32.** Confirmed via live web search (official NeurIPS proceedings page, arXiv
   1905.03222): author names, title, and venue match the citation added to `references.bib` as
   `romanopattersoncandes2019`. [NeurIPS proceedings](https://papers.nips.cc/paper/8613-conformalized-quantile-regression)

4. **The checkout-latency worked example.** Original synthetic scenario and dataset built for
   this pass: a gamma-distributed latency model driven by request load through a queueing-style
   utilization term, with shape shrinking as utilization rises to create the right-skew and
   heteroscedasticity the section's argument depends on. Every number in the prose (216 ms mean
   prediction, 434 ms P95 prediction, 452 ms measured P95, 400 ms SLA, the 90%/92%/74%/89%/100%
   coverage figures, the 0.3 ms calibration adjustment) was read directly from a script run
   against this simulation, not estimated or copied from any external source.

5. **The two Python code examples** (fitting `GradientBoostingRegressor(loss="quantile", ...)`
   and the split-conformal calibration loop) are short, mechanical usage of public library APIs
   (scikit-learn) and a five-line implementation of the standard CQR calibration formula from
   Romano et al. (2019), which the surrounding prose cites. Neither is copied from
   scikit-learn's own documentation examples or from the CQR paper's reference implementation;
   both were written fresh against this chapter's own variable names and dataset.

6. **Named technical terms without a dedicated citation** ("pinball loss" / "check function,"
   "quantile crossing," "split conformal prediction"): these are treated the same way the
   existing chapter treats "expected improvement" and "natural gradient descent," as established
   field vocabulary rather than single-source claims requiring their own citation. Quantile
   regression's origin is credited to Koenker and Bassett (1978); conformal prediction's origin is
   credited to Vovk, Gammerman, and Shafer (2005); CQR specifically is credited to Romano,
   Patterson, and Candès (2019). No further named-framework citation gap exists in the new
   content.

---

## Citation health by section

| Section | Paragraphs | Citations | Density | Status |
|---------|-----------|-----------|---------|--------|
| Quantile regression: predicting the tail, not the average | 8 | 1 | 1 per 8 paras | OK (concept has a single clean origin citation; remaining paragraphs build a self-contained worked example) |
| Conformal prediction: a coverage guarantee without a posterior | 11 | 2 | 1 per 5.5 paras | OK |

**Citation gaps**: None. Both new sections cite their originating source before extending into
the worked example, matching the density pattern used elsewhere in this chapter (for instance,
the "Bayesian hyperparameter optimization" section cites Akiba et al. once for Optuna and then
runs several paragraphs of worked example and tooling guidance without a further citation).

---

## Recommended action sequence

No action required. This pass introduced zero Critical or Warning findings. The one Note flag
above is a monitoring note, not a blocking issue, and needs no edit before publication.

---

## Addendum (2026-09-05): scoped audit of the NGBoost multimodality callout

A follow-up gap-analysis pass, cross-referencing this chapter against BAP's mixture-model
chapter, added one callout note inside the "NGBoost: a boosted model that outputs a
distribution" section (between the NGBoost predictive-interval discussion and the transition
into "Quantile regression"). No other content in the chapter changed in this pass.

**Content added**: a callout stating that NGBoost fits one distribution family (typically
Normal) at every point, which breaks down when the true conditional distribution is multimodal,
illustrated with a new, self-contained hypothetical (a two-path deployment pipeline: ~3-minute
fast path when a cached build image exists, ~13-minute slow path when the image must be
rebuilt, 40%/60% mix), showing a Normal-distribution NGBoost fit would report a mean near 9
minutes that few deployments report.

**Web verification performed**: two searches. (1) `"NGBoost" "single distribution family"
multimodal residuals mixture` confirmed NGBoost's own documentation/paper describes it as usable
with "any family of distributions with continuous parameters," meaning one chosen family per
model, not a per-point mixture output; no evidence turned up that NGBoost natively fits
multimodal/mixture output distributions, supporting the callout's claim rather than
contradicting it. (2) A search for the callout's closing phrase ("average of two clusters is
not itself a plausible outcome") returned no matching or near-matching result; results confirm
the underlying statistical point (a bimodal distribution's mean can fall in a low-density gap
between its two peaks) is standard, uncredited textbook knowledge, not a single-source claim
that needs a citation.

**Findings**: 🔴 Critical: 0. 🟡 Warning: 0. 🔵 Note: 0. The two-path deployment scenario, its
numbers (3 min, 13 min, 40%, 60%, 9 min mean), and the callout's wording are original to this
pass, not drawn from BAP, NGBoost's own paper or docs, or any search result. No citation gap:
NGBoost is cited via @duan2020ngboost earlier in the same section, and this callout makes no new
named-framework or statistical claim that needs its own citation.

Verdict: **PUBLICATION-READY**. No action required.
