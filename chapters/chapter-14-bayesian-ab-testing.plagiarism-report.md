# Plagiarism & Attribution Audit Report

**Document**: chapter-14-bayesian-ab-testing.md
**Audit date**: 2026-09-05
**Audited by**: /plagiarism-check

---

## Scope note

`published-repo/chapters/` had no prior `.plagiarism-report.md` for this chapter (only
`quarto-book/chapters/chapter-14-bayesian-ab-testing.plagiarism-report.md` existed, covering the
chapter's pre-existing content). This audit is scoped to the three sections added in this pass:
"Sample ratio mismatch: checking the split before trusting the comparison," "Multiple metrics,
multiple chances to be wrong," and "Hierarchical A/B testing: partial pooling across segments."
All figures, code, and worked numbers in these sections are original (own dataset, own random
seeds, own simulated numbers). The chapter's pre-existing sections were not re-audited here; see
the quarto-book report for that history.

---

## Executive summary

Three new sections were checked for direct quotes, statistics, named methodologies, and
attributed claims. One citation error was caught and fixed during this audit: a stricter
significance-threshold claim ($p < 0.001$ for sample ratio mismatch) had been attributed to
Fabijan et al. 2019 from memory, but live verification showed that paper documents SRM's
taxonomy of causes without stating this specific threshold as a rule. The threshold traces to
Kohavi, Tang, and Xu's 2020 practitioner reference instead. The citation was split correctly:
`[@kohavitangxu2020]` for the threshold recommendation, `[@fabijan2019]` for the taxonomy of
causes, and both entries were verified against live sources (KDD proceedings page and the book's
publisher record) before being added to `references.bib`. The author list on the Fabijan
citation was also wrong in the first draft (a fabricated author lineup) and was corrected
against the paper's own author list via dblp and the paper's PDF. No other Critical or Warning
findings surfaced. Verdict: **PUBLICATION-READY**.

---

## Audit statistics

| Metric | Value |
|--------|-------|
| New content word count | ~1,750 |
| New sections analyzed | 3 |
| New paragraphs | ~40 |
| New citations added | 2 (`@fabijan2019`, `@kohavitangxu2020`) |
| Phrases extracted for verification | 6 |
| Web searches executed | 5 |
| 🔴 Critical issues | 0 (1 found and fixed during audit) |
| 🟡 Warning issues | 0 |
| 🔵 Note flags | 1 |
| ✅ Verified clear | 5 |

---

## 🔴 Critical issues (found and fixed during this audit)

```
[Sample ratio mismatch, SRM significance-threshold paragraph]
Severity:  🔴 (wrong citation, caught pre-publication)
Original:  "...recommends a stricter significance threshold than the conventional 0.05 for
           this specific check, on the order of p < 0.001... [@fabijan2019]. That same
           source catalogs the usual culprits..."
Source:    Fabijan, A. et al., "Diagnosing Sample Ratio Mismatch in Online Controlled
           Experiments" (KDD '19), https://dl.acm.org/doi/10.1145/3292500.3330722
Match:     WRONG CITATION, verified via a full-text PDF read (exp-platform.com mirror); the
           paper documents SRM causes and a worked chi-square example but never states a
           p < 0.001 rule. The threshold is Kohavi, Tang & Xu's (2020) documented
           recommendation instead.
Fix type:  1 (citation correction)
Fixed:     Split into two sentences: the threshold claim now cites
           [@kohavitangxu2020] (Kohavi, Tang, Xu, "Trustworthy Online Controlled
           Experiments," Cambridge University Press, 2020), and the taxonomy-of-causes
           claim keeps [@fabijan2019], correctly matched to what that paper documents.
```

```
[references.bib, fabijan2019 entry]
Severity:  🔴 (fabricated author list, caught pre-publication)
Original:  author = {Fabijan, Aleksander and Dmitriev, Pavel and McFarland, Colin and
           Vermeer, Lukas and Holmstrom Olsson, Helena and Bosch, Jan}
Source:    dblp record, https://dblp.org/rec/conf/kdd/FabijanGGOQVD19.html; confirmed
           against the paper's own byline in the exp-platform.com PDF.
Match:     WRONG. The paper's author list is Fabijan, Gupchup, Gupta, Omhover, Qin,
           Vermeer, Dmitriev. Three of the six names in the first draft do not appear on
           this paper at all.
Fix type:  1 (citation correction)
Fixed:     author = {Fabijan, Aleksander and Gupchup, Jayant and Gupta, Somit and
           Omhover, Jeff and Qin, Wen and Vermeer, Lukas and Dmitriev, Pavel A.}
```

---

## 🟡 Warning issues

None.

---

## 🔵 Note flags

```
[Multiple metrics section, lottery-ticket opening analogy]
Phrase:   "Buying ten lottery tickets instead of one does not raise any single ticket's
          odds, but it does raise the odds that at least one of them pays out."
Reason:   Lottery-ticket analogies for multiple-comparisons intuition are a common device
          in statistics writing generally. WebSearch found no close match to this wording;
          the closest results were unrelated lottery-odds explainer articles. Recorded as a
          note rather than clear because the underlying analogy pattern is widely used,
          even though this specific sentence is original.
```

---

## Verified passages

- **SRM chi-square worked example** (Weeks 1 through 5, redirect-timeout bug, 15,000
  visitors per week, 6% drop rate): original simulated dataset, seed 303, no external
  source.
- **Multiple-metrics simulation** (k = 1 through 20 metrics, 30,000 replicate experiments,
  Normal approximation to win probability): original simulation, seed 404; the
  $1-(1-\alpha)^k$ closed-form callback correctly cross-references this chapter's own
  Chapter 2 content (verified by reading `chapter-02-hypothesis-testing.md`'s
  multiple-comparisons section, which independently derives the same 64.2% figure at
  k = 20).
- **Hierarchical segment dataset and PyMC model** (organic, paid search, referral, and
  email segments; no-pooling, complete-pooling, and hierarchical models; `az.compare()`
  table): original simulated dataset, seed 707; model code follows a standard three-way
  pooling comparison structure (a well-known pattern in Bayesian hierarchical-modeling
  pedagogy, for example Gelman and Hill's textbook treatment of the pooling spectrum)
  applied here to original data, not copied from any single source.
- **Non-centered parameterization cross-reference to Chapter 9**: verified against
  `chapter-09-bayesian-regression.md`'s own callout, which introduces the technique without
  external citation; this chapter's phrasing follows the same uncited convention for
  consistency, correctly framed as an internal cross-reference rather than a new external
  claim.
- **Goodness-of-fit chi-square formula cross-reference to Chapter 2**: verified against
  `chapter-02-hypothesis-testing.md`'s "Categorical association: chi-squared and goodness of
  fit" section; the formula and framing correctly match what that chapter established
  earlier.

---

## Citation health by section

| Section | Paragraphs | Citations | Density | Status |
|---------|-----------|-----------|---------|--------|
| Sample ratio mismatch | ~10 | 2 | 1 per 5 paras | OK |
| Multiple metrics | ~7 | 0 (internal cross-ref only) | n/a | OK (internal claims, not external) |
| Hierarchical A/B testing | ~10 | 0 (internal cross-refs to Ch. 9/10 only) | n/a | OK (methodology, not external claims) |

**Citation gaps**: None flagged. The multiple-metrics and hierarchical sections make no
external factual claims beyond internal cross-references to this book's own earlier chapters,
which were verified directly against those chapters' source files rather than requiring a web
citation.

---

## Recommended action sequence

1. Both Critical findings were fixed during this audit pass (citation split and author-list
   correction) before this report was finalized. No outstanding action required.
2. Carry the corrected `[@kohavitangxu2020]` and `[@fabijan2019]` entries through to
   `quarto-book/references.bib` when syncing this chapter.

---

## Addendum, 2026-09-05: latent-subgroup mixture section and divergences cross-reference

Scope for this pass: the new "Latent subgroups: what a mixture model finds when the segment
isn't logged" section, its `fig_latent_mixture` figure and PyMC mixture model, and the short
divergence cross-reference passage inserted into the existing "Hierarchical A/B testing"
section. Checked against BAP's mixture-model notebook (`code/Chp6/06_mixture_models.ipynb`,
Osvaldo Martin, github.com/aloctavodia/BAP), which this book's own research notes
(`task-todo/research-bap-ch6-8.md`) flagged as the closest prior treatment of label switching
and finite mixture models.

**Dataset and scenario**: original. BAP's mixture-model notebook fits every example to
`chemical_shifts_theo_exp.csv` (NMR chemical-shift data grouped by amino acid). This chapter's
new section fits a simulated checkout-time-change dataset (2,400 visitors, seed 505,
returning-visitor and first-time-visitor latent groups), an unrelated domain with its own
scenario, numbers, and variable names (`checkout_time_delta`, `true_weight_returning`,
`true_mean_first_time`, and so on, none carried over from BAP's `theta`, `mu`, or chemical-
shift naming).

**Code pattern**: paraphrased, not copied. BAP demonstrates the label-switching fix two ways: a
`pm.Potential` with a `-inf` penalty for unordered means, and, for its K = 3 through 6 models, an
`ordered` transform. This chapter uses only the `ordered`-transform approach (matching modern
PyMC's documented API, confirmed via a live search of PyMC's own discourse forum and API docs,
not reproduced from BAP's specific code), with different variable names, a different prior
(`Normal(0, 10)` versus BAP's setup), and a different `initval`. No BAP code was reproduced
verbatim; the transform call itself is standard PyMC usage shared across many independent
tutorials, not something a single source can claim.

**Named methodology check**: "label switching" is a named, citable concept (Category C). BAP's
notebook explains the problem but is a code companion, not the origin source, and citing BAP
itself would misattribute a decades-old statistical result to a textbook's code repository.
Live web search confirmed the canonical citation: Stephens, M. (2000), "Dealing with Label
Switching in Mixture Models," *Journal of the Royal Statistical Society Series B*, 62(4),
795-809, DOI 10.1111/1467-9868.00265. Added `stephens2000` to `references.bib` and cited it
inline at first use of the term in the new section. This closes what would otherwise have been
a 🟡 Warning (named methodology, no citation).

**Distinctive-phrase sweep**: the section's opening analogy ("A restaurant's average four-star
rating can hide two different dining rooms behind it...") returned no matches on live web
search; reads as original. No other 8+ word phrase in the new section returned a match.

**Divergences cross-reference passage**: verified against `chapter-09-bayesian-regression.md`'s
own `{#sec-divergences-funnel}` section. The new passage in this chapter states no new claim
about funnel geometry; it applies that section's existing, cited (`[@neal2003]`) material to
this chapter's own `sigma_delta` / `delta_offset` parameters and points the reader back to
Chapter 9 rather than re-deriving anything. No citation gap: this is an internal cross-
reference, the same pattern the base report above verified as acceptable for this chapter's
non-centered-parameterization callout.

**Findings**: 0 Critical, 1 Warning (resolved: added `[@stephens2000]`), 0 Notes needing manual
follow-up, remainder Clear.

**Verdict**: PUBLICATION-READY.
