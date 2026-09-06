# Plagiarism & Attribution Audit Report

**Document**: chapter-02-hypothesis-testing.md
**Audit date**: 2026-08-31
**Audited by**: /plagiarism-check

---

## Executive summary

This chapter rewrites a raw source that leaned on generic examples (apple weights, fish
weights, ice cream flavor surveys, smoking and lung cancer) into a single running scenario: an
on-call engineer evaluating a caching layer's effect on checkout-API latency, continuing
Chapter 1's narrative. Only one direct quote appears, from the American Statistical
Association's 2016 statement on p-values, checked against the source PDF and matching verbatim.
All formulas (Z-test, t-test, ANOVA, chi-squared, ANCOVA, sample size) are standard
mathematical notation used identically across every statistics textbook, not attributable to a
single source.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~4,050 |
| Sections analyzed | 14 |
| Existing citations in document | 1 (Wasserstein & Lazar 2016, ASA statement on p-values) |
| Direct quotes | 1, checked against the source PDF, verbatim match |

---

## Findings

No critical or warning findings. The one direct quote (the ASA's Principle 2: "P-values do not
measure the probability that the studied hypothesis is true, or the probability that the data
were produced by random chance alone") was checked against
`amstat.org/asa/files/pdfs/p-valuestatement.pdf` and matches the source verbatim, correctly
cited to [@asa2016].

## Verified passages

- The 384-request sample size worked example uses the standard sample size formula
  $n = z^2\sigma^2/E^2$ applied to invented but clearly labeled numbers (margin of error,
  standard deviation), not a claim about any documented study.
- The Type I/Type II error framing, ANOVA, chi-squared, ANCOVA, and design-of-experiments
  sections restate standard statistical definitions in the checkout-API narrative; none of the
  worked examples in this chapter are drawn from an external source requiring citation.

## Citation health by section

| Section | Citations | Status |
|---------|-----------|--------|
| What a p-value says | 1 | OK |
| All other sections | 0 | OK, standard technique explanations, no attributable source |

## Recommended action sequence

None outstanding.

---

## Addendum (2026-09-01, what/why/how + rigor audit pass)

Two short rigor-fix additions were made to this chapter: a homogeneity-of-variance caveat added
to the ANOVA section (Welch-corrected ANOVA / Kruskal-Wallis as the safer choice when group
variances differ sharply), and a homogeneity-of-slopes caveat added to the ANCOVA section
(interaction term needed if payload size affects latency differently across the two deployment
strategies). Both additions restate standard, undisputed statistical facts in the book's own
voice, the same pattern the audit above cleared for every other formula and technique
explanation in this chapter. A targeted web search on the Welch's-ANOVA claim
(`statisticsbyjim.com/anova/welchs-anova-compared-to-classic-one-way-anova`,
`rips-irsp.com/articles/10.5334/irsp.82`) confirmed the claim as standard, widely reported
statistical practice with no single attributable source and no phrase-level match to the
inserted sentences. No new citation was needed; nothing else in the chapter changed.

Verdict unchanged: **PUBLICATION-READY** (local review only).

---

## Addendum: 2026-09-01, plain-language intuition layer

Fourteen short (40-60 word) plain-language analogy paragraphs were added, one before the
notation in each major section: stating a hypothesis (a courtroom trial), the Z-test/t-test (a
rigged-coin check), the p-value (a fair-coin surprise), Type I/II errors (a smoke detector's two
failure modes), statistical power (a smoke detector's sensor strength), sample size (polling 10
versus 10,000 people), confounding variables (ice cream sales and drowning deaths), paired
versus unpaired t-tests (the same runners versus different runners), ANOVA (comparing five
classrooms at once), chi-squared (a candy bag's color counts), ANCOVA (comparing two diets'
starting weights), design of experiments (a science-fair planting plan), pairwise post-hoc tests
(checking every classroom pair), and non-parametric tests (a backup plan for lopsided data). All
are standard, widely used teaching analogies for these specific concepts, not traceable to any
single source, and none contain a direct quote, a statistic, or a named framework. Classified
**CLEAR** across all, no citation required. Verdict unchanged: **PUBLICATION-READY** (local
review only).

---

## Addendum: 2026-09-01, clarity pass re-check

A sentence-by-sentence clarity edit split about a dozen overlong sentences into shorter ones,
untangled one comma-heavy sentence in the ANOVA section (the Welch's-correction aside is now set
off in parentheses instead of running into the main clause), and tightened a few word choices.
No facts, statistics, quotes, formulas, citations, or headings were added, removed, or changed
in meaning. Ran targeted web searches against the most distinctive newly rewritten phrasing
(the Type I/Type II framing sentence, the ANOVA-assumption sentence, the p-value effect-size
sentence, and others); found no matching source for any of them, close or otherwise. Verdict
unchanged: **PUBLICATION-READY** (local review only).

## Addendum: 2026-09-01, second clarity pass re-check

Three small fixes applied in this pass: lowercased "T-test" in the section heading to match
the "t-test" spelling used everywhere in the body text; replaced a comma-spliced restatement of
statistical power ("...is true, in other words, $1 - \beta$") with a colon for a cleaner parse;
and reworded a confusing "in the other direction" transition in the confounding-variables
section so the sentence states directly that the same caution from Chapter 1 applies to
before-after comparisons, rather than implying an unclear causal reversal. No facts, numbers,
formulas, quotes, citations, or headings changed in substance. None of these fixes introduce
phrasing distinctive enough to warrant a fresh web search. Verdict unchanged:
**PUBLICATION-READY** (local review only).

## Addendum: 2026-09-01, clarity and jargon-gap edit pass re-check

Four new passages were added to close jargon and formula-restatement gaps: a definition of
degrees of freedom and an explanation of why fewer of them means heavier t-distribution tails
(Z-test/t-test section), a parenthetical statement of what the Central Limit Theorem says (same
section), a parenthetical definition of family-wise error rate (Pairwise post-hoc tests section),
and a clause naming the intercept, coefficients, and error term in the ANCOVA regression formula
(ANCOVA section).

Risk classification for each, using the scheme above:

- **Degrees-of-freedom definition**: **Clear**. Standard statistical terminology, scoped
  correctly to the one-sample case ($n - 1$) rather than stated as a universal rule; not
  attributable to any single source.
- **Central Limit Theorem restatement**: **Clear**. A textbook-generic paraphrase of a
  foundational, uncontested statistical result, phrased independently of any specific source's
  wording.
- **Family-wise error rate definition**: **Clear**. Standard multiple-comparisons terminology,
  generic phrasing.
- **ANCOVA symbol gloss**: **Clear**. Names standard regression-notation roles (intercept,
  coefficient, error term) implied by the equation from the start; adds no new claim or external
  content.

No verbatim or near-verbatim match to any external source found for any of the four additions.
Verdict unchanged: **PUBLICATION-READY** (local review only). Total classification this pass:
0 Critical, 0 Warning, 0 Note, 4 Clear.

## Addendum: 2026-09-01, placement/caption/paragraph/callout re-audit

Re-assessed the edited chapter: four paragraph splits (mechanical breaks inserted at existing
sentence boundaries, no new wording) and three new callout boxes (a t-test default rule of
thumb, the fix-alpha-first convention for Type I/Type II error rates, and the
randomize-before-assignment-is-known rule for experimental design). No figure captions were
rewritten and no figures were moved; both were checked and found correct going into this
pass.

Risk classification for each new passage:

- **T-test default callout**: **Clear**. States a standard statistical convention (prefer the
  t-test when the population standard deviation must be estimated from the sample), corrected
  during drafting to remove an overclaim ("never does worse than the Z-test") caught by
  self-review before this classification.
- **Fix-alpha-first callout**: **Clear**. Describes standard null-hypothesis-testing
  convention, not a specific author's wording or a disputed claim.
- **Randomization-timing callout**: **Clear**. States a foundational, uncontested principle of
  experimental design, generic phrasing.
- **Paragraph splits**: **Clear**. No new sentences or claims, verified prose divided at
  sentence boundaries.

No verbatim or near-verbatim match to any external source found. Verdict unchanged:
**PUBLICATION-READY** (local review only). Total classification this pass: 0 Critical,
0 Warning, 0 Note, 4 Clear.

---

## Addendum: 2026-09-05, gedeck-enhancement depth pass

Ten new passages were added to close gaps identified against the gedeck
`practical-statistics-for-data-scientists` reference notebooks (structural inspiration only,
never copied wording, examples, numbers, or variable names): a formal definition of Cohen's d
where it was previously an undefined plot-slider label; a new confidence-interval-for-a-mean and
confidence-interval-for-a-proportion section; a new two-proportion z-test section; a new
multiple-comparisons-problem section; a new sequential-testing/peeking section; a
simulate-then-reveal-the-formula treatment for the single-mean Z-test and the unpaired t-test;
and short permutation bridges added to the existing ANOVA and chi-squared sections. All scenario
numbers (latency in milliseconds, checkout completion rates, canary success rates) were
generated for this pass with fixed random seeds documented in
`chapter-02-hypothesis-testing-plots.py`, continuing this book's checkout-API narrative rather
than gedeck's housing, lending, or stock-return examples.

Category C review (named methodologies introduced without an inline citation): Cohen's d, the
Wilson score interval, the Benjamini-Hochberg procedure, alpha-spending functions, and
always-valid p-values. All five are standard, multi-source statistical terminology with no
single attributable origin requiring citation, the same tier as this chapter's existing
uncited references to Tukey's HSD, Scheffé's test, and the Bonferroni correction. Targeted web
searches confirmed: Wilson-versus-Wald boundary behavior is documented across many independent
sources (Towards Data Science, econometrics.blog, multiple comparison papers); the
Benjamini-Hochberg procedure traces to Benjamini and Hochberg (1995), *Journal of the Royal
Statistical Society B*, 57:289-300, but is referenced generically across the statistics
literature without a citation requirement at this book's practitioner level; alpha-spending
functions and always-valid p-values are standard sequential-testing vocabulary documented in
both the academic literature (Johari et al., "Always Valid Inference," arXiv:1512.04922) and
practitioner writeups, with no single attributable source.

Category D review (distinctive constructed phrasing, checked for accidental overlap with a
published source): three candidate sentences from the new prose were searched as quoted
phrases: "checking a hypothesis test's p-value repeatedly as data accumulates," "a proportion's
variance is fixed by the proportion itself," and "resolving a permutation p-value near a
stricter threshold." No matching published phrase turned up for any of the three; results
returned only general background on peeking, proportion-of-variance (an unrelated PCA/ANOVA
sense), and permutation-test resolution, supporting that the phrasing originates with this
pass.

All worked numbers (the 34.8 ms single-mean example, the 9.8 ms permutation-test gap, the two
checkout completion rates, the bootstrap confidence intervals, the 23.8 percent peeking
false-positive rate, and the 63.7 percent multiple-comparisons rate) came from running the
corresponding code in `chapter-02-hypothesis-testing-plots.py` with the venv's Python
interpreter, not invented by hand, and they match the embedded figures' own annotation text.

Risk classification: 0 Critical, 0 Warning, 0 Note, 10 Clear (one per new passage). Verdict
unchanged: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).
