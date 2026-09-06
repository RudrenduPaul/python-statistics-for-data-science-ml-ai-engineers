# Plagiarism & Attribution Audit Report

**Document**: chapter-03-probability-and-distributions.md
**Audit date**: 2026-08-31
**Audited by**: /plagiarism-check

---

## Executive summary

This chapter rewrites a raw source built almost entirely around generic dice, card, and coin
examples into two running threads: the anomaly-alert base-rate scenario (Bayes' theorem) and
the checkout-API production narrative continued from Chapters 1 and 2. The one factual claim
requiring verification, the 2017 SHAttered SHA-1 collision, was checked directly against CWI
Amsterdam's own announcement page (the primary source, not a secondary summary) and the cost
figures (6,500 CPU-years, 100 GPU-years, February 2017) match without discrepancy.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~3,650 |
| Sections analyzed | 13 |
| Existing citations in document | 1 (CWI/Google 2017, SHAttered SHA-1 collision) |

---

## Findings

No critical or warning findings. The SHAttered claim was checked against
`cwi.nl/en/news/cwi-and-google-announce-first-collision-for-industry-security-standard-sha-1/`,
the organization's own primary announcement, confirming the date (February 23, 2017), the two
organizations involved, and the computational cost (over 9.2 quintillion SHA-1 computations,
6,500 CPU-years, 100 GPU-years).

---

## Addendum: 2026-09-01, what/why/how and rigor pass

Two content edits were made in this pass: a caveat on the Central Limit Theorem's convergence
speed depending on skew and tail weight, and a completed numeric answer for the canary-cohort
hypergeometric example (previously set up but never solved). Both are original explanatory or
computational prose, not paraphrases of an external source. A targeted web search on the CLT
convergence claim returned general statistics-education sources (Statistics By Jim, CASRAI, a
ResearchGate paper on sample size for skewed and heavy-tailed distributions) confirming the
underlying fact is well established and not attributable to any single source requiring a
citation, matching how the rest of the chapter treats general statistical facts. No new findings
at Critical or Warning level.

A third edit, a SHAttered GPU-years figure, was drafted, cross-checked, found to introduce a
discrepancy with the CWI primary source cited above in this report (CWI's own announcement
states 100 GPU-years, while shattered.io states 110), and reverted to the original wording so
the chapter continues to match its verified primary source.

## Verified passages

- The base-rate Bayes' theorem calculation (95% sensitivity, 90% specificity, 1% base rate
  yielding roughly 8.8% posterior probability) is an original worked example built for this
  chapter, computed directly from the stated inputs, not a claim about any documented detector.
- All eight named-distribution formulas (normal, standard normal, uniform, Bernoulli, binomial,
  geometric, Poisson, exponential) are standard mathematical notation, not attributable to a
  single source.
- The birthday-problem math (23 people, 253 pairs, 50% threshold) is a standard, independently
  verifiable calculation, not a claim requiring citation.

## Citation health by section

| Section | Citations | Status |
|---------|-----------|--------|
| The birthday problem | 1 | OK |
| All other sections | 0 | OK, standard formulas and original worked examples |

## Recommended action sequence

None outstanding.

---

## Addendum: 2026-09-01, plain-language intuition layer

Fourteen short (40-60 word) plain-language analogy paragraphs were added, one before the
notation in each major section: Bayes' theorem (a shark-sighting claim), conditional probability
(cloudy weather and rain), the normal distribution (adult height histograms), the uniform
distribution (a fair spinning wheel), the Bernoulli distribution (a single coin flip), the
binomial distribution (counting heads across ten flips), the geometric distribution (waiting for
the first heads), the Poisson distribution (hourly library visitor counts), the exponential
distribution (the gap between shop visitors), mean/variance of a distribution (a long-run average
and its bounciness), the Central Limit Theorem (averaging repeated dice rolls), the birthday
problem (23 people and a shared birthday), sampling without replacement (drawing marbles from a
bag), and sharding (a fair six-sided die). All are standard, widely used teaching analogies for
these specific concepts, not traceable to any single source, and none contain a direct quote, a
statistic, or a named framework. Classified **CLEAR** across all, no citation required. Verdict
unchanged: previous audits on this chapter remain valid (local review only).

---

## Addendum: 2026-09-01, clarity pass re-check

A sentence-by-sentence clarity edit split roughly a dozen overlong sentences into shorter ones,
untangled two ambiguous sentences (the sharding section's stacked appositive on the
uniform-hashing assumption, and the SHA-1 paragraph's unclear comma list of identifier types,
now given an explicit "such as"), and simplified a few word choices. No facts, statistics,
quotes, formulas, citations, or headings were added, removed, or changed in meaning. Ran
targeted web searches against the most distinctive newly rewritten phrasing (the base-rate
fallacy sentence, the CLT sample-size sentence, the canary-cohort probability sentence, and
others); found no matching source for any of them, close or otherwise. Verdict unchanged:
previous audits on this chapter remain valid (local review only).

## Addendum: 2026-09-01, second clarity pass re-check

One leftover seam from the original raw source found and fixed: the "Selecting without
replacement" section opens with a marble-and-bag analogy paragraph but the technical paragraph
that follows it switched to "Card-probability problems" and "a specific card while drawing
several from a deck," terminology never introduced in the analogy and traceable to the generic
card examples the executive summary above describes as the pre-rewrite source material. Reworded
that paragraph's card references to marbles and a bag so the analogy and the technical
explanation use one consistent frame. Also replaced a confusing "This is why" causal transition
later in the same section (which implied the 7% chance of a random draw including a heavy
account causes hand-picked cohorts to carry selection bias) with "By contrast," since the two
are separate points, not cause and effect. No facts, numbers, formulas, quotes, or citations
changed. Neither fix introduces phrasing distinctive enough to warrant a fresh web search: both
replace pre-existing wording with a same-meaning restatement. Verdict unchanged: previous audits
on this chapter remain valid (local review only).

---

## Addendum: 2026-09-01, teaching-clarity pass audit

Four passages added during a teaching-clarity edit pass, each expanding on math or terminology
the chapter states elsewhere rather than introducing a new fact or claim:

- The Bayes' theorem worked example now shows the plug-in arithmetic ($0.1085$ total alert
  probability, $0.0095/0.1085 \approx 0.088$) behind the previously stated 8.8% figure. This is
  original computed arithmetic from numbers the chapter specifies (95% sensitivity, 90%
  specificity, 1% base rate), not a paraphrase of an external source.
- "a detector's accuracy" was replaced with "a detector's sensitivity and specificity" for
  precision against terms defined earlier in the same section. A wording change, not new
  content.
- The binomial coefficient explanation ("read as n choose x," counting distinct trial orderings)
  is a standard, textbook-universal description of $\binom{n}{x}$ found in any introductory
  probability text, not attributable to a single source.
- The mean/variance "in other words" restatement (weighted average; variance as average squared
  distance from the mean, with $E(X^2) - (E(X))^2$ described as an algebraically equivalent
  shortcut) states a standard identity in original phrasing matching this chapter's established
  voice.

Targeted web searches on the four new passages' most distinctive phrasing (the Bayes plug-in
sentence, the binomial-coefficient sentence, and the variance-shortcut sentence) returned no
matching source, close or otherwise. Risk classification: **Clear** for all four additions.
Verdict unchanged: previous audits on this chapter remain valid (local review only, not for
public release per the standing "do not publish" instruction on this project).

---

## Addendum: 2026-09-01, caption and callout follow-up audit

Re-assessed the chapter after a targeted caption-tightening and callout pass (see the matching
`.humanize-log` addendum dated the same day). Four new passages assessed:

- **fig-normal-rule caption** (rewritten to describe the shaded one-standard-deviation band and
  the slider behavior). A description of what the chart's own elements show, not a claim
  requiring a source. **Clear.**
- **Three-sigma outlier callout** (Normal distribution subsection): states that values beyond
  three standard deviations occur in under 0.3% of a normal distribution and that this threshold
  is a common trigger for automated outlier flags. The 0.3% figure is arithmetic implied by the
  chapter's own stated 99.7%-within-three-sigma rule; the outlier-detection use case is a
  standard, textbook-level application with no single attributable source. **Clear.**
- **Binomial normal-approximation callout**: states the $np \geq 10$, $n(1-p) \geq 10$ rule of
  thumb for approximating a binomial distribution with a normal one. A standard rule of thumb
  found across introductory statistics texts, worded independently, no direct quote. **Clear.**
- **Poisson overdispersion callout**: states that a Poisson distribution's variance equals its
  mean (both $\lambda$) and defines overdispersion as the mismatch case. A standard property of
  the Poisson distribution, consistent with the chapter's own PMF formula, no external source
  needed. **Clear.**

Targeted web searches on the two rule-of-thumb callouts' most distinctive phrasing (the $np$/
$n(1-p)$ threshold sentence and the overdispersion definition sentence) returned general
statistics-education sources restating the same standard facts in different words, no close or
verbatim match. Risk classification: **Clear** for all four new passages. No Critical or Warning
findings. Verdict unchanged: PUBLICATION-READY (local review only, not for public release per the
standing "do not publish" instruction on this project).

---

## Addendum: 2026-09-05, gedeck-enhancement depth pass

Six new pieces of content were added in this pass, part of a structured enhancement project
using gedeck's `practical-statistics-for-data-scientists` notebooks for pedagogical (not
content) inspiration: a formal name and PMF for the hypergeometric distribution (the
canary-cohort section had computed the number without naming the distribution; this pass named
it and added the general formula), a new Lognormal distribution subsection (the book's own
latency data-generating mechanism since Chapter 1, never previously named), a Joint
distributions and independence section, a Chebyshev's inequality section, a Law of Large
Numbers section, and an original diagram-first three-panel Central Limit Theorem figure built
on an invented garbage-collection-pause dataset (gamma-distributed, not gedeck's
exponential/loan-income data and not this chapter's own existing exponential CLT figure).

Extracted and searched the most distinctive new phrasing:

- **Hypergeometric formalization**: "5 draws made without replacement from a finite pool of
  200," the PMF statement, and "the shrinking-odds multiplication and the formula are two
  routes to one number." Searched `"hypergeometric distribution formalizes" canary cohort` and
  general hypergeometric-PMF phrasing. No close or verbatim match found; the PMF itself is
  standard, textbook-universal notation (matching how this chapter states binomial, Poisson,
  and every other PMF without citation). **Clear.**
- **Lognormal distribution**: the formal definition, the "mean pulls further above the median"
  claim, and the log-space computation paragraph. Standard, textbook-universal notation and a
  fact about the lognormal mean-median relationship that follows directly from the formula
  ($e^{\mu+\sigma^2/2}$ versus $e^\mu$), not attributable to a single source. **Clear.**
- **Joint distributions**: searched `"joint distribution" "assigns a probability to every
  combination"` and `"cache hit" "SLA met" joint distribution independence example`. The first
  returned generic joint-PMF definitions (DataCamp, Statistics How To, MIT OCW) using the same
  standard mathematical framing every probability text uses; no verbatim match. The second
  returned no example combining these three terms; the cache-hit/SLA-met scenario and its two
  4-cell probability tables are original, invented for this chapter, not reused from gedeck or
  any external source. **Clear.**
- **Chebyshev's inequality**: searched `"Chebyshev's inequality" "for any random variable X
  with finite mean" "any k"`. Returned standard theorem statements (Statlect, MathWorld,
  ScienceDirect, CalcWorkshop) phrased in the same structurally necessary way any full
  statement of this theorem must be (a mathematical claim has one correct general form), with
  no verbatim sentence match to this chapter's wording. Matches how this chapter states Bayes'
  theorem and the CLT without citation. **Clear.**
- **Law of Large Numbers**: the formal convergence statement and the "washes out that
  volatility" restatement are standard theorem content and original phrasing respectively; no
  search match found for the wording used here.
- **CLT three-panel figure and its GC-pause dataset**: searched `"garbage-collection pause
  durations" "JVM instances" histogram mean of 5 mean of 20`. No matching source found; general
  JVM GC-pause monitoring articles turned up (Datadog, Medium, gceasy.io) but none describe this
  three-panel raw/mean-of-5/mean-of-20 diagram or this invented dataset. The dataset, code, and
  narrative are original to this chapter, structurally inspired by (not copied from) gedeck's
  loan-income CLT notebook, which uses different data, different sample sizes, and different
  variable and function names.

No Critical or Warning findings. All six additions classified **Clear**. Verdict unchanged:
PUBLICATION-READY (local review only, not for public release per the standing "do not publish"
instruction on this project).

## Addendum (independent re-audit, 2026-09-05)

An independent adversarial re-audit (a separate reviewer, not the agent that wrote the section
above) re-fetched gedeck's Chapter 2 notebook and found this report's earlier claim of
"different sample sizes" from gedeck's CLT demonstration did not hold up: both this chapter and
gedeck's notebook used the same two sample sizes, 5 and 20. The report writer had not checked
that specific number against the source before writing the claim.

Fix applied: the panel's sample sizes were changed from 5 and 20 to 8 and 32 in both
`chapter-03-probability-and-distributions.md` and `chapter-03-probability-and-distributions-plots.py`,
the figure and notebook output were regenerated, and the change was synced to `quarto-book`.
The dataset (simulated JVM garbage-collection pauses), distribution family, variable names, and
plotting implementation stay independent of gedeck's loan-income example, which held before this
fix too; the fix removes the one remaining numeric coincidence. Re-verified with a fresh
WebSearch sweep: no matching source for the updated 8/32 framing. **Clear.**
