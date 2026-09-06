# Plagiarism & Attribution Audit Report

**Document**: chapter-01-descriptive-statistics.md
**Audit date**: 2026-08-31
**Audited by**: /plagiarism-check

---

## Executive summary

This chapter is a rewrite built around an original running example (simulated API-latency
data) rather than the source docx's generic examples, so structural overlap with any existing
published source is low by construction. Two Category B statistical claims and one Category A
direct quote were checked against source material. One quote did not match its source verbatim
and has been corrected; two historical figures (Literary Digest return count and Roosevelt's
final vote share) were off by rounding and have been corrected to the figures a second,
independent source confirms. The kidney-stone treatment numbers were cross-verified against two
independent sources and agree on both.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~3,230 |
| Sections analyzed | 10 |
| Paragraphs | ~34 |
| Phrases extracted for verification | 4 |
| Web searches and fetches executed | 6 |
| Critical issues | 0 (1 found, 1 fixed) |
| Warning issues | 0 (2 found, 2 fixed) |
| Note flags | 0 |
| Verified clear | 4 |
| Existing citations in document | 3 (Charig et al. 1986, Squire 1988, Google SRE Book 2017) |

---

## Findings and fixes applied

### Fix 1: direct quote did not match source verbatim (was Critical, now fixed)

**Section**: Measures of spread, variance and standard deviation
**Original**: "some requests are serviced quickly while others take longer, and a simple
average can obscure these tail latencies"
**Source**: [Google SRE Book, ch. 4](https://sre.google/sre-book/service-level-objectives/),
confirmed via WebFetch
**Match**: NEAR (merged two separate sentences into one, misrepresented as a single verbatim
quote)
**Fix applied**: replaced with the verbatim sentence, "a simple average can obscure these tail
latencies, as well as changes in them."

### Fix 2: Literary Digest return count off by rounding (was Warning, now fixed)

**Section**: Selection bias
**Original**: "got about 2.4 million back"
**Source**: [randomservices.org](https://www.randomservices.org/random/data/LiteraryDigest.html),
confirmed via WebFetch, corroborates the original WebSearch result of about 2,300,000
**Fix applied**: corrected to "about 2.3 million back."

### Fix 3: 1936 election result off by rounding (was Warning, now fixed)

**Section**: Selection bias
**Original**: "Roosevelt won with 62%"
**Source**: [randomservices.org](https://www.randomservices.org/random/data/LiteraryDigest.html),
confirmed via WebFetch: Roosevelt won 60.8% of the popular vote
**Fix applied**: corrected to "Roosevelt won with almost 61% of the popular vote."

---

## Verified passages

- **Kidney stone success rates** (small stones 93%/87%, large stones 73%/69%, combined
  78%/83%): cross-checked against two independent sources
  ([Wikipedia](https://en.wikipedia.org/wiki/Simpson%27s_paradox) and
  [bookdown.org](https://bookdown.org/pkaldunn/Book/PercentagesKStones.html)), both matching.
  Cited to [@charig1986].
- **Literary Digest sampling-frame description** (telephone directories, car registrations,
  subscriber rolls): general historical record, consistent across all sources checked. Cited to
  [@squire1988].
- **Statistical formulas** (mean, variance, standard deviation, covariance, correlation):
  standard mathematical notation, not attributable to any single source.
- **The running latency narrative, all worked examples, and the six interactive figures**: newly
  authored for this chapter, no source match expected or found.

---

## Citation health by section

| Section | Paragraphs | Citations | Status |
|---------|-----------|-----------|--------|
| Measures of spread | 4 | 1 | OK |
| Selection bias | 5 | 1 | OK |
| Simpson's paradox | 5 | 1 | OK |
| All other sections | ~20 | 0 | OK, conceptual and original content, no attributable source |

No run of 3+ paragraphs containing an unattributed claim that required a citation.

---

## Recommended action sequence

All fixes above are applied to the chapter file. No outstanding action before this chapter
moves to the humanize pass.

---

## Addendum: 2026-09-01, what/why/how and rigor audit pass

One new paragraph was added to "Measures of spread: variance and standard deviation," noting
that tail percentiles (p99 and higher) need enough requests per window to be stable and can
swing noisily on low-traffic endpoints or narrow windows. This is a general, widely documented
operational fact (confirmed via WebSearch against IBM, Aerospike, Redis, and OneUptime
engineering write-ups on p99 reliability), stated in original phrasing, and consistent with how
the rest of the chapter treats uncited operational knowledge (the "mean plus k std" threshold
convention two paragraphs earlier is also uncited). No verbatim or near-verbatim match found to
any source. Classified **CLEAR**, no citation required, no fix needed.

No other prose changes were made to this chapter in this pass. Verdict unchanged:
**PUBLICATION-READY** (local review only).

---

## Addendum: 2026-09-01, plain-language intuition layer

Nine short (40-60 word) plain-language analogy paragraphs were added, one before the notation
in each major section: mean/median/mode (a school-bus-lateness example), variance/standard
deviation (two students with the same test average but different consistency), covariance and
correlation (height versus shoe size), skewness/kurtosis (a class's test-score tail), population
and sample (a classroom standing in for a country), data types (temperatures versus favorite
colors), selection bias (judging a school lunch by complaints alone), mean imputation (filling a
missing test score with the class average), and Simpson's paradox (a baseball player's split
versus combined batting average). All nine are standard, widely used teaching analogies for
these specific statistical concepts, not traceable to any single source, and none contain a
direct quote, a statistic, or a named framework. Classified **CLEAR** across all nine, no
citation required. Verdict unchanged: **PUBLICATION-READY** (local review only).

## Addendum: 2026-09-01, clarity pass re-check

A sentence-by-sentence clarity edit split roughly a dozen overlong sentences into shorter ones,
fixed one ambiguous comma-spliced list (the missing-values paragraph's schema-drift example),
and simplified a handful of word choices. No facts, statistics, quotes, formulas, citations, or
headings were added, removed, or changed in meaning; one existing citation
([@googlesre2017]) was moved to sit directly after the quote it supports rather than at the end
of a longer sentence, which is a placement fix, not a content change. Ran targeted web searches
against the most distinctive newly rewritten phrasing (the standard-deviation/dashboard
sentence, the tracing-selection-bias sentence, the mean-imputation timeout sentence, and others);
found no matching source for any of them, close or otherwise. Verdict unchanged:
**PUBLICATION-READY** (local review only).

## Addendum: 2026-09-01, second clarity pass re-check

Two small fixes applied in this pass, both confined to the variance/standard-deviation section:
normalized three inconsistent shorthand forms of "standard deviation" ("mean-plus-$k$-std,"
"std-based," "std.-deviation") to the spelled-out term used elsewhere in the chapter, and
rewrote a dangling-modifier sentence ("...confirmed close to normal, worth a quick check before
trusting the number") into a clear main clause plus appositive. No facts, numbers, formulas,
quotes, citations, or headings were touched. Neither fix introduces phrasing distinctive enough
to warrant a fresh web search: both are close paraphrases of language covered and cleared in
the prior passes above. Verdict unchanged: **PUBLICATION-READY** (local review only).

## Addendum: 2026-09-01, clarity and jargon-gap edit pass re-check

Four new passages were added to close jargon and formula-restatement gaps: a plain-language
restatement of the variance formula plus a NumPy-vs-pandas `ddof` default note (Measures of
spread section), a parenthetical definition of percentile at first substantive use (same
section), a precision fix narrowing "standard deviation assumes a shape the data does not have"
to the mean-plus-k-std threshold specifically, and a short clause explaining what multiple
imputation does differently from mean imputation (Missing values section).

Risk classification for each, using the scheme above:

- **NumPy/pandas `ddof` default note**: **Clear**. States a documented library default
  (`numpy.var` uses `ddof=0`, `pandas.Series.var` uses `ddof=1`), verifiable directly against
  each library's own API reference, not phrased as a quote from either.
- **Percentile definition**: **Clear**. A standard, textbook-generic definition ("the value below
  which N% of observations fall"), not attributable to any single source.
- **Standard-deviation precision fix**: **Clear**. A rewording of existing chapter prose for
  accuracy, no new external content introduced.
- **Multiple imputation gloss**: **Clear**. A one-clause, generic description of a well-known
  statistical technique, not a quote or a specific author's phrasing.

No verbatim or near-verbatim match to any external source found for any of the four additions.
Verdict unchanged: **PUBLICATION-READY** (local review only). Total classification this pass:
0 Critical, 0 Warning, 0 Note, 4 Clear.

## Addendum: 2026-09-01, placement/caption/paragraph/callout re-audit

Re-assessed the edited chapter: four paragraph splits (mechanical breaks inserted at existing
sentence boundaries, no new wording) and three new callout boxes (skewness-magnitude rule of
thumb, population/sample notation convention, ordinal-averaging mistake). No figure captions
were rewritten and no figures were moved; both were checked and found correct going into this
pass.

Risk classification for each new passage:

- **Skewness rule-of-thumb callout**: **Clear**. States the standard Bulmer-style skewness
  threshold (|skew| < 0.5 roughly symmetric, beyond 1 highly skewed), a generic textbook
  convention not attributable to a single source or phrased as a quote.
- **Population/sample notation callout**: **Clear**. States the standard Greek-letter
  (population) vs. Latin-letter (sample) convention used across statistics generally, not a
  specific author's wording.
- **Ordinal-averaging callout**: **Clear**. Illustrates a mistake using the chapter's own
  log-severity example (`debug`/`info`/`warn`/`error`), not new external content.
- **Paragraph splits**: **Clear**. No new sentences or claims, verified prose divided at
  sentence boundaries.

No verbatim or near-verbatim match to any external source found. Verdict unchanged:
**PUBLICATION-READY** (local review only). Total classification this pass: 0 Critical,
0 Warning, 0 Note, 4 Clear.

## Addendum: 2026-09-05, gedeck-enhancement pass (robust statistics, Spearman correlation, new figure)

Two new sections were added: "Robust statistics: trimmed mean, IQR, and MAD" (inserted after
Measures of spread) and "Spearman correlation and correlation across many metrics" (inserted
after Covariance and correlation, including one new figure, `fig-metric-correlations`). Both
extend the chapter's own running checkout-API and cluster health-metric scenarios; no content,
wording, examples, or dataset names were reused from any external source.

Phrase extraction and web verification, by category:

- **Category B (statistics/quantitative claims)**: every number in both new sections (the
  mean/trimmed-mean/median/std/IQR/MAD comparison table, the alert-threshold flagged-share
  table, the utilization-vs-latency Pearson/Spearman comparison, and every correlation value
  in the new figure) is computed directly from this chapter's own simulated data using the
  book's existing `simulated_latency_ms` generator or a newly written, self-contained
  simulation. None of these numbers describe a specific company, dataset, or published study,
  so none require a citation. Classified **Clear**.
- **Category C (named methodologies)**: four named techniques appear without inline citation:
  the trimmed mean, the interquartile range, Spearman's rank correlation, and Kendall's tau.
  All four are standard, textbook-generic statistical techniques not attributable to a single
  paper, the same treatment given elsewhere in this chapter to Pearson correlation, variance,
  and standard deviation. Classified **Clear**. Tukey's 1.5×IQR outlier fence is named and
  attributed to Tukey by name in the prose; a web search confirmed the fence originates in
  Tukey's 1977 "Exploratory Data Analysis" and that 1.5 (inner fence) and 3.0 (outer fence) are
  the values Tukey proposed, matching the multiplier used in the chapter text. This is a
  universally taught, generic convention (the same standing as the skewness rule of thumb
  cleared in an earlier addendum), so no inline citation was added, consistent with that
  precedent. Classified **Clear**. The modified z-score outlier rule is formally cited to
  Iglewicz and Hoaglin (1993) with a new `references.bib` entry; a web search independently
  confirmed the formula (0.6745 times the deviation from the median, divided by the raw MAD)
  and the 3.5 threshold recommendation both trace to that source, and that Iglewicz and
  Hoaglin selected 3.5 because it gives roughly the same false-positive rate under normality
  that a classical |z| > 3 rule gives. The chapter states the algebraically equivalent form
  $M_i = (x_i - \tilde{x})/\text{MAD}$ using the chapter's own previously scaled MAD
  definition (1.4826 times the raw median absolute deviation) rather than the source's
  0.6745-times-raw-MAD form; this is a mathematical restatement, not a quotation, so no
  wording match risk applies. Classified **Clear**, citation present and verified.
- **Category D (distinctive phrases)**: no phrase of 8 or more consecutive words in either new
  section returned a match on manual review against the audit's established checklist. Both
  sections use this chapter's own scenario (checkout-API latency, cluster health metrics) and
  original sentence construction throughout. Classified **Clear**.

No Critical or Warning findings. Verdict unchanged: **PUBLICATION-READY** (local review only).
Total classification this pass: 0 Critical, 0 Warning, 0 Note, 3 Clear categories covering all
new prose, tables, formulas, and the new figure.
