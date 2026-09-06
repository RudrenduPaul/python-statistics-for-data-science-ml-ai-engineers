# Plagiarism & Attribution Audit Report

**Document**: chapter-07-decision-trees-random-forests.md
**Audit date**: 2026-09-05
**Audited by**: /plagiarism-check
**Fixes applied**: 1 Warning (Breiman citation added to the permutation-importance sentence), 2026-09-05.
Note: the importance-comparison figure's numeric values were also corrected after this audit ran,
from an early scratch-test run's numbers to the values the shipped figure renders
(0.141/0.150 Gini/entropy, -0.006 permutation, all for the planted batch-ID feature). This does
not change the audit's findings; the "Verified passages" section below still references the
pre-correction numbers. The decision-boundaries section's heading was also renamed during the
subsequent humanize pass, from "Comparing decision boundaries: a tree, a forest, and logistic
regression" to "Decision boundaries: a tree, a forest, and logistic regression compared" (fixing
a gerund-opener pattern). References to the old heading text below describe the audited content
correctly; only the heading wording changed.

---

## Executive summary

This chapter (Tree-Based Methods: Decision Trees and Random Forests) was audited in full, with
extra scrutiny on its two newest sections: "Permutation importance, and where Gini and entropy
importance mislead" and "Comparing decision boundaries: a tree, a forest, and logistic
regression." Both sections use the chapter's existing simulated canary-rollback classifier and
introduce no wording, example framing, dataset names, or numbers drawn from the
gedeck/practical-statistics-for-data-scientists companion repository the two sections were
structurally inspired by (its multi-model decision-boundary grid and its Gini/entropy/permutation
importance comparison, both built on lending data, not this book's rollback scenario).

No word-for-word or close-paraphrase matches were found against any external source for the new content.
One gap surfaced: the new importance-comparison section introduces permutation importance without
citing its origin, even though the source (Breiman, 2001) is the same paper cited earlier
in this chapter for random forests themselves, and every other named technique in the chapter
(random forests, BART, the Gini-importance cardinality bias) carries an origin citation. This is
a consistency gap, not a plagiarism risk, and is resolved with a one-line citation addition.

A long-standing factual claim elsewhere in the chapter (the ROC curve's origin in WWII radar
signal detection) checks out against multiple independent historical sources but is common
knowledge repeated across dozens of unrelated pages rather than attributable to one source; it is
noted for optional attribution but does not block publication.

**Verdict: NEEDS MINOR FIXES** (one Warning-level citation addition; zero Critical findings).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~6,917 |
| Sections analyzed (H2) | 15 |
| Phrases extracted for verification | 9 |
| Web searches executed | 6 |
| Critical issues | 0 |
| Warning issues | 1 |
| Note flags | 2 |
| Verified clear | 6 |
| Existing citations in document | 3 |

---

## Warning issues

[Permutation importance, and where Gini and entropy importance mislead - paragraph 3]
Severity:  Warning
Original:  "Permutation importance measures how much a fitted model's accuracy drops when one
feature's column is shuffled at random on a held-out validation split, breaking that feature's
link to the outcome while leaving the fitted model and every other column untouched."
Source:    Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32, the paper that
introduced permutation-based variable importance alongside random forests themselves, confirmed
via web search of secondary sources describing Breiman's original permutation-importance
procedure. This is the same paper cited elsewhere in this chapter as `@breiman2001randomforests`
for random forests (see "Random forests: decorrelating trees with random feature selection").
Match:     ORIGIN (technique correctly described, origin uncited in this section)
Fix type:  1 (add citation)
Fixed:     "Permutation importance measures how much a fitted model's accuracy drops when one
feature's column is shuffled at random on a held-out validation split, breaking that feature's
link to the outcome while leaving the fitted model and every other column untouched
[@breiman2001randomforests]."

This closes a consistency gap: random forests, BART, and the Gini-importance cardinality bias
each carry an origin citation in this chapter; permutation importance was the one named technique
introduced without one, even though its origin citation appears elsewhere in the document under
a different claim.

---

## Note flags

- Section: "Evaluating a classifier: precision, recall, and the ROC curve" (pre-existing
  content, not part of this session's additions). Phrase: "a name left over from its origin
  in World War II radar signal detection." Reason: Verified accurate against multiple
  independent sources (OncoDaily, NCBI Bookshelf, Circulation/AHA Journals, Wentz Wu's radar
  ROC-analysis writeup) describing ROC's WWII radar-operator origin. It is common knowledge
  repeated near-identically across dozens of unrelated technical and historical sources rather
  than traceable to one canonical citation, so no fix is required, but a one-line citation to a
  historical source (for example the NCBI Bookshelf chapter) would strengthen rigor if the author
  wants full attribution parity with the chapter's cited claims. Manual verification recommended
  only if the author wants to add that citation; not a blocker.
- Section: "Comparing decision boundaries: a tree, a forest, and logistic regression."
  Phrase: the meshgrid-predict-reshape-shade procedure described in the closing paragraph.
  Reason: This is a generic, widely documented plotting technique (confirmed via search
  against HackerNoon, GeeksforGeeks, MachineLearningMastery, and multiple Medium tutorials, all
  describing the identical meshgrid to flatten to predict to reshape to contourf workflow in
  their own words). No word-for-word match was found between this chapter's wording and any of
  those sources, and the technique itself is textbook-generic rather than protectable expression,
  consistent with the research notes' own assessment that this "compare several models on one
  meshgrid" structure is a reusable pattern, not text, safe to reproduce with original wording and
  data. No fix required.

---

## Verified passages

- @breiman2001randomforests for the random forest method itself ("the method Leo Breiman
  formalized in 2001") - citation present, correctly attributed, standard reference for the
  technique.
- @strobl2007bias for the correlated-predictor variable-importance bias claim in "Variable
  importance" - citation present; confirmed via the paper's own abstract that its scope covers
  both correlated-predictor bias and predictor-cardinality/scale bias, so its reuse in the new
  "Permutation importance..." section for the high-cardinality bias claim is a correct, not
  overreaching, application of the same source.
- @chipman2010bart for BART's origin ("introduced by Chipman, George, and McCulloch") - citation
  present and correctly attributed.
- The worked numeric examples throughout the chapter (RSS drop from 119.5 to 7.0, Gini index
  values, the 200-deployment confusion matrix, the 0.141/0.150/-0.006 importance scores) are all
  computed from this chapter's own labeled-simulated dataset, not external data claims, and
  require no citation.
- The new "Permutation importance, and where Gini and entropy importance mislead" section's
  worked example (the planted deployment batch ID) is original: no matching scenario, dataset
  name, or wording found in gedeck's Chapter 6 permutation/Gini/entropy importance comparison
  (which uses loan3000.csv/loan_data.csv.gz) or anywhere else searched.
- The new "Comparing decision boundaries" section's scenario and figure are original: no matching
  wording found against gedeck's Chapter 5 four-model decision-boundary comparison (which uses
  the same lending datasets) or any other searched source.

---

## Citation health by section

| Section | Paragraphs | Citations | Density | Status |
|---------|-----------|-----------|---------|--------|
| Regression trees and recursive binary splitting | 7 | 0 | 0 | LOW (no external claims requiring citation; methodology is textbook-standard) |
| Classification trees: Gini index, entropy, and error rate | 8 | 0 | 0 | LOW (same as above) |
| Trees, categorical predictors, and interactions | 5 | 0 | 0 | OK (no external claims) |
| Tree depth, overfitting, and cost-complexity pruning | 8 | 0 | 0 | OK (no external claims) |
| Why a single tree has high variance | 4 | 0 | 0 | OK (no external claims) |
| Bagging: bootstrap aggregation | 4 | 0 | 0 | OK (no external claims) |
| Random forests: decorrelating trees with random feature selection | 4 | 1 | 1 per 4 paras | OK |
| Out-of-bag error estimation | 4 | 0 | 0 | OK (no external claims) |
| Evaluating a classifier: precision, recall, and the ROC curve | 12 | 0 | 0 | NOTE (ROC/WWII historical claim, see Note flags) |
| Variable importance | 5 | 1 | 1 per 5 paras | OK |
| Permutation importance, and where Gini and entropy importance mislead | 9 | 1 before fix, 2 after fix | improves after fix | OK after fix |
| Comparing decision boundaries: a tree, a forest, and logistic regression | 6 | 0 | 0 | OK (no external claims; original worked example) |
| When to reach for a single tree, a forest, or boosting | 4 | 0 | 0 | OK (synthesis/decision-guide content) |
| A Bayesian perspective | 9 | 1 | 1 per 9 paras | OK |

Citation gaps: No run of 3+ consecutive paragraphs contains an uncited factual claim requiring
a source. The chapter's citation-light sections are methodology-explanation prose (regression
trees, classification-tree criteria, pruning, bagging) rather than claims that need external
sourcing, consistent with this book's established citation practice elsewhere.

---

## Recommended action sequence

1. Add [@breiman2001randomforests] after the first sentence introducing permutation importance
   in "Permutation importance, and where Gini and entropy importance mislead" (Warning fix above).
2. Optional, not required: add a historical citation for the ROC/WWII radar origin claim if full
   attribution parity across all historical asides is desired.
