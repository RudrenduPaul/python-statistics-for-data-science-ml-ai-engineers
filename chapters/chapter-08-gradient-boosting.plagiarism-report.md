# Plagiarism & Attribution Audit Report

**Document**: chapter-08-gradient-boosting.md
**Audit date**: 2026-09-05
**Audited by**: /plagiarism-check

---

## Executive summary

PLAGIARISM AUDIT: chapter-08-gradient-boosting.md
==============================
Document type: book chapter
Word count: ~3,827
Sections: 10 (H2 headings)
Paragraphs: ~68
Existing citations: 3 (Friedman 2001, Chen and Guestrin 2016, Ke et al. 2017)
Starting extraction phase...

This audit covers two additions made in this session to a chapter whose base content went
through six prior clarity/formatting passes and was audited clean each time (see the addendum
history preserved below and mirrored from the equivalent quarto-book report, which used the
chapter's earlier internal slug `chapter-boosting-gradient-boosting.md`; this file uses the
chapter's current filename going forward). This session added: (1) a short cross-reference
passage in "Boosting versus bagging" pointing to Chapter 7's `@fig-confusion-threshold`
section instead of re-deriving the confusion matrix, precision, recall, F1, or ROC-AUC
treatment, and (2) a new section, "Regularized versus unregularized: an overfitting curve,"
built on a new simulated dataset (production API records: queue depth, cache-miss rate, retry
count, payload size, predicting response latency) with new code comparing an unregularized and
a regularized gradient boosting configuration.

Neither addition introduces a new statistic, quote, or named framework requiring citation
beyond the chapter's existing three sources. The cross-reference passage names no new external
claim; it points to Chapter 7's own, separately cited, treatment. The new section describes a
simulated experiment specific to this book, using a dataset (API latency, not lending or S&P
500 data, and not gedeck's rollback-risk data used elsewhere in this chapter) and code written
fresh for this task. Two targeted web searches on distinctive phrases from the new section
("puts the chapter's last three levers... overfitting curve," "a boosting pipeline that logs
only the current round's training loss") returned no matching source.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the project's
standing "do not publish" instruction).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~3,827 |
| Sections analyzed | 10 |
| New paragraphs this session | 9 (2 in the cross-reference passage, 7 in the new section) |
| Phrases extracted for verification (new content) | 2 distinctive phrases |
| Web searches executed (new content) | 2 |
| Critical issues | 0 |
| Warning issues | 0 |
| Note flags | 0 |
| Verified clear | 2 (new passages) plus 3 (existing citations, re-confirmed present and unchanged) |
| Existing citations in document | 3 (Friedman 2001, Chen and Guestrin 2016, Ke et al. 2017) |

---

## Findings (this session's additions)

No critical, warning, or note-level findings.

- **Cross-reference passage** (in "Boosting versus bagging," two paragraphs after the log-loss
  discussion): points to Chapter 7's `@fig-confusion-threshold` section for the confusion
  matrix, precision, recall, F1, and ROC curve treatment instead of rebuilding it. No new
  statistic or named framework is introduced; the terms named (confusion matrix, precision,
  recall, F1, ROC curve) are the same terms Chapter 7 defines and cites where appropriate.
  **CLEAR**.
- **"Regularized versus unregularized: an overfitting curve"** (new section, between "Tree
  depth as a regularizer" and "XGBoost"): a simulated experiment on a new dataset built only
  for this comparison, with new code (`api_latency_data`, `fig_overfitting_curve` in the
  paired `-plots.py`/`-plots.ipynb`). No direct quotes, no uncited statistics (the numeric
  values shown are outputs of this session's own simulation, not claims sourced elsewhere), no
  named methodology beyond gradient boosting itself, which the chapter cites to Friedman 2001
  in the section immediately above. **CLEAR**.

Two web searches on distinctive phrases from the new section ("puts the chapter's last three
levers... overfitting curve" and "a boosting pipeline that logs only the current round's
training loss") returned no matching source, confirming original phrasing.

---

## Verified passages

- **The cross-reference passage's own claim** (that a boosted classifier is evaluated the same
  way Chapter 7's rollback classifier is): a direct, correct restatement of Chapter 7's own
  section, not a new claim requiring separate verification.
- **The overfitting-curve section's simulated dataset and code**: newly written for this
  session, explicitly labeled as simulated, no source match expected or found. Distinct from
  gedeck's lending and S&P 500 datasets, and distinct from the `rollback_risk_data` dataset
  used elsewhere in this chapter.
- **The three pre-existing citations** (Friedman 2001, Chen and Guestrin 2016, Ke et al. 2017):
  confirmed present, unchanged, and correctly placed relative to the sections that cite them.

---

## Citation health by section

| Section | Paragraphs | Citations | Status |
|---------|-----------|-----------|--------|
| Boosting versus bagging | 4 (was 2, plus 2 new) | 1 (Friedman) | OK |
| Regularized versus unregularized: an overfitting curve | 5 (new section) | 0 | OK, conceptual and original content describing a simulated experiment, no attributable source |
| XGBoost | 2 | 1 (Chen and Guestrin) | OK |
| LightGBM | 3 | 1 (Ke et al.) | OK |
| All other sections | ~17 | 0 | OK, conceptual and original content, no attributable source |

No run of 3+ paragraphs containing an unattributed claim that required a citation.

---

## Recommended action sequence

None. Both additions are clean as drafted.

---

## Prior audit history (base content, carried forward from the equivalent quarto-book report)

The base chapter (everything except this session's two additions) went through six prior
clarity/formatting passes, each re-running `/plagiarism-check` and finding zero new critical,
warning, or note-level issues: a sentence-by-sentence clarity pass, a second clarity pass
fixing two voice inconsistencies, a teaching-clarity pass (CART/GOSS/SHAP glosses, a corrected
log-loss gradient sign caught by anti-sycophancy self-check), a figure-first structure and
paragraph-formatting pass, and a second image/caption/paragraph/callout pass that corrected one
caption (`fig-model-comparison-size`) to state the forest-vs-boosting crossover the chart
shows. Full detail for each pass is preserved in
`.humanize-log-chapter-08-gradient-boosting.md`. All three citations (Friedman 2001, Chen and
Guestrin 2016, Ke et al. 2017) were confirmed live via WebSearch before drafting and remain
untouched through every pass, including this session's.

**Note**: this file lives at `published-repo/chapters/` under the chapter's current filename.
An earlier version of this report existed only at `quarto-book/chapters/` under the chapter's
older internal slug; this file supersedes it for the published-repo copy, and the sync step at
the end of this session's task copies this file back to `quarto-book/` so both folders match.
