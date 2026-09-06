# Plagiarism & Attribution Audit Report

**Document**: chapter-bayes-bart.md
**Audit date**: 2026-09-01
**Audited by**: /plagiarism-check

---

## Executive summary

This chapter is new content extending Chapter 7's canary-rollback classifier scenario into a
Bayesian treatment (BART). It carries one substantive external claim requiring verification:
the Chipman, George, and McCulloch (2010) default hyperparameters for the tree-structure,
leaf-value, and error-variance priors. That claim was checked live against three independent
sources (a course PDF from co-author Rob McCulloch's own site, a graduate econometrics
textbook, and the original paper's abstract page) during drafting and confirmed to match:
alpha = 0.95, beta = 2, k = 2, m = 200. The pymc-bart library facts referenced (the `BART`
distribution's API, `plot_variable_importance`, and the `method="backward"` option) were
confirmed against the library's own official documentation during drafting. No direct quotes,
no uncited statistics, and no named framework introduced without attribution.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~2,506 |
| Sections analyzed | 8 |
| Paragraphs | ~20 |
| Phrases extracted for verification | 2 |
| Web searches and fetches executed | 2 (during drafting) |
| Critical issues | 0 |
| Warning issues | 0 |
| Note flags | 0 |
| Verified clear | 2 |
| Existing citations in document | 1 ([@chipman2010bart], reused from Chapter 7's existing
  citation, no new bib entry needed) |

---

## Verified passages

- **BART default hyperparameters** (alpha = 0.95, beta = 2, k = 2, m = 200): confirmed against
  three independent sources during drafting, including a PDF hosted by original paper co-author
  Rob McCulloch. Cited to [@chipman2010bart], present in `references.bib`.
- **pymc-bart API facts** (the `BART` distribution signature, `plot_variable_importance`'s
  split-count heuristic, the `method="backward"` alternative): confirmed against
  pymc.io's official pymc-bart documentation during drafting.
- **All numeric worked-example values** (the canary-rollback simulation, the eight held-out
  deployments, the credible-interval widths): generated for this chapter's own simulated
  scenario, explicitly labeled as simulated, consistent with every other chapter in the book.
- **The "orders of magnitude slower" MCMC-versus-random-forest fitting-time comparison**: stated
  as a directional, hedged estimate rather than a precise benchmark number, consistent with how
  the rest of the book handles claims without a specific benchmark to cite.

---

## Citation health by section

| Section | Paragraphs | Citations | Status |
|---------|-----------|-----------|--------|
| The three priors doing the work | 5 | 1 | OK |
| All other sections | ~15 | 0 | OK, conceptual, worked-example, and tooling content with no
  additional attributable claim |

No run of 3+ paragraphs containing an unattributed claim that required a citation.

---

## Recommended action sequence

None. No fixes required before this chapter moves to the humanize pass.

---

## Addendum: re-run after what/why/how + rigor pass (2026-09-01)

This pass re-verified the `m = 200` claim in "Verified passages" above and found it stated with
more confidence than earned: `m = 200` is the Chipman, George, and McCulloch (2010) paper's
recommended default, but it is not `pymc-bart`'s own library default. Verified live via WebFetch
against the current `pymc-bart` API reference
(https://www.pymc.io/projects/bart/en/latest/api_reference.html) on 2026-09-01: the `BART`
distribution's signature is `m: int = 50`, `alpha: float = 0.95`, `beta: float = 2.0`, with no
separate `k` argument exposed on the distribution itself. A second search corroborated that
`pymc-bart` internally still uses `k = 2` inside its leaf-value-scale formula
(`sigma_mu = 0.5 / (k * sqrt(m))`), so the `alpha`/`beta`/`k` claims stand; only `m = 200` was
wrong as a `pymc-bart` default.

The chapter text ("The three priors doing the work" section) was corrected to say `pymc-bart`
matches Chipman, George, and McCulloch on `alpha`, `beta`, and `k`, but ships its own default of
50 trees rather than 200, and now explains that this is why the code example sets `m=100`
explicitly. No new citation needed: still covered by the existing [@chipman2010bart] entry in
`references.bib`. This correction also supersedes the "Verified passages" line above listing
`m = 200` as a confirmed `pymc-bart` fact; that line should be read as "`m = 200` is the paper's
recommendation, not the library's default" going forward.

No other findings from this pass. Verdict unchanged: **PUBLICATION-READY** (local review only,
per the standing "do not publish" instruction on this project).

---

## Addendum: re-check after sentence-level clarity pass (2026-09-01)

This pass edited roughly 20 sentences for clarity: splitting long compound sentences, removing
redundant phrasing (e.g. the raised-$k$/lowered-$k$ contrast), and reordering one paragraph in
"The credible interval as a decision rule" so the routing outcome leads before the explanation
of why. No formula, citation, number, heading, or fact was changed; the $\alpha=0.95$,
$\beta=2$, $k=2$, and corrected $m=50$-library-default/$m=200$-paper-default figures established
in the addendum above are unchanged and appear in the same places. No new quotes, statistics,
named frameworks, or attributed claims were introduced, so no new web verification was required.
The chapter's citation ([@chipman2010bart]) remains untouched at its original locations. Verdict
unchanged: **PUBLICATION-READY** (local review only, per the standing "do not publish"
instruction on this project).

## Addendum: re-check after second clarity pass (2026-09-01)

Five edits applied this pass: replaced an ambiguous "it" in the opening paragraph with its
referent, tightened one elliptical clause in the Fourier-series analogy, restructured the
default-hyperparameters paragraph to name all four hyperparameters (alpha, beta, k, m) up front
instead of asking the reader to count to "three of the four" without them enumerated, split a
"two differences that matter" sentence into two explicitly labeled differences, and split a
dense trailing clause in the variable-importance section into two sentences. All five are pure
rewording; the same alpha = 0.95, beta = 2, k = 2, m = 50-library-default/m = 200-paper-default
figures established in the earlier addendum are unchanged and appear in the same places. No
formula, citation, number, heading, or fact was changed. No new quotes, statistics, named
frameworks, or attributed claims were introduced, so no new web verification was required. The
chapter's citation ([@chipman2010bart]) remains untouched at its original locations. Verdict
unchanged: **PUBLICATION-READY** (local review only, per the standing "do not publish"
instruction on this project).

## Addendum: re-check after teaching-clarity pass (2026-09-01)

Three edits applied this pass, all additive plain-language clarifications: an "In other words"
restatement of "a full posterior distribution over the score" in the opening paragraph, a
subject-verb fix in the `pymc-bart` tuning-effort sentence (no factual content changed), and a
clause explaining "closed-form posterior" plus the literal expansion "MCMC (Markov chain Monte
Carlo)" on its first use in the running text. None of these three edits introduce a new
external fact, statistic, named framework, or quote; each restates or explains a claim this
report has covered in an earlier verification pass. No new web search was required.

Risk classification for the three new passages, using this report's existing scheme:

| Passage | Risk |
|---------|------|
| "In other words, BART hands back a whole range of plausible scores..." | Clear (original restatement of a claim this report verified in an earlier pass, describing what a posterior distribution provides; no external fact added) |
| "tuning `pymc-bart` rarely needs more than a glance at the tree count..." | Clear (rephrasing for grammatical correctness only; the underlying claim about tuning effort is the chapter's own practical observation, not a citable benchmark, consistent with how the rest of the chapter treats non-benchmarked practical claims) |
| "...there is no formula that converts the priors and the data directly into the posterior distribution... MCMC (Markov chain Monte Carlo)" | Clear (standard textbook definition of a standard term; MCMC's expansion is a dictionary fact, not requiring citation, consistent with how other acronyms are handled elsewhere in this book) |

No Critical, Warning, or Note flags from this pass. Verdict unchanged: **PUBLICATION-READY**
(local review only, per the standing "do not publish" instruction on this project).

## Addendum: re-check after figure-reordering and structure pass (2026-09-01)

This pass restructured all five figures (fig-ensemble-buildup, fig-tree-structure-prior,
fig-mcmc-trace, fig-bart-vs-rf-interval, fig-variable-importance) to appear immediately after
their section heading, ahead of the prose that names and interprets them, per this chapter's new
image-before-text convention. Each figure div now opens with one short orienting sentence written
for this pass, and the "what you just saw" naming sentences that used to precede each figure were
relocated to follow it. Three captions were rewritten to state the takeaway rather than just
describe the axes (fig-ensemble-buildup: now states the one-tree to two-hundred-tree progression;
fig-bart-vs-rf-interval: now states that point predictions agree but interval width varies by
deployment; fig-variable-importance: now states that canary error rate dominates and matches the
random forest's Gini ranking). Two captions (fig-tree-structure-prior, fig-mcmc-trace) stated
their takeaway from the prior pass and were kept as-is. All five PNGs were reopened and confirmed
against their surrounding prose a second time, including fig-mcmc-trace, which continues to show
two chains converging as intended (the bug this figure had before an earlier pass fixed it).

Roughly a dozen long paragraphs were split at natural breaks for readability, including the
opening hook paragraph, the Chipman/George/McCulloch defaults paragraph, and the closing
decision-rule paragraph. Two callout boxes were added: a `.callout-note` restating the
`m=50`-library-default-versus-`m=200`-paper-default fact verified in the addendum above, and a
`.callout-warning` restating the point, established earlier in this chapter, that BART does not
parallelize within a single MCMC chain the way a random forest's independent trees do. Both
callouts restate facts this report has verified before; neither introduces a new external claim,
statistic, quote, or named framework, so no new web verification was required. No formula,
citation, number, or fact was changed. The chapter's citation ([@chipman2010bart]) remains
untouched at its original location. Verdict unchanged: **PUBLICATION-READY** (local review only,
per the standing "do not publish" instruction on this project).

## Addendum: caption accuracy fixes, paragraph splits, four new callouts (2026-09-01)

This pass re-read every figure PNG directly against its caption and the surrounding prose,
rather than checking only the earlier verified facts. Two numeric mismatches between the prose
and the figures were found and corrected; neither introduces a new external claim, so both are
internal consistency fixes rather than attribution issues, but both are logged here since they
change what the text asserts about a figure's content.

- **fig-tree-structure-prior**: the body prose said the split probability falls "less than a
  one-in-five chance" by depth 3 "under the default penalty." The figure's static PNG (the
  slider's own starting position) is beta=1, where depth 3 is 0.24 by the chapter's own
  $\alpha(1+\text{depth})^{-\beta}$ formula ($0.95 \times 4^{-1} = 0.2375$), not under 0.2. At
  the library's default of beta=2, named later in the same section, depth 3 works out to
  $0.95 \times 4^{-2} \approx 0.06$, which is under one-in-five. Corrected the sentence to state
  the beta=1 figure's own value (under one-in-four) and added the beta=2 comparison (under
  one-in-ten) as a second sentence. Both numbers were computed directly from the formula cited in
  this chapter to [@chipman2010bart]; no new source needed.
- **fig-bart-vs-rf-interval**: the body prose claimed credible intervals stay tight for
  deployments "near the middle of the training data" and widen "toward the edge of what the
  model has seen." Reading the PNG directly, the five lowest-probability deployments carry the
  widest intervals in this set (several reaching to 0) and the highest-probability deployment
  carries the narrowest, the reverse of that claim. This chapter's figure does not show each
  deployment's position in the underlying four-feature training distribution, so the
  middle-versus-edge explanation is not something this figure can confirm or deny. Rewrote the
  paragraph to state only the checkable pattern (interval width does not track the point
  estimate; the lowest-probability deployments carry the widest intervals here) and removed the
  unconfirmed causal claim rather than replacing it with a different unverified one.

Caption rewrites (4 of 5) name axes and state values read directly off each PNG
(fig-ensemble-buildup, fig-tree-structure-prior, fig-mcmc-trace, fig-bart-vs-rf-interval).
fig-variable-importance's caption was extended to name its axes; its existing takeaway claim
(canary error rate dominating, matching Chapter 7's Gini ranking) was checked against the PNG
and confirmed accurate.

Four new callouts added (`.callout-tip` in "A sum of weak trees," `.callout-note` in "The
rollback classifier in pymc-bart," `.callout-tip` in "Variable importance from the posterior,"
`.callout-important` in "The credible interval as a decision rule"): all four are original
practical guidance, none introduces an external fact, statistic, or named framework, and none
duplicates a point covered by one of the chapter's five existing callouts. No new web
verification was required for any of the four.

Roughly 20 over-length paragraphs were split at sentence boundaries; no formula, citation, or
number besides the two corrections above was changed anywhere in this pass. The chapter's
citation ([@chipman2010bart]) remains at its original locations, now also backing the
beta=1/beta=2 depth-3 arithmetic in the corrected tree-structure-prior paragraph.

Risk classification for this pass:

| Passage | Risk |
|---------|------|
| Corrected fig-tree-structure-prior paragraph (beta=1 and beta=2 depth-3 values) | Clear (arithmetic from the chapter's own cited formula, independently recomputed this pass) |
| Corrected fig-bart-vs-rf-interval paragraph (interval-width pattern) | Clear (states only what the PNG directly shows; the removed middle/edge claim is not restored elsewhere) |
| Four new callouts | Clear (original practical commentary, no external claim) |

No Critical or Warning flags. Updated word count: ~2,850. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the standing "do not publish" instruction on this
project).

## Addendum: third callout added for coverage parity (2026-09-01)

The chapter carried two callouts against a roughly 2,500-word body, below the one-per-400-to-600
-word target set for this pass, and below the three each of Chapter 10 and Chapter 11 reached.
Added a `.callout-tip` after the R-hat/convergence-diagnostic paragraph in "How BART is fit,"
restating the point that R-hat needs checking across every parameter in the sum-of-trees model,
not only the one leaf-value trace this chapter walks through by hand. This restates a claim
this report's "Verified passages" section covers above (the R-hat convergence-diagnostic
description); no new external fact, statistic, quote, or named framework was introduced, so no
new web verification was required. No formula, citation, number, or fact was changed. Verdict
unchanged: **PUBLICATION-READY** (local review only, per the standing "do not publish"
instruction on this project).

## Addendum: light-touch gap pass, sigmoid-link explanation added (2026-09-05)

A cross-chapter audit (`task-todo/audit-my-chapters-5-15.md`) flagged one concrete, unaddressed
gap on its own merit: the code example in "The rollback classifier in pymc-bart" wraps `mu`
(BART's raw output) in `pm.math.sigmoid(mu)` without explaining why that step is needed, unlike
Chapter 4's careful treatment of the same logistic transform for classical logistic regression.
Added roughly 230 words directly after the code block's `X_train`/`m=100` explanatory paragraph:
a short code snippet computing `sigmoid` at five representative `mu` values, followed by what the
sigmoid does (`pmb.BART` models an unconstrained numeric function, not a probability directly),
why it matters (a script reading `idata`'s posterior summary without the transform could
misread `mu` as a probability, and a routing rule built on the credible-interval-width threshold
introduced later in the chapter would compare numbers on the wrong scale), and a closing note
that this is the identical logistic function Chapter 4 introduced, applied here to a tree-sum
instead of a linear predictor, plus the formula itself.

Verification performed for this addition:

- **Sigmoid arithmetic**: the five stated output values (`[0.12, 0.38, 0.50, 0.62, 0.88]` for
  `mu = [-2.0, -0.5, 0.0, 0.5, 2.0]`) were computed directly with `numpy` during drafting, not
  estimated. Confirmed the match to two decimal places.
- **Cross-reference to Chapter 4**: re-read `chapter-04-regression-modeling.md` directly
  (the section "Logistic regression applies the logistic function...") to confirm it states the
  formula $p(x) = 1/(1+e^{-(\beta_0+\beta_1 x)})$ before citing it here as the same conversion.
  Confirmed present at that location, so the cross-reference holds up and requires no new
  citation (an internal reference to this book's own prior chapter, not an external source).
- **`pmb.BART` modeling a continuous latent function**: this restates what the chapter's own
  existing code shows (`mu = pmb.BART(...)` produces the term that `pm.math.sigmoid`
  then transforms), not a new external claim about the library, so no new web verification was
  needed beyond the `pmb.BART` API facts this report verified earlier in this document.
- No statistic, named framework, direct quote, or attributed-to-an-organization claim appears in
  the new text. It is original engineering commentary plus one internal cross-reference and one
  standard textbook formula (the same treatment this report has extended to "MCMC
  (Markov chain Monte Carlo)" and "closed-form posterior" elsewhere in this chapter).

Risk classification:

| Passage | Risk |
|---------|------|
| Sigmoid code snippet and its stated output values | Clear (arithmetic independently verified this pass) |
| "`pmb.BART` models a continuous latent function..." | Clear (restates the chapter's own existing code, no new external claim) |
| "Skip the sigmoid, and a script reading `idata`'s posterior summary..." | Clear (original engineering reasoning, no external claim) |
| Cross-reference to Chapter 4's logistic function and formula | Clear (internal citation to this book's own verified content, re-confirmed this pass) |

No Critical, Warning, or Note flags from this pass. Updated word count: ~3,080. Verdict
unchanged: **PUBLICATION-READY** (local review only, per the standing "do not publish"
instruction on this project).
