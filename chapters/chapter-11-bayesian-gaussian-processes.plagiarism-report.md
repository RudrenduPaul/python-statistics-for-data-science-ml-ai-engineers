# Plagiarism & Attribution Audit Report

**Document**: chapter-bayes-gaussian-processes.md
**Audit date**: 2026-09-01
**Audited by**: /plagiarism-check

---

## Executive summary

This chapter has no direct quotes, no statistics requiring external attribution, and no
uncited named frameworks beyond standard Gaussian process terminology (RBF kernel, Matern
kernel, marginal likelihood maximization) covered by the chapter's one citation, Rasmussen and
Williams 2006, the standard textbook reference for this material. The worked numeric example
(the two-observation posterior computation) is original arithmetic, verified by running the
computation in this session rather than asserted, and one error caught that way (an incorrect
kernel value and posterior mean) was corrected before this audit ran. Library names
(scikit-learn, PyMC, GPyTorch, GPflow) are factual tool references, consistent with how earlier
chapters name Datadog, Honeycomb, and Jaeger without a citation.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~2,300 |
| Sections analyzed | 7 |
| Paragraphs | ~20 |
| Phrases extracted for verification | 0 (no quotes, no uncited statistics found) |
| Web searches executed | 0 (no new claims required verification beyond the numeric check) |
| Critical issues | 0 |
| Warning issues | 0 |
| Note flags | 0 |
| Verified clear | 1 (Rasmussen and Williams 2006, cited and verified in Chapter 6's audit) |
| Existing citations in document | 1 |

---

## Verified passages

- **GP math (kernel definition, posterior mean/covariance formula, marginal likelihood)**:
  standard, textbook-level mathematics, not attributable to any single source beyond the
  general reference cited above.
- **The two-observation worked example**: original arithmetic, computed and checked in this
  session, not sourced from any external example.
- **Library names and their documented capabilities** (scikit-learn's O(n^3) cost, GPyTorch
  and GPflow as the standard sparse/variational alternatives): factual, no citation needed,
  matches the pattern established across this book for naming production tools.

---

## Citation health by section

| Section | Paragraphs | Citations | Status |
|---------|-----------|-----------|--------|
| The kernel: the one modeling choice that matters | 3 | 1 | OK |
| All other sections | ~17 | 0 | OK, conceptual/original content and code, no attributable source |

No run of 3+ paragraphs containing an unattributed claim that required a citation.

---

## Recommended action sequence

None. No outstanding action before this chapter moves to the humanize pass.

---

## Addendum: re-run after what/why/how + rigor pass (2026-09-01)

A new two-sentence paragraph was added to "Choosing the length-scale" flagging that marginal
likelihood maximization can land on a local optimum (a short-length-scale peak treating
variation as signal vs. a long-length-scale peak explaining it away as noise), and naming
`GaussianProcessRegressor`'s `n_restarts_optimizer` argument as the practical mitigation.

- **Category B/C check**: the local-optima behavior is standard GP theory, covered by the
  chapter's existing Rasmussen and Williams 2006 citation; no new external claim requiring a
  fresh citation.
- **Category E check**: `n_restarts_optimizer` verified live via WebFetch against the current
  scikit-learn `GaussianProcessRegressor` documentation
  (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
  on 2026-09-01. Confirmed: default `0`, restarts sampled log-uniform from the kernel's bounds,
  purpose stated as escaping local optima in the log-marginal-likelihood surface. The new
  paragraph's description matches; wording is an independent paraphrase, not copied from the
  docs.
- No quotes, statistics, or uncited frameworks introduced elsewhere in this pass.

Updated word count: ~2,380. Verdict unchanged: **PUBLICATION-READY** (local review only, per
the standing "do not publish" instruction on this project).

---

## Addendum: re-check after sentence-level clarity pass (2026-09-01)

This pass edited roughly 20 sentences for clarity: splitting long compound sentences, replacing
ambiguous pronouns with their referents, and tightening awkward phrasing (including the
worked-example paragraph). No formula, citation, number, heading, or fact was changed; every
edit is a pure rewording of prose cleared earlier in this report, and the worked-example numbers
($33.8$ ms at $\rho=0.35$, $0.01$ mean and $400$ variance at $\rho=0.80$) still match the
figures the prior addendum verified. No new quotes, statistics, named frameworks, or attributed
claims were introduced, so no new web verification was required. The chapter's one citation
(Rasmussen and Williams 2006) remains untouched at its original location. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the standing "do not publish" instruction on this
project).

## Addendum: re-check after second clarity pass (2026-09-01)

Two edits applied this pass: added a short forward pointer in the opening "prior over functions"
section noting that length-scale and kernel shape get defined in full in the next section (the
terms were used there before their formal definition, which the pointer now flags for the
reader), and removed a redundant "one of the few... one of those rare cases" repetition in the
closed-form-posterior paragraph. Both are pure rewording; no formula, citation, number, heading,
or fact was changed. No new quotes, statistics, named frameworks, or attributed claims were
introduced, so no new web verification was required. The chapter's one citation (Rasmussen and
Williams 2006) remains untouched at its original location. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the standing "do not publish" instruction on this
project).

## Addendum: re-check after clarity/teaching-value editing pass (2026-09-01)

Four additive edits applied this pass, all aimed at closing gaps in unstated jargon or missing
background rather than adding new external material:

1. Restructured the opening "prior over functions" sentence and added a `Recall from Chapter 9`
   pointer naming $\beta_0$ and $\beta_1$ as the concrete coefficients Bayesian linear
   regression's prior sits over. **Attribution check**: this points to the book's own Chapter 9,
   not an external source; no citation required. Verified against Chapter 9's own text
   (`chapter-09-bayesian-regression.md`, line 16: $Y = \beta_0 + \beta_1 X + \varepsilon$) that
   the cross-reference is accurate rather than a fabricated recollection.
2. Added a parenthetical gloss for "Brownian motion" (a random walk with no well-defined slope
   anywhere). **Category B/C check**: standard mathematical terminology, the same class of
   textbook fact covered by the chapter's Rasmussen and Williams 2006 citation; no new citation
   needed.
3. Added two sentences unpacking why the log marginal likelihood is called "marginal" (the
   unobserved function values are integrated out). **Category B/C check**: standard GP theory,
   same coverage as above; no new citation needed.
4. Added one sentence at the first substantive use of "credible band" pointing back to Chapter
   9's credible interval. **Attribution check**: internal cross-reference to this book's own
   Chapter 9, not an external claim; no citation required.

No quotes, statistics, or named external frameworks were introduced. No web verification was
required, since nothing in this pass makes a claim beyond standard Gaussian process mathematics
or this book's own prior chapters. The chapter's one citation (Rasmussen and Williams 2006)
remains untouched at its original location, and the worked-example numbers are untouched.

Updated word count: ~2,480. Risk classification for this pass: **Clear** across all four
additions (0 Critical, 0 Warning, 0 Note). Verdict unchanged: **PUBLICATION-READY** (local
review only, per the standing "do not publish" instruction on this project).

## Addendum: re-check after figure-placement fix, caption rewrites, splits, callouts (2026-09-01)

This pass moved the `fig-extrapolation` figure div up into "From prior to posterior:
conditioning on data," directly after the worked-example paragraph that first cites it, fixing
a placement gap where the figure was cited in prose roughly 100 lines before its div appeared.
Rewrote four of five captions after re-reading each PNG, split roughly 14 over-length paragraphs
at sentence boundaries, and added four callouts (in "A prior over functions," "What the credible
band buys," "More than one input at a time," and "When the closed form runs out").

- **Category A/B/C/D/E check**: none of the new or moved sentences (the fig-extrapolation setup
  line, the four rewritten captions, the four callouts, the split-paragraph transitions)
  introduce a direct quote, a statistic, a named framework, a distinctive phrase resembling
  external prose, or an attributed claim. The captions describe what is visible in the PNGs
  reviewed this pass; the callouts are original practical commentary consistent with the three
  the chapter had going in. No web verification was required.
- **Fact/number/formula integrity check**: confirmed every formula, the worked-example numbers
  ($k(0.30,0.40)\approx 243$, posterior mean $\approx 33.8$ ms at $\rho=0.35$, $\approx 0.01$
  mean and $\approx 400$ variance at $\rho=0.80$), and the chapter's one citation (Rasmussen and
  Williams 2006) are unchanged, only relocated as whole blocks alongside the figure move.
- **Visual inspection**: all five figure PNGs (fig-prior-samples, fig-kernel-comparison,
  fig-posterior-fit, fig-length-scale-fit, fig-extrapolation) were re-read directly against the
  rewritten captions and the surrounding prose. The fig-extrapolation caption was checked
  against the visible posterior-mean flattening/drifting-down pattern past the shaded boundary,
  which the previous caption in this chapter's history did not mention; the new caption states
  it along with the caveat that it is not a forecast of improving latency, which the
  anti-sycophancy self-check in the matching humanize-log addendum confirms did not soften or
  drop any existing claim.

A prior addendum below claims this same figure-reorder work was completed on 2026-09-01, but the
file state found at the start of this pass still had the placement defect that addendum
describes fixing, so this addendum supersedes that record's figure-move claim specifically; its
caption, paragraph-split, and citation-integrity findings are consistent with what this pass
also found and are not disputed.

Updated word count: ~2,760. Risk classification for this pass: **Clear** (0 Critical, 0 Warning,
0 Note). Verdict unchanged: **PUBLICATION-READY** (local review only, per the standing "do not
publish" instruction on this project).

## Addendum: re-check after figure-reorder/caption/callout formatting pass (2026-09-01)

This pass reordered every figure div in the chapter to appear immediately after its section
heading (with one short orienting sentence ahead of it where needed), moved the interpretive
prose that used to precede each figure to follow it instead, rewrote five thin captions into
takeaway-stating captions, split several long paragraphs, and added three new callout boxes
(one `.callout-note` on kernel choice, one `.callout-warning` on the $O(n^3)$ cost, one
`.callout-tip` on `n_restarts_optimizer`).

- **Category A/B/C/D/E check**: none of the newly written sentences (orienting sentences,
  rewritten captions, callout text) introduce a direct quote, a statistic, a named framework, a
  distinctive phrase resembling external prose, or an attributed claim. Every new sentence is
  either a plain restatement of content covered elsewhere in this report (the caption rewrites
  and orienting sentences restate facts from the paragraphs immediately following them) or
  original practical commentary (the three callouts). No web verification was required.
- **Fact/number/formula integrity check**: confirmed every formula, the worked-example numbers
  ($k(0.30,0.40)\approx 243$, posterior mean $\approx 33.8$ ms at $\rho=0.35$, $\approx 0.01$ mean
  and $\approx 400$ variance at $\rho=0.80$), and the chapter's one citation
  (Rasmussen and Williams 2006) are unchanged and in their original form, only relocated as whole
  blocks alongside their section's figure reordering.
- **Visual inspection**: all five figure PNGs (fig-prior-samples, fig-kernel-comparison,
  fig-posterior-fit, fig-length-scale-fit, fig-extrapolation) were re-read and checked against
  the surrounding prose; no mismatch found.

Updated word count: ~2,650. Risk classification for this pass: **Clear** (0 Critical, 0 Warning,
0 Note). Verdict unchanged: **PUBLICATION-READY** (local review only, per the standing "do not
publish" instruction on this project).

## Addendum: report copied into published-repo, and light-touch periodic-kernel pass (2026-09-05)

`published-repo/chapters/` is this project's authoritative edit target (per the project's own
divergence-resolution decision); this report previously existed only under `quarto-book/chapters/`
even though `published-repo`'s copy of the chapter had moved ahead of every pass logged above (it
carries a `WhiteKernel`-omission callout and a full "More than one input at a time" ARD figure
and callout that this report's history never covered). This addendum copies the full history
above into `published-repo/chapters/chapter-11-bayesian-gaussian-processes.plagiarism-report.md`
unchanged and adds the audit for the one new section added in this pass.

**New content audited**: one new section, "Periodic structure: composing kernels for signals
that repeat," inserted after "More than one input at a time" and before "When the closed form
runs out." Adds one new figure (`fig-periodic-kernel`, RBF-only vs. RBF+periodic composite kernel
forecasting a held-out fourth day of a simulated daily latency cycle), a periodic-kernel formula,
one new code sample, and two callouts.

- **Category A (quotes)**: none.
- **Category B (statistics)**: the RMSE figures (about 13.8 ms for the RBF-only fit, about 1.2 ms
  for the RBF-plus-periodic fit) and the credible-band width ratio (about three times as wide)
  come from a simulated dataset generated and fit inside this session, the same category as the
  chapter's existing two-observation worked example. Independently recomputed by running the
  chapter's own `-plots.py` figure function in the same sequential order the script executes it
  in (later figures in the file consume the shared RNG object, so an isolated re-run would not
  reproduce the number a full run produces); the values printed match the numbers stated in the
  new prose.
- **Category C (named frameworks)**: the periodic kernel (`ExpSineSquared`) and the additive
  trend-plus-periodic composition device are standard Gaussian process terminology and technique,
  covered by the chapter's existing Rasmussen and Williams 2006 citation.
- **Category E (attributed claim)**: the new paragraph states that Rasmussen and Williams build a
  composite kernel this way to model the Mauna Loa atmospheric carbon dioxide record. Verified
  live via WebSearch on 2026-09-05: this is Section 5.4.3 of *Gaussian Processes for Machine
  Learning*, which builds a covariance function combining a long-term trend term, a periodic
  seasonal term, a term for medium-term irregularities, and a noise term to model that dataset.
  The chapter's revised wording ("a smooth trend term added to a periodic term plus further terms
  for medium-term drift and noise") matches this structure rather than understating it as a
  two-term sum. **VERIFIED**.
- **Category D (distinctive phrases)**: the periodic-kernel formula matches the standard
  `ExpSineSquared` definition used across the GP literature and in scikit-learn's own
  documentation; formulas are not attributable prose, consistent with how this report has always
  treated the chapter's RBF and Matern formulas. The commuter analogy paragraph and the two new
  callouts are original commentary, not matched to any external source.

No Critical or Warning findings. One Note-level attributed claim, independently confirmed above
and upgraded to Verified.

Updated word count: ~3,150. Verdict unchanged: **PUBLICATION-READY** (local review only, per the
standing "do not publish" instruction on this project).

## Addendum: GP-classification and full-Bayesian-hyperparameters gap-fill pass (2026-09-05)

**Scope**: an audit against BAP (Bayesian Analysis with Python) chapter 7 found this chapter's
biggest gap was GP classification, previously a single paragraph with no math, code, or figure.
Three additions close that gap: a new section "Full Bayesian hyperparameters: sampling the
length-scale instead of optimizing it" (one new two-panel figure, `fig-full-bayes-hyperparams`,
one PyMC code sample, one callout), a new section "When the outcome is a label: Gaussian process
classification" (one new figure, `fig-gp-classification-boundary`, one PyMC code sample, one
callout), and two compact callouts on GP-modulated Poisson GLMs and Kronecker-structured GPs
added to "When the closed form runs out." No existing section, figure, or callout was removed,
shortened, or reworded beyond the two-sentence intro tweak in "When the closed form runs out"
that now points at the new classification section instead of re-describing it inline.

- **Category A (quotes)**: none.
- **Category B (statistics)**: every number in both new sections (the length-scale posterior
  mean/interval, the noise posterior interval, the credible-band width ratios, the flagged-count
  29/90, the logistic-regression coefficient 0.030, the GP posterior probabilities at the range
  edges and midpoint) comes from datasets generated and models fit inside this session, using
  dedicated RNG seeds (107 and 207) independent of every other figure's draws so the numbers do
  not depend on script execution order. Recomputed outside the chapter's own `-plots.py` first,
  then reconfirmed by loading the edited script via `importlib` and calling its functions
  directly, and a third time from the executed `.ipynb` cell outputs: all three runs agree with
  the prose to the stated precision.
- **Category C (named frameworks)**: `pm.gp.Latent`, `pm.gp.Marginal`, `pm.math.invlogit`, NUTS,
  and `pm.gp.LatentKron` are PyMC API names, not attributable prose; the chapter's existing PyMC
  citation ([@salvatier2016pymc3], introduced in Chapter 9) covers the library. No new
  citation needed.
- **Category D (distinctive phrases)**: four of the most distinctive original phrases in the new
  content were checked directly via live WebSearch on 2026-09-05: "A prior on the length-scale,
  signal variance, and noise means none of the three collapses" (no matching or near-matching
  wording found; results returned general GP-hyperparameter-prior literature, unrelated
  wording), "algebraic shortcut, integrating the latent function out in closed form" (no match;
  unrelated closed-form-solution results), "A lookout watching one corridor for trouble" and
  "Picture two analysts handed the same eight readings" (no match on either; both original
  analogies), "A rule that only tracks how far along the corridor something is" (no match;
  results were about an unrelated accounting "corridor rule"), and the Kronecker-GP callout's
  framing against "sparse or variational approximation" (matched only to legitimate GP-scaling
  literature on the general Kronecker-GP concept, not to any specific sentence; the callout's own
  wording carries no citation, matching how this chapter treats `LatentKron` as a named PyMC
  technique rather than a specific paper's finding). **All CLEAR**.
- **Category E (attributed claims)**: none of the new content names an external organization,
  researcher, or study as a source; every claim is either a mechanical fact about the PyMC API
  (verified by running it) or a number computed inside this session.
- **BAP dataset/example avoidance check**: grepped the full chapter for every BAP ch.7 dataset
  and variable name to avoid (`space_flu`, `iris`, `coal`, `redwood`, `island`, `kline`,
  `sepal`) after drafting. Zero matches. The classification example (request payload size vs. a
  U-shaped anomaly-flag probability) and the full-Bayesian-hyperparameters example (a small
  concurrent-load-vs-latency subset, reusing this chapter's own running scenario) are both
  original to this book, matching the research file's recommendation.

No Critical or Warning findings. Updated word count: ~6,000. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the standing "do not publish" instruction on this
project).
