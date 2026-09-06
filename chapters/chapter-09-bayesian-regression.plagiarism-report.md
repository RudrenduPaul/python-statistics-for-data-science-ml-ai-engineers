# Plagiarism & Attribution Audit Report

**Document**: chapter-bayes-regression.md
**Audit date**: 2026-09-01
**Audited by**: self-directed audit (fork execution, following /plagiarism-check methodology)

---

## Executive summary

This chapter extends Chapter 4's existing Bayesian subsection and the checkout-API running
example with new worked numeric examples, all computed in this session from the paired
`-plots.py` script rather than invented for narrative convenience. Three new citations were
checked live: Carvalho, Polson & Scott (2010) on the horseshoe prior, Salvatier, Wiecki &
Fonnesbeck (2016) on PyMC3, and Gelman, Meng & Stern (1996) on posterior predictive checks, all
confirmed via WebSearch against independent bibliographic sources before citing. One factual
claim (ArviZ's default HDI probability of 94%) was independently verified against ArviZ's own
documentation before stating it. No fabricated statistics, no fabricated URLs.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the project's
standing "do not publish" instruction).

---

## Findings

- **Numeric consistency check**: an initial run of the conjugate Gaussian posterior used an
  assumed noise variance ($\sigma^2 = 60^2$) mismatched to the simulated data's residual scale
  ($\sigma \approx 1.4$), which would have produced a credible interval far wider than the
  matching OLS confidence interval and undermined the chapter's central numeric comparison. Found
  and corrected before writing the prose, then all posterior/interval numbers in the chapter were
  regenerated to match the corrected figures (verified via a script replicating the RNG call
  order used by `chapter-bayes-regression-plots.py`).
- **Citations**: [@parkcasella2008] reused from the existing `references.bib` (no duplicate
  entry added). Three new entries added to `.refs-chapter-bayes-regression.bib`.
- **Code listings**: the PyMC/ArviZ snippets in the chapter prose (`pm.Model`, `pm.sample`,
  `az.summary`, `pm.sample_posterior_predictive`, `az.plot_ppc`) use current, working API calls,
  not pseudocode, per the coordinator's explicit steering message mid-task.

---

## Verified passages

- Horseshoe prior mechanism (half-Cauchy local scale, Normal-half-Cauchy hierarchy): matches
  the mechanism described in [@carvalhopolsonscott2010]'s abstract and summary, phrased in this
  chapter's own words.
- ArviZ's default `hdi_prob = 0.94`: confirmed via ArviZ's own documentation.
- All worked numbers (posterior means/sds, credible intervals, OLS comparison, coverage rate,
  logistic posterior modes): computed in this session, traceable to
  `chapter-bayes-regression-plots.py`, not asserted.

---

## Recommended action sequence

No outstanding action. Chapter is ready for the coordinator's renumbering and merge pass.

---

## Addendum: what/why/how + rigor pass (2026-09-01)

Re-ran phrase extraction and web verification against the two new passages added during this
pass:

1. **Ridge/Lasso closed-form correction** (only Ridge's Gaussian-Gaussian pairing is conjugate;
   the Bayesian Lasso's posterior has no closed form beyond its mode and needs Park and Casella's
   Gibbs sampler to recover in full): checked against Park and Casella (2008), "The Bayesian
   Lasso," Journal of the American Statistical Association 103(482), and the paper's own scale-
   mixture-of-normals hierarchy, confirmed live via WebSearch this session. No new citation
   needed, `[@parkcasella2008]` was cited in this chapter before this pass began. **VERIFIED**.
2. **Horseshoe funnel-geometry caveat** (spike-and-tail shape produces a funnel-shaped posterior
   that a naive sampler struggles with, addressed in practice with a non-centered
   parameterization): confirmed via WebSearch against the general MCMC/hierarchical-model
   literature on funnel geometry and non-centered parameterization (Neal's funnel, Piironen and
   Vehtari's horseshoe-sampling work). This is standard MCMC-diagnostics knowledge common to
   every treatment of shrinkage-prior sampling, not attributable to one source, matching how this
   chapter's report treats the ASA p-value statement and ArviZ defaults elsewhere. No citation
   added. **CLEAR**.

No critical, warning, or note-level issues found in the new material. No new bib entries were
required; both fixes rest on citations this chapter carried before this pass began.

Verdict: **PUBLICATION-READY** (local review only, per the project's standing "do not publish"
instruction).

---

## Addendum: 2026-09-01 (clarity pass re-check)

Ran a fourth-pass sentence-by-sentence clarity edit on this chapter, the densest of the three
audited this session (Bayesian regression, Ridge/Lasso-as-priors, horseshoe prior, MCMC
diagnostics). Edits split long sentences, resolved an elliptical construction in the ASA
p-value passage, fixed a comma splice in the conjugate-posterior sentence, and addressed two
seams where earlier passes' additions read as bolted on rather than woven in: the MCMC
paragraph that introduced the sampler by name twice in a row (once via an analogy sentence,
again in the following paragraph) was merged into one continuous explanation, and the
"four friends guessing temperature" analogy was moved so it leads directly into the $\hat{R}$
definition it illustrates, with a bridging sentence added, instead of sitting ahead of a
generic transition line. No new facts, statistics, quotes, or named frameworks were introduced;
all four prior citations (Carvalho/Polson/Scott 2010, Salvatier/Wiecki/Fonnesbeck 2016, Gelman/
Meng/Stern 1996, Park and Casella 2008) remain untouched and correctly placed.

Spot-checked the two passages most likely to converge on standard textbook phrasing under a
clarity edit: the $\hat{R}$ convergence-diagnostic description and the confidence-interval
procedural-statement framing. Both checked live via WebSearch this session against Stan/rstan
documentation and standard statistics references; the chapter's wording matches the field's
shared concept, as expected for an unattributable standard result, without matching any single
source's phrasing.

No critical, warning, or note-level issues found. Verdict unchanged: **PUBLICATION-READY**
(local review only).

---

## Addendum: 2026-09-01 (teaching-clarity pass re-check)

Sixth pass, the first of this session's two to add new explanatory prose rather than only
reshape existing sentences: five glosses closing jargon and formula gaps a working data
scientist would hit (posterior-distribution notation tying $\mu_n, \tau_n^2$ to the Gaussian
posterior's mean and variance; a *half-Cauchy distribution* gloss; a *non-centered
parameterization* gloss; a *Gibbs sampler* / *scale-mixture representation* gloss, reordered
ahead of the chapter's later formal MCMC definition; and a note distinguishing ArviZ's 94%
highest-density interval from the 95% equal-tailed credible interval computed by hand earlier in
the chapter). Checked each against field sources this session:

1. **Half-Cauchy distribution properties** (heavy tail, mass concentrated near zero, positive
   support): standard distribution-theory fact, part of the horseshoe-prior mechanism this
   chapter cites to [@carvalhopolsonscott2010]. No new citation needed. **CLEAR**.
2. **Non-centered parameterization mechanism**: confirmed via WebSearch this session against the
   horseshoe/funnel-geometry literature (Piironen and Vehtari on horseshoe shrinkage, and general
   non-centered-parameterization treatments of Neal's funnel), consistent with the earlier
   funnel-geometry addendum in this report. Standard MCMC-diagnostics knowledge, not
   attributable to one source. **CLEAR**.
3. **Gibbs sampler as an MCMC method that updates one coefficient conditional on the others, and
   the Laplace prior's scale-mixture-of-normals representation**: rests on the same Park and
   Casella (2008) source this report's first addendum verified; no new claim beyond what that
   verification covers. **CLEAR**.
4. **ArviZ's HDI as the narrowest interval containing the stated probability, versus an
   equal-tailed interval**: confirmed via WebSearch this session against ArviZ's own
   documentation and API reference, which defines `hdi()` as computing the shortest interval
   containing the given probability mass. Builds on the 94%-default fact this report's
   "Verified passages" section covers. **CLEAR**.

No critical, warning, or note-level issues found in the new material. No new bib entries
required. One self-caught overclaim from drafting (an early version of the scale-mixture gloss
called the un-augmented conditional posterior "an unsolvable integral") was corrected before
this report was written; the corrected wording, "no closed form at all," was the version
checked above.

Verdict: **PUBLICATION-READY** (local review only, per the project's standing "do not publish"
instruction).

---

## Addendum: 2026-09-01 (second clarity pass re-check)

Ran a fifth-pass sentence-level clarity edit on this chapter, given extra scrutiny as the
densest of the three audited this session and the one with the most edit history. Fixes: two
duplicate italics removed ("credible interval" was italicized again here despite being
introduced and italicized in Chapter 4; "prior" was italicized a second time one paragraph after
its own first use), a factual mismatch corrected (prose said the posterior-narrowing figure
covers "four sample sizes" when the figure caption and the underlying plotting script both use
five: n = 10, 30, 100, 300, 1000, confirmed by reading `chapter-09-bayesian-regression-plots.py`
directly), a second-person pronoun in the net-making-factory analogy rewritten to match the
third-person framing every other analogy in this chapter uses, an MCMC forward-reference fixed
(the acronym was used once in the horseshoe-prior section roughly 50 lines before its own
spelled-out definition in the PyMC section, so the earlier mention now spells out "Markov chain
Monte Carlo" without assuming the reader has seen the acronym yet), and one confusing sentence
about prior conjugacy rewritten for directness. No new facts, statistics, quotes, or named
frameworks were introduced, and no numeric values were changed, only a miscount in the
surrounding prose. All four prior citations (Carvalho/Polson/Scott 2010, Salvatier/Wiecki/
Fonnesbeck 2016, Gelman/Meng/Stern 1996, Park and Casella 2008) remain untouched and correctly
placed.

No critical, warning, or note-level issues found. Verdict unchanged: **PUBLICATION-READY**
(local review only).

---

## Addendum: 2026-09-01 (figure/structure/paragraph formatting pass re-check)

Seventh pass: a structural formatting pass, not a content pass. All seven figures (
`fig-posterior-narrowing`, `fig-ci-repeated-experiments`, `fig-credible-interval-single`,
`fig-prior-shapes`, `fig-shrinkage-profile`, `fig-bayesian-logistic`,
`fig-posterior-predictive-check`) were reordered so each figure div appears before the prose
that names and interprets it, five thin captions were rewritten to state a takeaway instead of
only describing axes, every paragraph in the chapter was split to roughly 3-4 lines, and three
Quarto callout boxes were added (a `.callout-note` on non-centered parameterization, a
`.callout-important` on the confidence-vs-credible-interval distinction, and a `.callout-warning`
on MCMC divergence checks). No new facts, statistics, or citations were introduced; all four
citations this chapter carried before this pass (Carvalho/Polson/Scott 2010, Salvatier/Wiecki/
Fonnesbeck 2016, Gelman/Meng/Stern 1996, Park and Casella 2008) remain untouched and correctly
placed.

One numeric correction: the posterior mode for the timeout-probability logistic coefficient was
stated in prose as 0.311, but a direct visual read of
`chapter-bayes-regression-fig-bayesian-logistic.png` (annotated on the chart itself as "posterior
mode = 0.307") showed the prose number did not match the figure. Corrected to 0.307 in both
places it appears (the interpretive sentence and the figure caption).

The two new callout sentences most likely to echo standard textbook phrasing (the confidence-
interval-vs-credible-interval distinction, and the horseshoe non-centered-parameterization
summary) were checked live via WebSearch this session (queries: `"a credible interval is a
direct probability statement about the parameter"` and `"one of the most common misreadings"
confidence interval credible interval`). No verbatim or near-verbatim phrase match turned up
against any single source; the underlying concept is standard, widely documented statistical
knowledge (matching prior addenda's treatment of the ASA statement and $\hat{R}$ description),
and both callouts condense this chapter's own cleared prose rather than assert a new claim. No
citation needed. **CLEAR**.

No critical, warning, or note-level issues found in this pass. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the project's standing "do not publish"
instruction).

---

## Addendum: 2026-09-01 (figure/caption/paragraph/callout audit pass re-check)

Eighth pass, a structural audit re-run independent of the prior pass, checking captions against
the referenced PNGs directly rather than trusting the earlier pass's summary. Found all 7
figure divs correctly placed and all 7 captions accurate against their charts, so no placement
or caption edits were made. Split 11 paragraphs that still ran 5 to 7 source lines (opening
hook, shared-model-versus-prior, n=40 numbers, horseshoe shape-versus-mechanism, Ridge-conjugate-
versus-Lasso-mode, logistic-likelihood complication, MCMC approximation, 94%-HDI-versus-95%,
posterior-predictive-check definition, the fig-posterior-predictive-check lead-in, and the
closing paragraph) at existing sentence boundaries, and added two `.callout-tip` blocks to the
two sections that had no callout (a posterior-mean-versus-OLS sanity check, and a posterior-
predictive-check reminder).

No new facts, statistics, or citations were introduced; all four citations this chapter carries
(Carvalho/Polson/Scott 2010, Salvatier/Wiecki/Fonnesbeck 2016, Gelman/Meng/Stern 1996, Park and
Casella 2008) remain untouched and correctly placed. The two new callout sentences restate
claims the chapter's own cleared prose makes elsewhere (the posterior-mean-to-OLS convergence
discussed at n=1000, and the "checking the posterior alone would have missed it" point closing
the posterior-predictive-check section) in a shorter form, so no independent web verification
was needed beyond confirming they introduce no new claim. **CLEAR**.

Checked every paragraph split for dropped caveats or technical drift: none found. Each split
sentence pair still carries the same qualifiers it carried before the split (for example, the
Ridge/Lasso split keeps "only its mode does" attached to the Lasso sentence rather than losing it
to the paragraph break).

No critical, warning, or note-level issues found in this pass. Verdict unchanged:
**PUBLICATION-READY** (local review only, per the project's standing "do not publish"
instruction).

## Addendum: 2026-09-05, light-touch pass, new "Prior predictive checks" section

Added one new section, "Prior predictive checks," between "Did the sampler converge?" and
"Posterior predictive checks," plus its supporting figure
(`chapter-bayes-regression-fig-prior-predictive-check`) and figure function
(`fig_prior_predictive_check`) in `chapter-09-bayesian-regression-plots.py` and the paired
notebook. This closes a gap the chapter-audit flagged: the chapter covers posterior predictive
checks in depth but never checks a prior before fitting anything.

- **Category C (named methodology)**: "prior predictive check" is standard modern Bayesian
  workflow terminology. Web search traced the concept to Box (1980) and Rubin (1984), with the
  contemporary reference practitioners cite being Gabry, Simpson, Vehtari, Betancourt, and Gelman,
  "Visualization in Bayesian Workflow," Journal of the Royal Statistical Society Series A,
  182(2):389-402, 2019 (confirmed live via WebFetch against
  https://academic.oup.com/jrsssa/article/182/2/389/7070184, which resolved and matched the
  title/authors/volume/issue/pages/year/DOI). Added as
  `[@gabrysimpsonvehtaribetancourtgelman2019]` in `references.bib`, cited once at the point the
  new section names the practice, matching this chapter's existing pattern of citing named
  Bayesian-workflow concepts (Gelman/Meng/Stern 1996 for posterior predictive checks,
  Carvalho/Polson/Scott 2010 for the horseshoe, Salvatier/Wiecki/Fonnesbeck 2016 for PyMC).
  **Fix applied** (Fix type 1, add citation).
- **Category D (distinctive phrases)**: web search on the section's original phrasing ("a prior
  that looks harmless," "before the model has looked at a single request," the figure's framing)
  found no exact or near-exact match to any existing source. **CLEAR**, original prose.
- **API accuracy check**: verified `pm.sample_prior_predictive`'s signature directly against
  PyMC's own documentation before treating the code listing as correct. The first draft used a
  `samples=500` keyword argument; the real parameter is `draws`. Corrected to `draws=500` before
  this report was written, so the shipped code listing is accurate.
- **Figure and code**: the new figure was generated by running the updated `-plots.py` and the
  updated `.ipynb` through the project's own `.venv` (confirmed image output rendered), not
  invented or hand-drawn. No new dataset was introduced; the figure reuses the same timeout
  logistic model and payload centering already established earlier in the chapter.

No critical, warning, or note-level issues remain. Verdict unchanged: **PUBLICATION-READY**
(local review only, per the project's standing "do not publish" instruction).

---

## Addendum: BAP-gap-driven depth pass (2026-09-05)

Four new sections added: robust regression via a Student-t likelihood, hierarchical/multilevel
regression across five services, a from-scratch Metropolis-Hastings walkthrough, and a
divergences/funnel figure. Scope: ~3,100 new words. Audited against `task-todo/research-bap-ch1-2.md`,
`research-bap-ch3-4.md`, `research-bap-ch6-8.md`, and `audit-my-bayesian-chapters-current.md`'s
list of BAP datasets, variable names, and staging devices to avoid.

**Dataset and scenario check**: no reuse of BAP's Anscombe quartet, `chemical_shifts.csv`,
`babies.csv`, `iris.csv`, `fish.csv`, or its 8-group synthetic hierarchical seed
(`np.random.seed(314)`, M=8, N=20, one group truncated to n=1). This chapter's own checkout-API
payload/latency framing was extended instead: a retry-storm outlier scenario (6 of 60 requests,
each spiked 150-300 ms) for robust regression, and five named backend services (cart, pricing,
inventory, shipping, fraud) with fraud-service deliberately starved to n=4, for the hierarchical
section. Neither scenario, seed, group count, nor starvation pattern matches BAP's.

**Variable-naming check**: no Greek-letter PyMC variable names (`α`, `β`, `ϵ`, `ν` as literal
labels) and no `pm.math.dot`/`theano.shared`/old `sd=` kwarg carried over from BAP's PyMC3/Theano
code. All PyMC snippets use PyMC 5 syntax (`sigma=`, `pm.Model(coords=...)`, `dims=`), per the
task brief's instruction.

**Category A (quotes)**: two internal self-quotes ("explores many candidate coefficient values by
trial and error...", "more than a handful of divergences") both point back to this same chapter's
own earlier prose, not an outside source. No outside quotation marks used. **CLEAR**.

**Category B (statistics)**: every number in the four new sections (8.32, 9.48, 9.5, 10.10, 0.87,
9.54, 9.55, 9.00, 70.2%, 16.8%, etc.) is a value computed directly by
`chapter-09-bayesian-regression-plots.py`'s new figure functions on synthetic, seeded data, not an
outside claim requiring citation. Verified by rerunning the script and cross-checking every
number quoted in prose against the script's live output before writing it down.

**Category C (named frameworks)**: Student-t robust regression, Metropolis-Hastings, NUTS, and
non-centered/centered parameterization are all named and cited: [@lange1989] for robust-t
modeling, [@metropolis1953] and [@hastings1970] for the Metropolis-Hastings algorithm,
[@hoffmangelman2014] for NUTS, [@neal2003] for the funnel/reparameterization idea, and
[@gelmanhill2007] for hierarchical/multilevel regression generally. All five entries added to
`references.bib` and confirmed against known bibliographic facts (author names, journal, volume,
year, DOI) before citing; none fabricated.

**Category D (distinctive phrases)**: five of the most technical, distinctive 8+ word phrases from
the new sections were run through WebSearch: the NUTS-trajectory description, the HMC-gradient
description, the funnel-narrowing description, the "rejected proposal is not wasted computation"
framing, and the "borrow strength" hierarchical-modeling phrase. No matching wording turned up in
any search; the closest hits were either generic, multiply-attributed statistical terminology (a
phrase like "borrowing strength" appears in Gelman & Hill, in Stan's own documentation, and in
dozens of other sources, attributable to no single origin) or concerned an unrelated domain
(patents on vehicle trajectory planning, for the "curving back on itself" NUTS phrase). A further
targeted search against Osvaldo Martin's own BAP text and against Betancourt-style funnel and
divergence explanations confirmed this chapter's wording, scenario, notation (`log_tau`, `delta`,
`z` rather than BAP's Greek-letter or Stan's `theta`/`phi` convention), and step-size-mismatch
diagnostic (an original construction for this chapter, not found in any source checked) hold up
as this chapter's own. **CLEAR**.

**Figure and code check**: all four new figures were generated by running the updated `.py`
script through the project's own `.venv` (`fig_robust_regression`, `fig_hierarchical_regression`,
`fig_metropolis_walkthrough`, `fig_divergences_funnel`), confirmed rendered, then mirrored into the
paired `.ipynb` with static PNG cell outputs and executed end to end via `jupyter nbconvert
--execute` with zero errors. No figure or number was invented apart from the script.

No critical, warning, or note-level issues found in the new content. Verdict: **PUBLICATION-READY**
(local review only, per the project's standing "do not publish" instruction).
