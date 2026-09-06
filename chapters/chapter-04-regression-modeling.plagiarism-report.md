# Plagiarism & Attribution Audit Report

**Document**: chapter-04-regression-modeling.md
**Audit date**: 2026-08-31
**Audited by**: /plagiarism-check

**Addendum, 2026-09-01 (plain-language layer pass)**: six short intuition paragraphs added
(least-squares line fitting, confidence vs. prediction intervals, logistic curve, adjusted
R-squared/AIC/BIC, regularization, Bayesian prior/posterior), each a generic teaching analogy
(coffee-shop wait times, dimmer switch, science-fair judging, suitcase packing, guessing a
stranger's age) with no direct quotes, statistics, or named frameworks introduced. No web-search
sweep was warranted; none of these fall into Category A-E of the audit protocol. Classified
CLEAR by inspection. Verdict unchanged: PUBLICATION-READY.

---

## Executive summary

This chapter rewrites a raw source built around a generic housing-price example and a
disease-prediction logistic regression example into the checkout-API narrative continued from
Chapters 1 to 3: OLS predicting latency from payload size (a direct callback to Chapter 1's
correlation discussion), logistic regression predicting timeout probability from load, and
regularization applied to a multi-predictor latency model. Three foundational citations were
added for regularization and model selection (Hoerl & Kennard 1970 for ridge regression,
Tibshirani 1996 for the lasso, Akaike 1974 for AIC), each verified via WebSearch against the
published journal record. The raw source's stated formula for the Cp statistic was mathematically
garbled (it defined Cp as "the ratio of log-likelihood to number of parameters," which is not
Mallows' Cp or any standard model-selection statistic); this chapter drops the incorrect formula
rather than propagate it, and covers AIC and BIC instead, which the raw source also introduced
correctly.

Verdict: **PUBLICATION-READY** (local review only, not for public release per the standing
"do not publish" instruction on this project).

---

## Audit statistics

| Metric | Value |
|--------|-------|
| Total word count | ~3,050 |
| Sections analyzed | 8 |
| Existing citations in document | 3 (Hoerl & Kennard 1970, Tibshirani 1996, Akaike 1974) |

---

## Findings

No critical or warning findings. All three citations were checked via WebSearch against
publisher/journal records (Technometrics, JRSS-B, IEEE Transactions on Automatic Control) and
the author names, years, volumes, and page ranges match the published record.

## Verified passages

- All regression formulas (OLS, confidence/prediction interval standard errors, logistic
  function, R-squared, adjusted R-squared, AIC, BIC, Lasso/Ridge/Elastic Net objectives) are
  standard mathematical notation, not attributable to a single source beyond the three cited
  foundational papers.
- The worked coefficient example ("a coefficient of 9 on payload size...") is computed directly
  from the chapter's own simulated dataset (see the matching `-plots.py` file), not a claim
  about a documented study.

## Corrected error from the raw source

- **Cp statistic**: the source docx defined it as "the ratio of the log-likelihood of the model
  to the number of parameters," which does not match Mallows' Cp or any standard definition.
  This chapter omits the incorrect formula and treats AIC and BIC as the model-selection
  criteria, both of which the source also covered and defined correctly.

## Citation health by section

| Section | Citations | Status |
|---------|-----------|--------|
| Measuring fit: R-squared, adjusted R-squared, AIC, and BIC | 1 | OK |
| Regularization: Lasso, Ridge, and Elastic Net | 2 | OK |
| All other sections | 0 | OK, standard formulas and original worked examples |

## Recommended action sequence

None outstanding.

---

## Addendum: "A Bayesian perspective" section (2026-08-31)

A new closing section was added covering Bayesian linear regression, the Ridge-as-Gaussian-prior
and Lasso-as-Laplace-prior equivalence, and Bayesian logistic regression. Audited on its own:

- **Ridge/Lasso-as-posterior-mode claim**: confirmed via WebSearch against multiple independent
  academic sources (a Bayesian-lasso literature review, Penn State STAT 897D course notes) that
  this equivalence traces to Tibshirani's own 1996 lasso paper, cited in this chapter as
  `[@tibshirani1996]`. No new claim introduced beyond what that citation supports.
- **New citation added**: Park, T. and Casella, G. (2008), "The Bayesian Lasso," *Journal of the
  American Statistical Association*, 103(482), 681-686, DOI 10.1198/016214508000000337. Verified
  via WebSearch cross-referencing three independent listings (Google Scholar, Taylor & Francis
  Online, ResearchGate), all matching on volume, issue, page range, and DOI. Added to
  `.refs-chapter04.bib` as `parkcasella2008`.
- **New figure's math**: the posterior-mean and posterior-variance formulas underlying
  `fig_bayesian_ridge` (Gaussian-Gaussian conjugate update) were checked numerically in this
  session: at a near-flat prior (prior variance = 1,000,000), the posterior mean matches the
  chapter's own OLS slope estimate to three decimal places, confirming the flat-prior-recovers-OLS
  claim made in the text rather than asserting it without verification.
- No direct quotes, no new statistics beyond the verified citations above, no fabricated URLs.

Verdict for the addition: **CLEAR**. No critical or warning findings.

---

## Addendum: what/why/how and rigor pass (2026-09-01)

Two targeted edits, both audited fresh:

- **BIC citation fix**: the AIC/BIC paragraph previously cited the AIC/BIC penalty comparison to
  `[@akaike1974]` alone, misattributing a claim about BIC's behavior to Akaike's own AIC paper.
  Verified via WebSearch that BIC originates from a separate paper: Schwarz, G. (1978),
  "Estimating the Dimension of a Model," *The Annals of Statistics*, 6(2), 461-464, DOI
  10.1214/aos/1176344136, confirmed against Project Euclid's own listing. Added as `schwarz1978`
  to the new fragment file `chapters/.refs-chapter04-whywhyhow.bib` (per this pass's instruction
  to use a per-chapter fragment rather than editing `references.bib` directly) and cited inline
  alongside the existing `[@akaike1974]`.
- **Logistic-regression coefficient interpretation**: added two sentences explaining the concrete
  consequence of reading a log-odds coefficient as a direct probability shift. Content is a
  standard property of the logistic curve's shape (flattest near 0 and 1, steepest near its
  midpoint), not attributable to a single source, and introduces no new statistic or named
  framework requiring citation.

No direct quotes, no fabricated URLs, no new statistics beyond the two items above.

Verdict for this addition: **CLEAR**. No critical or warning findings.

---

## Addendum: sentence-level clarity pass (2026-09-01)

A fourth-pass clarity edit reworded roughly 20 sentences throughout the chapter: splitting
overlong sentences (the OLS-assumptions paragraph, the confidence/prediction-interval closing
paragraph, the logistic-regression coefficient warning, the AIC/BIC ranking sentence, the
Ridge/Lasso mechanism paragraph, the Bayesian-Ridge/Lasso-equivalence sentence, the Bayesian
logistic-regression closing paragraph), removing filler openers ("Note that"), and replacing one
ambiguous appositive with a direct restatement. No formula, citation, figure reference, fact, or
number was changed. No new claim, statistic, quote, or named framework was introduced, so no web
search was warranted for this pass; every edit is a syntactic restructuring of content this
report previously verified. Spot-checked the edited passages against the citations verified above
(Akaike 1974, Schwarz 1978, Tibshirani 1996, Hoerl & Kennard 1970, Park & Casella 2008): each
citation's placement and the claim it supports are unchanged.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: figure-first restructure and callout pass (2026-09-01)

Every figure div was moved to appear immediately after its section heading (or after the
minimal formula/definition setup a chart's axis labels require), with all interpretive prose,
including the sentence originally introducing each figure by name, moved to follow the image.
All seven captions were rewritten to state the chart's takeaway rather than only describe its
axes. Roughly a dozen overlong paragraphs were split to stay within 3-4 lines each. Four new
callout boxes (`.callout-tip` on residual-plot diagnostics, `.callout-note` on choosing a
confidence versus prediction interval, `.callout-warning` on misreading a logistic coefficient
as a probability shift, `.callout-note` on Lasso's arbitrary selection within a correlated
group) were added.

Audited against Categories A-E: no direct quotes, no new statistics, no new named framework or
methodology, and no attributed claim lacking a source. Every callout restates a claim this
report verified as CLEAR in an earlier addendum (the CI/PI operational guidance at lines
103-106 of the original draft; the beta-1 misreading consequence from the "what/why/how and
rigor pass" addendum above; the Lasso-versus-Ridge correlated-group behavior from the chapter's
regularization section) in new, shorter phrasing. The residual-plot tip is standard,
textbook-universal diagnostic practice not attributable to a single source. No formula,
citation, fact, or number was changed; the rho-latency example does not appear in this chapter.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: figure-placement reversal, caption, and callout follow-up (2026-09-01)

This pass moved five figure divs (OLS, CI/PI, logistic, R-squared, regularization) from
immediately after their section heading, per the prior figure-first restructure addendum, to
immediately before the paragraph carrying their `@fig-id` reference, per updated placement
guidance requiring a figure to sit after the minimum background needed to read it. `fig-aic-bic`
and `fig-bayesian-ridge` needed no placement change. This is a reordering of prose verified in
prior addenda, not new content, so no fresh citation audit applies to the moved text itself.

New or rewritten passages assessed against Categories A-E:

- **Five rewritten captions** (OLS, CI/PI, logistic, R-squared, regularization): each now names
  the plotted axes and legend colors (for example, "Latency (ms) plotted against payload size
  (KB), with the fitted OLS line in red"). A description of what a chart the chapter's own
  `-plots.py` script generates displays, not a claim requiring a source. **Clear.**
- **Six paragraph splits** (OLS lead-in, confidence/prediction-interval analogy versus
  definitions, the AIC/BIC usage-guidance sentence, the regularization intro, and two sentences
  in the Bayesian-perspective section): each split falls at an existing sentence boundary with no
  wording change beyond the split itself. Not new content. **Clear.**
- **AIC/BIC comparability callout**: states that AIC and BIC scores are only comparable across
  models fit on the same dataset and outcome variable. A standard property that follows directly
  from the AIC/BIC formulas this chapter defines, consistent with Akaike (1974) and Schwarz
  (1978), both cited earlier in this chapter. **Clear.**
- **Statistical-versus-practical-significance callout**: restates, in shorter form, the point the
  surrounding paragraph makes (a large sample can make a small, unimportant coefficient
  statistically significant). Not a new claim. **Clear.**
- **Bayesian prior-choice callout**: states that a prior is a choice that can bias a result if
  picked without justification. A standard caution in Bayesian statistics consistent with this
  section's own flat-prior-versus-informative-prior discussion, no new citation needed. **Clear.**

Targeted web searches on the five new caption sentences and the three new callout sentences
returned no matching source, close or otherwise; the AIC/BIC comparability and prior-choice
statements returned general statistics-education sources restating the same standard facts in
different words. Risk classification: **Clear** for all passages touched in this pass. No
Critical or Warning findings. Verdict unchanged: PUBLICATION-READY (local review only, not for
public release per the standing "do not publish" instruction on this project).

---

## Addendum: teaching-clarity pass (2026-09-01)

Five passages added during a teaching-clarity edit pass, each expanding on a formula or claim the
chapter states elsewhere rather than introducing a new fact:

- The logistic-function "in other words" restatement (behavior of $p(x)$ at the extremes of $x$,
  and the correction for a negative $\beta_1$) is original explanatory prose describing standard
  sigmoid-function behavior, not attributable to a single source.
- The adjusted R-squared mechanism restatement and the AIC/BIC $-2\ln(\hat{L})$ restatement both
  describe standard, textbook-universal properties of these well-known formulas.
- The $\lambda$ regularization-strength definition, including the $\lambda = 0$ edge case, is a
  standard fact about the Lasso and Ridge objective functions covered in the same sources cited
  in this section (Tibshirani 1996; Hoerl & Kennard 1970).
- The posterior-mode-versus-mean clarification in the Bayesian-perspective section (Gaussian
  posteriors are symmetric so mode equals mean; Laplace-prior posteriors are not symmetric in
  general, so only the mode matches the Lasso estimate) states a standard property of these two
  distribution families in original phrasing. No new citation was added; the claim sits inside
  the section supported by Tibshirani (1996) and Park & Casella (2008), verified in the addendum
  above.

Targeted web searches on the most distinctive new phrasing (the logistic-extremes sentence, the
adjusted-R-squared mechanism sentence, and the posterior-mode-versus-mean sentence) returned no
matching source, close or otherwise. Risk classification: **Clear** for all five additions.
Verdict unchanged: previous audits on this chapter remain valid, chapter overall remains
**PUBLICATION-READY** (local review only, not for public release per the standing "do not
publish" instruction on this project).

---

## Addendum: second clarity pass (2026-09-01)

Five small prose edits following the second clarity pass (see the humanize log's matching
addendum for the full list). Audited each against Categories A-E: no direct quotes, no new
statistics, no new named framework, and no distinctive phrase requiring a web-search sweep.
Two edits were pure terminology consistency (changing "noise feature" to "noise predictor" in
the regularization section to match the term used everywhere else in this chapter and echoed in
Chapter 5's opening paragraph); the other three were pronoun/parsing fixes with no change to
claims, citations, or numbers. Spot-checked all five edited passages against the citations
verified in prior addenda (Akaike 1974, Schwarz 1978, Tibshirani 1996, Hoerl & Kennard 1970,
Park & Casella 2008): unaffected.

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY**.

---

## Addendum: multiple regression, categorical encoding, interactions, VIF, and classification
metrics gap-fill pass (2026-09-05)

This pass closes the chapter's largest documented gap (per the gedeck-comparison audit): the
prose never formally introduced multiple regression, categorical/dummy encoding, or interaction
terms even though the paired `-plots.py` file fit models with 3-5 predictors from the start. Six
new sections were added: multiple regression (an omitted-variable-bias worked example),
categorical predictors and dummy encoding, binning a high-cardinality categorical predictor by
residual (an original example using API endpoint paths, a different domain from gedeck's
zip-code treatment), interaction terms, multicollinearity and the variance inflation factor, and
a classification-metrics groundwork section (accuracy, precision, recall) tied to the existing
logistic regression example. Six new figures were added to the paired
`-plots.py`/`-plots.ipynb` files, one per section, and one new citation.

**New citation**: Marquardt, D.W. (1970), "Generalized Inverses, Ridge Regression, Biased Linear
Estimation, and Nonlinear Estimation," *Technometrics*, 12(3), 591-612, DOI
10.1080/00401706.1970.10488699, cited for the variance inflation factor. Verified via the
CrossRef API (`api.crossref.org`), an independent bibliographic metadata registry rather than a
publisher's own paywalled page (the Taylor & Francis abstract page returned an HTTP 403 to a
live WebFetch attempt): title, author, journal, volume, issue, page range, and year all match
what is cited in the chapter and recorded in `references.bib`.

**Category A (direct quotes)**: none found in the new sections.

**Category B (statistics/quantitative claims)**: every number in the six new sections (payload
coefficients of 9.95/9.26/9.20 ms/KB across the three nested models; the 0.78 payload-load
correlation; the 16.3 ms raw versus 8.3 ms adjusted us-west gap; the 29,039-versus-47,537 SSE
contrast in the interaction figure; the VIF values from 1.12 to above 1,300; the 0.662-to-0.833
R-squared jump in the endpoint-binning example; the precision/recall pairs at each of the five
confusion-matrix thresholds) is computed directly by the paired `-plots.py` script from this
chapter's own simulated data, verified numerically in this session by executing the module's
functions independently and cross-checking the printed values against every number quoted in the
prose (see the Verified passages section below). None of these are claims about an external,
documented study, so none require a citation, the same treatment this report has given every
prior worked-example number in this chapter.

**Category C (named frameworks/methodologies)**: "dummy variable trap," "confusion matrix," and
"variance inflation factor" all appear without an inline citation beyond the one added above.
WebSearch confirmed all three are generic, multiply-attested statistics vocabulary (Wikipedia,
Statology, LearnDataSci, and an MPRA working paper for the dummy variable trap; Penn State's
STAT 462, Statistics How To, and ScienceDirect Topics for the variance inflation factor and its
conventional 5/10 thresholds), the same "textbook-universal, no single owner" category this
report applied earlier to this chapter's residual-plot and AIC/BIC-comparability callouts.
**Clear.**

**Category D (distinctive phrases)**: targeted WebSearch queries on several of the new section's
more specific sentences (the omitted-variable-bias explanation, the ruler analogy opening the VIF
section, the leaner-fleet interaction-term framing) returned no matching source, close or
otherwise. A separate search on gedeck's own repository and its zip-code/high-cardinality
treatment confirmed no wording overlap with this chapter's endpoint-path binning section, which
uses an unrelated domain (API routes, not zip codes) and original prose throughout. **Clear.**

**Category E (attributed claims without a source)**: none found; the new sections attribute no
claim to an outside organization or researcher beyond the Marquardt citation above.

## Verified passages (this pass)

- All six new figures' underlying numbers were reproduced independently in this session by
  importing `chapter-04-regression-modeling-plots.py` and calling its functions directly (not by
  trusting the prose), confirming: the 9.95/9.26/9.20 ms/KB coefficient sequence and the 0.78
  payload-load correlation (`multi_predictor_latency_sample`); the 16.3 ms raw versus 8.3 ms
  adjusted us-west gap and the underlying per-region payload/load averages (11.8 KB/55.4% for
  us-west versus 11.0 KB/53.4% for us-east); the 29,039/29,038 SSE pair at the smallest
  interaction multiplier and the 47,537/29,038 pair at the largest; the VIF sequence
  1.12/1.02/1.12 through 1,397/1.02/1,397 across the five correlation-strength steps; the
  0.662-to-0.833 R-squared improvement and the 8.65/23.82/31.57 ms bin-dummy coefficients in the
  endpoint-binning example; and the five threshold rows of true/false positive and negative
  counts, accuracy, precision, and recall in the confusion-matrix figure.
- The VIF formula ($\text{VIF}_j = 1/(1-R_j^2)$) and the interaction-term formula
  ($Y = \beta_0 + \beta_1 X_1 + \gamma D + \delta(X_1 \times D) + \varepsilon$) are standard
  regression notation, not attributable to a single source beyond Marquardt (1970) for VIF.
- The endpoint-path domain (`/api/v1/checkout`, `/api/v1/fraud/score`, and 26 similar routes) and
  every latency number attached to them are original to this chapter's simulation, not drawn from
  any published API catalog or dataset.

## Citation health by section (this pass)

| Section | Citations | Status |
|---------|-----------|--------|
| Multiple regression: more than one predictor at a time | 0 | OK, standard OLS extension and original worked example |
| Categorical predictors and dummy encoding | 0 | OK, standard notation and original worked example |
| Binning high-cardinality categorical predictors by residual | 0 | OK, original worked example, different domain from the source material's zip-code treatment |
| Interaction terms | 0 | OK, standard notation and original worked example |
| Multicollinearity and the variance inflation factor | 1 (Marquardt 1970) | OK |
| Classification metrics: accuracy, precision, and recall | 0 | OK, standard, multiply-attested definitions and original worked example |

Verdict for this pass: **CLEAR**. No critical or warning findings. Verdict for the chapter
overall remains: **PUBLICATION-READY** (local review only, not for public release per the
standing "do not publish" instruction on this project).
