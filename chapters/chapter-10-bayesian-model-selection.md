# Bayesian Model Selection and Comparison

Chapter 5 closed with a claim it did not prove: that PSIS-LOO and WAIC answer the same
question k-fold cross-validation does, from inside a Bayesian model instead of outside one.
This chapter proves it, with the same latency data Chapter 5 used, fit through PyMC instead of
scikit-learn, and compared through ArviZ instead of a manual cross-validation loop.

Start with the shape of the problem. A Bayesian model does not hand back one fitted line; it
hands back thousands of posterior draws, each one a slightly different plausible line through
the data.

Asking "how well does this model generalize" now has to mean something like "how well
does the *whole cloud* of plausible lines predict a point it has not seen." That answer gets
averaged over every draw in that cloud, and the average has a name: the *log pointwise
predictive density*.

## The log pointwise predictive density

The chart below computes this quantity for every observation in a Bayesian linear regression of
checkout-API latency on payload size and concurrent-request count (the same two predictors
Chapter 4 used), sorted from worst-predicted to best-predicted.

::: {#fig-lpd-per-point}
```{=html}
<iframe src="../_generated/chapter-bayes-model-selection-fig-lpd-per-point.html" width="100%"
        height="520" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Log pointwise predictive density per observation, sorted worst to best. A handful of points sit
well below the rest: these are the requests the posterior struggles to explain.
:::

Picture a thousand students who took the same class but understand it slightly differently,
all answering one tricky question. Some nail it, some miss. Averaging how the whole room did
shows how well the class grasped it. The log pointwise predictive density does the same for one
data point: it averages how well every plausible version of the model predicts it.

For one held-out observation $y_i$, the log pointwise predictive density averages the
likelihood of $y_i$ across every posterior draw $\theta_s$, then takes the log:

$$\text{lpd}_i = \log \frac{1}{S}\sum_{s=1}^{S} p(y_i \mid \theta_s)$$

In other words, a draw where $y_i$ looked plausible contributes a large number to the average,
and a draw where $y_i$ looked implausible contributes almost nothing. The log of that average
stays close to zero when most of the posterior's plausible lines predicted $y_i$ well, and drops
sharply negative when they did not.

@fig-lpd-per-point sorts every observation in that fit this way, from worst explained to best
explained.

Recall from Chapter 1 that a distribution's tail carries information a summary statistic can
hide. The same is true here: a single average lpd across all observations can look acceptable
while a handful of points (the red bars in the figure) are barely explained at all.

In production this is a diagnostic worth acting on directly: those points are candidates for
missing predictors, mismeasured inputs, or a different regime the model has not encountered
enough to learn from.

:::{.callout-tip}
Do not judge a Bayesian model by its average lpd alone. A handful of badly explained points can
hide behind an otherwise acceptable-looking mean; check the per-point lpd, not just the summary
number.
:::

## WAIC: an information criterion built from the posterior

Two students both score 90 on a test. One wobbled wildly across practice tests, 70 one week
and 100 the next. The other stayed near 90 every time. The wobbly student's true skill is less
certain, so that 90 deserves less trust. WAIC works the same way: it discounts a model's fit by
how much its predictions wobble.

*WAIC*, the Watanabe-Akaike information criterion, turns the lpd into a single number that
penalizes complexity the same way AIC does.

Chapter 4 covered AIC's closed-form penalty for classical models; WAIC estimates that same
penalty directly from posterior draws instead of counting parameters. For each observation, the
*effective number of parameters* it contributes is the variance of its log-likelihood across
posterior draws:

$$p_{\text{waic},i} = \text{Var}_s\big[\log p(y_i \mid \theta_s)\big]$$

In other words, an observation whose predicted likelihood swings wildly from one posterior draw
to the next contributes more effective flexibility to the model than one whose predicted
likelihood barely changes across draws.

That matches the intuition behind "effective parameters": the quantity measures how much the
model's fit to that point depends on which particular draw got used, not how many coefficients
get counted directly. Summed and combined with the total lpd,

$$\widehat{\text{elpd}}_{\text{waic}} = \sum_i \text{lpd}_i - \sum_i p_{\text{waic},i}$$

gives an estimate of expected log predictive density on new data: the raw fit, penalized for how
much flexibility the model spent getting there.

## PSIS-LOO: the more trustworthy alternative

The chart below runs a diagnostic this section builds toward on a version of the latency dataset
with two intentionally extreme outliers planted in it.

::: {#fig-khat-diagnostic}
```{=html}
<iframe src="../_generated/chapter-bayes-model-selection-fig-khat-diagnostic.html" width="100%"
        height="520" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Pareto k-hat per observation on a dataset with two planted outliers. Every ordinary point sits
near zero; one outlier sits in the 0.5-0.7 watch range, closer to 0.7 than to 0.5, and the other
clears 0.7 outright.
:::

Leave-one-out testing is like practicing free throws: take one shot out of ten, guess whether
you'd make it based on the other nine, then check. Doing that literally, refitting a whole
model once per data point, is too slow. PSIS-LOO is a shortcut that gets nearly the same answer
without refitting anything.

WAIC works well in most everyday cases but degrades when a model is close to misspecified or
when a handful of observations wield outsized influence on the fit.

*Pareto-smoothed importance sampling leave-one-out cross-validation* (PSIS-LOO) is the
alternative Vehtari, Gelman, and Gabry (2017) built to answer the same question with better
guarantees. It is the default `arviz.loo()` computes [@vehtarigelmangabry2017; @kumar2019arviz].

::: {.callout-important}
Default to PSIS-LOO over WAIC in practice. `az.compare()` uses it as the default measure, and
PSIS-LOO's $\hat{k}$ diagnostic flags the influential-point and near-misspecification failures
that leave WAIC's number looking fine when it should not.
:::

The idea starts from a shortcut. Literally leaving out one observation, refitting, and repeating
$n$ times is precisely as expensive for a Bayesian model as LOOCV was for the linear models in
Chapter 5. PSIS-LOO instead reuses the posterior draws from the single full-data fit,
reweighting them to approximate what a leave-one-out posterior would have looked like.

The importance weight for observation $i$ under draw $s$ is proportional to $1/p(y_i \mid
\theta_s)$: a draw that explained $y_i$ well needs little adjustment to stand in for the
leave-one-out posterior, and a draw that explained $y_i$ poorly needs a much larger adjustment.

Raw importance weights like this have a well-known failure mode: a handful of huge weights can
dominate the average and make the estimate unstable.

PSIS smooths the largest weights by fitting a *generalized Pareto distribution*, built
specifically to model how extreme values in a tail behave, to the largest handful of weights for
each observation. It then replaces each of those raw, noisy weights with the smoothed value the
fitted distribution predicts for a weight of that rank, rather than the observed value itself.

Summing each observation's PSIS-smoothed pointwise density across the whole dataset, the same
way the lpd values summed into $\widehat{\text{elpd}}_{\text{waic}}$ above, produces
$\widehat{\text{elpd}}_{\text{loo}}$: an estimate of expected log predictive density built from
an approximate leave-one-out posterior instead of the full-data posterior WAIC uses.

That smoothing step produces a free diagnostic as a side effect: the shape parameter of the
fitted Pareto distribution, called $\hat{k}$, tells you directly whether the importance-sampling
approximation can be trusted for that observation.

::: {.callout-tip}
Read $\hat{k}$ as a traffic light: below roughly 0.5 the PSIS-LOO estimate is reliable, between
0.5 and 0.7 it is still usable but worth watching, and above 0.7 it should not be trusted.
:::

Above that 0.7 line, the observation is exerting enough influence on the posterior that
PSIS-LOO's estimate for it should not be trusted. The fix is to refit the model with that one
observation excluded outright: the expensive operation PSIS-LOO exists to avoid doing for every
point.

@fig-khat-diagnostic shows this directly: most points sit near zero, one clears the 0.5 watch
line, and one clears 0.7 outright, on a dataset built with two planted outliers to make the
warning visible rather than described in the abstract.

## WAIC and PSIS-LOO against a training-fit baseline

The chart below fits three versions of the latency model and compares training log-likelihood
against elpd_waic and elpd_loo for each.

::: {#fig-waic-ploo-complexity}
```{=html}
<iframe src="../_generated/chapter-bayes-model-selection-fig-waic-ploo-complexity.html"
        width="100%" height="560" style="border:1px solid #ddd; border-radius:6px;"
        loading="lazy"></iframe>
```

Training log-likelihood rewards every added feature, noise included. elpd_waic and elpd_loo
both reward the useful feature and penalize the fifteen noise features once there are enough of
them to visibly overfit.
:::

@fig-waic-ploo-complexity fits three versions of the latency model: one with payload size alone
(underfit, missing a predictor that matters), one with payload size and concurrent-request
count (the model Chapter 4 settled on), and one with fifteen added features that are pure noise.

Fifteen is a deliberately large number of noise features to add. That much noise is what makes
the divergence between "training fit" and "expected fit on new data" visible on a chart, rather
than a difference so small it would need a table of numbers to see at all (the next section
builds that table, for a single noise feature, where the gap stays that small).

Training log-likelihood climbs every time a feature is added, since a model with more
parameters can always fit its own training data at least as well as one with fewer.

Both elpd_waic and elpd_loo tell a sharper story. They jump when concurrent-request count is
added, because that feature carries information about latency that matters, and they fall once
the fifteen noise features are added on top, because those features let the model fit quirks of
this particular training sample that will not repeat in a new one.

That is the entire value of these two quantities in one comparison: they reward a predictor for
containing signal, not merely for existing, and they punish a model for spending its flexibility
on noise.

:::{.callout-tip}
Training log-likelihood always improves when a feature is added, noise included, so it cannot
tell overfitting from a meaningful gain on its own. Compare elpd_waic or elpd_loo instead.
:::

## When WAIC and PSIS-LOO disagree

Every comparison in this chapter so far has WAIC and PSIS-LOO pointing the same way: agreeing
on which model to prefer and differing only in how large the gap is. That agreement holds in
the ordinary case, and it can still break once a single observation turns unusual enough.

::: {#fig-waic-loo-disagreement}
```{=html}
<iframe src="../_generated/chapter-bayes-model-selection-fig-waic-loo-disagreement.html"
        width="100%" height="480" style="border:1px solid #ddd; border-radius:6px;"
        loading="lazy"></iframe>
```

Left: elpd_waic ranks payload_concurrent_noise slightly ahead of payload_concurrent; elpd_loo
ranks them the other way once one observation's importance ratios turn heavy-tailed. Right: the
k-hat diagnostic for that same observation clears 0.7.
:::

@fig-waic-loo-disagreement refits payload_concurrent and payload_concurrent_noise from earlier
in this chapter, this time with one observation given a small, rare chance per posterior draw of
being badly missed by the model, the same mechanism @fig-khat-diagnostic used to produce a high
k-hat.

By elpd_waic, payload_concurrent_noise comes out ahead: -200.6 against payload_concurrent's
-202.4, a gain of 1.8. By elpd_loo, the ranking flips: payload_concurrent_noise falls to -204.3,
a loss of 1.9 against the same baseline.

The reason sits in how each quantity is built. WAIC's penalty for the flagged observation,
$p_{\text{waic}}$, is the sample variance of its log-likelihood across posterior draws, and here
that variance comes out at 0.229, unremarkable next to the rest of the model. PSIS-LOO's
Pareto-tail fit sees something the plain variance misses: the k-hat for that same observation
reaches 0.945, well past the 0.7 line this chapter treats as a warning throughout.

A rare but severe miss barely moves a sample variance computed across thousands of draws, since
one bad observation contributes only a small share to an average over all of them. PSIS-LOO's
tail-shape estimate responds to the same miss directly, because it is built from the largest
weights specifically rather than averaged across every draw. The 0.229-versus-0.945 gap between
the two metrics is that difference, made visible on one observation.

:::{.callout-important}
When WAIC and PSIS-LOO disagree on which model is better, trust PSIS-LOO's ranking and read its
k-hat values before trusting either number. A k-hat past 0.7 on the observation driving the
disagreement means the next step is refitting without that point. The k-hat is the diagnostic
built to catch this kind of disagreement, so let it make the call.
:::

In production terms, a flagged k-hat here is the same signal @fig-khat-diagnostic's outlier
gave: an observation the posterior is not confidently accounting for. WAIC has no equivalent
signal to raise; it reports one penalized score and stops. A team that only checked WAIC on this
dataset would have quietly picked the noise-augmented model to deploy, on a ranking one unstable
observation produced by chance.

## Reading a model comparison

Here is what that comparison looks like the way a PyMC and ArviZ workflow produces it in
practice, rather than as three separate bar charts. Fitting each candidate model and computing
PSIS-LOO looks like this:

```python
import pymc as pm
import arviz as az

def fit_latency_model(payload_kb, concurrent, latency, use_concurrent=True, use_noise=False):
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        beta_payload = pm.Normal("beta_payload", mu=0, sigma=10)
        mu = intercept + beta_payload * payload_kb
        if use_concurrent:
            beta_concurrent = pm.Normal("beta_concurrent", mu=0, sigma=10)
            mu = mu + beta_concurrent * concurrent
        if use_noise:
            beta_noise = pm.Normal("beta_noise", mu=0, sigma=10)
            mu = mu + beta_noise * noise_feature
        sigma = pm.HalfNormal("sigma", sigma=10)
        pm.Normal("latency_obs", mu=mu, sigma=sigma, observed=latency)
        idata = pm.sample(2000, tune=1000, idata_kwargs={"log_likelihood": True})
    return idata

idata_payload_only = fit_latency_model(payload_kb, concurrent, latency, use_concurrent=False)
idata_payload_concurrent = fit_latency_model(payload_kb, concurrent, latency)
idata_payload_concurrent_noise = fit_latency_model(
    payload_kb, concurrent, latency, use_noise=True
)

comparison = az.compare({
    "payload_only": idata_payload_only,
    "payload_concurrent": idata_payload_concurrent,
    "payload_concurrent_noise": idata_payload_concurrent_noise,
})
```

::: {.callout-warning}
`noise_feature` is not a parameter of `fit_latency_model`; it works only because Python resolves
it as a free variable from the enclosing scope when `use_noise=True` runs. Reuse this pattern in
your own code only after passing the array in explicitly, or a `NameError` is one refactor away.
:::

Running the equivalent computation directly from posterior draws (the approach this chapter's
figures use, to keep every number reproducible without an MCMC sampler) on 80 simulated
requests produces the table `az.compare()` would print:

| Model | elpd_loo | SE | d_elpd | dSE |
|---|---:|---:|---:|---:|
| payload_concurrent | -272.6 | 6.0 | 0.0 | 0.0 |
| payload_concurrent_noise | -273.4 | 6.1 | 0.8 | 0.7 |
| payload_only | -345.4 | 12.5 | 72.8 | 10.9 |

::: {#fig-elpd-compare}
```{=html}
<iframe src="../_generated/chapter-bayes-model-selection-fig-elpd-compare.html" width="100%"
        height="520" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The same three models plotted by elpd_loo with a standard-error bar around each point, best
model first. `payload_concurrent` and `payload_concurrent_noise` overlap heavily;
`payload_only` sits far enough left that its error bar never comes close to the other two.
:::

@fig-elpd-compare puts the table's first three columns on an axis: each point is a model's
elpd_loo, and the horizontal bar through it is one standard error in each direction. The overlap
between the top two error bars is the same "well within noise" story the `d_elpd`/`dSE` columns
tell numerically; the gap between those two and `payload_only`'s bar is the same seven-standard-
error separation, visible directly rather than computed from the table.

Read the last two columns first: `d_elpd` is the gap between each model's elpd_loo and the best
model's, and `dSE` is the standard error on that specific gap. That is a different quantity from
the plain `SE` column, which is the standard error of each model's own elpd_loo estimate
considered on its own, with no reference to any other model in the table.

::: {.callout-warning}
Compare `d_elpd` against `dSE`, not against the plain `SE` column. `SE` describes one model's
own elpd_loo estimate in isolation; it says nothing about whether that model's gap from another
model is large or small.
:::

The comparison between `payload_only` and the two better models is decisive: a 72.8-point gap
against a standard error of 10.9 is roughly seven standard errors, not a result that would flip
on a different random sample.

The comparison between `payload_concurrent` and `payload_concurrent_noise` is the opposite
story: a 0.8-point gap against a standard error of 0.7 is well within noise.

PSIS-LOO is not claiming the noise feature hurts the model here. It is correctly refusing to
reward a feature that added no signal, which is the right answer. A table that reported only
elpd_loo without dSE would tempt a reader into treating a coin flip as a finding.

## Posterior predictive checks: a different question

::: {#fig-ppc-check}
```{=html}
<iframe src="../_generated/chapter-bayes-model-selection-fig-ppc-check.html" width="100%"
        height="480" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Left: the observed latency distribution (bold) against 25 datasets simulated from the fitted
model's posterior (thin). Right: the number of simulated requests over 150ms, against the four
observed; almost none of 5,000 simulated datasets produced that many.
:::

Every diagnostic in this chapter so far, lpd, WAIC, PSIS-LOO, has asked one question in
different ways: how well does this model predict data it has not seen, compared with another
model asking the same question. A posterior predictive check asks something else: does this
model's own generated data look anything like the data it was fit to, with no other model in the
room to compare against.

@fig-ppc-check fits the chapter's usual payload-and-concurrency latency model to a dataset where
a small share of requests hit a cold-start path, an unwarmed connection pool, say, adding a large
delay on top of the usual latency. Posterior predictive checking draws parameter values from the
posterior, simulates a full new dataset from each draw, and compares those simulated datasets
against the one on hand [@gelmanmengstern1996]. In PyMC and ArviZ:

```python
with latency_model:
    ppc = pm.sample_posterior_predictive(idata, random_seed=6)

az.plot_ppc(ppc, kind="kde")
```

The left panel of @fig-ppc-check overlays the observed latency density against densities from 25
simulated datasets. The bulk of the distribution matches well; the observed density's right
shoulder sits past where almost every simulated curve ends.

A visual overlay like that raises a suspicion without settling it, so the right panel turns that
suspicion into a number. Pick a test statistic that targets the part of the data the model might
be getting wrong, here the count of requests over 150ms, compute it on the observed data
($T_{\text{obs}} = 4$) and on each of 5,000 simulated datasets, and read off the *Bayesian
p-value*:

$$p_B = P(T_{\text{sim}} \geq T_{\text{obs}} \mid y_{\text{obs}})$$

A model whose posterior predictive distribution reflects the data well should produce this
statistic near the observed value about as often as not, putting $p_B$ somewhere in the middle
of its range. Here it came out at 0 out of 5,000: not one simulated dataset produced four or
more requests over 150ms, against an average of well under one per simulated dataset.

Why this is worth running even after WAIC and PSIS-LOO have picked a winner: those two
quantities only ever compare candidates against each other. If every candidate model shares the
same Gaussian-noise assumption, and none of them expect a cold-start tail, PSIS-LOO will still
confidently rank one of them best, because best-among-flawed-options is the question it is built
to answer. A posterior predictive check asks a different one: is the winner any good on its own
terms.

:::{.callout-tip}
Run a posterior predictive check alongside PSIS-LOO or WAIC, on whichever model they picked.
Model comparison says which candidate predicts better; a posterior predictive check says whether
every candidate is missing the same thing.
:::

For this dataset, the fix looks like Chapter 9's noise-assumption swap: a heavier-tailed or
mixture likelihood built to reproduce occasional large delays, checked against the same
$T_{\text{obs}} = 4$ test statistic until $p_B$ lands somewhere unremarkable instead of at the
edge of its range.

## Bayesian model averaging

::: {#fig-model-averaging}
```{=html}
<iframe src="../_generated/chapter-bayes-model-selection-fig-model-averaging.html" width="100%"
        height="480" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Predicted latency for one new request (low payload, high concurrency) under each of the three
candidate models, plus the weight-blended posterior predictive distribution. payload_only's
prediction sits apart from the other two; its near-zero weight keeps it from moving the blend
despite that gap.
:::

@fig-model-averaging picks up the three models from the "Reading a model comparison" table and
asks a different question than `az.compare()` answers. `az.compare()` ranks models and leaves an
analyst to pick the best row. Model averaging blends every candidate's posterior predictive
distribution into one, weighted by how much predictive support each model earned.

The weights come from elpd_loo directly, turning each model's elpd_loo into a weight and
normalizing so the weights sum to one, the softmax construction pseudo-BMA weighting is built on
[@yaovehtarisimpsongelman2018]:

$$w_k = \frac{\exp(\widehat{\text{elpd}}_{\text{loo},k})}{\sum_j \exp(\widehat{\text{elpd}}_{\text{loo},j})}, \qquad p(y_{\text{new}} \mid \text{data}) = \sum_k w_k\, p(y_{\text{new}} \mid M_k, \text{data})$$

Plugging in the elpd_loo values from earlier gives payload_concurrent a weight of 0.70,
payload_concurrent_noise 0.30, and payload_only a weight indistinguishable from zero. The
seven-standard-error elpd_loo gap between payload_only and the other two translates into an
exponentially small weight: adding a model to the blend does not hand it a share of the vote no
matter how badly it lost.

```python
weights = np.exp(elpd_loo - elpd_loo.max())
weights /= weights.sum()
membership = rng.choice(len(models), size=n_draws, p=weights)
blended_draws = np.concatenate([
    predictive_draws[k][membership == k] for k in range(len(models))
])
```

For a new request with a low payload size and near-peak concurrency, the three models disagree
by a wide margin. payload_only, blind to the concurrency spike, predicts 66.9ms; both
payload_concurrent and payload_concurrent_noise predict about 82.5ms. The weighted blend lands
at 82.5ms, close to the 82.4ms picking payload_concurrent alone from `az.compare()`'s top row
would give.

The near-agreement is informative on its own. Model averaging pulls a prediction away from the
single best model only when the models it is blending disagree with each other
and hold comparable weight. payload_only disagrees plenty here but holds no weight, so it changes
nothing; payload_concurrent_noise holds meaningful weight (30%) but agrees closely with
payload_concurrent on this prediction, since its one added coefficient is a coin flip centered
near zero. Averaging earns its keep specifically when two live candidates make different
predictions, a narrower condition than simply having more than one row in the table.

:::{.callout-note}
Reach for a weighted blend instead of `az.compare()`'s top row when two or more models sit
within a few standard errors of each other and a downstream decision needs the full predictive
uncertainty a single point forecast cannot supply. When one model is standard errors ahead,
picking it outright and blending it end up in the same place.
:::

## Bayes factors, and why this chapter did not lead with them

```python
log_bf10 = (
    log_marginal_likelihood(X_payload_concurrent, latency, sigma, prior_var=100)
    - log_marginal_likelihood(X_payload_only, latency, sigma, prior_var=100)
)
```

```text
log_bf10 = 76.08
```

Every comparison so far in this chapter has asked which model predicts new data with more
accuracy. There is an older question sitting underneath it: given the data collected, which
model was more probable to have produced it, integrating over every value each model's
parameters could plausibly have taken rather than evaluating at the value the data happened to
favor. The *Bayes factor* answers that question directly, and the code above computes it for
the same payload-only-versus-payload-and-concurrency comparison this chapter opened with: a log
Bayes factor of 76.08 decisively favors including concurrent-request count, the same conclusion
the 72.8-point elpd_loo gap from earlier reached from a different direction.

Why reach for this on top of a predictive-accuracy comparison the chapter has made: a Bayes
factor answers a question that does not need a held-out prediction task to define what "better"
means, only which generative story is more probable. That framing suits a strict either/or
engineering decision, does a rollout ship the new pricing formula or keep the old one, where no
obvious "predicts better on new data" framing is available.

How to compute it: this chapter's models are conjugate Gaussian, so the marginal likelihood
integral has a closed form, $y \sim \mathcal{N}(0,\ \sigma_{\text{prior}}^2 XX^\top + \sigma^2
I)$, and the ratio of two such densities is the Bayes factor with no sampler needed. Most
Bayesian workflows do not get this shortcut. One route embeds both candidates inside a single
*encompassing model* with a binary indicator $z$: $z=0$ selects $M_0$'s coefficient structure and
$z=1$ selects $M_1$'s, both feeding one shared likelihood, with a prior $P(z=1)=0.5$. Fitting
that joint model, the posterior odds $P(z=1 \mid y) / P(z=0 \mid y)$ equal the Bayes factor
because the prior odds start at 1:

$$BF_{10} = \frac{p(y \mid M_1)}{p(y \mid M_0)} = \frac{P(z=1 \mid y) / P(z=0 \mid y)}{P(z=1) / P(z=0)}, \qquad p(y \mid M_k) = \int p(y \mid \theta, M_k)\, p(\theta \mid M_k)\, d\theta$$

A second route fits each candidate separately with a sampler built to produce a marginal
likelihood estimate as a byproduct of sampling, since NUTS does not. Both routes exist because
the marginal likelihood integral above rarely has a closed form outside a conjugate setup like
this chapter's.

| prior_var | log BF (noise feature vs. not) | d_elpd_loo (noise feature vs. not) |
|---:|---:|---:|
| 0.1 | -0.05 | -0.12 |
| 1 | -0.39 | -0.97 |
| 10 | -1.26 | -0.95 |
| 100 | -2.38 | -0.84 |
| 1,000 | -3.52 | -0.84 |
| 10,000 | -4.67 | -0.84 |

That table reruns the payload_concurrent-versus-payload_concurrent_noise comparison from
earlier, a feature with no effect on the data-generating process, across a widening prior on the
noise coefficient. As the prior widens, the Bayes factor swings from barely favoring exclusion
(-0.05) to strongly favoring it (-4.67), purely from prior width, with nothing about the data
changing. The elpd_loo gap on the same comparison barely moves once the prior is wide enough to
stop constraining the fit (-0.97 to -0.84).

Sample size tells a matching story. Tripling the dataset, then tripling it again, with the same
underlying effect between payload_only and payload_concurrent, pushes the log Bayes factor from
76.1 to 167.6 to 509.9: a number that keeps growing without a natural stopping point, answering
"is this decisive" only against itself. elpd_loo's gap grows too, but it comes bundled with dSE,
a standard-error count that stays a portable measure of confidence no matter how much data
produced it.

:::{.callout-important}
Bayes factors answer a legitimate question but are sensitive to two things predictive-accuracy
metrics mostly are not: the width of the prior on any coefficient the comparison hinges on, and
the sample size, in a way that keeps growing without a reference scale. Reach for WAIC or
PSIS-LOO by default; reach for a Bayes factor only when the question is which generative story is
more probable, not which model predicts better, and the prior has been chosen and defended as
carefully as the likelihood [@kassraftery1995].
:::

## When this is worth the setup cost

Fitting three Bayesian models with PyMC and comparing them with ArviZ is more machinery than
running `cross_val_score` three times with scikit-learn. For routine model tuning, such as
choosing a regularization strength or comparing two or three feature sets, Chapter 5's k-fold
cross-validation remains the faster, more standardized default.

::: {.callout-note}
Reach for PSIS-LOO when the model has to be Bayesian anyway, when the dataset is too small for
k-fold refitting to stay stable, or when the $\hat{k}$ diagnostic's influential-point warning is
worth having on its own. For routine feature or regularization comparisons, k-fold CV is faster.
:::

PSIS-LOO earns its cost in cases k-fold CV cannot reach cleanly: when the model needs to be
Bayesian for other reasons, such as a hierarchical structure across services or customer
segments, or the Bayesian A/B testing framework Chapter 14 builds in full.

It also earns its cost when the dataset is small enough that refitting $k$ times introduces
meaningful variance in the CV estimate itself, or when the $\hat{k}$ diagnostic's built-in
warning about influential points is worth having for its own sake.

Chapter 5 ended by promising that Part 3 would need PSIS-LOO for this reason. Chapter 14's
"Hierarchical A/B testing" section is where that promise gets collected, using `az.compare()` to
check whether adding a segment-level effect to an experiment's model improves its predictions or
only makes it more elaborate.

## What "closer to the truth" means

This chapter has repeated one claim in different forms: some models sit closer to the process
generating the data than others, and lpd, WAIC, and PSIS-LOO each measure distance from that
target in their own way. That claim has a formal name.

*Kullback-Leibler divergence* measures how much information is lost when one distribution, $q$,
stands in for another, $p$:

$$D_{KL}(p \parallel q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

A model that assigns high probability everywhere the data-generating process does loses little;
a model that assigns low probability somewhere that process visits often loses a great deal, no
matter how well it does everywhere else.

Akaike's original insight behind AIC [@akaike1974], and Watanabe's generalization behind WAIC
[@watanabe2010], is that expected log predictive density is, up to a constant that does not
depend on which model is being scored, the same quantity as negative KL divergence from a fitted
model to the unknown data-generating process. Maximizing elpd and minimizing KL divergence to
that unknown process are the same optimization problem, seen from opposite ends.

@fig-lpd-per-point, @fig-waic-ploo-complexity, and @fig-elpd-compare have been building this
formal version of the claim one figure at a time. elpd_waic and elpd_loo are the closest
computable stand-in this chapter has for how far a model sits from the process that produced the
data, the one quantity that never shows up in a dataset no matter how much of it gets collected.

## References {.unnumbered}

::: {#refs}
:::
