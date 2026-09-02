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

Running the equivalent computation directly from posterior draws (the approach this chapter's
figures use, to keep every number reproducible without an MCMC sampler) on 80 simulated
requests produces the table `az.compare()` would print:

| Model | elpd_loo | SE | d_elpd | dSE |
|---|---:|---:|---:|---:|
| payload_concurrent | -272.6 | 6.0 | 0.0 | 0.0 |
| payload_concurrent_noise | -273.4 | 6.1 | 0.8 | 0.7 |
| payload_only | -345.4 | 12.5 | 72.8 | 10.9 |

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
Bayesian A/B testing framework is where that promise gets collected, using `az.compare()` to
check whether adding a segment-level effect to an experiment's model improves its predictions or
only makes it more elaborate.

## References {.unnumbered}

::: {#refs}
:::
