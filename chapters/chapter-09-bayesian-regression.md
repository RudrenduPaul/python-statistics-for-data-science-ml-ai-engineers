# Bayesian Linear Regression and Regularization

Chapter 4 fit the checkout-API latency data with a single number for the payload-size slope:
9.55 milliseconds per kilobyte, computed by OLS from 40 requests.

Its Bayesian perspective section closed with a promise: a full posterior distribution over that
slope, not just its most likely value, and a claim that a credible interval means something a
confidence interval does not. This chapter cashes that promise in, with the numbers behind it.

## The posterior over a single coefficient

Before seeing any data, a doctor guessing a patient's height has some idea it is probably
between four and seven feet, not twenty. That starting guess, before evidence updates it, is
what a *prior* captures in a Bayesian model.

Bayesian linear regression starts from the same model OLS does,
$Y = \beta_0 + \beta_1 X + \varepsilon$, with $\varepsilon$ normally distributed with variance
$\sigma^2$.

The difference is what happens to $\beta_1$ before any data arrives: OLS treats it
as an unknown constant to be estimated; Bayesian regression treats it as a random quantity with
a prior distribution, $\beta_1 \sim \text{Normal}(\mu_0, \tau_0^2)$. That distribution
describes which values are plausible before the checkout API has served a single request.

:::{.callout-tip}
A quick sanity check for whether a prior mattered at all: compare the posterior mean to the
OLS estimate. If the two are close, the data dominated; if they differ, the prior is still
shaping the answer.
:::

When the prior and the likelihood are both Gaussian and $\sigma^2$ is known, the posterior is
also Gaussian, $\beta_1 \mid \text{data} \sim \text{Normal}(\mu_n, \tau_n^2)$, and its mean and
variance have a closed form that needs no sampling:

$$
\tau_n^2 = \left(\frac{1}{\tau_0^2} + \frac{S_{xx}}{\sigma^2}\right)^{-1}, \qquad
\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{S_{xx}\hat\beta_1}{\sigma^2}\right)
$$

where $S_{xx} = \sum_i (x_i - \bar x)^2$ and $\hat\beta_1$ is the OLS estimate. In other words,
the posterior mean is a weighted average of the prior mean and the OLS estimate, with weights
set by how much each one is trusted: a tight prior ($\tau_0^2$ small) pulls the posterior toward $\mu_0$; a
lot of data ($S_{xx}$ large relative to $\sigma^2$) pulls it toward $\hat\beta_1$ instead.

::: {#fig-posterior-narrowing}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-posterior-narrowing.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The posterior over the payload-size coefficient at n = 10, 30, 100, 300, and 1000 requests.
More data narrows the posterior and pulls it away from the prior toward the OLS estimate.
:::

@fig-posterior-narrowing traces the posterior starting from a weakly informative prior,
$\beta_1 \sim \text{Normal}(0, 3^2)$, centered on zero because the direction of the
payload-latency relationship is left open before the data speaks, paired with a cautious
pre-data assumption of 20 milliseconds for the residual noise's standard deviation, across
five sample sizes drawn from the same simulated request stream Chapter 4 used.

At $n=10$, the posterior mean sits at 8.52 milliseconds per kilobyte with a posterior standard
deviation of 0.99: still wide, and noticeably left of the true slope of 9.5, since the prior
still has meaningful influence at this sample size. By $n=1000$, the posterior mean lands at
9.48 milliseconds per kilobyte with a posterior standard deviation of 0.12, closely matching
the true slope this simulation was built from.

This is because as $S_{xx}$ grows, the $\tau_n^2$ formula above gives the prior's precision,
$1/\tau_0^2$, less and less weight relative to the data's. Eventually the posterior mean
converges on the OLS estimate regardless of where the prior started.

Recall from Chapter 4 that a flat prior recovers the OLS estimate outright; this figure shows
the same convergence happening gradually as data accumulates instead.

## What a credible interval claims

At $n=40$, the same sample size Chapter 4's OLS fit used, the posterior mean for the
payload-size coefficient is 9.55 milliseconds per kilobyte with a posterior standard deviation
of 0.06, giving a 95% credible interval of $[9.44, 9.66]$. Chapter 4's OLS fit on the same data
gave a 95% confidence interval of $[9.47, 9.63]$.

The two intervals nearly coincide numerically. What they claim is not the same thing.

Think of a net-making factory where 95% of nets come out wide enough to catch a fish of the
true size, though there is no way to tell whether any single net pulled off the line is one of
the good ones or one of the 5% that is too small. A 95% confidence interval is a statement
about a procedure, not about this particular interval.

::: {#fig-ci-repeated-experiments}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-ci-repeated-experiments.html" width="100%" height="620"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

100 independent 95% confidence intervals from repeated 40-request experiments. Green intervals
contain the true slope; red ones miss it, close to the 95% the procedure promises.
:::

@fig-ci-repeated-experiments runs the same 40-request experiment 100 times, computes a fresh
95% confidence interval each time, and marks which ones contain the true slope of 9.5.

96 of the 100 intervals contain the true value, close to the nominal 95%. That is what "95%
confidence" means: run the experiment many times, and about 95% of the intervals it produces
will bracket the truth.

It says nothing about whether any single interval, including the one Chapter 4 computed, is one
of the 95% that succeeded or one of the roughly 5% that missed. There is no way to tell from the
interval alone.

A 95% credible interval claims something different and, for a single dataset, more direct.

::: {#fig-credible-interval-single}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-credible-interval-single.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

One posterior, one 95% credible interval: a direct probability statement about the
coefficient, conditional on the one dataset collected.
:::

@fig-credible-interval-single shows the one posterior this section has been building, with the
95% credible region shaded.

Given the observed data and the stated prior, there is a 95% posterior probability that the true
coefficient lies in $[9.44, 9.66]$. That is a claim about the parameter, conditional on the one
dataset collected, not a claim about how a repeated procedure would behave across hypothetical
datasets that were never observed.

::: {.callout-important}
A confidence interval is a statement about a procedure repeated many times; a credible interval
is a direct probability statement about the parameter, given this one dataset. Conflating the
two is one of the most common misreadings in applied statistics.
:::

Recall the ASA's 2016 statement on p-values from Chapter 2: a large share of the confusion
around frequentist inference comes from reading a frequentist quantity as if it made a direct
probability statement about the parameter. A frequentist interval does not make that statement;
a credible interval does.

## Ridge, Lasso, and the horseshoe as priors

Chapter 4 showed that Ridge regression is the posterior mode under an independent Gaussian
prior on each coefficient, and the Lasso is the posterior mode under an independent Laplace
prior, citing Park and Casella's treatment of the Bayesian Lasso [@parkcasella2008].

::: {#fig-prior-shapes}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-prior-shapes.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Ridge (Gaussian), Lasso (Laplace), and horseshoe prior densities at increasing scale. The
horseshoe's spike at zero is taller and its tails heavier than either alternative.
:::

@fig-prior-shapes puts the two priors side by side with a third, newer alternative: the
*horseshoe prior* [@carvalhopolsonscott2010].

Think of packing a suitcase: Ridge squeezes every item a little smaller. The Lasso throws out
anything under a certain size entirely. The horseshoe does both at once, crushing clutter down
small while leaving the items that matter close to full size.

The horseshoe prior has a spike at zero, taller than either the Gaussian's or the Laplace's,
and tails that stay heavier than both further out.

It achieves this shape by giving each coefficient its own local shrinkage weight, drawn from a
*half-Cauchy distribution* (a distribution over positive numbers that piles up most of its mass
near zero but tapers slowly enough that an occasional large value stays plausible), multiplied
by a shared global shrinkage parameter. Most coefficients get a tiny local weight and are
crushed toward zero, while a few get a large local weight and pass through with almost no
shrinkage at all.

That same spike-and-tail shape makes the horseshoe's posterior harder to sample than either
alternative: the local and global shrinkage weights interact to form a funnel-shaped geometry
that a naive Markov chain Monte Carlo sampler can get stuck exploring.

::: {.callout-note}
PyMC and similar tools fit horseshoe models with a *non-centered parameterization*: the
sampler draws a standardized version of each shrinkage weight and rescales it afterward,
instead of sampling the weight directly at its natural scale. That keeps the funnel from
tightening around the sampler as it explores.
:::

::: {#fig-shrinkage-profile}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-shrinkage-profile.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Shrinkage applied by Ridge, Lasso, and the horseshoe prior, as a function of the raw
coefficient. The horseshoe is the only one of the three that treats small and large
coefficients differently.
:::

@fig-shrinkage-profile makes the practical difference concrete: it plots the shrunk coefficient
each prior produces against the raw, unregularized value that would come out of an ordinary
regression with no penalty at all.

Ridge shrinks every coefficient by the same proportion, large and small alike. The Lasso applies
a soft threshold, subtracting a fixed amount from every coefficient and zeroing out anything
smaller than that amount. The horseshoe does neither: it shrinks small coefficients almost
completely to zero while leaving large coefficients nearly untouched.

For a checkout API with dozens of candidate predictors, most of them irrelevant (a request ID
hash, a client library version string encoded as a number, a rarely populated optional field),
the horseshoe's behavior is often closer to what an analyst wants: aggressive shrinkage on the
noise predictors, none on the ones that matter.

Which of the three to reach for depends on what is known about the predictors before fitting
anything. Ridge is the right default when most predictors are expected to matter at least a
little and none should be forced all the way to zero, for instance a set of correlated latency
sub-measurements (network time, queue time, handler time) that all plausibly contribute.

The Lasso is the right default when sparsity itself is the goal and a small number of
predictors are expected to matter, since it can zero coefficients out entirely. That is useful
for turning a wide feature set into a short list an engineer can act on.

The horseshoe is worth the added computation over the Lasso specifically when the analyst also
wants the surviving coefficients to stay close to their unshrunk values instead of the Lasso's
uniform soft-thresholding, which shrinks a predictor that matters a great deal by the same
fixed amount as one that barely matters at all.

## Bayesian logistic regression, and reaching for PyMC

Only Ridge has a closed-form Bayesian posterior: a Gaussian prior combined with a Gaussian
likelihood is *conjugate*, so the two combine algebraically into another Gaussian, the same
derivation the opening section of this chapter worked through directly.

The Lasso's Laplace prior is not conjugate to a Gaussian likelihood, so its full posterior has
no closed form. Only its mode does, and that mode is where the Lasso point estimate comes from.

Recovering the full Bayesian Lasso posterior is what Park and Casella's *Gibbs sampler* was
built to do: an MCMC method, covered in more depth below, that updates one coefficient at a time
by drawing from its distribution conditional on the current values of all the others.

That update is only possible in closed form here because the Laplace prior can be rewritten as a
mixture of Gaussians with a random scale. This *scale-mixture representation* gives each
coefficient's conditional distribution a known, recognizable form to sample from, rather than
one with no closed form at all [@parkcasella2008].

Logistic regression adds a further complication on top of that: its likelihood is not Gaussian,
so a Gaussian prior on its coefficients does not produce a Gaussian posterior, or any other named
distribution with a known formula.

No choice of prior fixes this, since conjugacy is a property of a specific prior paired with a
specific likelihood, and no standard prior is conjugate to the logistic likelihood.

Without a clean formula available, fitting a Bayesian logistic regression means approximating
the posterior instead of deriving it, through Markov chain Monte Carlo (MCMC) sampling run by a
probabilistic programming library rather than by hand [@salvatier2016pymc3].

MCMC explores many candidate coefficient values by trial and error, lingering near ones that
fit the data well and drifting away from ones that do not, until the pattern of visits traces
out the posterior.

Recall the timeout-probability logistic model from Chapter 4, predicting whether a checkout
request times out from its payload size. In PyMC, the Bayesian version of that model looks like
this:

```python
import pymc as pm
import arviz as az

with pm.Model() as timeout_model:
    beta0 = pm.Normal("beta0", mu=0, sigma=10)
    beta1 = pm.Normal("beta1", mu=0, sigma=3)
    logit_p = beta0 + beta1 * payload_kb_centered
    pm.Bernoulli("timeout", logit_p=logit_p, observed=timeout_observed)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=11)

az.summary(trace, var_names=["beta0", "beta1"])
```

`pm.sample` runs the No-U-Turn Sampler, PyMC's default MCMC algorithm, across four independent
chains and returns a set of posterior draws rather than a single number. `az.summary` then
reports the posterior mean, standard deviation, and 94% highest-density interval for each
parameter, computed directly from those draws.

That 94% is ArviZ's own default width, not the 95% used earlier in this chapter; the two are
different conventions, not different results for the same quantity.

It is also a different kind of interval: a *highest-density interval* (HDI) is the narrowest
range that contains the stated probability, rather than the equal-tailed interval computed by
hand for the conjugate Gaussian posterior earlier in this chapter.

For a symmetric posterior like that one, the two nearly coincide. For a skewed posterior they can
differ noticeably: an equal-tailed interval always cuts the same probability from each tail
regardless of the posterior's shape, while the HDI does not.

::: {.callout-tip}
For a symmetric, unimodal posterior it rarely matters which interval type gets reported. For a
skewed or multimodal one, prefer the HDI: an equal-tailed interval can exclude the single most
probable value if enough posterior mass sits in one tail.
:::

::: {#fig-bayesian-logistic}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-bayesian-logistic.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Posterior for the timeout-probability logistic coefficient, at three prior widths. All three
land on nearly the same posterior mode once 300 requests have been observed.
:::

@fig-bayesian-logistic shows what that posterior looks like for the payload-size coefficient,
computed here by evaluating the posterior on a fine grid instead of sampling it, since the
model has only one predictor and a grid gives a precise posterior where MCMC gives an
approximation.

The posterior mode lands at 0.307 regardless of whether the prior's standard deviation is 10, 3,
or 1. With 300 observed requests, the likelihood dominates enough that a moderately informative
prior barely moves the answer.

This is a useful thing to know before reaching for a Bayesian model at all: the extra machinery
buys the most when data is scarce, a prior carries meaningful information, or the answer needs a
full distribution rather than a point estimate for a downstream decision. It does not pay off
automatically in every case, regardless of how much data is on hand.

### Did the sampler converge?

Unlike the conjugate Gaussian case earlier in this chapter, `pm.sample` gives no guarantee its
draws represent the posterior correctly. MCMC can fail quietly: a chain can get stuck in one
region of the parameter space, or several chains can wander off and never agree with each other.
Either way, the output still looks like a normal set of posterior draws unless it is checked.

Two numbers from `az.summary` catch most of these failures before they reach a report. Picture
four friends separately guessing the outdoor temperature, then comparing notes: similar guesses
mean the average is trustworthy, but one friend far off from the rest is a warning sign.

$\hat{R}$ works on that same logic: it compares the variance within each of the four chains to
the variance across all four, and a value above 1.01 means the chains have not mixed and have not
converged on the same distribution.

::: {#fig-rhat-diagnostic}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-rhat-diagnostic.html" width="100%"
        height="540" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Four simulated MCMC chains for one parameter. Well-mixed, all four land on the same
distribution and $\hat{R}$ stays near 1.00; poorly-mixed, one chain gets stuck near a
different value for the whole run and $\hat{R}$ climbs well past the 1.01 warning line.
:::

@fig-rhat-diagnostic makes the "four friends guessing a temperature" analogy concrete: the
well-mixed view shows four chains started from different values converging onto the same band
and staying there, the pattern behind an $\hat{R}$ near 1.00. The poorly-mixed view keeps that
same setup except one chain never leaves the region it started in, and $\hat{R}$ rises well
above the 1.01 line as a direct consequence, since the variance across chains dwarfs the
variance within any single one of them.

`ess_bulk`, the effective sample size, estimates how many independent draws the correlated MCMC
samples are worth; an effective sample size under a few hundred, even with thousands of raw
draws, means the posterior summary is noisier than it looks.

::: {.callout-warning}
A good $\hat{R}$ does not guarantee a good `ess_bulk`: chains can agree with each other on
average while each one is still highly autocorrelated internally. Check both diagnostics
before trusting a posterior summary, not just the one that happens to look reassuring.
:::

::: {.callout-warning}
PyMC also reports divergent transitions during sampling, a diagnostic flagging regions of the
posterior its step size could not navigate. More than a handful of divergences means the model
or its priors need reworking, not a number to note and move past.
:::

## Prior predictive checks

::: {#fig-prior-predictive-check}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-prior-predictive-check.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Forty draws from the timeout model's priors alone, before any request data, at two prior widths
for the payload-size coefficient. The narrower prior (standard deviation 0.5) keeps most curves
gentle; the wider one (standard deviation 10) produces curves that swing from near 0% to near
100% timeout probability within a couple of kilobytes.
:::

@fig-prior-predictive-check draws forty pairs of $(\beta_0, \beta_1)$ from the timeout model's
priors, before the model has looked at a single request, and plots the timeout-probability curve
each pair implies against payload size.

A prior predictive check asks a question the posterior predictive check later in this chapter
cannot: do the model's stated assumptions describe plausible outcomes before any data gets
involved? A posterior predictive check compares simulated data to observed data after fitting; a
prior predictive check compares simulated data to what an analyst would expect to see, using
nothing but the priors, a routine step in the modern Bayesian workflow literature
[@gabrysimpsonvehtaribetancourtgelman2019].

The distinction matters because a prior that looks harmless written down as $\text{Normal}(0, 0.5^2)$
can imply something an analyst would reject on sight once it is drawn out as a curve. At standard
deviation 0.5, 38 of the forty curves above stay in a believable range: a gradual rise in timeout
probability spread over several kilobytes of payload size. At standard deviation 10, 37 of the
forty curves jump from near-certain success to near-certain timeout across a two- or
three-kilobyte window, a claim no engineer running this checkout API would sign off on before
seeing a single request.

The earlier posterior-narrowing figure in this chapter showed that a prior still shapes the
posterior meaningfully at n = 10. A prior that implies impossible curves does the same kind of
shaping on a small dataset, and the model gives no warning that it happened. A posterior
predictive check would eventually catch a resulting bad fit, but only after data collection and
model fitting are both done. A prior predictive check catches it before either starts.

Computing one takes the same PyMC model defined earlier for the timeout classifier, minus the
`observed` argument:

```python
with timeout_model:
    prior_checks = pm.sample_prior_predictive(draws=500, random_seed=11)

az.plot_ppc(prior_checks, group="prior")
```

`pm.sample_prior_predictive` draws parameter values straight from the priors, the same way the
figure above was built, and simulates timeout outcomes from those draws without touching
`timeout_observed` at all. Nothing here depends on the request data existing yet; the check can
run as soon as the priors are written down.

Formally, a prior predictive check draws $\theta^{(s)} \sim p(\theta)$ for $s = 1, \dots, S$ sets
of parameters directly from the prior, then draws simulated data $\tilde y^{(s)} \sim
p(y \mid \theta^{(s)})$ from the model's likelihood conditioned on each draw. Checking the
resulting $\tilde y^{(s)}$ against domain knowledge is what makes this a check of the prior. A
check of the fit runs the same comparison against the data observed once fitting is done.

::: {.callout-tip}
Run a prior predictive check before fitting anything, and a posterior predictive check after.
One asks whether the assumptions are reasonable before data arrives; the other asks whether the
fitted model reproduces the data that did.
:::

## Posterior predictive checks

A weather model can be confident in its own numbers and still be wrong if it never checks
itself against what happened outside. A posterior predictive check is that reality check for a
statistical model.

A posterior tells an analyst what the model believes about its parameters. It says nothing
about whether the model itself is a reasonable description of the data.

A *posterior predictive check* closes that gap: draw parameter values from the posterior,
simulate new data from the model using those draws, and compare the simulated data to what was
observed [@gelmanmengstern1996]. In PyMC and ArviZ, that workflow is two calls:

```python
with timeout_model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=11)

az.plot_ppc(ppc, kind="cumulative")
```

::: {#fig-posterior-predictive-check}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-posterior-predictive-check.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Residuals (observed latency minus the fitted line) against data simulated from the fitted
model's posterior, averaged over 200 replications under two different noise assumptions. Only
the log-normal assumption reproduces the observed right tail.
:::

@fig-posterior-predictive-check runs this check on a regression model for latency itself
rather than the timeout classifier, comparing two different assumptions about the noise term.

The first assumes $\varepsilon$ is normally distributed, the default behind both OLS and the
conjugate Gaussian posterior used earlier in this chapter. The second draws $\varepsilon$ from
a distribution matching the log-normal noise Chapter 1 established for this data.

Plotting raw latency would bury the noise term's shape under the much larger spread the
payload-size predictor contributes on its own, so @fig-posterior-predictive-check subtracts the
fitted line first and compares residuals, averaged over 200 simulated replications to smooth
out the noise a single draw would carry. Above 3 milliseconds of residual, the observed data
has 18 requests.
The Gaussian-noise model's average is 10, undershooting the tail the observed data has; the
log-normal-noise model's average is 18.9, matching it closely. Recall from Chapter 1 that
checkout-API latency is right-skewed by construction, a hard floor near zero and an unbounded
upside; a symmetric noise model cannot reproduce that shape no matter how its variance is tuned.

Switching the noise assumption to match the log-normal shape closes most of the gap in the tail.
That is the kind of mismatch a posterior predictive check exists to catch before a model with a
wrong noise assumption gets used to set an alerting threshold or a service-level objective.

A model can have a perfectly reasonable-looking posterior over its coefficients and still make
this mistake, because the coefficients and the noise assumption are separate modeling choices.
Checking the posterior alone would have missed it; checking what the model predicts against what
happened did not.

:::{.callout-tip}
A well-behaved posterior over a model's parameters is not proof the model fits the data. Run a
posterior predictive check before trusting any downstream decision built on that fit.
:::

## Robust regression: when a handful of points sway the whole fit {#sec-robust-regression}

Every regression in this chapter so far has assumed $\varepsilon$ is well behaved: log-normal
after Chapter 1's correction, but never wild. A retry storm breaks that assumption on purpose.

::: {#fig-robust-regression}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-robust-regression.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Same 60 requests, two likelihoods: the Normal-likelihood fit tracks the six retry-storm requests
it should ignore; the Student-t-likelihood fit does not.
:::

@fig-robust-regression fits the checkout API's payload-latency relationship on 60 requests, six of
which each picked up an extra 150 to 300 milliseconds from an upstream service timing out and
retrying mid-request, unrelated to how large their own payload happened to be. Everything else
about the data matches the log-normal noise this book has used since Chapter 1.

**What a Student-t likelihood is.** Swapping the Normal likelihood for a Student-t likelihood
keeps the same linear model, $Y = \beta_0 + \beta_1 X + \varepsilon$, and changes only the assumed
shape of $\varepsilon$. A Student-t distribution with a low degrees-of-freedom parameter $\nu$ has
heavier tails than a Normal: values several standard deviations from the center are unlikely
under either distribution, but far less unlikely under the Student-t. That one difference is what
makes the fit behave differently around outliers.

**Why it matters.** The Normal-likelihood fit in @fig-robust-regression comes out at 8.32
milliseconds per kilobyte, more than a full millisecond below the checkout API's true
per-kilobyte cost of 9.5, dragged there by six contaminated points out of sixty. A
capacity-planning number built from that slope understates how much a payload increase will cost,
and nothing about the fit's diagnostics flags the problem on its own: six points out of sixty is
rarely enough to fail a quick residual check. The Student-t fit lands at 9.48, within two
hundredths of a millisecond of the true value, without a single request removed from the dataset
by hand.

**How to compute it.** Fitting a Student-t likelihood by maximum a posteriori estimation needs an
optimizer rather than the closed-form update from the top of this chapter, since the Student-t is
not conjugate to a Normal likelihood the way the Gaussian prior on $\beta_1$ was. In PyMC:

```python
import pymc as pm

with pm.Model() as robust_model:
    beta0 = pm.Normal("beta0", mu=0, sigma=20)
    beta1 = pm.Normal("beta1", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=10)
    nu = pm.Exponential("nu", 1 / 10) + 1
    mu = beta0 + beta1 * payload_kb
    pm.StudentT("latency", nu=nu, mu=mu, sigma=sigma, observed=latency_ms)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=11)
```

Letting `nu` carry its own prior, rather than fixing it, lets the data decide how heavy the tails
need to be: a dataset with no outliers at all pushes `nu`'s posterior toward large values, where
the Student-t and the Normal are nearly indistinguishable, and PyMC ends up fitting something
close to the plain Normal-likelihood model without anyone choosing between the two ahead of time
[@lange1989]. The figure above fixes $\nu = 4$ and solves for the fit directly by evaluating the
Student-t log-density on a grid of candidate coefficients and taking the maximum, the same
closed-form-versus-optimization distinction this chapter has drawn since it first reached for
PyMC: a model with a known likelihood shape can be solved directly, and letting an extra parameter
float usually cannot.

**Formal notation.** With a Student-t likelihood, the model changes only its second line:

$$
Y_i \mid \beta_0, \beta_1, \sigma, \nu \sim \text{StudentT}\left(\nu,\ \beta_0 + \beta_1 X_i,\ \sigma\right)
$$

$\nu$ controls tail weight: as $\nu \to \infty$, the Student-t converges to a Normal with the same
location and scale, and low values of $\nu$, in the range of 3 to 7, produce visibly heavier
tails without the distribution losing a finite mean. That convergence is why a Normal-likelihood
model is a special case of a Student-t model rather than a separate one; letting $\nu$ float is a
strictly more flexible choice that costs one extra parameter to estimate.

::: {.callout-tip}
A quick way to decide whether robust regression is worth the extra parameter: refit with a
Student-t likelihood and check whether the coefficient estimates move by more than a rounding
error. If they barely move, the data has no meaningful outliers and the Normal likelihood was
fine all along.
:::

## Hierarchical regression: letting every service borrow strength from the rest {#sec-hierarchical-regression}

The horseshoe prior earlier in this chapter shrinks a coefficient toward zero within a single
regression. This section shrinks something different: an entire group's coefficient toward the
rest of the population, when that group has too little data to stand on its own.

::: {#fig-hierarchical-regression}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-hierarchical-regression.html" width="100%" height="500"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

No pooling, complete pooling, and partial pooling for five backend services' payload-size
slopes. Partial pooling shrinks fraud-service's noisy four-request estimate hardest, since it
has the least data to stand on.
:::

@fig-hierarchical-regression fits the same payload-to-latency relationship separately for five
services behind the checkout API: cart-service, pricing-service, inventory-service,
shipping-service, and fraud-service, a newly launched check with only 4 logged requests so far,
against the other four services' 50 to 140 requests each.

**What no pooling, complete pooling, and partial pooling are.** Three ways to combine five
separate datasets into slope estimates, in increasing order of how much information they share
across services. No pooling fits each service on its own data alone, as if the other four
services did not exist. Complete pooling does the opposite: it throws every service's requests
into one dataset and fits a single shared slope, as if the five services behaved identically.
Partial pooling, the hierarchical option, sits between the two: each service's slope is treated
as drawn from a shared population distribution, so a service's own data and the population's
typical behavior both inform its final estimate, weighted by how much data that service has to
offer.

**Why it matters.** fraud-service's no-pooling slope comes out at 10.10 milliseconds per
kilobyte with a standard error of 0.87; four requests are nowhere near enough to pin down a slope
with any confidence, and the value this simulation was built from is 9.5. Complete pooling
ignores fraud-service's own signal and reports 9.54 for every service, including
shipping-service, whose true slope of 9.1 sits meaningfully below that shared number. Partial
pooling gives fraud-service a slope of 9.55, close to the population's typical behavior rather
than the noisy four-request estimate, while still reporting 9.00 for shipping-service, grounded
in shipping-service's own 50 requests rather than forced to match everyone else. An engineer who
trusts fraud-service's no-pooling estimate on day one would build a capacity plan around a slope
0.6 milliseconds per kilobyte higher than its neighbors, on evidence equivalent to four coin
flips. Partial pooling does not throw fraud-service's four requests away; it just declines to
trust them past the weight four requests can carry.

**How to compute it.** The figure above computes partial pooling with a closed-form shortcut: the
same posterior-precision formula from the top of this chapter,

$$
w_j = \frac{1/\text{se}_j^2}{1/\text{se}_j^2 + 1/\tau^2}, \qquad
\hat\beta_{1,j}^{\text{partial}} = w_j\, \hat\beta_{1,j}^{\text{no-pool}} + (1-w_j)\, \bar\beta_1
$$

applied one level up. In place of a single coefficient's prior mean and variance, $\bar\beta_1$
and $\tau^2$ are the population's mean slope and between-service variance, estimated here from
the five no-pooling slopes themselves. A service with a small standard error, meaning it has a
lot of its own data, gets a weight $w_j$ close to 1 and mostly keeps its own estimate;
fraud-service's $w_j$ comes out at 0.12, so 88% of its final estimate comes from the population
rather than its own four requests.

A full Bayesian fit estimates all of this jointly instead of in two separate steps, and is what a
production model would use:

```python
import pymc as pm

coords = {"service": ["cart", "pricing", "inventory", "shipping", "fraud"]}

with pm.Model(coords=coords) as hierarchical_model:
    mu_beta1 = pm.Normal("mu_beta1", mu=0, sigma=5)
    tau_beta1 = pm.HalfNormal("tau_beta1", sigma=2)
    beta1 = pm.Normal("beta1", mu=mu_beta1, sigma=tau_beta1, dims="service")
    beta0 = pm.Normal("beta0", mu=0, sigma=20, dims="service")
    sigma = pm.HalfNormal("sigma", sigma=10)
    mu = beta0[service_idx] + beta1[service_idx] * payload_kb
    pm.Normal("latency", mu=mu, sigma=sigma, observed=latency_ms)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=11)
```

`pm.Model(coords=...)` names the service dimension once, and `dims="service"` tells PyMC that
`beta1` holds one value per service rather than one value overall. `mu_beta1` and `tau_beta1` are
the population-level mean and standard deviation the five services' slopes are drawn from;
fitting them jointly with each service's own `beta1` lets a service with little data borrow the
others' information automatically, rather than through the two-step shortcut used to draw the
figure above.

**Formal notation.**

$$
\beta_{1,j} \sim \text{Normal}(\mu_{\beta_1}, \tau_{\beta_1}^2), \qquad
Y_{ij} = \beta_{0,j} + \beta_{1,j} X_{ij} + \varepsilon_{ij}
$$

for service $j = 1, \dots, 5$ and request $i$ within that service. $\tau_{\beta_1}^2$ controls how
much pooling happens: as $\tau_{\beta_1}^2 \to 0$, every service is pulled toward the same slope,
recovering complete pooling; as $\tau_{\beta_1}^2 \to \infty$, each service's slope moves
independently of the others, recovering no pooling. Partial pooling is what happens in between,
with the data itself setting how far toward either extreme $\tau_{\beta_1}^2$ should sit, rather
than an analyst choosing one of the two extremes ahead of time [@gelmanhill2007].

::: {.callout-note}
The tighter $\tau_{\beta_1}$ is estimated to be, the harder every service's slope gets pulled
toward the shared mean, and the harder that joint model becomes to sample: this is the same
geometry the divergences section later in this chapter builds a figure around.
:::

## How MCMC explores a posterior, step by step {#sec-mcmc-metropolis-hastings}

Earlier in this chapter, MCMC got one sentence: it "explores many candidate coefficient values by
trial and error, lingering near ones that fit the data well." That sentence is true and shows no
mechanism. This section builds one from scratch, on a problem small enough to check against the
closed-form formula from the top of the chapter.

::: {#fig-metropolis-walkthrough}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-metropolis-walkthrough.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A from-scratch Metropolis sampler targeting a 25-request posterior this chapter has a closed-form
answer for. The step-by-step view shows individual proposals accepted and rejected; the long-run
view shows 3,000 post-burn-in samples against the closed-form density.
:::

@fig-metropolis-walkthrough runs a Metropolis sampler against the payload-size coefficient's
posterior on a fresh 25-request sample, small enough to keep the walkthrough short and paired
with the same conjugate-Normal formula this chapter derived at the top: a posterior mean of 9.48
milliseconds per kilobyte and a posterior standard deviation of 0.08, computed directly with no
sampling involved.

**What a Metropolis sampler is.** At every step, the sampler proposes a new candidate value near
its current position, decides whether to move there, and either moves or stays put. Repeated
thousands of times, the fraction of iterations spent near any given value converges to that
value's posterior density, without the sampler ever computing the posterior's normalizing
constant, the part of Bayes' theorem that is usually hardest to get in closed form.

**Why it matters.** The "step-by-step mechanics" view in @fig-metropolis-walkthrough shows the
first 40 iterations, each one either a green accepted move or a red rejected one. The first eight
tell the whole story:

| Iteration | Current | Proposed | Accept? | Next |
|---|---|---|---|---|
| 1 | 0.000 | -0.011 | No | 0.000 |
| 2 | 0.000 | 0.147 | Yes | 0.147 |
| 3 | 0.147 | 0.090 | No | 0.147 |
| 4 | 0.147 | 0.160 | Yes | 0.160 |
| 5 | 0.160 | 0.070 | No | 0.160 |
| 6 | 0.160 | 0.153 | No | 0.160 |
| 7 | 0.160 | 0.298 | Yes | 0.298 |
| 8 | 0.298 | 0.336 | Yes | 0.336 |

A rejected proposal is not wasted computation the way it might look at first glance: staying put
is the correct behavior whenever the proposed point explains the data worse than the current one,
and counting the repeated value again is what gives it the right weight in the final histogram.
Run for 4,000 iterations, this sampler accepted 70.2% of its proposals and produced 3,000
post-burn-in draws with a sample mean of 9.48 and a sample standard deviation of 0.08, matching
the closed-form answer to two decimal places without the sampler ever being told what that answer
was.

**How to compute it.** The acceptance step is the whole algorithm:

```python
def log_posterior(beta1):
    log_lik = norm.logpdf(latency_ms, loc=beta0 + beta1 * payload_centered, scale=2.0).sum()
    log_prior = norm.logpdf(beta1, loc=0, scale=5.0)
    return log_lik + log_prior

current, current_lp = 0.0, log_posterior(0.0)
samples = []
for _ in range(4000):
    proposal = current + rng.normal(0, step_size)
    proposal_lp = log_posterior(proposal)
    if np.log(rng.uniform()) < proposal_lp - current_lp:
        current, current_lp = proposal, proposal_lp
    samples.append(current)
```

Working in log-space avoids underflow once the likelihood involves more than a handful of
observations, and comparing a log-uniform draw against the log-ratio is equivalent to comparing a
plain uniform draw against the ratio itself. This proposal is symmetric, a Normal random walk
centered on the current value, which is what makes a plain ratio of posterior densities the
correct acceptance rule; an asymmetric proposal needs a correction term for however lopsided the
proposal is, the Hastings half of Metropolis-Hastings [@metropolis1953; @hastings1970].

**Why NUTS improves on this.** A plain random walk like the one above has to be told a step size
by hand, and one step size rarely suits every part of a posterior equally well: too large, and it
proposes moves the posterior keeps rejecting; too small, and it accepts almost everything but
crawls, taking many iterations to move anywhere. The No-U-Turn Sampler PyMC reaches for by
default, what `pm.sample` runs unless told otherwise, uses the posterior's gradient to propose
distant moves that stay probable, adapts its own step size automatically during warmup, and stops
each proposed trajectory once it starts curving back on itself instead of running a fixed number
of steps every time [@hoffmangelman2014]. For the single-coefficient posterior in this
walkthrough, plain Metropolis works fine; the posterior is one bump with no odd geometry to trip
over. For the horseshoe prior's spike-and-tail shape from earlier in this chapter, or the
hierarchical model's population-level variance parameter from the previous section, a fixed-step
random walk struggles in the regions those models most need explored with care. The next section
shows why.

## Why divergences happen: the funnel {#sec-divergences-funnel}

This chapter has asserted, twice, that certain models need a non-centered parameterization to
sample well: the horseshoe prior earlier in this chapter, and the hierarchical model's
population-level variance in the previous section. Neither assertion has had a supporting
picture until now.

::: {#fig-divergences-funnel}
```{=html}
<iframe src="../_generated/chapter-bayes-regression-fig-divergences-funnel.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The same funnel-shaped posterior, sampled two ways. Centered, 16.8% of draws sit where a single
step size is more than three times too big for the local geometry. Non-centered, none do.
:::

@fig-divergences-funnel draws 5,000 samples from a simple two-parameter posterior that shows up
whenever a hierarchical model estimates a population-level standard deviation, here called
$\tau$, jointly with an individual group's deviation from the population mean. The same
underlying distribution is drawn two different ways: once in its natural, centered coordinates,
and once in a reparameterized, non-centered form.

**What the funnel is.** Plot $\log \tau$ against a group's deviation from the population mean,
and the region of high posterior probability narrows sharply as $\log \tau$ decreases: when the
population-level spread is small, every group's deviation is pulled close to zero too, since the
two are directly linked (the deviation's own conditional distribution has standard deviation
$\tau$). The result looks like a funnel, wide where $\tau$ is larger and narrowing to a point
where it is small.

**Why it matters.** A sampler moving through the centered coordinates has to use one step size
across the whole funnel. Calibrate that step size to move efficiently through the wide part, and
it becomes badly oversized the moment the chain wanders into the neck: @fig-divergences-funnel
marks 16.8% of the centered draws as sitting where a step size tuned to the funnel's bulk is more
than three times too large for the neck's own scale. An HMC or NUTS sampler in that spot flags
the problem directly: its numerical integrator becomes unstable partway through a proposed
trajectory, and PyMC reports that step as a divergence rather than folding a bad value into the
trace unnoticed. Ignore divergence warnings, and the reported posterior for $\tau$ tends toward
larger values than its true posterior holds, since the sampler under-visits the narrow region it
cannot navigate with a step size built for the wide part. The "more than a handful of
divergences" warning box earlier in this chapter was pointing at this same problem, without yet
showing what a handful of divergences looks like or why they cluster where they do.

**How to compute it.** Centered and non-centered parameterizations describe the same distribution
in different coordinates.

$$
\text{Centered:} \quad \log\tau \sim \text{Normal}(0, 1.2^2), \qquad \delta \sim \text{Normal}(0, \tau^2)
$$

$$
\text{Non-centered:} \quad \log\tau \sim \text{Normal}(0, 1.2^2), \qquad z \sim \text{Normal}(0, 1), \qquad \delta = z \cdot \tau
$$

where $\delta$ is a group's deviation from the population mean. In the non-centered form, $z$'s
own conditional distribution never depends on $\tau$; its scale stays fixed at 1 no matter what
$\tau$ is, so a single step size that works well for $z$ at one value of $\tau$ works just as well
at every other value. @fig-divergences-funnel's non-centered panel confirms this directly: none
of its draws are flagged, by construction, since nothing about $z$'s scale changes as $\tau$
shrinks.

In PyMC, the non-centered version replaces a direct draw with an offset and a rescale:

```python
with pm.Model() as centered:
    tau = pm.HalfNormal("tau", sigma=3)
    delta = pm.Normal("delta", mu=0, sigma=tau, dims="service")

with pm.Model() as non_centered:
    tau = pm.HalfNormal("tau", sigma=3)
    z = pm.Normal("z", mu=0, sigma=1, dims="service")
    delta = pm.Deterministic("delta", z * tau, dims="service")
```

Both models describe the same prior on `delta`. Only the second one gives NUTS a posterior
shaped like a simple, round hill to move through instead of a funnel, which is why a non-centered
parameterization is worth reaching for by default whenever a hierarchical model's population-level
scale might come out small [@neal2003].

PyMC flags a transition as divergent when its Hamiltonian simulation's discretization error
crosses a fixed threshold partway through a step, a numerical check unrelated to $\hat{R}$ or
`ess_bulk`: a model can show a good $\hat{R}$ and good `ess_bulk` overall while still logging
divergences clustered in one region of parameter space, since both summary statistics average
over the whole chain and a funnel's neck can be a small enough region to barely move either one.

::: {.callout-warning}
Divergences cluster where a model's posterior geometry is hardest, not at random. A handful
scattered across many spots is a nuisance; a cluster tells an analyst precisely where to look.
Reparameterizing, or tightening the prior on the offending scale parameter, usually clears both
at once.
:::

## What carries forward

Every other predictive method in this book gets the same treatment in the chapters that follow: a
Bayesian counterpart with its own posterior, its own priors, and its own honest accounting of
what the extra computation buys over a point estimate.

The equivalence this chapter opened with (Ridge as a Gaussian prior, Lasso as a Laplace prior) is
not a special case specific to linear regression.

It is the first instance of a pattern that recurs through cross-validation, splines, trees, and
boosting alike: a classical method that produces one best answer, and a Bayesian version of the
same method that produces a full distribution over answers, at the cost of more computation and
more to explain.

## References {.unnumbered}

::: {#refs}
:::
