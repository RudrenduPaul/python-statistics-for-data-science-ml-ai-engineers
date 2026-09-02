# Cross-Validation and Model Selection

Chapter 4 left one question unanswered. The latency model there added a regularization penalty
controlled by a strength $\lambda$, and Lasso zeroed out a noise predictor once $\lambda$ grew
large enough. But how large is large enough?

Picking $\lambda$ by looking at how well the model fits the data it was trained on cannot answer
that question. A large enough model always fits its own training data better as it grows more
flexible, whether or not it has learned anything that generalizes. *Cross-validation* is the
tool that answers it instead, by estimating how a model performs on data it has never seen.

This chapter works through four questions that come up whenever a model has a knob to tune:

1. *Why can't training error tell me which model is best?*
   A model that fits its training data perfectly is not a good model; it may have memorized
   noise specific to that sample.
2. *How do I estimate performance on unseen data without waiting for new data to arrive?*
   The validation-set approach, leave-one-out cross-validation, and k-fold cross-validation are
   three ways to answer this using only the data in hand.
3. *How many folds should I use?*
   The choice trades bias against variance, and the practical answer is narrower than it looks.
4. *What does cross-validation not fix?*
   It estimates performance under the assumption the data is independently and identically
   distributed; when that assumption breaks, so does the estimate.

## Why training error is the wrong yardstick

::: {#fig-train-test-error}
```{=html}
<iframe src="../_generated/chapter-cv-fig-train-vs-test-error.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Training error falls the whole way as the model grows more flexible, but test error is
U-shaped: it falls alongside training error through roughly degree 2, then rises once added
flexibility starts fitting noise that will not repeat in new data.
:::

Studying only the practice questions that will appear on a test can make anyone feel ready
without proving the material stuck; the test itself is what settles that. Cross-validation
checks a model the same way, against data it never got to study.

Suppose the latency model from Chapter 4 is fit with an increasingly flexible curve rather than
a straight line, a polynomial of growing degree.

@fig-train-test-error fits polynomials of degree 1 through 12 to the same simulated latency
data, measuring error both on the data the model was trained on and on a large, separate batch
of held-out data the model never saw during fitting.

Training error falls the entire way from degree 1 to degree 12; a flexible enough curve can
always snake closer to its own training points.

Test error tells a different story: it falls alongside training error while the model is
capturing the underlying relationship between payload size and latency, then turns and rises
once additional flexibility starts fitting noise that will not repeat in new data.

This is because training error measures how well a model memorized one specific sample, while
test error measures whether what it learned holds up elsewhere, and those are not the same
question.

That gap between the two curves is invisible to anyone who only tracks training error: without
checking held-out performance, there is no way to know which side of the curve's minimum a model
sits on.

::: {.callout-note}
A model's training error can only fall, or stay flat, as flexibility increases, whether or not
the added flexibility captures anything that will repeat in new data. Held-out data is the only
way to tell those two cases apart.
:::

## The validation-set approach

The simplest fix is to hold out part of the data before fitting anything: split the dataset into
a training set and a validation set, fit the model on the training set, and measure error on the
validation set the model never touched during fitting.

This is honest, in that the validation error is a fair estimate of performance on unseen data,
and it is cheap, in that it requires fitting the model only once per candidate.

It is also noisy. With a modest dataset, say 40 or 50 observations, a 50/50 split leaves only 20
to 25 points on each side.

Recall from Chapter 1 that a sample is a subset of a population, and any subset carries sampling
variability. A validation set that happens to include a few unusually easy or unusually hard
requests will systematically over- or underestimate error.

A different random split of the same data can produce a meaningfully different validation
error, and, in borderline cases, a different choice of model.

:::{.callout-tip}
The smaller the dataset, the noisier a single validation split gets. Below a few hundred
observations, k-fold cross-validation is worth the extra fitting cost over a one-time split.
:::

## Leave-one-out cross-validation

*Leave-one-out cross-validation* (LOOCV) removes the arbitrariness of a single split by
repeating the validation-set idea $n$ times, once for every observation in the dataset: fit the
model on all but one point, predict that one point, record the error, and move to the next
point.

Averaging the $n$ resulting errors gives a single estimate of test performance that does not
depend on which particular split happened to occur, since every observation gets its turn as
the validation set of precisely one fold.

LOOCV was formalized by Stone (1974), who framed cross-validation as a general way to assess a
statistical prediction rule using the data on hand rather than a fresh sample [@stone1974].

The cost is computational: LOOCV requires fitting the model $n$ times over, once for every
observation, which is expensive once $n$ grows into the thousands or the model itself is costly
to fit.

Certain linear models, ordinary least squares among them, have a closed-form shortcut: a formula
computes what each leave-one-out error would have been directly from the single full-data fit,
without literally leaving out and refitting each point in turn. That shortcut does not extend to
most of the models later chapters cover, including trees and gradient boosting.

::: {.callout-tip}
For OLS and other models with this closed-form shortcut, computing the leave-one-out error takes
one fit, not $n$ separate fits. Check whether a library exposes that shortcut before defaulting
to a manual loop over every observation.
:::

## k-fold cross-validation

*k-fold cross-validation* is the practical middle ground between a single validation split and
LOOCV. Instead of leaving out one observation at a time, the data is divided into $k$ roughly
equal folds.

Each fold takes a turn as the validation set while the model trains on the remaining $k-1$
folds, and the $k$ resulting error estimates are averaged.

LOOCV is the special case where $k$ equals $n$; a single validation split is the special case
where $k$ equals 2 and the split happens only once instead of $k$ times.

::: {#fig-validation-variance}
```{=html}
<iframe src="../_generated/chapter-cv-fig-validation-strategy-variance.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A single validation split produces the widest spread of estimated test error across repeats.
5-fold and 10-fold CV narrow that spread considerably, and LOOCV narrows it further still.
:::

@fig-validation-variance repeats each strategy, a single validation split, 5-fold CV, 10-fold
CV, and LOOCV, 60 times on freshly simulated data, showing the spread of the resulting
test-error estimate for each strategy across those repeats.

## Choosing k: the bias-variance trade-off

It is natural to conclude from @fig-validation-variance that LOOCV, having the narrowest spread,
is the best choice every time. The full picture is more balanced.

LOOCV's $n$ training folds each contain nearly the entire dataset, so the $n$ fitted models are
highly correlated with one another and with the model fit on all the data.

Averaging their errors gives a nearly unbiased estimate of test error. That estimate can still
have high variance, though, because averaging highly correlated numbers does not reduce
variance the way averaging independent numbers does.

A small $k$, by contrast, trains each fold's model on a visibly smaller slice of the data. That
biases the error estimate upward (a model trained on 80% of the data usually performs a little
worse than one trained on 100% of it), but it keeps the $k$ folds' models less correlated with
each other.

In practice, $k=5$ or $k=10$ has become the standard choice because both sit in the range where
the bias from training on a slightly smaller fold stays small while the correlation-driven
variance problem that affects LOOCV is largely avoided. This is an empirically supported default
rather than a theorem.

:::{.callout-tip}
Treat $k=5$ or $k=10$ as the starting point for any cross-validation setup, not a fixed rule.
Both keep the bias from smaller training folds low without the variance problem that comes from
LOOCV's highly correlated fits.
:::

A dataset small enough that even a 10th of it is a meaningful loss of training data is a case
worth checking by hand rather than defaulting to $k=10$ automatically.

## Cross-validation for model selection: picking lambda

::: {#fig-lambda-selection}
```{=html}
<iframe src="../_generated/chapter-cv-fig-lambda-selection.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Cross-validated error traces a U-shape in lambda: the minimum marks the value cross-validation
would select, and that minimum firms up, becoming less sensitive to the particular split, once
k grows from 3 to 10.
:::

Return to the question this chapter opened with: what value of $\lambda$ should the latency
model's Lasso penalty use? @fig-lambda-selection answers it directly, computing k-fold
cross-validated error across a range of $\lambda$ values for the same latency model, with a
toggle between $k=3$ and $k=10$.

The curve is U-shaped for the same reason @fig-train-test-error was: too small a $\lambda$
barely regularizes the model at all, leaving it free to fit noise the way an unregularized OLS
fit would; too large a $\lambda$ shrinks every coefficient toward zero so aggressively that the
model underfits, losing useful signal along with the noise.

The $\lambda$ at the bottom of the curve is the value cross-validation recommends. Switching the
toggle from $k=3$ to $k=10$ shows the curve firming up.

With only 3 folds, each fold's error estimate rests on a smaller, noisier training set, so the
curve's minimum can shift meaningfully from one run to the next. That instability stabilizes as
$k$ grows.

This same procedure, sweeping a candidate value and picking the one with the lowest
cross-validated error, generalizes far past $\lambda$: choosing a polynomial's degree, a
smoothing parameter, or the number of trees in a forest are all the same search, applied to a
different knob.

## What cross-validation does not fix

Cross-validation estimates test error under one working assumption: that the data is
independently and identically distributed, so a fold selected at random is a fair stand-in for
future data. Two common situations break that assumption in ways cross-validation cannot detect
on its own.

The first is data leakage: information from the validation fold sneaking into the training
process through some channel other than the model fit itself.

A feature engineered from the full dataset before the split, for instance a normalization
computed across every observation rather than fit only on the training fold, leaks information
about the validation fold into the training procedure. That leak makes cross-validated error
look better than it will on data the pipeline has never seen at all.

::: {.callout-warning}
Any preprocessing step that uses information from the full dataset before splitting, such as a
global normalization or a target-encoded category, leaks validation information into training.
The result is cross-validated error that looks better than what the pipeline will see once it
runs in production on data it has never processed before.
:::

The second is non-independent data, most commonly time series. Recall the request-latency logs
from Chapter 1: consecutive requests during a traffic spike are not independent draws from a
fixed distribution.

They are correlated in time, and shuffling them into random folds lets a model trained on data
from *after* a validation point predict that earlier point, an advantage it will never have in
production.

:::{.callout-important}
Shuffling time-ordered data into random folds lets a model trained on later observations
predict an earlier one, an advantage it will never have once it runs in production. Use a
rolling-origin or blocked split for anything measured over time.
:::

Time-ordered data calls for a variant such as rolling-origin or blocked cross-validation, where
every training fold precedes its validation fold in time, rather than the random k-fold split
this chapter has used throughout.

## A Bayesian perspective

Cross-validation is not the only way to ask whether a model will generalize; it is the
frequentist answer to a question a Bayesian framework poses differently.

Recall from Chapter 4's Bayesian perspective that a Bayesian model treats each coefficient as a
range of plausible values, the posterior distribution, rather than one fitted number. Model
comparison in that framework asks how well the posterior predicts data it has not seen, the same
underlying question cross-validation answers, approached from a different direction.

The Bayesian analog most directly comparable to classical LOOCV is *leave-one-out
cross-validation computed from posterior draws*, abbreviated PSIS-LOO, where a draw is one
sampled parameter value out of the many an MCMC fit produces in place of a single point
estimate.

Instead of literally retaking a test with one question removed each time, imagine estimating
that score from patterns in the one full attempt on record. That shortcut, applied to model
fitting, is what PSIS-LOO does.

Refitting a Bayesian model $n$ times, once per left-out observation, is every bit as expensive
as classical LOOCV, and just as impractical for anything but small datasets.

Vehtari, Gelman, and Gabry (2017) addressed this with Pareto smoothed importance sampling
(PSIS), a technique that approximates each leave-one-out fit from the posterior draws of the
single full-data fit, avoiding the need to refit the model at all in most cases
[@vehtarigelmangabry2017].

The approximation is not reliable by default when a handful of observations wield an unusually
large influence on the fit. The method's own diagnostic, a Pareto shape-parameter estimate
called Pareto-k, flags those cases directly, and an observation whose k value crosses the
paper's threshold needs a direct refit rather than a trusted PSIS estimate.

The resulting number is the *expected log predictive density* (ELPD): a higher value means the
model's posterior assigns higher probability, on average, to data it did not train on, the
direct Bayesian counterpart to a lower cross-validated mean squared error.

::: {#fig-elpd-by-complexity}
```{=html}
<iframe src="../_generated/chapter-cv-fig-elpd-by-complexity.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

ELPD rises with model complexity while the model is still capturing the underlying structure,
peaking around degree 2, then falls as added flexibility starts fitting noise, the mirror image
of the test-error curve in the first figure of this chapter.
:::

@fig-elpd-by-complexity computes an ELPD approximation across the same range of model complexity
as @fig-train-test-error, and the two curves tell the same story from different directions: one
falls after a point, the other rises after a point, and both mark the same transition from a
model that generalizes to one that has started fitting noise.

A closely related quantity, the Watanabe-Akaike information criterion (WAIC), estimates the same
out-of-sample predictive accuracy from a different approximation [@vehtarigelmangabry2017].

It tends to agree closely with PSIS-LOO on well-behaved models, though WAIC has no equivalent to
the Pareto-k diagnostic, which is one reason the `arviz` and `loo` software packages default to
PSIS-LOO.

None of this Bayesian machinery replaces classical k-fold cross-validation for most day-to-day
model tuning: it requires committing to a full Bayesian model with priors on every parameter,
which is a heavier lift than fitting a Lasso path, and k-fold CV remains simpler to explain to a
stakeholder who has never heard the word posterior.

The Bayesian route earns its keep when a project has adopted a Bayesian model for other reasons,
most notably the Bayesian A/B testing framework Part 3 of this book builds next.

There, PSIS-LOO becomes the natural way to check whether added model complexity (an extra
covariate, a hierarchical structure across segments) improves predictions rather than just
fitting the observed experiment more closely.

## References {.unnumbered}

::: {#refs}
:::
