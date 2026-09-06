# Bayesian Approaches to Gradient Boosting

Chapter 9 turned a regression coefficient into a posterior distribution with a single choice
of prior. Chapter 12 turned a random forest into BART by putting a regularizing prior on tree
structure and leaf values, then sampling the ensemble with MCMC.

Gradient boosting resists the same move. It is worth being specific about why, before looking
at what a Bayesian practitioner does instead.

## Why boosting does not have a clean posterior

A Bayesian model update, in every case this book has covered, works by combining a prior belief
with a likelihood to get a posterior belief, then sampling from or computing that posterior
directly.

Conjugate linear regression does this in closed form: one formula, plugged in once, with no
iteration and no sampling required. Bayesian logistic regression and BART do it with MCMC
(Markov chain Monte Carlo), drawing samples that, in the long run, represent the posterior even
when no formula for it exists.

Both routes have the same shape: a well-defined target distribution exists, and the machinery's
job is to characterize it.

Gradient boosting, as Chapter 8 built it, does not define a target distribution at all. Recall
the mechanism: start from a constant prediction, compute the negative gradient of the loss with
respect to the current predictions, fit a small tree to that gradient, add it to the running
total with a learning rate applied, and repeat [@friedman2001].

Every step is a deterministic optimization move, a greedy step toward lower training loss, not
a draw from any distribution. There is no prior on "what the ensemble should look like" doing
any work in that loop, and no posterior falls out of it when the loop finishes.

XGBoost's regularization term penalizes leaf weights and leaf count [@chen2016xgboost], which
looks Bayesian in spirit. A penalty term is, after all, mathematically equivalent to a prior's
effect on a point estimate, the same equivalence Chapter 9 showed for Ridge and the Bayesian
Lasso.

But the equivalence stops there: nothing in XGBoost's fitting procedure ever samples, so the
output is a single fitted ensemble, a point estimate rather than a distribution over ensembles.

::: {.callout-note}
The core distinction: boosting optimizes greedily toward one fitted ensemble. It never defines
a target distribution to sample from, which is why it has no natural posterior the way linear
regression or trees do.
:::

This is not a minor implementation gap that a future release will close. It is a structural
property of what boosting optimizes: a single, sequentially-constructed function, chosen
greedily at each step to reduce loss the fastest. Nothing in the loop can represent more than
one plausible ensemble at a time.

Two practical threads exist despite this, and they solve different problems. Bayesian
hyperparameter optimization treats the *tuning* of a boosted model as a Bayesian decision
problem, leaving the model itself untouched.

NGBoost, covered later in this chapter, changes what the model predicts (a full distribution
instead of a point value) without changing how the ensemble is built. Neither one gives
gradient boosting the kind of posterior BART has over its ensemble.

## Bayesian hyperparameter optimization

Think of tuning a recipe's oven temperature and bake time. Instead of testing combinations at
random, a good cook uses what worked or flopped in earlier batches to guess where to try next.
That is Bayesian hyperparameter search in one sentence.

Suppose the rollback-risk model from Chapter 8 needs its learning rate and max tree depth
tuned. A grid search would try every combination on a fixed grid. A random search would sample
combinations uniformly.

Both treat every untried combination as equally worth trying next. That wastes evaluations on
combinations a reasonable person would expect to be bad, given what earlier trials revealed.

A Bayesian approach treats the relationship between hyperparameters and validation loss as an
unknown function to be modeled, the same framing Chapter 11 introduced for a different problem
(interpolating a noisy curve).

Here, the function being modeled is
$f(\text{learning rate}, \text{max depth}) = \text{validation loss}$, known only at the handful
of points evaluated so far.

In other words, plug in a learning rate and a max depth and this function would return how well
a model trained with those settings performs on held-out data, if only it could be written down
directly.

It cannot: the only way to learn its value at a given point is to train a model there and
measure the result, which is what makes every trial expensive.

A Gaussian process fits a posterior belief over that function from the trials run so far.
Recall from Chapter 11 that this means assigning a posterior belief not to a single number but
to every curve consistent with the points observed so far, weighted by how well each curve
fits.

As covered further down, a different kind of surrogate model can play the same role. An
*acquisition function* then scores every untried combination by how promising it is to try
next, balancing two things: how low the surrogate's mean prediction is there (exploitation) and
how uncertain the surrogate still is there (exploration).

*Expected improvement* (EI), the most common acquisition function, computes the expected amount
by which a new trial would beat the best result seen so far, under the surrogate's current
posterior.

::: {#fig-bo-search-trajectory}
```{=html}
<iframe src="../_generated/chapter-bayes-boosting-fig-search-trajectory.html" width="100%"
        height="580" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A Bayesian search's trials on a simulated validation-loss surface, learning rate vs. max depth.
Early trials spread out to explore; later trials cluster around the region that turns out to be
promising. Move the slider.
:::

@fig-bo-search-trajectory shows this loop run on a simulated validation-loss surface over
learning rate and max depth for the rollback-risk model, seeded with four random trials and
then extended one trial at a time by maximizing expected improvement. Move the slider forward
and watch where each new point lands.

Early trials spread out, since with almost nothing observed yet the surrogate's uncertainty
dominates the expected-improvement score everywhere. As more trials accumulate, later points
cluster around the region that turns out to be promising.

Occasionally the search still probes a far corner of the space where uncertainty remains high,
even though the mean prediction there looks unremarkable. That occasional probe is expected
improvement doing its job.

A point with mediocre predicted loss but wide uncertainty can still beat a point with slightly
better predicted loss and near-zero uncertainty, because the first point might turn out to be
excellent. The second point, by contrast, is unlikely to turn out much better than the best
trial found so far.

::: {#fig-expected-improvement}
```{=html}
<iframe src="../_generated/chapter-bayes-boosting-fig-expected-improvement.html" width="100%"
        height="560" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The GP posterior mean, the trials evaluated, and the expected-improvement curve that picks the
next trial. EI peaks not at the best point seen so far but where promise and uncertainty
overlap. Move the slider.
:::

@fig-expected-improvement isolates one hyperparameter (learning rate, with max depth held
fixed) and shows the GP surrogate's posterior mean, the true underlying loss curve (which the
search never sees directly), the trials evaluated so far, and the expected-improvement curve
that decides where the next trial lands.

Note where expected improvement peaks relative to the trials evaluated so far. It is not
highest at the best point found so far, since that point's uncertainty has shrunk from being
explored.

It is not highest in the region furthest from any trial either, since that area is uncertain
but the surrogate's mean prediction offers no reason to expect it is good. Instead, it peaks
somewhere the surrogate is both moderately optimistic and still uncertain enough to be worth
checking.

In practice, nobody hand-writes this loop. `Optuna`, the most actively maintained and widely
adopted Bayesian hyperparameter optimization library in Python today, defaults to a
tree-structured Parzen estimator (TPE) rather than a Gaussian process as its surrogate model
[@akiba2019optuna].

A TPE models the *distribution of hyperparameters* that produced good versus bad results,
rather than modeling the loss surface directly the way a GP does. It scales better to many
hyperparameters and to non-numeric (categorical) ones, at the cost of a less interpretable
surrogate than a GP's.

::: {.callout-tip}
A rule of thumb on which surrogate to reach for: a GP-based optimizer such as `scikit-optimize`
or `BoTorch` works well up to roughly a dozen or two hyperparameters, past which the covariance
matrix grows expensive and the surrogate's fit degrades. TPE's independence assumption between
hyperparameters is what lets Optuna scale past that range.
:::

`scikit-optimize` (`skopt`) offers a more traditional GP-based Bayesian optimizer, closer to
what @fig-bo-search-trajectory demonstrates directly.

For problems where the objective function is expensive enough that every evaluation counts (a
model that takes hours to train, not milliseconds), `BoTorch` is the name worth knowing for
GP-based Bayesian optimization at a larger scale, built on PyTorch and commonly used for tuning
deep learning training runs rather than gradient-boosted trees.

::: {.callout-note}
Tooling changes fast in this space. `scikit-optimize`'s maintainers archived the repository in
2024, and `GPyOpt` has seen little active maintenance in recent years. Check a library's
commit history before depending on it for new work.
:::

A worked call against the rollback-risk model, in Optuna's own idiom:

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    lr = trial.suggest_float("learning_rate", 0.01, 0.35, log=True)
    depth = trial.suggest_int("max_depth", 1, 10)
    model = GradientBoostingClassifier(learning_rate=lr, max_depth=depth, n_estimators=200)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_log_loss")
    return -score.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print(study.best_params)
```

::: {.callout-warning}
`cross_val_score` with `scoring="neg_log_loss"` returns a negative number, since scikit-learn's
convention is that a higher score is always better. Dropping the sign flip in
`return -score.mean()` above leaves Optuna minimizing the wrong quantity, pushing the search
toward the worst hyperparameters instead of the best.
:::

Thirty trials with this search routinely land close to the region a much larger grid search
would have found. Every trial after the first handful is chosen using what the earlier trials
revealed, the same logic @fig-bo-search-trajectory shows visually.

## NGBoost: a boosted model that outputs a distribution

Picture two weather forecasters. One says "72 degrees tomorrow" and stops there. The other says
"72 degrees, but anywhere from 65 to 79 wouldn't surprise me." NGBoost is built to be the
second kind of forecaster.

A separate, more direct answer to "how do I get uncertainty out of a boosted model" is to
change what the model predicts, not how it is tuned. `NGBoost` (Natural Gradient Boosting) fits
a boosted ensemble the usual way, one tree at a time correcting the ensemble's current
mistakes.

But instead of predicting a single number, it predicts the parameters of a full probability
distribution [@duan2020ngboost]. For a continuous target, that typically means a Normal
distribution's mean and variance.

For a rollback-probability target, it means a Bernoulli parameter (the probability that a given
deployment triggers a rollback) trained against a proper scoring rule, a loss function whose
minimum is reached only when the predicted probability matches the true probability, so the
model has no incentive to hedge toward 50% or overstate its confidence.

That training target keeps the probability calibrated, though a single Bernoulli parameter
carries no separate variance term the way the continuous case does.

The "natural" in the name refers to natural gradient descent, a technique for taking
optimization steps that account for the geometry of the distribution's parameter space rather
than treating every parameter as an independent, equally-scaled coordinate.

That geometry-aware approach turns out to matter a great deal for training stability when the
thing being predicted is a distribution's parameters rather than a single value.

::: {#fig-ngboost-width}
```{=html}
<iframe src="../_generated/chapter-bayes-boosting-fig-ngboost-width.html" width="100%"
        height="560" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

NGBoost's predicted mean (solid blue) tracks the point-estimate model's forecast (dashed red)
closely at every payload size on the x-axis, but only NGBoost adds a shaded interval around it.
That interval stays narrow inside the dense training-data region (roughly 5 to 25 KB) and
widens sharply for payload sizes below and above it, since fewer training examples there leave
the model less certain about the predicted rollback probability. Move the slider.
:::

@fig-ngboost-width contrasts a point-estimate boosted model's prediction with an NGBoost-style
predictive interval, on payload size ranging outside the region where training data was dense.

The point-estimate model's dashed line has no opinion about where it is extrapolating. The
number it reports at a payload size of 40 KB looks just as confident as the number it reports
at 15 KB, even though the model saw plenty of 15 KB examples during training and almost none
near 40 KB.

NGBoost's interval widens specifically in the region past the dense training data. That happens
because its distributional prediction carries a variance term that the fitting procedure is
free to inflate when the ensemble's trees disagree more, or when nearby training examples were
sparser.

This is not a full posterior over the ensemble the way BART's MCMC samples are. It is a single
fitted model that happens to output a distribution rather than a point. That distinction
matters for what you can and cannot do with it.

::: {.callout-important}
NGBoost's predictive interval is not the same as a full Bayesian posterior over the ensemble.
It estimates spread at each point; it does not tell you how differently the fitted ensemble
itself might have turned out under a different training run.
:::

NGBoost's interval tells you the model's estimate of its own predictive spread at each point.
It does not tell you how much the fitted ensemble itself might have looked different under a
different random training run. That second question is closer to what BART's tree-to-tree
posterior variation captures.

NGBoost asks the ensemble to output an entire distribution's parameters at once. A narrower,
older, and in practice more commonly deployed answer asks it to output a single number instead:
one specific quantile of that distribution, trained directly against that target.

## Quantile regression: predicting the tail of the distribution

::: {#fig-quantile-regression}
```{=html}
<iframe src="../_generated/chapter-bayes-boosting-fig-quantile-regression.html" width="100%"
        height="580" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Checkout latency against request load, with a fixed mean-regression line (dashed red) and a
quantile-regression line (solid blue) that climbs as the target quantile rises. At 1,500
requests per second the mean line sits at 216 ms, comfortably under a 400 ms SLA; the 95th-
percentile line sits at 434 ms, over it. Move the slider.
:::

Picture a checkout service with a 400 millisecond latency SLA, on-call rotation included. At
1,500 requests per second, a model trained to predict mean latency reports 216 ms. An engineer
reading that number, and nothing else, would call the service healthy with room to spare.

The 95th-percentile latency measured directly from held-out traffic near that same load is
452 ms, past the SLA. A quantile-regression model trained to target the 95th percentile predicts
434 ms at that load, close to the measured figure and on the correct side of the threshold the
mean model missed by 184 ms.

That gap comes from the shape of a right-skewed distribution: the mean and the 95th percentile
are different numbers by construction, and @fig-quantile-regression shows both lines drawn from
the same data. A well-fit mean model optimizes for the center of the distribution; catching the
tail was never its target.

Quantile regression fits a model to a chosen quantile $\tau$ of the conditional distribution of
$Y$ given $X$, rather than to its conditional mean [@koenkerbassett1978]. Setting $\tau = 0.5$
targets the median; setting $\tau = 0.95$ targets a value only 5% of outcomes are expected to
exceed, which is the number an SLA or a capacity-planning budget is usually written against.

::: {.callout-note}
An SLA defined on P95 or P99 latency is a statement about a tail quantile. A model trained to
minimize squared error targets the mean, which is the wrong target whenever the number feeding
a paging threshold or a capacity plan is that tail quantile.
:::

Provisioning a fleet off the mean-regression forecast in the scenario above would leave the
service under capacity until the SLA breach shows up in a monitoring dashboard, well after the
load that caused it has arrived. Sizing off the quantile-regression forecast catches the same
breach at 1,500 requests per second, while load is still a planning question, before it turns
into an incident.

The mechanism behind this is the same gradient-boosted tree ensemble Chapter 8 built, with one
change: the loss function. Ordinary boosting minimizes squared error, which is minimized by the
conditional mean. Quantile regression instead minimizes the *pinball loss* (also called the
check function), which is minimized by the conditional $\tau$-quantile instead:

$$
L_\tau(y, \hat{y}) =
\begin{cases}
\tau (y - \hat{y}) & \text{if } y \geq \hat{y} \\
(1 - \tau)(\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}
$$

For $\tau = 0.95$, underpredicting a high observed value costs 19 times as much as overpredicting
it by the same amount ($\tau / (1 - \tau) = 0.95 / 0.05$). That asymmetry is what pulls the
fitted curve up toward the tail of the distribution.

```python
from sklearn.ensemble import GradientBoostingRegressor

p95_model = GradientBoostingRegressor(
    loss="quantile", alpha=0.95, n_estimators=200, max_depth=3, learning_rate=0.05,
)
p95_model.fit(load_train.reshape(-1, 1), latency_train)
p95_model.predict([[1500]])
```

Swapping `loss="squared_error"` for `loss="quantile"` and setting `alpha` to the desired $\tau$
is the entire change; everything else about fitting, tuning, and reading a `GradientBoostingRegressor`
carries over unchanged from Chapter 8. `sklearn.linear_model.QuantileRegressor` offers the same
idea for a linear model when the relationship does not need trees.

::: {.callout-warning}
Two quantile models fit independently, one per $\tau$, carry no constraint that keeps them
ordered. A model fit for $\tau = 0.50$ can predict a higher value than a model fit for
$\tau = 0.95$ at some point in the input space, a problem known as quantile crossing. Fitting
every quantile of interest from the same ensemble (as most boosting libraries allow) or sorting
the predictions after the fact are the two common fixes.
:::

## Conformal prediction: a coverage guarantee without a posterior

::: {#fig-conformal-coverage}
```{=html}
<iframe src="../_generated/chapter-bayes-boosting-fig-conformal-coverage.html" width="100%"
        height="620" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A fixed-width interval built from calibration residuals (dashed red) against a conformalized
quantile-regression interval (shaded blue) at a 90% target. The fixed-width interval covers 92%
of held-out points overall but only 74% in the busiest quarter of the load range; the
conformalized interval holds close to 90% in both the busiest and the quietest quarter. Move the
slider.
:::

The same checkout-latency setup can also be asked for a 90% prediction interval, a natural
follow-up question once a single P95 number is in hand. A naive approach takes the mean model's
calibration-set residuals, computes their spread, and adds a fixed multiple of that spread above
and below every future prediction. Measured against held-out data, that interval covers 92% of
points overall, comfortably past the 90% target, which is the number a dashboard checking only
the aggregate would report.

Restricted to the busiest quarter of the load range (above roughly 1,300 requests per second),
the same fixed-width interval's coverage drops to 74%, well short of the 90% it was built for,
in the traffic band closest to the provisioned ceiling. In the quietest quarter it swings the
other way and covers 100% of points, spending width the calibration set never needed there. A
single aggregate coverage number hid both problems.

@fig-conformal-coverage builds the interval differently: fit two quantile-regression models from
the previous section, one for a low quantile and one for a high quantile bracketing the target
coverage, then calibrate the gap between them against a held-out calibration set the models never
trained on. The result is *conformalized quantile regression*, or CQR [@romanopattersoncandes2019],
one member of the conformal-prediction family Vovk, Gammerman, and Shafer built into a general
framework for distribution-free prediction [@vovkgammermanshafer2005].

At the same 90% target, the conformalized interval covers 90% of points in the busiest quarter
and 89% in the quietest, against the fixed-width interval's 74% and 100%. The quantile-regression
models stretch the raw interval with load on their own; calibration only nudges that width by an
amount small enough to state directly here: 0.3 ms on each side, in this run, because the
underlying quantile models started close to the right size before calibration touched them.

::: {.callout-note}
Conformal prediction's coverage guarantee rests on one condition: the calibration data and the
future data it is applied to must be exchangeable, a weaker requirement than picking the right
likelihood or the right prior. The underlying model can be misspecified and the guarantee still
holds.
:::

The calibration step itself: hold out a set the model never trained on, score how far each
calibration point fell outside the raw interval $[\text{lo}(x), \text{hi}(x)]$ produced by the
low- and high-quantile models,

$$
E_i = \max\bigl(\text{lo}(x_i) - y_i,\; y_i - \text{hi}(x_i)\bigr),
$$

and take $\hat{q}$ as the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of those scores
across the $n$ calibration points. That slightly inflated fraction is what keeps the guarantee
correct at finite $n$; a plain $1 - \alpha$ quantile only reaches that guarantee in the limit as
$n \to \infty$. The interval reported for a new point is
$[\text{lo}(x) - \hat{q},\; \text{hi}(x) + \hat{q}]$.

```python
import numpy as np

lo_cal, hi_cal = lo_model.predict(X_cal), hi_model.predict(X_cal)
scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
n = len(scores)
q_hat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)

lo_new = lo_model.predict(X_new) - q_hat
hi_new = hi_model.predict(X_new) + q_hat
```

Under only the assumption that the calibration and test points are exchangeable, this interval
covers the true value at least $1 - \alpha$ of the time, no matter how badly `lo_model` and
`hi_model` are misspecified. That guarantee holds without requiring the model to be correct,
which is a different promise than the credible intervals Chapters 9 and 12 built.

A Bayesian credible interval states the range containing $1 - \alpha$ of the posterior
probability mass, under the prior and likelihood chosen for that model. When the model is a good
description of the data, a credible interval and a conformal interval ask close to the same
question and tend to agree. When the model is wrong, such as a linear-Gaussian assumption applied
to the right-skewed latency data used throughout this section, only the conformal interval's
coverage promise still holds.

::: {.callout-important}
What conformal prediction gives up for that guarantee is the same thing NGBoost's predictive
interval gives up: it says nothing about how the fitted model itself would have looked under a
different training sample. A credible interval, when the model is right, answers that broader
question; a conformal interval answers only "will this specific interval contain the next point."
:::

Bayesian hyperparameter optimization, NGBoost, quantile regression, and conformal prediction
solve four separate problems that surface when boosting a model without a posterior: tuning it
efficiently, getting a calibrated predictive spread out of it, targeting a specific tail quantile
directly, and wrapping any of the above in an interval with a coverage guarantee that does not
depend on the model being right. None of the four requires the ensemble to have a posterior over
it in the first place.

## Where the field stands, honestly

A thinner, research-stage thread frames boosting itself as an approximate form of Bayesian
inference. The sequence of trees, under certain loss functions and with the right
regularization scheme, can be shown to trace out something resembling a posterior via
functional gradient descent.

Functional gradient descent means gradient descent carried out directly in the space of
functions (the ensemble itself is the object being optimized at each step) rather than in a
fixed set of parameters. That is the same underlying view of boosting Chapter 8 built the
mechanism from.

Separately, some work has extended BART's additive-tree framework to allow the
sequential-fitting style this chapter describes, rather than BART's fully parallel MCMC
updates.

None of this has produced a standard, widely adopted library the way PyMC has for general
Bayesian models, ArviZ has for model comparison, or `pymc-bart` has for BART itself. Treat it
as an active research direction rather than something ready to replace XGBoost or LightGBM in
production today.

If the goal is boosting's flexibility plus a full posterior over predictions, the honest answer
today is not a Bayesian version of XGBoost. It is BART, covered in full in Chapter 12.

:::{.callout-tip}
As a rule of thumb: if a project needs a full posterior over predictions with a boosting-like
ensemble, reach for BART (Chapter 12) rather than trying to graft Bayesian machinery onto
XGBoost or LightGBM. Bayesian hyperparameter optimization and NGBoost solve narrower problems
along the way, not this one.
:::

BART is not gradient boosting: its trees are fit by Gibbs-sampling-style backfitting against
the ensemble's residual, not by gradient descent on a loss function. Recall from Chapter 12
what that means in practice: cycling through the trees one at a time, refitting each one to
whatever residual the rest of the ensemble leaves behind.

But it delivers the same practical outcome a reader reaching for "Bayesian boosting" is usually
after: an additive-tree ensemble with a full posterior, mature tooling, and a track record on
structured tabular data close to what XGBoost and LightGBM handle.

Bayesian hyperparameter optimization and NGBoost solve two narrower problems along the way:
tuning a boosted model efficiently, and getting a calibrated predictive interval out of one,
without requiring a fundamentally different training procedure.

Between the three, most of what a practitioner reaching for "Bayesian boosting" needs is
covered, even without a single library that does all of it at once.

## References {.unnumbered}

::: {#refs}
:::
