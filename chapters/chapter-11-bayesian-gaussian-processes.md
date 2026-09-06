# Bayesian Nonlinear Regression: Gaussian Processes

A fitted smoothing spline, from Chapter 6, hands back one curve: the single best estimate of how
latency depends on load, given the requests observed so far. It says nothing about how much that
curve might have looked different if a different sample of requests had been logged instead.

A *Gaussian process*, or *GP*, answers the same curve-fitting question with a full distribution
over curves rather than one winner. That distributional answer means every prediction the GP
makes carries its own built-in statement of how much to trust it.

This chapter works through what that distribution is, how it updates once data arrives, and what
it costs to get.

1. *What does it mean for a distribution to be over functions, not numbers?*
   The idea sounds abstract until it is sampled from directly.
2. *What is a kernel, and what does it control?*
   The kernel is the one modeling choice a Gaussian process asks for, and it decides almost
   everything about the fitted curve's behavior.
3. *How does conditioning on data turn a prior over functions into a posterior?*
   For a Gaussian process, unlike most Bayesian models in this book, the answer has a closed
   form: no MCMC sampling required.
4. *What does the fitted model buy over a smoothing spline?*
   A credible band at every point, not just at the ones near the training data, and an honest
   account of where the model runs out of evidence.

## A prior over functions

The chart below draws five functions from a Gaussian process prior with a squared-exponential
(RBF) kernel, at four different length-scale settings.

::: {#fig-prior-samples}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-prior-samples.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Short length-scales produce curves that wiggle rapidly across the full range shown; long
length-scales produce curves that barely move across that same range. The slider steps through
that whole spectrum, from short to long.
:::

Imagine asking a hundred friends to sketch a smooth curve through five dots, without telling
them how. Each sketches it differently between the dots but agrees closely at the dots
themselves. A Gaussian process describes that whole bundle of possible sketches at once, not
just one.

Suppose a length-scale and a kernel shape are picked before a single data point is observed; the
next section defines both in full. That choice alone defines a *prior*, but it looks different
from the priors used earlier in this book.

Recall from Chapter 9 that Bayesian linear regression puts a prior over a handful of
coefficients, such as $\beta_0$ and $\beta_1$.

A Gaussian process prior sits directly over the space of functions consistent with the chosen
kernel instead: infinitely many candidate curves, not a handful of numbers. Drawing a sample
from this prior means drawing one entire function at once, not one number.

@fig-prior-samples shows this directly: at a short length-scale, the sampled functions wiggle
rapidly; at a long length-scale, they barely move across the same range. No data has been
observed yet in any of these curves.

This is the space of functions the model considers plausible before it sees anything. The
length-scale is the single knob that controls how large that space is.

:::{.callout-tip}
Sample a few curves from a candidate prior before fitting anything to observed data. If none of
them look like plausible latency curves, the kernel or length-scale needs to change before a
single observation is involved.
:::

## The kernel: the one modeling choice that matters

The chart below draws samples from an RBF kernel next to two Matern kernels at the same
length-scale, so the smoothness assumption each one encodes is visible directly rather than
described in the abstract.

::: {#fig-kernel-comparison}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-kernel-comparison.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The x-axis is scaled utilization and the y-axis is the sampled function value, same as the
previous figure. The slider swaps the kernel behind the four sampled curves while holding the
length-scale fixed: at the RBF setting shown here, every curve bends smoothly with no sharp
corners; switching to Matern nu=1.5 makes the curves visibly jagged; Matern nu=2.5 lands between
the two, smoother than nu=1.5 but not as smooth as the RBF curve.
:::

Think of two students sitting near each other in class: they tend to get similar grades, since
they share notes and study together. Students across the room barely influence each other's
scores. A kernel measures that kind of closeness between points: how much knowing one value
should shift a guess about a nearby one.

The *kernel*, or covariance function, defines how correlated the function's value at one point
is with its value at another. The most common choice, the *RBF kernel* (also called the squared-
exponential kernel), makes two points highly correlated when they are close together and lets
that correlation decay smoothly as the distance between them grows:

$$k(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right).$$

The *length-scale* $\ell$ sets how quickly that correlation decays: a short $\ell$ means even
nearby points are treated as loosely related, producing the wiggly samples in the previous
figure. A long $\ell$ means points far apart still move together, producing near-flat samples.

The *signal variance* $\sigma_f^2$ sets the overall vertical scale of the function's swings. In
other words, $\ell$ controls how fast the function is allowed to change, and $\sigma_f^2$
controls how far it is allowed to swing while changing.

The RBF kernel comes with an assumption worth naming directly: every function it can produce is
infinitely differentiable, meaning perfectly smooth at every scale, with no kinks anywhere. That
is a strong assumption.

The *Matern kernel* relaxes it with a smoothness parameter $\nu$: at $\nu = 1/2$ it produces
functions as rough as Brownian motion (a random walk that changes direction at every instant and
has no well-defined slope anywhere), at $\nu = 3/2$ or $\nu = 5/2$ it produces functions with a
controlled, finite amount of smoothness, and as $\nu \to \infty$ it converges to the RBF kernel.

@fig-kernel-comparison steps through all three: the perfectly smooth RBF curve, the visibly
rougher Matern nu=1.5 curve, and the Matern nu=2.5 curve between them.

Recall the saturation curve from Chapter 6: latency rises smoothly, without kinks, as
utilization approaches capacity. An RBF kernel's smoothness assumption fits that shape well.

A metric with sudden regime changes, such as a step function triggered by an autoscaling rule,
would be a better match for a rougher Matern kernel or an explicit change-point model. The
kernel is a direct claim about the shape of the process being modeled, worth choosing
deliberately rather than leaving at a library's default [@rasmussenwilliams2006].

::: {.callout-note}
The kernel is not a tuning knob to leave at its library default. It is a stated claim about
whether the process being modeled is smooth everywhere or allowed to bend sharply.
:::

## From prior to posterior: conditioning on data

The chart below fits this same model to the concurrent-load-versus-latency data from Chapter 6,
at growing sample sizes.

::: {#fig-posterior-fit}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-posterior-fit.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The x-axis is utilization (rho) and the y-axis is latency in milliseconds; the dotted line is
the true curve the observations were generated from. At the slider's starting point of five
observations, the posterior mean tracks that true curve closely through the smooth,
low-utilization part of the range, but the credible band widens sharply above roughly 80%
utilization, where the curve bends fastest and the five points give it the least to go on.
Dragging the slider toward sixty observations narrows the band further across the whole range,
though it never closes all the way at that same high-utilization edge.
:::

Think of guessing a friend's height from two friends whose heights you know, weighted by how
close in age each is to your friend. Closer ages pull the guess harder. Conditioning on data
works the same way: nearby observations pull a Gaussian process's prediction toward them, and
the pull fades with distance.

A closed-form answer is available here: a Gaussian process with a Gaussian observation noise
model is one of the few cases in Bayesian statistics where the posterior can be written down
directly. Stack the observed inputs into $X$ and the observed outputs into $y$.

The joint distribution of the observed values and any new prediction point is multivariate
normal by construction, since every finite collection of points from a Gaussian process is
multivariate normal by definition. Standard multivariate-normal conditioning then gives the
posterior mean and covariance at a new point $x_*$ in closed form:

$$\mu(x_*) = k(x_*, X)\left[K(X, X) + \sigma_n^2 I\right]^{-1} y$$

$$\sigma^2(x_*) = k(x_*, x_*) - k(x_*, X)\left[K(X, X) + \sigma_n^2 I\right]^{-1} k(X, x_*)$$

In other words, the posterior mean at a new point is a weighted average of the observed $y$
values, where the weights come from how correlated the new point is with each observed point
under the kernel. The posterior variance shrinks wherever nearby observations pin the function
down, and stays close to the prior variance wherever they do not.

A tiny worked example makes the formula concrete before trusting it to a plotted curve. Suppose
two requests have been observed: one at utilization $\rho = 0.30$ with latency $29$ ms, another
at $\rho = 0.40$ with latency $33$ ms. The kernel is RBF with $\ell = 0.1$ and $\sigma_f^2 = 400$,
plus a small observation-noise variance $\sigma_n^2 = 4$.

The kernel gives $k(0.30, 0.40) = 400 \cdot e^{-0.5} \approx 243$: substantial correlation, but
well short of the maximum possible value of $400$, since the two points sit one length-scale
apart.

Plugging both observations into the posterior-mean formula and querying at $\rho = 0.35$,
right between the two, gives a posterior mean of about $33.8$ ms, not the simple average of $29$
and $33$, which would be $31$.

This is worth sitting with: because the two training observations are correlated with each other
(through the same off-diagonal kernel entry just computed), the model does not treat them as two
independent votes to be averaged.

It treats the second, higher observation as carrying information the first one only partially
confirms, and the weighted combination in the formula can push the posterior mean slightly past
the range of the observed data.

Querying at $\rho = 0.80$, far from either observation relative to $\ell = 0.1$, gives a
posterior mean of about $0.01$, essentially reverted to the prior mean. The posterior variance
there is about $400$, matching the prior variance closely, since the model has no evidence at
all in that region.

This is the same shrinkage-toward-the-prior behavior @fig-extrapolation shows next, worked here
on the full simulated dataset rather than by hand on just two points. The chart fits this same
model to load data that never exceeded 75% utilization, then extends the prediction grid out to
97%.

::: {#fig-extrapolation}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-extrapolation.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Past the shaded boundary, the last utilization observed in this dataset, the posterior mean
flattens and drifts down instead of continuing the rising trend. The credible band widens to
several times its width just inside that boundary. Both show the model admitting it has no
evidence past that point, not a forecast that latency improves at higher load.
:::

No sampling algorithm is required to reach this answer; a matrix inversion produces it directly.
That inversion is also the computational bottleneck: it costs $O(n^3)$ in the number of
observations, which is why a plain Gaussian process becomes expensive past a few thousand
points.

::: {.callout-warning}
The $O(n^3)$ matrix inversion is the practical limit on a plain Gaussian process, not the model
itself. A fit that is fast at a few hundred rows can become impractical once the dataset grows
into the thousands.
:::

`scikit-learn`'s `GaussianProcessRegressor` implements this closed-form fit directly and is the
standard entry point for a dataset in that range:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF(length_scale=0.2, length_scale_bounds=(0.02, 2.0)) + WhiteKernel(1.0)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(rho.reshape(-1, 1), latency)
mean, std = gp.predict(grid, return_std=True)
```

:::{.callout-warning}
Drop the `WhiteKernel` term from the code above and the kernel matrix $K(X, X)$ can turn
near-singular whenever two training points sit closer together than the length-scale, making
the matrix inversion in the posterior formulas fail or return garbage. The noise term is not
just modeling honesty, it is what keeps that inversion numerically stable.
:::

@fig-posterior-fit fits this scikit-learn model to the concurrent-load-versus-latency data from
Chapter 6, at growing sample sizes. The *credible band* shown there is the functional
counterpart of the credible interval Chapter 9 built for a single coefficient: instead of one
interval, there is now one at every point along the curve.

## Choosing the length-scale: marginal likelihood, or a prior on the kernel itself

The chart below holds the length-scale fixed instead of letting the optimizer choose it.

::: {#fig-length-scale-fit}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-length-scale-fit.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

At the length-scale of 0.02 shown here, too short, the posterior mean (orange) chases individual
points. The credible band swells and pinches unevenly along the curve instead of growing
smoothly with distance from the data. Drag the slider toward the optimized value from the
previous figure to watch both settle down. Push it well past that point instead, and the mean
flattens into a near-straight line that misses the saturation bend near high utilization.
:::

The length-scale is not something to guess by eye. The standard approach, *marginal likelihood
maximization*, sometimes called *empirical Bayes* in this context, picks the length-scale (and
signal variance, and noise variance) that make the observed data most probable under the GP
prior. It does this by numerically maximizing the log marginal likelihood.

The term "marginal" refers to the unobserved function values being integrated out
algebraically. What is left is a likelihood that depends only on the length-scale, signal
variance, and noise variance, not on any specific sampled curve.

This is what happens inside `GaussianProcessRegressor.fit()` by default whenever the kernel's
bounds allow the optimizer to move; the code example above used it.

@fig-length-scale-fit shows what happens when that optimization step is skipped and the
length-scale is instead held fixed: too short, and the posterior mean chases individual noisy
points; too long, and it smooths straight through the saturation curve's bend near high
utilization.

Marginal likelihood maximization returns a single best length-scale, a point estimate of a
hyperparameter, which is a mild inconsistency in an otherwise fully Bayesian model.

A stricter Bayesian treatment puts a prior on the length-scale itself and samples from its
posterior along with everything else, using a probabilistic programming library such as PyMC
rather than `scikit-learn`'s closed-form fit. A prior on the length-scale breaks the closed form
this section relied on, which is why that stricter treatment needs a sampler instead.

In practice, marginal likelihood maximization is close enough for most tabular problems: it is
the default in essentially every GP library for this reason. Treating the length-scale as fully
uncertain matters most when the dataset is small enough that different plausible length-scales
would lead to meaningfully different conclusions.

The optimizer behind that maximization is not guaranteed to land on the best peak. The marginal
likelihood surface can have more than one local maximum: commonly a short-length-scale peak that
treats most of the variation in the data as signal, and a long-length-scale peak that explains
the same variation away as observation noise.

A poorly chosen starting point can leave the fit stuck at the worse of the two
[@rasmussenwilliams2006].

`GaussianProcessRegressor`'s `n_restarts_optimizer` argument, left at its default of zero in the
code example above, reruns the search from several randomly chosen starting points and keeps the
best result. It is worth turning on whenever a fitted length-scale looks suspicious, rather than
trusting a single optimization run.

::: {.callout-tip}
Leave `n_restarts_optimizer` at zero only for a first pass. Once a fit is trusted for anything
beyond exploration, rerun it with several restarts and confirm the length-scale does not move.
:::

## Full Bayesian hyperparameters: sampling the length-scale instead of optimizing it

The chart below fits the same RBF-plus-noise kernel to eight concurrent-load observations
twice: once the way every fit so far in this chapter has, by optimizing the length-scale to a
single best value, and once by placing a prior on the length-scale, signal variance, and noise,
then sampling their posterior with PyMC.

::: {#fig-full-bayes-hyperparams}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-full-bayes-hyperparams.html" width="100%"
        height="480" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Left: both fits predict a similar posterior mean, but the full-Bayesian band (blue) runs about
1.6 times wider than the point-estimate band (orange) near the observed points and close to
twice as wide at the highest utilization shown. Right: the sampled length-scale posterior
spreads from roughly 0.09 to 0.41, with a mean of 0.22, while the point estimate (dashed
vertical line) lands at 0.15, near the lower edge of that spread rather than at its center.
:::

Picture two analysts handed the same eight readings. One reports the single trend line that
best explains them and states a margin of error around that one line. The other is not sure the
trend's steepness is the one number the first analyst found, so she carries several
plausible steepnesses forward and reports a wider, more honest margin as a result. Both analysts
have the same eight readings. The second is doing the job the PyMC fit below does for the
length-scale.

A prior on the length-scale, signal variance, and noise means none of the three collapses to a
single number before the posterior is computed. `pm.gp.Marginal` still exploits the same
closed-form conditioning this chapter opened with, since the observation likelihood is still
Gaussian, but now that conditioning happens conditional on each sampled hyperparameter draw
rather than on one optimized value, and NUTS explores the resulting three-dimensional
hyperparameter posterior directly:

```python
import pymc as pm

coords = {"obs_id": np.arange(len(rho))}
with pm.Model(coords=coords) as full_bayes_model:
    rho_data = pm.Data("rho_data", rho, dims="obs_id")
    ell = pm.Gamma("ell", alpha=2.0, beta=8.0)
    eta = pm.HalfNormal("eta", sigma=50.0)
    sigma_n = pm.HalfNormal("sigma_n", sigma=10.0)
    cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ls=ell)
    gp = pm.gp.Marginal(cov_func=cov)
    gp.marginal_likelihood("latency_obs", X=rho_data[:, None], y=latency, sigma=sigma_n)
    trace = pm.sample(1000, tune=1000, chains=4, target_accept=0.95)

    f_pred = gp.conditional("f_pred", Xnew=grid[:, None])
    pred = pm.sample_posterior_predictive(trace, var_names=["f_pred"])
```

`gp.marginal_likelihood` plays the same role here that `.fit()` played for `scikit-learn`
earlier, except the length-scale, signal variance, and noise passed into it are now random
variables with priors instead of numbers an optimizer chose. `gp.conditional`, together with
`pm.sample_posterior_predictive`, then produces a predictive distribution that carries the
hyperparameters' own uncertainty forward into every prediction, rather than conditioning on one
fixed setting the way `.predict()` does.

The `Gamma(alpha=2, ...)` shape for the length-scale prior above is not specific to this
example: it is a common choice across the Gaussian process literature (documented in
Betancourt's Gaussian process case studies and PyMC's own example gallery) because it keeps
positive support while pulling the prior away from implausibly small length-scales, a mode that
would otherwise let the fit chase noise.

@fig-full-bayes-hyperparams shows what that costs and buys on this eight-point dataset. The
length-scale posterior's 95% interval runs from about 0.09 to 0.41, comfortably containing the
point estimate of 0.15 near its lower end but also reaching more than twice as far. The noise
posterior tells the same story: its 95% interval, roughly 1.8 ms to 14.1 ms, comfortably
contains the 3 ms noise this dataset was generated with, but says nothing sharper than that
range with only eight points to work from.

Eight points is a small-N situation, the case the previous section flagged as the one where the
difference matters most. `scikit-learn`'s optimizer has to commit to a single
length-scale even though the data barely constrains which one is right; PyMC's posterior instead
reports that the data barely constrains it, and widens every downstream prediction to match.
Neither fit is wrong. The point estimate is a reasonable summary of a distribution the full
Bayesian fit makes visible in full.

That visibility costs compute: sampling a three-parameter posterior with NUTS took several
seconds on eight points in the run behind this figure, against a fraction of a second for the
optimizer's single search. On the hundreds-to-low-thousands datasets this book has worked with
throughout, where the marginal likelihood surface has enough data to pin the length-scale down
tightly, that cost buys little the point estimate did not report on its own. It earns its keep
specifically when the dataset is small enough, or the decision resting on the fit is high-stakes
enough, that the gap between "here is one plausible length-scale" and "here is how much the data
lets that length-scale vary" changes what a reader should conclude.

::: {.callout-tip}
Run the full Bayesian fit on a handful of representative small subsets before committing to it
for an entire pipeline. If the point-estimate and full-posterior predictive bands stay close
across those subsets, the extra sampling cost is not buying anything the optimizer did not
provide on its own.
:::

## What the credible band buys: honest extrapolation

@fig-extrapolation, shown earlier in this chapter, is the situation Chapter 6 flagged as
dangerous for an alerting threshold: the region of highest interest is the region with the
least data.

Past the observed range, the credible band widens sharply, which is the correct behavior: the
kernel's correlation between a query point and the training data decays with distance.

Far enough past the last observation, the posterior reverts toward the prior: wide and
uninformative. A smoothing spline's fitted curve, by contrast, keeps extrapolating along
whatever polynomial trend the boundary basis functions imply, with no built-in signal that the
curve has left the region it was trained on.

Neither method can conjure information about untested load levels out of nothing. The difference
is that the Gaussian process says so, in a number that can be checked against an alert
threshold, and the spline does not.

:::{.callout-important}
A widening credible band past the observed range means the model is reporting, in a number that
can be checked against an alert threshold, that it has run out of evidence, not that the model
is malfunctioning.
:::

## More than one input at a time

Every figure in this chapter used a single predictor, utilization, to keep the plots readable
as two-dimensional curves. Nothing about the kernel or the posterior formula requires that.
Chapter 4's OLS model predicted latency from payload size and concurrent-request count together.

A Gaussian process handles the same two-predictor setup by measuring distance between points in
that two-dimensional input space instead of along a single line, typically with a separate
length-scale per input dimension (an "automatic relevance determination" kernel, in the
terminology `scikit-learn` and most other GP libraries use).

A short length-scale on one dimension and a long length-scale on another is itself a form of
variable-importance signal: the model has learned that the function changes quickly along the
first input and barely reacts to the second, without anyone hand-coding that conclusion in
advance.

:::{.callout-tip}
An automatic-relevance-determination kernel only makes length-scales comparable across
dimensions when every input is standardized first. Fit the same kernel to payload size in raw
bytes and utilization on a 0-1 scale, and the length-scales say more about the units chosen than
about which predictor matters.
:::

::: {#fig-ard-length-scales}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-ard-length-scales.html" width="100%" height="480"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Two predictors, payload size and concurrent-request utilization, on inputs standardized to the
same scale so the two length-scales are directly comparable. Concurrent-request utilization
fits a length-scale under 1, since latency swings sharply as utilization approaches capacity.
Payload size fits a length-scale over 20, since latency barely moves across its whole observed
range once utilization is accounted for.
:::

@fig-ard-length-scales makes the point concrete: an automatic-relevance-determination kernel
fit to both predictors at once returns a length-scale twenty-five times longer for payload size
than for concurrent-request utilization, without anyone telling the model in advance which
predictor mattered more.

The same $O(n^3)$ cost applies regardless of how many input dimensions there are, since it
depends only on the number of observations, not the number of predictors. That is one reason a
GP scales more gracefully with feature count than it does with dataset size.

:::{.callout-note}
After fitting a multi-dimensional GP, check the optimized length-scales against each input's
own range before reading anything else. A length-scale several times wider than the range of
its input is a cheap early sign that dimension is not doing much work in the fit.
:::

## Periodic structure: composing kernels for signals that repeat

The chart below fits two kernels to three days of hourly latency and asks both to predict the
fourth, unobserved day.

::: {#fig-periodic-kernel}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-periodic-kernel.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The dotted line is the true mean latency behind the data: a slow four-day drift plus a
business-hours peak near 2pm each day. Markers show the first three days, the only data either
model sees; the shaded region marks the fourth day, held out. The RBF-only fit (orange, dashed)
never reproduces the afternoon bump: its mean drifts downward across the whole fourth day, and
the single highest point it does reach in that window falls at 2am, half a cycle from the true
2pm peak, with a credible band about three times as wide as the alternative. The RBF-plus-periodic
fit (blue) tracks the true curve through the fourth day, peak included, and closes most of the
way in on that day's latency values.
:::

A commuter who has watched traffic build for three straight afternoons does not need a fourth
afternoon to guess when tomorrow's rush hour starts. An RBF kernel, on its own, cannot make that
same guess: it only measures how close two points sit on the input axis, so hour 38 and hour 62,
both mid-afternoon on different days, look no more related to it than hour 38 and hour 50.

Every kernel used so far in this chapter, RBF and Matern alike, shares that property: correlation
depends only on distance between two points, with no notion that some pairs of points sit in the
same phase of a repeating cycle. Chapter 7's rollback classifier treats hour of day as a feature
carrying signal on its own; a GP asked to model a metric with that same daily rhythm needs a
kernel built to notice it.

The *periodic kernel*, most commonly the `ExpSineSquared` form, supplies that missing structure
directly:

$$k_{\text{periodic}}(x, x') = \sigma_f^2 \exp\!\left(-\frac{2 \sin^2\!\left(\pi |x - x'| / p\right)}{\ell^2}\right).$$

The period $p$ sets how often the pattern repeats (24 hours for a daily cycle, 168 for a weekly
one); the length-scale $\ell$ sets how much the shape is allowed to drift from one cycle to the
next, the periodic kernel's counterpart to the RBF length-scale from earlier in this chapter.

A periodic kernel alone would treat every day as a rescaled copy of every other day forever, with
no way to represent the slow four-day drift in the figure above. Kernels compose: adding a
periodic term to an RBF term lets one part of the sum absorb the smooth trend while the other
absorbs the daily reshaping, and the two are estimated together from the same data. Rasmussen and
Williams build a kernel this way, a smooth trend term added to a periodic term plus further terms
for medium-term drift and noise, to model the rise and seasonal fall in the Mauna Loa atmospheric
carbon dioxide record, one of the standard worked examples of kernel composition in the Gaussian
process literature [@rasmussenwilliams2006].

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel

kernel = (
    RBF(length_scale=48.0, length_scale_bounds=(10.0, 200.0))
    + ExpSineSquared(length_scale=1.0, periodicity=24.0, periodicity_bounds="fixed")
    + WhiteKernel(1.0)
)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=6)
gp.fit(hours.reshape(-1, 1), latency)
```

Fixing `periodicity_bounds="fixed"` at 24 hours instead of letting the optimizer search for it is
deliberate. The length-scale in earlier sections came from data because nothing else pins it
down; a metric's period, by contrast, is usually known ahead of time from the process generating
it (a day, a week, a billing cycle), and three or four observed cycles is a thin basis for the
same marginal-likelihood search to also discover that period from noise. Fix what is known ahead
of time and let the optimizer spend its search on what is not.

@fig-periodic-kernel shows the payoff on the held-out fourth day: the RBF-only fit's root-mean-
squared error against the true curve is about 13.8 ms, against about 1.2 ms for the fit with the
periodic term added, and the RBF-only credible band stays roughly three times as wide across that
same day. Both models see the identical three days of data; only the kernel's assumptions differ.

:::{.callout-tip}
Add a periodic term whenever a metric is known to repeat on a fixed cycle, hour of day or day of
week, rather than asking a plain RBF or Matern kernel to infer that structure from a handful of
observed cycles. Fix the period at its known value and let the optimizer fit the length-scale and
noise around it.
:::

## When the outcome is a label: Gaussian process classification

The chart below fits a Gaussian process classifier to ninety synthetic request-payload
observations, each labeled as flagged or not flagged by an anomaly filter, and compares it
against a plain logistic regression fit to the same data.

::: {#fig-gp-classification-boundary}
```{=html}
<iframe src="../_generated/chapter-bayes-gp-fig-classification-boundary.html" width="100%"
        height="520" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The x-axis is request payload size in kilobytes and the y-axis is the fitted probability of
being flagged as anomalous; markers near the bottom and top show individual clear (0) and
flagged (1) observations, jittered so overlapping points stay visible. The true flag rate
(dotted) peaks near both edges of the range and drops close to zero in the middle. The GP
posterior mean (blue) tracks that same shape, rising back above 0.65 at both edges after
falling under 0.10 near the middle, with the credible band (shaded) widening wherever fewer
observations sit nearby. The logistic regression fit (orange, dashed) stays close to flat across
the whole range, moving only from about 0.26 to 0.38, since one coefficient cannot make the flag
rate rise on both sides of a safe middle range at once.
:::

A lookout watching one corridor for trouble notices someone slipping in near the entrance and
someone else causing a commotion near the far end, with a quiet stretch in between where nothing
happens. A rule that only tracks how far along the corridor something is, and moves its alarm
threshold one direction as that distance grows, cannot describe both trouble spots without also
raising the alarm through the quiet stretch between them. A logistic regression's single
coefficient is that kind of one-direction rule.

A Gaussian process classifier keeps the same latent function this chapter has built up all
along, a smooth curve drawn from a GP prior and reshaped by conditioning on data, but routes
that curve through a link function before it reaches the observed outcome instead of reading it
off directly. The sigmoid link squeezes the latent function's unbounded output into a valid
probability between 0 and 1, and a Bernoulli likelihood then treats each observed flag as a coin
flip weighted by that probability.

The engineering consequence shows up directly in the fitted logistic regression coefficient
behind the orange curve: 0.030, close enough to zero that the model barely distinguishes any
payload size from any other. Averaged across two flagged regions pulling in opposite directions,
a single-direction rule settles for a probability that hovers near the overall flag rate
everywhere, useless for either clearing traffic with confidence or catching it. A filter built on
that fit would let small and large payloads through at close to the same rate as the safe middle
range, defeating the purpose of screening on payload size at all. The GP classifier's latent
function is free to bend twice, so it recovers both flagged regions and the safe range between
them from the same ninety labels.

Fitting this model means giving up the closed form the rest of this chapter relied on. A
Bernoulli likelihood is not Gaussian, so `pm.gp.Marginal`'s algebraic shortcut, integrating the
latent function out in closed form, no longer applies. `pm.gp.Latent` samples the latent function
directly instead, and NUTS explores its posterior jointly with the length-scale:

```python
import pymc as pm

coords = {"obs_id": np.arange(len(payload_kb))}
with pm.Model(coords=coords) as gp_classifier:
    payload_data = pm.Data("payload_data", payload_scaled, dims="obs_id")
    ell = pm.Gamma("ell", alpha=2.0, beta=3.0)
    cov = pm.gp.cov.ExpQuad(1, ls=ell)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=payload_data[:, None], dims="obs_id")
    p = pm.Deterministic("p", pm.math.invlogit(f), dims="obs_id")
    pm.Bernoulli("flagged_obs", p=p, observed=flagged, dims="obs_id")
    trace = pm.sample(1000, tune=1000, chains=4, target_accept=0.9)

    f_pred = gp.conditional("f_pred", Xnew=grid_scaled[:, None])
    pred = pm.sample_posterior_predictive(trace, var_names=["f_pred"])
```

`pm.math.invlogit` is the sigmoid link, applied to the sampled latent function `f` rather than
to a linear combination of coefficients the way ordinary logistic regression applies it. Once
sampling finishes, `gp.conditional` extends that same latent function to new payload sizes, and
`expit` on the resulting samples turns them back into probabilities for the curve in
@fig-gp-classification-boundary.

The length-scale posterior for this fit comes out short, a mean of about 0.40 on the
standardized payload scale, short enough to let the fitted curve bend twice within the
observed range instead of settling on one smooth trend. A longer length-scale prior would push
the fit back toward something closer to the logistic regression curve, unable to represent both
flagged regions at once. Formally, the model places a GP prior on a latent function and a
Bernoulli likelihood on top of a sigmoid-transformed draw from it:

$$f \sim \mathcal{GP}(0, k(x, x')), \qquad p(x) = \sigma(f(x)), \qquad y_i \sim
\text{Bernoulli}(p(x_i)).$$

Every kernel this chapter has covered, RBF, Matern, or a periodic-plus-trend composite, plugs
into $k(x, x')$ here the same way it plugged into the regression posterior earlier; only the
likelihood on top of the latent function changes.

::: {.callout-note}
If the credible band around a fitted classification boundary looks too narrow in a region far
from any observation, adding a linear kernel term the way this chapter composes RBF and
periodic kernels earlier is a standard fix: it lets the model's uncertainty grow honestly with
distance from the data instead of collapsing back toward the prior mean too quickly on the
probability scale.
:::

## When the closed form runs out

Everything in this chapter relied on a Gaussian process with Gaussian observation noise, which
is the one case with a closed-form posterior.

Two situations break that closed form and push toward the tools Chapter 9 introduces for other
models: a non-Gaussian likelihood, the classification case the previous section just walked
through in full, and a dataset too large for the $O(n^3)$ matrix inversion to finish in
reasonable time.

Both push toward approximate inference: variational methods that summarize the posterior with a
simpler distribution, or sparse GP approximations that summarize the training data itself with a
smaller set of representative "inducing points." `GPyTorch` and `GPflow` are the two libraries
most commonly reached for once a problem crosses either of those lines.

Both implement sparse and variational GP methods that scale well past the point where
`scikit-learn`'s closed-form implementation becomes impractical, typically somewhere in the
range of several thousand to tens of thousands of training points, depending on available
memory.

::: {.callout-note}
A GP can earn its keep as a covariance structure buried inside a larger model, not only as the
direct regression surface. A smoothly time-varying incident rate, for instance, can ride inside
a Poisson GLM as a latent log-rate:
`rate_gp = pm.gp.Latent(cov_func=...)`, `log_rate = rate_gp.prior("log_rate", X=time_bins)`, then
`pm.Poisson("incidents", mu=pm.math.exp(log_rate), observed=incident_counts)`. The GP supplies a
covariance structure over the latent quantity being modeled, a changepoint-free alternative to
hand-coding a step change in the rate, and still needs `pm.gp.Latent` rather than
`pm.gp.Marginal`, since a Poisson likelihood is as non-Gaussian as a Bernoulli one.
:::

::: {.callout-note}
When inputs sit on a regular multi-dimensional grid, such as latency broken out by region and
hour of day, a Kronecker-structured GP (`pm.gp.LatentKron`, built from one covariance function
per axis) scales past the plain $O(n^3)$ cost flagged earlier in this chapter without falling
back to sparse or variational approximation. It exploits the grid's separability directly, at
the cost of applying only when every combination of axis values is observed.
:::

For a dataset in the hundreds or low thousands, which covers most of the tabular, single-service
metrics this book has worked with, the closed-form fit this chapter walked through is enough.

:::{.callout-tip}
Reach for `GPyTorch` or `GPflow` only once the dataset size or the likelihood forces the issue.
For the tabular sizes common in this book, the closed-form `scikit-learn` fit is easier to set
up and easier to debug.
:::
