# Moving Beyond Linearity: Regression Splines

Chapter 4 fit a straight line through payload size and checkout-API latency, and the line
worked: the relationship was close to linear across the range of payload sizes the service saw.

Concurrent request load is a different story. At low load, adding a few more simultaneous
requests barely moves latency. Past some point, the same increase in load produces a much
sharper rise, because the service is running out of capacity to process requests as fast as
they arrive. A straight line cannot represent both of those behaviors with one slope.

This chapter works through the tools that handle a curved relationship without abandoning
regression altogether:

1. *Why does a linear fit fail here, specifically?*
   The shape of the failure points directly at what kind of flexibility the model needs.
2. *Can a higher-degree polynomial just fix it?*
   Sometimes, but polynomials misbehave in a specific and predictable way near the edges of the
   data.
3. *What is a basis function, and why does almost every fix in this chapter use one?*
   Step functions, polynomials, and splines are all the same idea in different clothing.
4. *What is a regression spline, and how many knots does it need?*
   Piecewise polynomials joined smoothly at chosen points, with a meaningful trade-off in how
   many points to choose.
5. *What does a smoothing spline buy that a hand-placed knot cannot?*
   A penalty term, the same idea Chapter 4 used for Lasso and Ridge, applied to curve
   roughness instead of coefficient size.
6. *Is there a way to smooth a curve without choosing knots or a penalty at all?*
   Local regression fits a small model at every point instead of one model for the whole range,
   at a cost that shows up when it comes time to serve predictions.
7. *What happens once a second predictor needs the same curve-fitting treatment as the first?*
   A generalized additive model sums one function per predictor, and a partial residual plot
   checks whether each function is earning its keep.

## Why a straight line breaks down: concurrent load and latency

Queueing theory gives a simple, well-established model for what happens as a server approaches
its capacity: mean latency grows roughly in proportion to $1 / (1 - \rho)$, where $\rho$ is
utilization, the ratio of incoming load to the service's processing capacity.

::: {#fig-linear-misfit}
```{=html}
<iframe src="../_generated/chapter-splines-fig-linear-misfit.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

An OLS line captures the general upward trend but misses both ends: it underpredicts latency at
low load, dipping toward zero while observed latency holds near 20 ms, then overshoots badly as
load climbs toward the top of the range shown, precisely the region where an accurate prediction
matters most for an alerting threshold.
:::

Think of a coffee shop with one register: with few customers, each order gets served quickly
and one more customer barely lengthens the line. Once the shop gets busy enough, each new
customer adds a noticeably longer wait, since the register is running out of room to keep up.

@fig-linear-misfit forces an OLS line through simulated latency data generated from this
saturation curve, confirming the mismatch the queueing-theory relationship predicts.

This is not a linear relationship: it stays nearly flat for most of the range and then rises
steeply as load approaches capacity. At $\rho = 0.5$, the service is running at half capacity
and latency stays close to its baseline.

At $\rho = 0.9$, the denominator has shrunk to 0.1, and latency runs roughly five times its
$\rho = 0.5$ level. At $\rho = 0.99$, the denominator shrinks to 0.01, and latency runs roughly
ten times higher again, about fifty times the $\rho = 0.5$ baseline.

:::{.callout-tip}
A relationship that stays flat, then bends sharply upward past some threshold, is a signature
of a system approaching a capacity limit. A straight line will always underfit one end of that
curve.
:::

## Polynomial regression and its instability at the edges

::: {#fig-poly-degree}
```{=html}
<iframe src="../_generated/chapter-splines-fig-poly-degree.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The fit improves through degree 4 to 6, tracking the saturation curve closely, then starts
wobbling through the middle of the range and swinging near the high-load boundary as degree
keeps rising: the Runge phenomenon in practice, worst in precisely the region an operator
cares about most.
:::

Picture a piece of wire bent to touch as many dots on a page as possible: bending it enough to
hit every dot near the middle can send the ends whipping away from the true shape, even while
the middle looks great. Polynomials with a high degree can do the same thing to a curve.

The most direct fix is to add polynomial terms: $Y = \beta_0 + \beta_1 X + \beta_2 X^2 +
\beta_3 X^3 + \dots + \varepsilon$. A quadratic or cubic term lets the fitted curve bend, and for
many datasets a low-degree polynomial captures the curvature well.

It is natural to expect that adding still more degree terms should only improve the fit
further. Surprisingly, that is not what happens.

As the polynomial degree rises, the fitted curve starts to oscillate wildly near the boundaries
of the data, even while fitting the interior points almost perfectly. This is known as the
Runge phenomenon, after Carl Runge's 1901 demonstration that interpolating a smooth function at
evenly spaced points with a high-degree polynomial can diverge near the edges of the interval.

That divergence happens even though the underlying function itself is well-behaved [@runge1901].

@fig-poly-degree fits polynomials of increasing degree to a small sample of the load-latency
data and shows this directly: degree 4 to 6 tracks the saturation curve reasonably well, but by
degree 14 the fitted curve is chasing individual points, swinging above and below the true
relationship through the middle of the range and again near the high-load edge, the region an
operator cares about most. A small sample size makes the swings easier to see; the same
instability is present with more data, just harder to spot at a glance.

:::{.callout-warning}
A lower training error at a higher polynomial degree is not evidence of a better fit. Check the
curve near the edges of the data before trusting a high-degree polynomial, since that is where
the Runge phenomenon shows up first.
:::

## Basis functions: the idea underneath every fix in this chapter

::: {#fig-basis-functions}
```{=html}
<iframe src="../_generated/chapter-splines-fig-basis-functions.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The polynomial basis grows without bound and covers the whole range at once, so a change
anywhere reshapes the curve everywhere. The step-function basis is hard-edged flat boxes, and
the cubic spline basis is smooth, overlapping bumps, each nonzero over only part of the range,
the property that lets a spline bend locally without disturbing the fit elsewhere.
:::

Think of building a curved wall out of straight LEGO bricks stacked at different angles: no
single brick bends, but the right combination of simple, straight pieces can approximate almost
any shape. A basis function is one of those simple building-block pieces.

Every technique in this chapter, polynomials included, is a special case of the same general
strategy: instead of predicting $Y$ directly from $X$, transform $X$ through a set of *basis
functions* $b_1(X), b_2(X), \dots, b_K(X)$ and fit a linear model on the transformed inputs,

$$Y = \beta_0 + \beta_1 b_1(X) + \beta_2 b_2(X) + \dots + \beta_K b_K(X) + \varepsilon.$$

A polynomial regression is the special case where $b_k(X) = X^k$. A *step function* is the
special case where each $b_k(X)$ is an indicator variable for whether $X$ falls in a particular
range, for instance whether concurrent load falls between 60% and 70% of capacity.

Step functions are easy to interpret and immune to the boundary instability polynomials show,
since each region is fit independently. But they produce a fitted curve with hard jumps at every
cut point, not the smooth saturation curve the underlying queueing behavior has.

:::{.callout-note}
Step functions fit best when the underlying relationship has a true jump, a pricing tier or a
rate limit that kicks in above a threshold. Forcing hard edges onto a smooth process like
queueing latency trades away the shape the data has.
:::

A regression spline's own basis functions, covered next, are a third option: instead of a global
power series or a set of hard-edged boxes, each one is a smooth bump that is nonzero over only
part of the range, what mathematicians call its support.

@fig-basis-functions plots the individual basis functions for each family side by side: the raw
ingredients that go into a fit, separate from the fitted curve they eventually produce.

The right choice of basis functions is what separates a fit that tracks the data's shape from
one that merely bends in its general direction.

::: {.callout-tip}
The shape of a basis function is a direct clue to how a fit built from it will misbehave. A
basis that spans the whole range, like a polynomial term, lets one distant point tilt the fit
everywhere. A basis with local support, like a spline's bumps, confines that influence to a
neighborhood around the point.
:::

The shape of the spline basis functions is the reason a regression spline can bend sharply near
the saturation point without the polynomial's boundary instability: each bump only influences
the fit in its own local region. A change near one knot does not ripple across the entire curve
the way a change to a degree-9 polynomial's coefficients does.

## Regression splines: piecewise polynomials joined at knots

::: {#fig-knot-count}
```{=html}
<iframe src="../_generated/chapter-splines-fig-knot-count.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Three or four knots track the saturation curve well; a dozen knots start chasing noise between
individual points, the same overfitting pattern a high-degree polynomial shows, without the
boundary oscillation.
:::

A *regression spline* fits a separate low-degree polynomial, typically cubic, within each
region of $X$, then constrains those pieces to join smoothly at boundary points called *knots*.

For a cubic spline, "smoothly" means the function value, its first derivative (the curve's
slope), and its second derivative (how sharply it is bending) all match at each knot, so the
fitted curve has no visible seam even though it is built from separate pieces.

The number and placement of knots controls the trade-off directly. Too few knots and the spline
cannot bend enough to follow the saturation curve, similar to a low-degree polynomial.

Too many knots and the spline starts fitting noise in the data between nearby points, similar to
a high-degree polynomial. The spline does not show the boundary oscillation problem, though,
since each piece only has to behave well within its own region.

@fig-knot-count fits regression splines with a growing number of evenly spaced knots to the
load-latency data.

Two decisions are hiding inside "the number and placement of knots": where to put them, and how
many to use. Placement is usually the easier call: space the knots at even quantiles of the
predictor, the same percentile idea from Chapter 1 generalized to any fraction of the data
rather than just the 90th or 99th, instead of at even intervals of its range.

That puts more knots where the data is dense and fewer where it is sparse, so no single knot
ends up governing a region with only a handful of observations.

:::{.callout-tip}
Place knots at even quantiles of the predictor, not at even intervals of its range. That puts
more knots where the data is dense and avoids a knot that has to govern a region with only a
handful of observations.
:::

The count is the harder call. It follows the same rule Chapter 5 established for any tuning
parameter: try a range of knot counts, cross-validate each one, and pick the count with the
lowest cross-validated error rather than the one that looks best on the training fit.

## Natural cubic splines: constraining the boundaries

A cubic regression spline is still a cubic polynomial in its two outermost regions, which means
it inherits some of the same boundary instability polynomials show, particularly if the last
knot sits close to the edge of the observed data.

A *natural cubic spline* adds one more constraint: the function must be linear, not merely
cubic, beyond the two boundary knots. This trades a small amount of flexibility at the extremes
for a curve that behaves predictably when extrapolating.

That matters directly here, since the highest-load observations in any given hour of traffic
are the region where the next hour's traffic might land slightly beyond what has been observed
so far.

::: {#fig-natural-boundary}
```{=html}
<iframe src="../_generated/chapter-splines-fig-natural-boundary.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Both curves are fit to the same knots and agree everywhere the data covers. Past the last
knot, the unconstrained cubic spline keeps curving on its own terms; the natural cubic spline
continues in a straight line instead, matching the value and slope it held at the boundary.
:::

@fig-natural-boundary fits both curves to load-latency data cut off at a utilization of 0.85,
standing in for a fitting run that has not yet seen the busiest traffic, then extends both past
that cutoff. They track each other up to the last knot, since both interpolate the same four
points; past it, the unconstrained spline is free to keep bending however the shape of its last
piece dictates, while the natural spline holds to the straight line implied by its boundary
condition. Neither is a substitute for more data. The natural spline is simply the more
conservative choice for the region beyond it.

::: {.callout-important}
A model that swings unpredictably past the edge of its training data is a poor choice for the
region an operator is most likely to page someone about. The natural boundary constraint exists
specifically to prevent that kind of swing.
:::

## Smoothing splines: a penalty instead of a knot count

::: {#fig-smoothing-lambda}
```{=html}
<iframe src="../_generated/chapter-splines-fig-smoothing-lambda.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A small lambda produces a wiggly curve that chases individual points. A large lambda smooths
that noise-chasing away, leaving a simple curve that keeps the gentle bend of the saturation
shape all the way to the high-load edge.
:::

Choosing how many knots to use, and where to place them, is itself a decision that can be made
or avoided. Imagine a strict art teacher who lets a sketch curve freely but takes off points for
every sharp, jagged wiggle. A smoothing spline judges a curve the same way: it rewards a close
fit to the data but charges a penalty for bending too sharply too often.

A *smoothing spline* sidesteps knot selection by placing a knot at essentially every unique data
point and then controlling flexibility with a penalty on roughness, minimizing

$$\sum_{i=1}^{n} (y_i - f(x_i))^2 + \lambda \int f''(t)^2 \, dt$$

where the first term is the usual sum of squared errors and the second term penalizes how much
the fitted function curves, since a large second derivative means the function is bending
sharply.

This is the same idea Chapter 4 used for Lasso and Ridge regression: a tuning parameter
$\lambda$ trades fit against complexity, except here the penalty targets the curvature of an
entire function rather than the size of a fixed set of coefficients.

@fig-smoothing-lambda shows the smoothing spline at several values of $\lambda$, tracing the
path from near-interpolation of every point at small $\lambda$ to a smooth, simple curve at
large $\lambda$ that keeps the saturation bend the data itself carries. The penalty targets
how sharply the function bends from point to point. Push $\lambda$ high enough and the fit
settles at a single global cubic curve, the least flexible shape this basis can take, and
that shape still rises to track the saturation curve at high utilization.

Recall from Chapter 5's discussion of cross-validation that a tuning parameter like $\lambda$
should not be picked by eye. In practice, $\lambda$ is chosen the same way the regularization
strength for Lasso or Ridge was chosen: by cross-validating over a range of candidate values and
picking the one that minimizes estimated test error, not training error.

## Local regression: smoothing without choosing knots

::: {#fig-loess-span}
```{=html}
<iframe src="../_generated/chapter-splines-fig-loess-span.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

At a span of 0.08, the local fit reaches roughly 270 ms at the high-load edge, close behind the
saturation curve; widen the span to 0.7 and the same edge tops out near 170 ms, more than 100 ms
short, because the wider window is now borrowing points from the flatter middle of the range to
compute the fit at the edge.
:::

Every technique so far in this chapter commits to something before the curve gets fit: a knot
count, a set of knot locations, or a roughness penalty. Local regression, usually called LOESS,
skips that commitment. Instead of fitting one curve to the whole range at once, it fits a small,
disposable regression near every point where a prediction is wanted, using only the data closest
to that point, then discards the small fit and moves on to the next point.

Picture reading a street map by covering everything outside a two-block radius of wherever you
stand: what sits ten blocks away does not help cross the next intersection, and it may not even
belong to the same neighborhood. Local regression treats a scatter plot the same way. It answers
"what does latency look like right here" using only the load values nearby, then answers the same
question ten load-levels over with a different local neighborhood built from scratch.

Local regression's cost in production runs opposite to every spline in this chapter. A regression
spline's basis functions get evaluated once against a fixed set of coefficients, so the model that
ships to a serving layer is a short list of numbers: cheap to store, cheap to evaluate at 2 a.m.
Local regression has no such list to ship. Every prediction re-solves a small weighted regression
against a slice of the training data, so the training data itself has to sit somewhere reachable
at prediction time, in addition to sitting there during fitting. A spline can run on a device with
no memory of the data that trained it. Local regression, by construction, keeps that data on hand
for the life of the deployment.

Fitting a local regression at a query point $x_0$ follows four steps:

1. Pick a *span*: the fraction of the data that counts as nearby, for instance 0.3, meaning the
   closest 30% of points by distance in $X$.
2. Weight those nearby points with a kernel that falls to zero at the edge of the window, most
   commonly the tricube weight $w_i = \left(1 - \left(\frac{|x_i - x_0|}{d}\right)^3\right)^3$,
   where $d$ is the distance to the farthest point still inside the span.
3. Fit a weighted least-squares regression, usually a straight line or a quadratic, using those
   weights.
4. Read the fitted value at $x_0$ off that local regression, then discard the fit and repeat for
   the next $x_0$.

Formally, the local fit at $x_0$ minimizes

$$\sum_{i=1}^{n} w_i(x_0) \left(y_i - \beta_0 - \beta_1 x_i\right)^2,$$

a weighted version of ordinary least squares where the weights shift with every query point
[@cleveland1979].

@fig-loess-span fits local regression to the load-latency data at four span values.

::: {.callout-tip}
A LOESS span plays the role a spline's knot count or a smoothing spline's $\lambda$ plays: a
narrow span chases individual points the way too many knots do, and a wide span oversmooths the
way too large a $\lambda$ does. Cross-validate the span the same way Chapter 5 cross-validated a
knot count or a penalty, rather than picking one by eye.
:::

A narrow span stays close to the data near each query point and follows the saturation curve into
the high-load region, since only nearby points get a vote there. A wide span borrows points from
the flatter middle of the range to smooth the edge, and ends up cutting the top of the curve off,
understating the load level where a false sense of headroom costs the most.

::: {.callout-note}
Local regression and a smoothing spline solve the same problem from opposite directions. A
smoothing spline keeps every observation as a candidate knot and fits one global penalized
function in a single pass. Local regression answers one query point at a time and keeps no global
function at all. Both remove the knot-count decision, and both replace it with their own tuning
knob: span for local regression, $\lambda$ for the smoothing spline.
:::

## Generalized additive models: splines with more than one predictor

::: {#fig-gam-components}
```{=html}
<iframe src="../_generated/chapter-splines-fig-gam-components.html" width="100%" height="480"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The payload term comes out as a straight line, matching Chapter 4's own finding that payload size
and latency move together in close to a fixed ratio. The load term keeps the saturating bend every
figure in this chapter has shown so far. One model, two predictors, and each keeps the shape that
fits it.
:::

Every spline built in this chapter so far has depended on one predictor: load. Chapter 4's own
regression model used two, payload size and concurrent load, fit together as a single straight-line
surface. Neither chapter has shown how to give a second predictor the same curve-fitting treatment
the first one just got.

A *generalized additive model*, or GAM, is what happens when a regression sums a separate function
per predictor instead of a separate coefficient per predictor:

$$Y = \beta_0 + f_1(X_1) + f_2(X_2) + \dots + f_p(X_p) + \varepsilon,$$

where each $f_j$ can be a straight line, a step function, or a spline, chosen on a
predictor-by-predictor basis rather than forced to be the same shape for every column in the
model.

Chapter 4's regression had to pick one shape, a line, and apply it to both payload size and load
at once, because ordinary least squares fits one global linear surface. That worked for payload
size, since Chapter 4 found the two moved together in close to a fixed ratio, but it is the wrong
shape for load, which this chapter has spent several sections showing bends sharply as utilization
climbs. A GAM removes the requirement to pick one shape for every predictor: keep the cheap,
interpretable line where a predictor earns it, and reserve a spline for the one predictor whose
relationship needs the extra flexibility, instead of spending that flexibility everywhere by
default.

A GAM is fit by *backfitting*, an iterative procedure that estimates one term at a time against
whatever the other terms have not yet explained:

1. Start with every non-intercept term at zero.
2. Pick one term, say the payload term. Compute the residual left over once every other term's
   current estimate is subtracted from $y$, then fit that one term (a line, in this case) against
   the residual.
3. Center the newly fit term to a mean of zero across the training data, so the intercept alone
   carries the overall average and the terms stay identifiable against each other.
4. Move to the next term, say the load term. Compute the residual left over once the (now updated)
   payload term is subtracted from $y$, then fit a regression spline against that residual, using
   the same quantile-placed knots this chapter has used throughout.
5. Repeat steps 2 through 4 until the term estimates stop moving.

For the load-payload-latency data behind @fig-gam-components, ten backfitting passes recover a
payload coefficient of 8.90 ms per kilobyte, within a tenth of a millisecond of the 9.0 ms/KB value
built into the simulation, while the load term settles into the same saturating shape shown
earlier in @fig-knot-count and @fig-smoothing-lambda. Two terms converge fast because each update
is a short calculation, an OLS slope for the line and a spline fit for the curve; a GAM with more
predictors needs correspondingly more passes.

### The partial residual plot: reading a spline term's contribution

::: {#fig-gam-partial-residual}
```{=html}
<iframe src="../_generated/chapter-splines-fig-gam-partial-residual.html" width="100%" height="480"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Subtract the payload term's straight-line contribution and the intercept from every observation,
and what remains lines up with the load term's own fitted spline almost point for point, evidence
that the leftover pattern is the saturating bend the load term is meant to capture, and not just
whatever noise the payload term failed to absorb.
:::

A GAM's summary output reports one fitted curve per term, but a curve on its own does not say
whether that curve tracks a pattern in the data or chases noise the rest of the model left behind.
A *partial residual plot* answers that question for one term at a time, holding every other term
fixed.

The consequence of skipping this check shows up whenever a GAM's coefficient or curve shape
changes after a new predictor gets added. Without a per-term diagnostic, there is no way to tell
whether the change reflects a different relationship in the data or the new predictor simply
absorbing variation the old term used to soak up on its own. A partial residual plot gives each
term a way to defend its own shape, term by term, instead of trusting the model's overall fit
statistic to vouch for every term at once.

Computing a partial residual for one term, say the load spline term $f_{\text{load}}$, takes two
steps once the full model is fit:

1. Remove every other term's fitted contribution from each observation, but leave the term under
   inspection out of the subtraction: $r_i = y_i - \hat\beta_0 - \hat{f}_{\text{payload}}(x_i)$.
2. Plot $r_i$ against load, and overlay the fitted curve $\hat{f}_{\text{load}}(\text{load})$ on
   the same axes. If the term is capturing something in the data, the scattered points should
   track the curve; if the term is fitting noise the other terms left behind, the points scatter
   around the curve without following its shape.

@fig-gam-partial-residual runs this check on the load term from the model above. The partial
residuals trace the same saturating bend the fitted spline term shows, evidence that the spline is
earning its added flexibility rather than absorbing leftover noise the linear payload term could
not explain.

The partial residual plot is a paraphrase of a general diagnostic used across regression modeling
wherever a model sums more than one predictor's effect together: isolate one term by removing the
rest, then check whether what remains matches what that term claims to be doing. Applied to the
load-payload-latency data here, the same check works for any spline term a GAM adds, not only the
one worked through in this section.

::: {.callout-important}
A spline term with no partial residual check behind it is an unverified claim about the data. Run
this plot on every spline term in a GAM before shipping it, especially a term added after an
earlier version of the model passed review.
:::

The backfitting-plus-partial-residual pattern generalizes past two terms without changing shape:
add a third predictor, give it its own $f_3$, and the same iterative procedure and the same
per-term diagnostic apply unchanged. What does change is compute: each additional spline term adds
another knot-selection and cross-validation decision, and a GAM with a dozen predictors needs a
dozen of those decisions made and checked, one term at a time, per [@hastietibshirani1990].

## A Bayesian perspective

Every spline in this chapter produces a single fitted curve: one best estimate of the
relationship between load and latency, with no built-in statement of how much that curve might
have looked different under a slightly different sample of requests.

Gaussian process regression takes the smoothing-spline idea and gives it a full probability
distribution instead of a point estimate.

Instead of settling on one curve through the data, imagine sketching hundreds of plausible
curves by hand, keeping only the ones that pass close to the known points, and looking at how
much the surviving curves still disagree with each other elsewhere. A Gaussian process turns
that intuition into math.

A Gaussian process defines a prior distribution not over parameters, as in Chapter 4's Bayesian
linear regression, but directly over functions: before seeing any data, the model treats every
smooth function consistent with a chosen *kernel* as plausible.

The kernel encodes assumptions like how quickly the function is allowed to change
[@rasmussenwilliams2006].

The most common choice, the squared-exponential kernel, has a single *length-scale* parameter
that plays much the same role as a smoothing spline's $\lambda$: a short length-scale allows the
function to bend quickly across small changes in load, a long length-scale forces it to change
slowly.

:::{.callout-tip}
A Gaussian process's length-scale is a tuning parameter, not a value to set by eye. Cross-validate
it against held-out load-latency pairs the same way Chapter 5 cross-validated $\lambda$ for the
smoothing spline.
:::

Conditioning that prior on the observed load-latency pairs produces a posterior distribution
over functions, with a posterior mean curve that looks similar to a smoothing spline fit.

::: {#fig-gp-posterior}
```{=html}
<iframe src="../_generated/chapter-splines-fig-gp-posterior.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The credible band widens noticeably at the high-load edge, where observations are sparser, an
honest signal that the model is least certain in the region that matters most, something a
single fitted spline curve cannot say about itself.
:::

That posterior also provides something the smoothing spline does not offer on its own: a
credible band around the curve at every load level, wide where data is sparse or noisy and
narrow where the model has seen consistent evidence.

@fig-gp-posterior shows a Gaussian process fit to the load-latency data at several
length-scales, with the 95% credible band shaded around the posterior mean. Watch what happens
at the high-load edge: the band widens, right where a single fitted spline curve would stay
quiet about its own uncertainty.

The credible band is not free. A Gaussian process's computational cost grows roughly with the
cube of the number of observations, since fitting one requires inverting an $n \times n$ matrix.

::: {.callout-note}
A Gaussian process's fitting cost grows with the cube of the number of observations. A few
hundred to a few thousand points is a practical ceiling without a sparse approximation.
:::

That cost makes a direct GP fit impractical past a few thousand points without an approximation
method. For a load-latency curve built from a full day of production traffic sampled down to a
few hundred representative points, a GP is a reasonable choice.

For a curve fit directly on millions of raw request logs, a smoothing spline or a sparse GP
approximation is usually the more practical tool.

## References {.unnumbered}

::: {#refs}
:::
