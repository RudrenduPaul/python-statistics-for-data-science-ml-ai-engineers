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
specifically to prevent that failure mode.
:::

## Smoothing splines: a penalty instead of a knot count

::: {#fig-smoothing-lambda}
```{=html}
<iframe src="../_generated/chapter-splines-fig-smoothing-lambda.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A small lambda produces a wiggly curve that chases individual points, and a large lambda
flattens the fit back toward a straight line, recovering something close to the OLS fit that
opened this chapter.
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
path from near-interpolation of every point at small $\lambda$ to a nearly straight line at
large $\lambda$.

Recall from Chapter 5's discussion of cross-validation that a tuning parameter like $\lambda$
should not be picked by eye. In practice, $\lambda$ is chosen the same way the regularization
strength for Lasso or Ridge was chosen: by cross-validating over a range of candidate values and
picking the one that minimizes estimated test error, not training error.

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
