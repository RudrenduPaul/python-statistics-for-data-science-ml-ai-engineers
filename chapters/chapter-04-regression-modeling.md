# Regression Modeling

Recall from Chapter 1 that payload size and checkout-API latency showed a strong positive
correlation, and that correlation alone could not say how much of the relationship was causal.

*Regression modeling* is the tool that goes one step further: instead of a single number
summarizing how two variables move together, it produces a formula that predicts one variable
from another, and a set of diagnostics that say how much to trust that formula.

This chapter works through six questions that come up whenever a prediction needs to be backed
by more than intuition:

1. *How do I fit a line through the data, and what does "best" mean?*
   Ordinary least squares answers both, along with the assumptions that make the answer
   trustworthy.
2. *How uncertain is a prediction, and does that uncertainty shrink with more data?*
   Confidence intervals and prediction intervals answer two different versions of that
   question.
3. *What happens when the outcome is a category instead of a number?*
   Logistic regression extends the same machinery to binary outcomes.
4. *How do I know if a model is a good fit, and not just a complicated one?*
   R-squared, adjusted R-squared, AIC, and BIC each answer a version of that question.
5. *What does a coefficient mean, and when does a significant one not matter?*
   Statistical significance and practical significance are not the same thing.
6. *What happens when there are more predictors than the data can support?*
   Regularization techniques trade a small amount of bias for a model that generalizes.

## Ordinary least squares regression

Picture drawing one straight line through a scatter of dots so it stays as close as possible to
every dot at once. For each dot, measure how far the line misses it, square that distance so
big misses count more, then pick the line that makes the total as small as possible.

*Ordinary least squares* (OLS) regression fits a line through a set of points by minimizing the
sum of the squared differences between observed and predicted values. For predicting latency
from payload size,

$$Y = \beta_0 + \beta_1 X_1 + \varepsilon$$

where $Y$ is latency, $X_1$ is payload size, $\beta_0$ and $\beta_1$ are the coefficients to
estimate, and $\varepsilon$ is the error term capturing everything the model does not explain.

OLS finds the values of $\beta_0$ and $\beta_1$ that minimize the sum of squared residuals, the
gap between each observed latency and the value the line predicts for that request's payload
size.

::: {#fig-ols}
```{=html}
<iframe src="../_generated/chapter-04-fig-ols-fit.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Latency (ms) plotted against payload size (KB), with the fitted OLS line in red. Dragging the
noise slider higher scatters the dots further from the line: the fitted slope barely moves, but
R-squared falls sharply, since a noisier relationship leaves more of the variance unexplained
even while the slope itself stays close to correct.
:::

@fig-ols shows this fit at several noise levels: the fitted line keeps finding roughly the right
slope no matter how much scatter surrounds it, but R-squared, the share of variance the line
explains, drops as that scatter grows.

OLS rests on five assumptions: **linearity** (the true relationship is a straight line, not a
curve), **independence** (one observation's error does not influence another's), and
**homoscedasticity** (the spread of residuals stays constant across the range of the predictor).

It also assumes **normality** (residuals are normally distributed) and **no multicollinearity**
(predictors are not near-perfect linear combinations of each other).

::: {.callout-tip}
A residual plot, predicted values against residuals, is the fastest practical check on
linearity and homoscedasticity: a random scatter around zero supports both assumptions, while a
funnel shape or a curve says one of them needs a closer look.
:::

Recall from Chapter 1 that latency is right-skewed with a variance that likely grows alongside
payload size. That means homoscedasticity and normality are assumptions worth checking rather
than assuming for this specific relationship.

When they fail badly enough, a generalized linear model or a transformation of the outcome
variable is the usual fix, a topic beyond this chapter's scope but worth knowing about.

## Confidence intervals and prediction intervals

Suppose the model needs to answer two different questions: what is the typical latency for
requests with a 10 KB payload, and what latency should a single new 10 KB request expect? These
are not the same question, and they get different intervals.

Guessing the average height in a school gets more precise the more students get measured;
guessing the height of the next person through the door stays uncertain no matter how many
others were measured before.

A *confidence interval* estimates a range likely to contain the true mean latency at that
payload size. A *prediction interval* estimates a range likely to contain one future
observation, which is wider because it accounts for both the uncertainty in estimating the mean
and the natural variability of any single request around that mean.

::: {#fig-ci-pi}
```{=html}
<iframe src="../_generated/chapter-04-fig-ci-vs-pi.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Latency (ms) against payload size (KB): the confidence band (red) narrows steadily as sample
size grows, since a larger sample pins down the mean more precisely. The prediction band (blue)
barely narrows at all, because no amount of data shrinks the irreducible variability of one new
request.
:::

@fig-ci-pi shows both intervals as sample size grows, confirming the pattern the definitions
above predict. The confidence interval only has to account for how precisely the mean is known,
and precision improves with sample size.

The prediction interval also has to account for how far any individual request can land from
that mean, a spread that does not shrink no matter how much data the model has seen.

::: {.callout-note}
A capacity-planning question about typical load calls for a confidence interval. A question
about the worst latency any single request might see calls for a prediction interval. Reporting
the narrower one for the wrong question understates the true range of outcomes.
:::

## Logistic regression

*Logistic regression* models a binary outcome, success or failure, timeout or no timeout,
rather than a continuous one. Think of a dimmer switch instead of an on-off light switch:
probability slides smoothly from near-zero to near-certain along an S-shaped curve instead of
jumping straight from 0% to 100%.

That shape also keeps every predicted probability between 0 and 1. Logistic regression applies
the *logistic function* to map any input to a value between 0 and 1, interpreted as a
probability:

$$p(x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

In other words, when $\beta_1$ is positive, as with load predicting a timeout, $-(\beta_0 +
\beta_1 x)$ becomes a large negative number as $x$ grows, the exponential term shrinks toward
zero, and $p(x)$ approaches 1; as $x$ shrinks, the exponential term grows without bound and
$p(x)$ approaches 0.

A negative $\beta_1$ reverses that direction, but the exponential term can never make $p(x)$
fall below 0 or rise above 1 regardless of sign, which is the property the dimmer-switch analogy
above was pointing at.

Suppose the team wants to model the probability that a request times out as a function of
concurrent load on the service.

::: {#fig-logistic}
```{=html}
<iframe src="../_generated/chapter-04-fig-logistic-timeout.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

P(timeout) against concurrent load: dots along the top and bottom edges are individual observed
requests (timed out or not), and the S-shaped curve is the fitted probability. A steeper curve
flips the service from healthy to overwhelmed over a narrower band of load; a shallower curve
spreads that same transition across a wider range.
:::

@fig-logistic shows the fitted curve at several steepness values: at low load the probability of
a timeout sits near zero, and beyond the midpoint of the curve timeouts become close to certain.

Logistic regression's coefficients are estimated by maximum likelihood rather than by
minimizing squared error, and they are interpreted differently than an OLS coefficient: $\beta_1$
represents the change in the log-odds of the outcome for a one-unit increase in $x$, not a
direct change in probability.

::: {.callout-warning}
Reading $\beta_1$ as if it were a direct probability shift is the most common misreading of a
logistic coefficient. Because the curve is flat near 0 and 1 and steepest near its midpoint, the
same one-unit increase in load can move the timeout probability by a fraction of a percentage
point near the tails or by ten or more percentage points near the midpoint.
:::

A single coefficient cannot report "the" change in probability without saying where on the
curve the service currently sits. Recall Chapter 2's Type I and Type II error framework:
choosing where along this curve to draw the "predict timeout" threshold is the same trade-off
between false positives and false negatives that any classification decision has to make.

## Measuring fit: R-squared, adjusted R-squared, AIC, and BIC

*R-squared* is the share of variance in the outcome that the model explains, ranging from 0 (no
explanatory power) to 1 (the model explains every observed difference). Adding predictors to a
model can only raise R-squared or leave it unchanged, even when those predictors are pure noise,
since OLS will always find some small, spurious pattern to exploit in a finite sample.

Think of a science-fair judge who docks a point for every extra gadget bolted onto a display,
even ones that did not help the demonstration. *Adjusted R-squared* corrects for this by
penalizing additional predictors:

$$\bar{R}^2 = 1 - (1 - R^2) \frac{n-1}{n-p-1}$$

where $n$ is the sample size and $p$ is the number of predictors. In other words, as $p$ grows
relative to $n$, the fraction $(n-1)/(n-p-1)$ grows too, which inflates the unexplained-variance
term $(1 - R^2)$ enough to counteract R-squared's automatic rise; a predictor that does not truly
reduce unexplained variance now pulls adjusted R-squared down instead of leaving it unchanged.

::: {#fig-r2}
```{=html}
<iframe src="../_generated/chapter-04-fig-r2-vs-adjusted.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Value (0 to 1) against the number of random noise predictors added to the model. R-squared
(blue) always rises as predictors are added, even useless ones. Adjusted R-squared (red)
penalizes that added complexity, flattening and then declining once the noise predictors stop
paying for themselves.
:::

@fig-r2 adds twenty random, unrelated noise predictors to a latency model one at a time and
tracks both metrics: R-squared climbs the whole way, while adjusted R-squared flattens and then
declines once the noise predictors stop paying for the complexity they add.

*AIC* (Akaike's Information Criterion) and *BIC* (Bayesian Information Criterion) address the
same overfitting risk from a different angle, useful for comparing entirely different candidate
models rather than only tracking one model as terms are added:

$$\text{AIC} = -2 \ln(\hat{L}) + 2k, \qquad \text{BIC} = -2 \ln(\hat{L}) + k \ln(n)$$

where $\hat{L}$ is the model's maximized likelihood and $k$ is the number of parameters. In other
words, $-2\ln(\hat{L})$ measures how poorly the model fits: a model that assigns a higher
likelihood to the observed data produces a smaller value here, so this term shrinks as fit
improves.

The $2k$ or $k\ln(n)$ term adds back a fixed cost for every parameter, so a model only earns a
lower AIC or BIC by fitting well enough to outweigh the complexity it adds, not by fitting at
all.

AIC comes from Akaike (1974) [@akaike1974]; BIC comes from a separate derivation by Schwarz
(1978), starting from a Bayesian argument about model dimension rather than Akaike's
information-theoretic one [@schwarz1978].

Both penalize additional parameters; BIC penalizes them more heavily as sample size grows, which
tends to favor simpler models than AIC does on large datasets. Lower values of either indicate a
better balance of fit against complexity.

The two are best used to rank several candidate models (a payload-only model, a
payload-plus-region model, a payload-plus-region-plus-time-of-day model) against each other,
rather than read in isolation.

::: {.callout-tip}
AIC and BIC scores are only comparable across models fit to the same dataset and the same
outcome variable. A lower AIC on a model trained on a different sample, or on a transformed
outcome, is not a meaningful comparison, even though the two numbers can be placed side by side.
:::

::: {#fig-aic-bic}
```{=html}
<iframe src="../_generated/chapter-04-fig-aic-bic.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Both AIC and BIC reward adding region, which carries a measurable effect, but BIC penalizes the
noise predictor, time of day, more heavily than AIC does, and that gap widens as the sample size
grows.
:::

@fig-aic-bic ranks those three candidate models, where region carries a measurable effect on
latency and time of day is pure noise, and tracks AIC and BIC for each as the sample size grows.

Drag the slider from a small sample toward a large one and watch the gap open up: at n=30, AIC
and BIC mostly agree on which model looks best.

By n=400, BIC has pulled further away from rewarding the model with the noise predictor added.
That is the same "BIC penalizes complexity more heavily as sample size grows" claim made above,
now shown concretely instead of merely asserted.

## Model interpretation: statistical significance versus practical significance

A regression coefficient states how much the outcome changes for a one-unit change in a
predictor, holding everything else in the model constant. A coefficient of 9 on payload size
means each additional kilobyte of payload is associated with roughly 9 additional milliseconds
of latency, on average.

Whether that coefficient is *statistically significant* is a question Chapter 2 covered: compute
a p-value for the coefficient and compare it against a threshold.

Recall from Chapter 2's discussion of the ASA statement on p-values that statistical
significance says nothing about the size of an effect. The same caution applies directly here: a
coefficient can be statistically significant, with a p-value far below 0.05, while representing
a change too small to act on.

A 0.1 ms latency improvement, for example, can clear statistical significance across ten million
requests and still not justify shipping. *Practical significance* asks the second question a
statistically significant coefficient does not answer on its own: given the size of this effect,
does it change any decision.

::: {.callout-tip}
With a large enough sample, almost any nonzero coefficient clears the significance bar. The
question worth asking is not whether a p-value is small, but whether the coefficient's size is
big enough to change a decision that matters.
:::

## Regularization: Lasso, Ridge, and Elastic Net

Suppose the latency model grows to include payload size, deployment region, time of day, and
endpoint identity, several of which move together because larger payloads cluster in certain
regions or certain endpoints.

*Regularization* adds a penalty term to the OLS objective that shrinks coefficients, trading a
small amount of bias for a model that generalizes better to new data.

Packing a suitcase with a rule that charges a fee per item keeps only the items worth their
weight; Lasso's fee is steep enough to drop an item to zero, Ridge's just makes every item
lighter.

*Lasso* regularization (short for least absolute shrinkage and selection operator) adds an L1
penalty, the sum of the absolute values of the coefficients [@tibshirani1996]:

$$\text{minimize} \quad \text{SSE} + \lambda \sum_j |\beta_j|$$

Here $\lambda$ is the *regularization strength*, a value the analyst chooses rather than one OLS
estimates from the data: $\lambda = 0$ adds no penalty and reduces the formula back to plain
OLS, and larger values of $\lambda$ shrink coefficients more aggressively.

*Ridge* regularization adds an L2 penalty, the sum of the squared coefficients, instead
[@hoerlkennard1970]:

$$\text{minimize} \quad \text{SSE} + \lambda \sum_j \beta_j^2$$

*Elastic Net* combines both penalties, controlled by a mixing parameter $\alpha$:

$$\text{minimize} \quad \text{SSE} + \alpha \lambda \sum_j |\beta_j| + (1-\alpha) \lambda \sum_j \beta_j^2$$

The practical difference between Lasso and Ridge shows up clearly once several correlated
predictors are in the model together.

::: {#fig-regularization}
```{=html}
<iframe src="../_generated/chapter-04-fig-regularization-paths.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Coefficient value for each predictor, comparing Lasso (red) against Ridge (blue) as the
regularization strength grows. As regularization strengthens, Lasso zeroes out the noise
predictor and one of the two correlated region indicators, while Ridge only shrinks every
coefficient toward zero without eliminating any of them.
:::

@fig-regularization fits both to a latency model with two region indicators correlated with
payload size and one pure noise predictor (time of day), tracking every coefficient as the
regularization strength $\lambda$ grows.

This is because the two penalties shrink coefficients differently. Ridge's squared penalty
shrinks every coefficient smoothly, which handles multicollinearity well: correlated predictors
end up sharing the credit for their combined effect.

But it never removes a variable from the model entirely, which can make the model harder to read
when many predictors are involved. Lasso's absolute-value penalty can drive a coefficient to
zero once $\lambda$ is large enough, effectively performing feature selection as part of fitting
the model.

::: {.callout-note}
Lasso tends to keep only one predictor from a group of correlated predictors and zero out the
rest, largely by which one the optimization happens to favor first. Ridge splits the credit
evenly across the group instead. Elastic Net exists to balance those two behaviors when a model
has both correlated and irrelevant predictors at once.
:::

Regularization comes at a cost: shrinking coefficients toward zero introduces bias, trading some
fit on the training data for a model that is less likely to have memorized noise specific to
that sample.

The section below approaches that same trade-off from a different angle, by treating a
coefficient not as a single fixed number to estimate but as a quantity with its own probability
distribution, updated as evidence accumulates.

## A Bayesian perspective

Every model in this chapter has produced a single best-fitting number for each coefficient: an
OLS slope, a logistic-regression weight, a Lasso or Ridge estimate shrunk toward zero.

*Bayesian linear regression* asks a different question: instead of one number, what is the full
range of coefficient values consistent with the data, and how much more likely are some values
than others?

Guessing a stranger's age before meeting them gives a wide starting range; hearing they have
kids in college narrows that range sharply. The wide starting guess is a prior, the narrowed one
after new evidence is a posterior.

::: {.callout-note}
A prior is a choice, not a free parameter. An overly narrow or badly centered prior can bias a
result as much as too little data can, so a Bayesian analysis is only as trustworthy as the
reasoning behind the prior it starts from.
:::

It starts from the same model as OLS, $Y = \beta_0 + \beta_1 X_1 + \varepsilon$, but treats
$\beta_1$ as a random quantity with a *prior* distribution describing what is plausible before
seeing the data, then updates that prior into a *posterior* distribution once the observed
latencies are in hand.

Under a flat, uninformative prior, the posterior mean lands on the OLS estimate. Bayesian
regression takes the assumptions OLS makes about the coefficients and states them explicitly, as
a prior, instead of leaving them buried in the method.

Recall Ridge and Lasso from the section above. Both turn out to be Bayesian estimates in
disguise. Placing an independent Gaussian prior on each coefficient and taking the posterior
mode, the single most probable value, recovers the Ridge estimate.

Placing an independent Laplace (double-exponential) prior instead and taking the posterior mode
recovers the Lasso estimate [@tibshirani1996; @parkcasella2008].

A Gaussian posterior is symmetric, so its mode and mean are the same value, which is why the
flat-prior-recovers-OLS claim above is stated in terms of the mean while the Ridge equivalence
just given is stated in terms of the mode: for a Gaussian posterior the two coincide.

A Laplace-prior posterior is not symmetric in general, so its mode and mean can differ, and it is
specifically the mode, not the mean, that matches the Lasso estimate. From this angle, the
regularization strength $\lambda$ in both formulas is the precision of the prior: how tightly
that prior clusters around zero before any data arrives.

::: {#fig-bayesian-ridge}
```{=html}
<iframe src="../_generated/chapter-04-fig-bayesian-ridge.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

As the prior variance shrinks, equivalent to $\lambda$ growing, the posterior on the
payload-size coefficient narrows and its center moves toward zero, tracing the same shrinkage
Ridge regression's penalty term produces.
:::

@fig-bayesian-ridge shows this pattern directly for the payload-size coefficient: one full
distribution at a time, instead of one point estimate, tracing out the same shrinkage Ridge
regression produces as the prior tightens.

Logistic regression does not offer the same clean arithmetic. Because the logistic function is
nonlinear, the posterior over its coefficients has no closed form the way Ridge and Lasso's
Gaussian and Laplace posteriors do.

Fitting a Bayesian logistic regression means approximating that posterior instead, typically
with Markov chain Monte Carlo sampling or a Laplace approximation around the posterior mode.

The added computational cost buys something a point estimate cannot: a *credible interval* on
each coefficient, a range that directly states the probability the true effect falls inside it.
A standard error, by contrast, is only as trustworthy as the model's specification, and its
validity also depends on the sample being large enough for its asymptotic guarantees to hold.

That trade of more computation for a fuller picture of uncertainty is worth returning to. Part 3
takes it up in full, this time for the practical case of deciding between two live variants of a
product rather than fitting a single regression model.

## References {.unnumbered}

::: {#refs}
:::
