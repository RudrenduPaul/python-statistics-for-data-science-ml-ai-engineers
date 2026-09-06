# Regression Modeling

Recall from Chapter 1 that payload size and checkout-API latency showed a strong positive
correlation, and that correlation alone could not say how much of the relationship was causal.

*Regression modeling* is the tool that goes one step further: instead of a single number
summarizing how two variables move together, it produces a formula that predicts one variable
from another, and a set of diagnostics that say how much to trust that formula.

This chapter works through eight questions that come up whenever a prediction needs to be backed
by more than intuition:

1. *How do I fit a line through the data, and what does "best" mean?*
   Ordinary least squares answers both, along with the assumptions that make the answer
   trustworthy.
2. *How uncertain is a prediction, and does that uncertainty shrink with more data?*
   Confidence intervals and prediction intervals answer two different versions of that
   question.
3. *What changes once more than one predictor, or a category instead of a number, joins the
   model?*
   Multiple regression, dummy encoding, and interaction terms turn one straight line into
   something that can represent a fleet of servers and a workload at once, and the variance
   inflation factor flags when two of those predictors have started saying the same thing.
4. *What happens when the outcome is a category instead of a number?*
   Logistic regression extends the same machinery to binary outcomes.
5. *Once a model predicts a category, how is it graded?*
   Accuracy, precision, and recall turn a fitted threshold into a scorecard.
6. *How do I know if a model is a good fit, and not just a complicated one?*
   R-squared, adjusted R-squared, AIC, and BIC each answer a version of that question.
7. *What does a coefficient mean, and when does a significant one not matter?*
   Statistical significance and practical significance are not the same thing.
8. *What happens when there are more predictors than the data can support?*
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

::: {.callout-note}
Multicollinearity does not bias the coefficient estimates themselves; it inflates their standard
errors, so an individual predictor's coefficient can look statistically insignificant even when
the group of correlated predictors together explains a meaningful share of the variance. Ridge
regression, covered later in this chapter, addresses that instability directly by shrinking
correlated coefficients together instead of letting one absorb an unstable share of the effect.
The variance inflation factor section further down this chapter gives a direct way to measure
how much multicollinearity is inflating a given predictor's standard error, before deciding
whether Ridge, or simply dropping a redundant predictor, is the fix a given model needs.
:::

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

## Multiple regression: more than one predictor at a time

The OLS section above ties latency to a single suspect, payload size. An on-call engineer
diagnosing a slow endpoint rarely gets that luxury: payload size, concurrent load, and
deployment region all move at once, and blaming the one variable a single-predictor model can
see risks pointing at the wrong knob to turn.

::: {#fig-multiple-regression}
```{=html}
<iframe src="../_generated/chapter-04-fig-multiple-regression.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The estimated coefficient on payload size across three models fit to the same 300 requests, with
a dotted reference line at the 9 ms/KB effect this chapter's simulation was built to produce. A
payload-only model overstates that effect; adding concurrent load and then region pulls the
estimate back toward the reference line.
:::

@fig-multiple-regression fits three models to the same 300 simulated requests, adding one
predictor at a time. A model that sees only payload size estimates its effect at 9.95 ms per
additional kilobyte. Adding concurrent load to the model pulls that estimate down to 9.26 ms.
Adding the two region indicators on top moves it to 9.20 ms, close to the 9 ms per kilobyte
relationship this chapter's simulation was built to produce.

*Multiple regression* extends ordinary least squares from one predictor to several, fitting a
single equation that estimates each predictor's own contribution to the outcome while holding
the others fixed:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \varepsilon$$

Each $\beta_j$ answers a narrower question than the simple-regression slope did: not "how does
$Y$ change as $X_j$ changes," but "how does $Y$ change as $X_j$ changes, with every other
predictor in the equation held at a fixed value." Fitting proceeds the same way OLS always does,
minimizing the sum of squared residuals, just over more coefficients at once.

Concurrent load and payload size move together in this simulation, with a correlation of 0.78,
since heavier payloads tend to arrive during the same high-load batch windows. A payload-only
model has no predictor to credit for load's own share of the slowdown, so it folds that share
into payload's coefficient instead. A chunk of the 9.95 ms the single-predictor model reports is
load's own effect, showing up under payload's name because the two move together in this sample.
A capacity-planning decision that trims payload size expecting the full 9.95 ms per kilobyte in
savings would come up short, closer to the 9.20 ms the fuller model estimates.

::: {.callout-note}
Recall Chapter 2's treatment of confounding variables: a coefficient's value can shift once the
right control joins the model. Payload's coefficient shifting from 9.95 to 9.20 ms as concurrent
load and region enter the equation above is the same phenomenon, now inside a regression instead
of a group comparison.
:::

## Categorical predictors and dummy encoding

Deployment region is not a number a regression can multiply by a coefficient. There is no
meaningful sense in which "eu-west" is twice "us-east," so a categorical predictor needs a
different encoding before it can enter the equation above at all.

::: {#fig-region-dummy-effect}
```{=html}
<iframe src="../_generated/chapter-04-fig-region-dummy-effect.html" width="100%" height="480"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Gray bars show each region's raw average latency; blue bars show what the full model from the
section above predicts for that region once payload size and concurrent load are held at their
sample averages. The two views mostly agree for eu-west but diverge for us-west, where the raw
gap overstates region's own contribution.
:::

@fig-region-dummy-effect compares two views of the same three regions. For us-east, both bars
read about 142 ms, since us-east serves as the reference level and contributes nothing beyond the
intercept. For us-west, the raw group average runs 16.3 ms above us-east's, but the model's own
coefficient on the us-west indicator is only 8.3 ms. For eu-west, the raw gap (18.6 ms) and the
model's coefficient (17.9 ms) land close together.

The gap between us-west's two bars traces back to the sample itself: the us-west requests in this
draw happen to carry a slightly higher average payload (11.8 KB against us-east's 11.0 KB) and
slightly higher average load (55.4 against 53.4), so part of us-west's raw latency gap comes from
payload and load, not region. Once the model holds those two predictors fixed, only 8.3 ms of the
16.3 ms raw gap is left for the region indicator to explain.

A *dummy variable* recodes a $k$-level categorical predictor into $k-1$ binary indicator columns,
each equal to 1 when an observation belongs to that level and 0 otherwise, with one level chosen
as the reference against which every other level is compared:

$$Y = \beta_0 + \beta_1 X_1 + \gamma_1 D_{\text{us-west}} + \gamma_2 D_{\text{eu-west}} +
\varepsilon$$

Here $D_{\text{us-west}}$ equals 1 for a request served from us-west and 0 otherwise, and
$D_{\text{eu-west}}$ works the same way for eu-west; us-east, with no indicator of its own, is the
reference level baked into the intercept $\beta_0$. In other words, $\gamma_1$ and $\gamma_2$
state how much higher or lower latency runs for each named region compared to us-east, holding
payload size and load fixed, the same reading the figure above puts numbers to.

::: {.callout-warning}
Encoding a $k$-level category with all $k$ indicator columns, instead of $k-1$, creates a new
column fully determined by the others: every request's set of indicators sums to 1. Paired with
an intercept, that is a perfect multicollinearity problem the fitting routine cannot resolve,
sometimes called the dummy variable trap. Dropping one level as the reference avoids it.
:::

## Binning high-cardinality categorical predictors by residual

Deployment region only has three levels. A checkout API's endpoint path is a different kind of
categorical predictor: dozens of distinct routes, from `/api/v1/checkout` to
`/api/v1/fraud/score`, each called a different number of times. Dummy-encoding every one of them
means one coefficient per endpoint, several estimated from a handful of requests or fewer.

::: {#fig-endpoint-residual-binning}
```{=html}
<iframe src="../_generated/chapter-04-fig-endpoint-residual-binning.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Endpoints sorted left to right by their median residual from a payload-only model, colored by the
bin they fall into. Slide the bin count and the same sorted bars regroup into more or fewer
color-coded clusters, without changing any endpoint's underlying position.
:::

@fig-endpoint-residual-binning sorts 24 endpoints that appear in a sample of 350 requests by
their median residual from a payload-only latency model, then groups them into four color-coded
bins by cumulative request count. The endpoints that run fastest for their payload size,
`/api/v1/inventory/reserve` and `/api/v1/checkout` among them, land in the leftmost bin, roughly
20 ms below the payload-only model's prediction. The ones that run slowest, including
`/api/v1/fraud/score`, `/api/v1/notifications/subscribe`, and `/api/v1/auth/refresh`, land in the
rightmost bin, roughly 18 ms above it.

The technique behind that figure fits a control model without the endpoint predictor at all,
extracts each request's residual, computes the median residual for every endpoint that shows up
in the sample, sorts endpoints by that median from most negative to most positive, then bins the
sorted list into a small number of groups by cumulative row count rather than by an arbitrary
cutoff:

$$e_i = y_i - \hat{y}_i, \qquad \tilde{e}_g = \text{median}\{e_i : \text{endpoint}(i) = g\}$$

where $e_i$ is request $i$'s residual from the control model and $\tilde{e}_g$ is endpoint $g$'s
median residual. In other words, an endpoint's median residual measures how much slower or
faster that endpoint runs than a request's payload size alone would predict, and sorting by that
number before binning groups endpoints that behave alike, whatever each route happens to be
called.

Some endpoints in this sample of 350 requests show up only once or twice, `/api/v1/payments/verify`
and `/api/v1/pricing/quote` each appear a single time, and four of the catalog's 28 endpoints do
not appear at all. A coefficient estimated from one observation would move wildly from one
sample to the next; refitting with the four-bin grouping instead of 27 individual endpoint dummies
raises this model's R-squared from 0.66 to 0.83, with the slowest bin's coefficient landing at
31.6 ms above the fastest bin's baseline, a stable, interpretable summary a single rare endpoint's
own noisy coefficient could not offer on its own.

::: {.callout-tip}
The same residual-then-bin technique applies to any high-cardinality categorical predictor with
uneven traffic across levels, product SKU codes, device model IDs, or sensor IDs, not only
endpoint paths. The predictor does not need to be about location for the method to work; it needs
only a control model to compute residuals against and a category with more levels than the data
can support one coefficient each for.
:::

## Interaction terms

Suppose eu-west runs on a leaner fleet than us-east, fewer machines behind a comparable amount of
client traffic, so a given rise in concurrent load costs eu-west more latency than it costs
us-east. A model with one common load coefficient for every region has no way to represent "load
matters more in one region than another."

::: {#fig-interaction-slopes}
```{=html}
<iframe src="../_generated/chapter-04-fig-interaction-slopes.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Observed latency by concurrent load, split by region, with a dashed common-slope line ignoring
region entirely and two solid lines from a model that lets each region carry its own slope. As
the slider raises eu-west's load sensitivity, the dashed line increasingly misses both regions
while the two solid lines keep tracking each one.
:::

@fig-interaction-slopes fits four versions of this scenario, each drawing eu-west's servers as
more load-sensitive than us-east's by a growing multiplier. At the smallest multiplier, where the
two regions barely differ, a single common slope and two region-specific slopes fit almost
identically: 29,039 against 29,038 in summed squared residuals. At the largest multiplier, the
two regions' slopes have pulled apart to 0.35 ms per point of load in us-east against 0.97 ms per
point in eu-west, and the common-slope model's fit has degraded to 47,537 against the
interaction model's steady 29,038.

An *interaction term* multiplies two predictors together, letting one predictor's slope depend on
the level of another instead of forcing a single shared slope across every group:

$$Y = \beta_0 + \beta_1 X_1 + \gamma D + \delta (X_1 \times D) + \varepsilon$$

Here $D$ is 1 for a eu-west request and 0 for us-east, so the fitted slope on load works out to
$\beta_1$ in us-east and $\beta_1 + \delta$ in eu-west. In other words, $\delta$ is not an
adjustment to the intercept the way $\gamma$ is; it is an adjustment to the slope itself, active
only for the group the indicator flags.

A single shared load coefficient, fit without the interaction term, reports one average slope
that under-predicts eu-west's load sensitivity and over-predicts us-east's once the two regions
diverge enough, understating the risk a capacity-planning decision for eu-west's fleet would need
to see.

::: {.callout-note}
An interaction term adds a parameter to the model the same way any predictor does, and the
Measuring fit section later in this chapter's caution about noise predictors inflating R-squared
applies to interaction terms just as much as to plain ones. Add an interaction because a
mechanism explains why one predictor's effect should depend on another, not because trying one
more term is easy.
:::

## Multicollinearity and the variance inflation factor

Two rulers that always agree cannot tell a carpenter anything a single cut did not establish on
its own. Two predictors that move together carry much of the same information, and separating
their individual effects gets harder the more closely they agree, the same "no multicollinearity"
assumption named earlier in this chapter, now with a way to check it.

::: {#fig-vif}
```{=html}
<iframe src="../_generated/chapter-04-fig-vif.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Variance inflation factor for three predictors on a log scale, as payload size and a stand-in
response-size field are pushed to move together more tightly. Dotted and dashed reference lines
mark the conventional VIF cutoffs of 5 and 10; concurrent load, unrelated to either one, barely
moves off the floor.
:::

@fig-vif tracks the variance inflation factor for three predictors, payload size, concurrent
load, and a response-size field, as payload size and response size are pushed to move together
more tightly. At a correlation of 0.32, both hover at 1.12, indistinguishable from three unrelated
predictors. At a correlation of 0.91, both climb past 5, the conventional threshold marked by the
dotted line. By a correlation of 1.00, both predictors' VIF has climbed above 1,300. Concurrent
load, uncorrelated with either one throughout, holds at 1.02 the entire time.

The *variance inflation factor* for predictor $j$, a diagnostic introduced by Marquardt (1970)
[@marquardt1970] alongside his work on ridge regression, is computed by regressing that predictor
on every other predictor in the model and reading off the resulting R-squared:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the R-squared from that auxiliary regression, not from the model predicting the
outcome. In other words, a VIF of 5 for payload size means payload's coefficient carries a
standard error five times larger than it would carry if payload size were completely uncorrelated
with every other predictor in the model, the concrete cost of multicollinearity named earlier in
this chapter's OLS assumptions.

A high VIF does not mean a predictor lacks a meaningful relationship with the outcome; it means
the data cannot cleanly separate that predictor's effect from a correlated one's, so the
coefficient on either could shift substantially with a different sample. Shipping a model that
reports payload's effect as reliable, when its VIF sits above 1,300, would overstate how
confidently that coefficient's specific value can be trusted.

::: {.callout-tip}
VIF thresholds of 5 and 10 are conventions, not fixed rules, the same way Chapter 2 treats alpha
of 0.05 and power of 0.8 as defaults rather than requirements. A VIF just above 5 on a predictor
central to the question a model exists to answer deserves more scrutiny than the same VIF on a
predictor headed for removal regardless.
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
between false positives and false negatives that any classification decision has to make. The
next section puts numbers on that trade-off directly.

## Classification metrics: accuracy, precision, and recall

Fitting the logistic curve above is only half the job. A fitted model still needs a threshold,
a probability above which it predicts a timeout, and once that threshold is set, some way to
grade how well its predictions matched what happened.

::: {#fig-confusion-threshold}
```{=html}
<iframe src="../_generated/chapter-04-fig-confusion-threshold.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Counts of the four possible outcomes for 400 requests at a chosen threshold: correctly flagged
timeouts, wrongly flagged healthy requests, correctly cleared healthy requests, and missed
timeouts. Raising the threshold shrinks the false-positive bar and grows the false-negative bar,
while accuracy, shown in the corner, barely moves.
:::

@fig-confusion-threshold counts those four outcomes for the same 400 requests at five different
timeout-prediction thresholds. At a loose threshold of 0.20, the model catches 149 of 155
timeouts (a recall of 0.961) but wrongly flags 61 healthy requests along the way (a precision of
0.710). At a strict threshold of 0.80, precision climbs to 0.933, but recall falls to 0.626, now
missing 58 timeouts to avoid raising 7 false alarms.

A *confusion matrix* organizes every prediction into four counts: a true positive is a timeout
the model predicted and the request confirmed, a false positive is a timeout the model predicted
that the request did not confirm, a true negative is a healthy request the model correctly
cleared, and a false negative is a timeout the model missed entirely. From those four counts,
three summary metrics follow:

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad
\text{precision} = \frac{TP}{TP + FP}, \qquad
\text{recall} = \frac{TP}{TP + FN}$$

In other words, accuracy is the share of every prediction that matched what happened; precision
is, of every request the model flagged as a timeout, the share that confirmed; and recall is, of
every request that confirmed as a timeout, the share the model caught. This use of "precision"
names a share of correct positive predictions, a different sense from the estimation precision
discussed earlier in this chapter, where the word described how tightly a confidence interval
clustered around a parameter.

A false positive here means paging an on-call engineer or triggering a failover for a request
that would have completed on its own, wasted effort and, repeated often enough, alert fatigue
that makes the next legitimate page easier to miss. A false negative means a timeout goes
unflagged, so no fallback kicks in and a user-facing request hangs with nothing catching it. A
team that only tracks accuracy, 0.833 at the loosest threshold in the figure above and 0.838 at
the strictest, would conclude the threshold barely matters, missing that precision and recall
each swing by more than twenty percentage points across that same range.

::: {.callout-tip}
This section covers only the introductory vocabulary a logistic regression's threshold needs.
Chapter 7 builds the fuller classification-diagnostics toolkit, receiver operating
characteristic curves and the area under them, on top of the confusion matrix defined here.
:::

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

::: {#fig-significance}
```{=html}
<iframe src="../_generated/chapter-04-fig-significance-vs-sample-size.html" width="100%" height="480"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The estimated effect (0.1 ms) never moves. Only its 95% confidence interval does: wide and
straddling zero at small sample sizes, narrow and clear of zero once the sample is large enough.
The marker turns red the moment the p-value crosses 0.05, at a sample size where 0.1 ms is still
too small to act on.
:::

@fig-significance holds that same 0.1 ms effect fixed and grows only the sample size. Between
100,000 and 500,000 requests the confidence interval pulls away from zero and the p-value drops
below 0.05, so the effect becomes statistically significant without becoming any larger. Nothing
about the effect changed; only the precision of the estimate did.

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

::: {.callout-warning}
Lasso and Ridge penalize coefficient magnitude directly, so a predictor measured in KB and a
predictor that is a 0/1 region indicator get penalized on entirely different scales unless every
predictor is standardized first. Skipping that step lets whichever predictor happens to carry
the largest raw scale absorb most of the penalty, regardless of how much it explains.
:::

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

As the prior variance shrinks, equivalent to $\lambda$ growing, the fitted line on the left
visibly flattens away from the fixed, dashed OLS reference, while the posterior on the right
narrows and its center moves toward zero, tracing the same shrinkage Ridge regression's
penalty term produces.
:::

@fig-bayesian-ridge shows this pattern from two angles at once: the left panel grounds it in
the fitted line itself, which pulls away from the OLS reference as the prior tightens, and the
right panel shows the full distribution behind that line, instead of one point estimate,
tracing out the same shrinkage Ridge regression produces as the prior tightens.

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
