# Probability and Distributions

Suppose an anomaly detector pages the on-call engineer for the checkout API at 3 a.m. Before
reaching for a laptop, there is a question worth asking first: given that the alert fired, how
likely is it that something is wrong?

*Probability* is the branch of mathematics that answers questions shaped like this one.
*Distributions* describe the range of values a random quantity, like a latency measurement or
a request count, tends to take.

This chapter works through five questions that come up whenever uncertainty needs a number
attached to it:

1. *Given some new evidence, how should a prior belief update?*
   Bayes' theorem is the formal answer, and it is easy to get wrong by instinct alone.
2. *Which named distribution matches the mechanism producing a number?*
   Nine distributions cover most of what shows up in production systems, each tied to a
   specific generating process rather than chosen by looks.
3. *Why does the normal distribution show up everywhere, even when the underlying process
   is not normal at all?*
   The Central Limit Theorem explains the pattern.
4. *How surprising is a coincidence?*
   The birthday problem shows that collisions happen far sooner than intuition expects, with
   consequences for anything that generates random identifiers.
5. *How do sampling without replacement and uniform random assignment behave?*
   The mechanics behind drawing a canary cohort or assigning a request to a shard.

## Bayes' theorem: what an alert tells you

Suppose a friend says they saw a shark at the beach. Before panicking, it helps to know how
rare shark sightings there usually are in the first place.

Bayes' theorem combines that starting belief with the new claim to land on a better, updated
guess.

*Bayes' theorem* describes how to update the probability of an event given new evidence:

$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$$

Here $P(A \mid B)$ is the probability of $A$ given that $B$ has happened, $P(B \mid A)$ is the
probability of observing $B$ if $A$ were true, and $P(A)$ is the *prior*, the probability of
$A$ before any evidence at all.

Roughly speaking, the theorem takes a starting belief and reshapes it in light of what was
just observed.

::: {#fig-bayes}
```{=html}
<iframe src="../_generated/chapter-03-fig-bayes-base-rate.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

P(true incident given an alert) as the base rate of true incidents falls, holding detector
sensitivity and specificity fixed. At a 1% base rate the probability collapses to about 8.8%,
even from a detector that looks accurate on paper.
:::

@fig-bayes works through a concrete version of the alert example. Suppose the anomaly
detector has a sensitivity of 95% (it fires on 95% of true incidents) and a specificity of 90%
(it correctly stays quiet 90% of the time when nothing is wrong).

It is natural to assume a 95%-sensitive detector means a firing alert is 95% likely to be a
true incident. That assumption misses the *base rate*, the underlying frequency of incidents,
which in a stable production system might be as low as 1% of any given hour.

Bayes' theorem gives the correct answer:

$$P(\text{incident} \mid \text{alert}) = \frac{P(\text{alert} \mid \text{incident}) \, P(\text{incident})}{P(\text{alert})}$$

Plugging in the numbers: the 90% specificity leaves a 10% chance of a false alarm on any hour
with no incident, so $P(\text{alert}) = (0.95)(0.01) + (0.10)(0.99) = 0.1085$, the chance of
an alert firing at all, whether or not there is a true incident.

Dividing the true-incident share of that total, $(0.95)(0.01) = 0.0095$, by $0.1085$ gives
$0.0095 / 0.1085 \approx 0.088$. With a 1% base rate, this works out to roughly 8.8%, not 95%.

::: {.callout-important}
A detector's sensitivity and specificity are different numbers from the probability that a
positive alert is a true one. Confusing them is called the base-rate fallacy, and it is the
same mechanism behind alert fatigue: a detector tuned for high sensitivity on a rare event
will generate more false alarms than true ones unless specificity is also high enough.
:::

That is why on-call teams learn to distrust a noisy alert regardless of its advertised
accuracy.

## Conditional probability

If it is cloudy outside, the chance of rain is higher than on a random day.

Conditional probability is the math behind that shift: how likely something is once you know
something else is true, rather than how likely it is in general.

*Conditional probability*, the probability of one event given that another has occurred, is
the mechanism behind Bayes' theorem rather than a separate idea:

$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}$$

In other words, conditioning on $B$ shrinks the space of outcomes under consideration down to
only those where $B$ happened, and then asks what share of that smaller space also satisfies
$A$.

For the alert example, $P(\text{incident} \mid \text{alert})$ restricts attention to the
world where the alert fired, then asks what fraction of that world contains a true incident.

Every named distribution in the rest of this chapter, and every hypothesis test in Chapter 2,
rests on this same conditioning logic, whether or not the formula is written out explicitly.

## Matching a distribution to the mechanism

A distribution is not a shape to pick because it looks right; it follows from the process
that generated the data. Nine distributions cover most of what a production system produces.

### Normal distribution

Plot the heights of thousands of adults and the result is usually a hump shape: many people
near the middle height, few extremely short or extremely tall people.

The normal distribution is the mathematical version of that hump, and it shows up constantly.

The *normal distribution*, or Gaussian, is a continuous, symmetric, bell-shaped distribution
defined entirely by its mean $\mu$ and standard deviation $\sigma$:

$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Recall from Chapter 1 that checkout-API latency is right-skewed, not normal, because it is
dominated by network and downstream I/O variance with a hard floor near zero.

Contrast that with an in-memory hash lookup with no I/O: its latency is the sum of many
small, roughly independent sources of jitter (cache line access, branch prediction, garbage
collector pauses). Sums of many small independent effects tend toward normal, the mechanism
the Central Limit Theorem formalizes later in this chapter.

::: {#fig-normal-rule}
```{=html}
<iframe src="../_generated/chapter-03-fig-normal-empirical-rule.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The shaded band marks one standard deviation on either side of the mean, and roughly 68% of
the curve's area falls inside it. Dragging the standard-deviation slider stretches or
compresses the whole curve, but the same 68-95-99.7 split of the area holds at every scale.
:::

The *68-95-99.7 rule* says roughly 68% of a normal distribution's mass falls within one
standard deviation of the mean, 95% within two, and 99.7% within three. @fig-normal-rule shows
this rule holding at every scale of standard deviation.

::: {.callout-tip}
A value more than three standard deviations from the mean occurs in under 0.3% of a normal
distribution, which is why three-sigma bands are a common trigger for automated outlier
alerts. The threshold only holds if the underlying metric is close to normal in the first
place.
:::

### Standard normal distribution

The *standard normal distribution* is a normal distribution with $\mu = 0$ and $\sigma = 1$.
Converting any normal value $x$ into a *z-score*, $z = (x - \mu) / \sigma$, expresses it in
standard deviations from the mean.

::: {#fig-zscore}
```{=html}
<iframe src="../_generated/chapter-03-fig-standard-normal-zscore.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The tail probability P(Z > z) beyond a given z-score. Moving the threshold from 0.5 to 3.0
standard deviations shrinks the tail from roughly 31% of the distribution to about 0.1%.
:::

@fig-zscore shows why a z-score threshold makes a good anomaly trigger: the probability of a
value landing that far out drops fast, so a fixed threshold like $z > 3$ flags a small share
of normal variation while still catching outliers that matter. A p99 checkout latency of
340ms against a service mean of 200ms and a standard deviation of 40ms gives a z-score of 3.5,
worth a page regardless of whether the metric is in milliseconds or requests per second.

::: {.callout-tip}
That is what makes z-scores useful for comparing values across services with different
latency scales: a z-score of 3 means the same thing (three standard deviations above typical)
whether the underlying metric is measured in milliseconds or requests per second.
:::

### Uniform distribution

Spin a fair wheel with numbers 1 through 100 marked evenly around the edge, and every number
has the same chance of coming up.

The uniform distribution describes that kind of situation: every value in a range is just as
likely as every other value.

The *uniform distribution* assigns equal probability to every value within a range and none
outside it:

$$f(x) = \frac{1}{b-a} \quad \text{for } a \le x \le b$$

A feature-flag rollout that assigns each user a uniformly random number between 0 and 1, then
enables the feature for anyone below a chosen threshold, is a direct application.

::: {#fig-uniform}
```{=html}
<iframe src="../_generated/chapter-03-fig-uniform-rollout.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A uniform draw on [0, 1] split by a rollout threshold. Moving the threshold from 0.05 to 1.00
grows the treatment share from 5% of users to all of them, matching the threshold value.
:::

@fig-uniform shows the mechanism directly: because every draw is equally likely anywhere in
[0, 1], the share of users below a threshold equals the threshold. A 10% canary rollout is
just a threshold of 0.10, and doubling that threshold to 0.20 doubles the canary's size, no
distributional surprises involved.

Every user has an equal chance of landing in the treatment group, which is the completely
randomized design Chapter 2 introduced for experiment assignment.

### Bernoulli distribution

Flipping a single coin has two outcomes: heads or tails.

The Bernoulli distribution describes any single yes-or-no event like that, one trial with a
fixed chance of success, whether the event is a coin flip, a single dice roll landing on six,
or a single request timing out.

The *Bernoulli distribution* describes a single trial with two outcomes, success or failure,
with probability $p$ of success:

$$f(x) = p^x (1-p)^{1-x} \quad \text{for } x \in \{0, 1\}$$

::: {#fig-bernoulli}
```{=html}
<iframe src="../_generated/chapter-03-fig-bernoulli-trial.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A single Bernoulli trial's two possible outcomes, at success probabilities from 10% to 90%.
The two bars always sum to 100%, since a Bernoulli trial has no third outcome.
:::

@fig-bernoulli is the building block behind every other distribution in this section: a
binomial count is a sum of Bernoulli trials, a geometric count is the wait for the first
Bernoulli success, and the sharding logic later in this chapter reduces to a Bernoulli trial
per shard when only one shard's outcome is in question. Whether a single request times out,
whether a single canary request lands in the treatment group, and whether a single
feature-flag check evaluates true are all Bernoulli trials.

### Binomial distribution

Flip a coin 10 times and count how many land heads. The total could be anywhere from 0 to 10,
and some totals are far more likely than others.

The binomial distribution describes that pattern: the number of successes across a fixed
number of repeated yes-or-no trials.

The *binomial distribution* counts the number of successes across $n$ independent Bernoulli
trials, each with the same success probability $p$:

$$f(x) = \binom{n}{x} p^x (1-p)^{n-x} \quad \text{for } x = 0, 1, \dots, n$$

Here $\binom{n}{x}$, read as *n choose x*, counts the number of distinct orders in which $x$
successes can occur among $n$ trials. Multiplying that count by $p^x (1-p)^{n-x}$, the
probability of any one specific order, gives the total probability of ending up with $x$
successes, no matter which trials they land on.

::: {#fig-binomial-conversions}
```{=html}
<iframe src="../_generated/chapter-03-fig-binomial-conversions.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Number of conversions out of a fixed 50-visitor bucket, by the true underlying conversion
probability. Both the center and the spread of the distribution shift as that probability
rises.
:::

The number of failed requests in a batch of 100, or the number of users who convert out of a
fixed-size A/B test bucket, is binomial. @fig-binomial-conversions fixes the bucket at 50
visitors and moves the true conversion probability, showing how the distribution of possible
conversion counts shifts and spreads as that probability rises.

Part 3 builds directly on this distribution: a Bayesian A/B test of a conversion rate treats
the number of conversions as binomial and updates a prior belief about the underlying
conversion probability as data arrives.

::: {.callout-warning}
The binomial model assumes every trial shares the same success probability and that trials are
independent. A batch of requests hit by a partial outage breaks both assumptions at once:
failures cluster together, so the count of failures shows more variance than a binomial model
predicts, the same overdispersion pattern the Poisson section below describes.
:::

::: {.callout-tip}
When both $np$ and $n(1-p)$ are at least about 10, a binomial distribution is close enough to
normal for a normal approximation to work well in a pinch, a direct consequence of the Central
Limit Theorem covered later in this chapter.
:::

### Geometric distribution

Flipping a coin over and over while waiting for the first heads could take one flip or could
take many.

The geometric distribution describes how many attempts it typically takes to reach the first
success, when each attempt carries the same fixed chance of working.

The *geometric distribution* models the number of trials needed to reach the first success,
given a fixed per-trial success probability $p$:

$$f(x) = p (1-p)^{x-1} \quad \text{for } x = 1, 2, 3, \dots$$

::: {#fig-geometric}
```{=html}
<iframe src="../_generated/chapter-03-fig-geometric-retries.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The distribution of attempts needed before a flaky call succeeds, and how its expected value
rises sharply as reliability drops.
:::

A flaky downstream call that succeeds with probability $p$ on each attempt, retried until it
finally works, produces a geometric number of attempts. @fig-geometric shows how the expected
number of retries climbs as the per-attempt success probability falls.

::: {.callout-note}
Useful for setting a retry budget: a call with a 30% success rate needs, on average, more than
three attempts before it succeeds.
:::

### Poisson distribution

A library counts how many people walk in during any given hour. Some hours see three
visitors, others see none, others see ten.

The Poisson distribution describes that pattern of counts: events happening independently at
some steady average rate over a fixed stretch of time.

The *Poisson distribution* models the count of events occurring in a fixed interval, given a
known average rate $\lambda$:

$$f(x) = \frac{\lambda^x e^{-\lambda}}{x!} \quad \text{for } x = 0, 1, 2, \dots$$

::: {#fig-poisson}
```{=html}
<iframe src="../_generated/chapter-03-fig-poisson-shape.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The Poisson distribution's shape at low request rates versus high ones. At high enough
$\lambda$, the shape looks close to normal, another instance of the Central Limit Theorem's
reach.
:::

Requests arriving at an API endpoint in a one-second window is the standard example: if
requests arrive independently of one another at a steady average rate, the count in any given
second is Poisson. @fig-poisson shows the distribution's shape shifting as the average rate
grows.

::: {.callout-note}
A Poisson distribution's variance always equals its mean, both equal to $\lambda$. When a
measured count's variance runs much higher than its mean, that mismatch, called
overdispersion, is a sign the events are not arriving independently and a Poisson model no
longer fits.
:::

### Exponential distribution

If visitors arrive at a shop at some steady average rate, the gap between one visitor and the
next is sometimes short, sometimes long.

The exponential distribution describes how long that wait tends to be, the time between
events that happen at a steady rate.

The *exponential distribution* models the time until an event occurs, given a constant rate
$\lambda$:

$$f(x) = \lambda e^{-\lambda x} \quad \text{for } x \ge 0$$

::: {#fig-exponential}
```{=html}
<iframe src="../_generated/chapter-03-fig-exponential-wait.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Wait-time density at rates from 0.25 to 4 events per unit time. A higher rate pulls the whole
curve toward zero: the mean wait is always $1/\lambda$, so the probability of a short wait
climbs fast as the rate increases.
:::

@fig-exponential also gives a direct way to compute with the distribution: at a constant rate
of one event every two units of time ($\lambda = 0.5$), the mean wait is 2 units, and the
probability the next event arrives within one unit is $1 - e^{-0.5} \approx 39\%$, read
straight off the shaded area under the curve up to that point.

The time between consecutive requests in a Poisson arrival process is exponentially
distributed. So is the time until a service instance fails, under the assumption that failure
risk does not change with how long the instance has been running (the distribution's defining
*memoryless* property).

::: {.callout-warning}
This assumption is often wrong for hardware, which tends to wear out over time, but reasonable
for many software failure modes that are triggered by external conditions rather than
accumulated wear.
:::

### Lognormal distribution

::: {#fig-lognormal}
```{=html}
<iframe src="../_generated/chapter-03-fig-lognormal-latency.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The checkout-API latency density from Chapter 1, redrawn as a lognormal curve at growing
spread. As sigma rises, the mean (dashed line) pulls further above the median (dotted line);
at sigma = 0.90 the two sit far apart, while at sigma = 0.20 they nearly coincide.
:::

@fig-lognormal is the same latency data Chapter 1 built its running example from
(`RNG.lognormal(mean=np.log(45), sigma=0.35, ...)`), redrawn here as a density curve instead
of a histogram. Every response-time figure in this book, and the payload and duration figures
in several later chapters, is generated the same way, but this chapter never gave that
generating mechanism its own name.

A quantity behaves lognormally whenever its logarithm, not the quantity itself, follows a
normal distribution. Response times are the standard example: a request's latency is the
product of many independent multiplicative slowdowns (a slower disk read, a retry, a queueing
delay), and multiplying many positive random factors together produces the same right-skewed
hump this book's latency figures have shown since Chapter 1: a floor at zero, a peak near the
typical case, and a long tail of rare, much slower requests.

That shape carries a direct engineering consequence: the mean of a lognormal metric always
sits above its median, and the gap between the two grows with the spread. A dashboard that
reports only "average response time" over a lognormal metric will read higher than what most
requests experienced, which is why Chapter 1 recommended the median or a trimmed statistic for
latency reporting in the first place. @fig-lognormal makes that gap visible: dragging sigma
from 0.20 to 0.90 pulls the mean further past the median while the median itself barely moves.

Computing with a lognormal metric means working in log space first. Taking the natural log of
every observed latency turns the right-skewed lognormal data back into ordinary normal data,
at which point every tool built for the normal distribution (a z-score threshold, a confidence
interval built with the CLT) becomes usable again. Chapter 9's regression diagnostics handle
payload-size noise the same way: fit the model on the log scale, then convert the predictions
back with an exponential.

The *lognormal distribution* describes a random variable $X$ such that $\ln(X)$ follows a
normal distribution with mean $\mu$ and standard deviation $\sigma$:

$$f(x) = \frac{1}{x \sigma \sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}} \quad \text{for } x > 0$$

In other words, $\mu$ and $\sigma$ describe the normal distribution of the logged values, not
the mean and standard deviation of $X$ itself; the mean of $X$ works out to
$e^{\mu + \sigma^2/2}$, which is why it always lands above the median, $e^\mu$.

::: {.callout-note}
A lognormal variable is never negative and has no upper bound, matching two facts about
latency that a normal distribution would get wrong: a request cannot finish in negative time,
and a rare request can take far longer than a typical one with no theoretical ceiling.
:::

## Mean and variance of a distribution

A distribution's mean is its long-run average value if the random process ran forever.

Its variance measures how spread out the outcomes tend to be around that average, the same
bounciness idea from Chapter 1, applied to a known mathematical shape instead of a dataset.

Two numbers summarize any distribution's center and spread. The *mean*, or expected value, is

$$E(X) = \sum x \, P(X=x)$$

and the *variance* is

$$\text{Var}(X) = E(X^2) - \big(E(X)\big)^2$$

In other words, the mean weights each possible outcome by how often it occurs, and the
variance is the average squared distance between an outcome and that mean.

$E(X^2) - (E(X))^2$ is an algebraically equivalent shortcut for computing that average
squared distance without having to subtract the mean from every outcome first.

::: {.callout-warning}
That shortcut is convenient by hand but risky in floating-point code: when the mean is large
relative to the spread, $E(X^2)$ and $(E(X))^2$ are two large, nearly equal numbers, and
subtracting them can lose precision to cancellation. A running algorithm like Welford's method
avoids the problem by updating variance incrementally instead of computing the two terms
separately.
:::

Recall from Chapter 1 that these are the population-level counterparts of the sample mean and
sample variance computed from data. Every distribution introduced in this chapter has a known
formula for both, which is part of what makes naming the right distribution useful in the
first place.

## Chebyshev's inequality: a bound that needs no assumptions

::: {#fig-chebyshev}
```{=html}
<iframe src="../_generated/chapter-03-fig-chebyshev-bound.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Two ways to bound how far a metric can stray from its mean. The dashed line assumes the
metric is normal; the solid line makes no assumption about its shape at all, and pays for that
safety with a much looser bound.
:::

@fig-chebyshev compares two answers to the same question: how much of a metric's mass can sit
more than $k$ standard deviations from its mean? At $k = 3$, a normal metric puts under 0.3%
of its mass out that far, the three-sigma rule from earlier in this chapter. The bound that
makes no assumption about the metric's shape only guarantees under 11.1%, a much weaker but
far more widely applicable claim.

Most metrics an on-call team watches, queue depth, retry counts, a newly added latency
percentile with no history yet, have not been checked for normality, and some (the
right-skewed latency metric this chapter has used throughout) are known not to be normal. An
alerting threshold built on the normal-distribution three-sigma rule can be quietly wrong for a
metric like that: it will either fire too often or miss the tail it was meant to catch, because
the metric's tail is thicker than a normal distribution's. Chebyshev's inequality gives a
threshold that holds regardless of the metric's shape, at the cost of being far more
conservative.

Computing a Chebyshev bound needs only the mean and standard deviation, quantities every
monitoring system tracks. Divide 1 by $k^2$: a threshold of 2 standard deviations bounds the
tail at 25%, a threshold of 3 standard deviations bounds it at about 11.1%, and a threshold of
5 standard deviations bounds it at 4%. No histogram, percentile estimate, or distributional
assumption is needed to compute any of those numbers.

*Chebyshev's inequality* states that for any random variable $X$ with finite mean $\mu$ and
variance $\sigma^2$, and any $k > 0$:

$$P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}$$

In other words, no matter what shape a distribution takes, so long as it has a finite mean and
variance, the chance of landing more than $k$ standard deviations from the mean can never
exceed $1/k^2$.

::: {.callout-tip}
Chebyshev's bound is loose enough that it should not replace a normal-distribution threshold
once a metric is known to be close to normal. It earns its place for a metric whose shape has
not been checked yet, or is known to be far from normal, where a distribution-free bound is
worth more than a tight bound resting on the wrong assumption.
:::

## Joint distributions and independence

::: {#fig-joint}
```{=html}
<iframe src="../_generated/chapter-03-fig-joint-cache-sla.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Two joint distributions over the same two variables, cache outcome and SLA outcome, sharing
the same marginal rates (60% hit, 70% meets SLA) but distributing that probability across the
four combinations in sharply different ways.
:::

@fig-joint tracks two variables at once for the same request: whether it hit the cache (hit or
miss) and whether it met its latency SLA (met or missed). The "assumed independent" scenario
multiplies the two marginal rates cell by cell, 60% hit times 70% met gives 42% of requests
landing in the hit-and-met cell. The "observed" scenario keeps the same two marginal rates but
concentrates far more of the probability in that same cell, 51% instead of 42%, because a
cache hit and an SLA-met outcome are not happening independently of each other.

A *joint distribution* over two random variables $X$ and $Y$ assigns a probability to every
combination of their outcomes at once, capturing how the two move together, something each
variable's own distribution cannot show on its own. Chapter 1's correlation coefficient is a
single number summarizing how far a joint distribution's shape strays from what independence
would predict; this section spells out that comparison directly, cell by cell.

Knowing only that 60% of requests hit the cache and 70% meet their SLA answers a narrower
question than "of the requests that hit the cache, what share met their SLA?" Two variables
can share identical marginal rates while pointing to opposite engineering conclusions: in the
independent scenario, caching has no bearing on whether a request meets its SLA, but in the
observed scenario, a cache hit often comes packaged with an SLA success, evidence that the
cache is helping reduce tail latency, a causal signal rather than a coincidence of two
unrelated rates.

Checking independence is a single multiplication: compute the product of the two marginal
probabilities, $P(\text{hit}) \times P(\text{SLA met})$, and compare it to the joint
probability, $P(\text{hit and SLA met})$, read directly off the joint table. When the two
numbers match, as in the "assumed independent" scenario, the variables carry no information
about each other. When they diverge, as in the "observed" scenario (42% predicted against 51%
observed), that gap is the signal a correlation coefficient is built to detect.

Formally, two events $A$ and $B$ are *independent* when their joint probability factors into
the product of their individual probabilities:

$$P(A \cap B) = P(A) \, P(B)$$

In other words, knowing that $A$ happened changes nothing about how likely $B$ is, and vice
versa; conditioning on one event leaves the other's probability unchanged.

::: {.callout-warning}
A joint distribution's two marginal distributions never reveal how the variables relate to
each other; two sharply different joint tables, like the two shown in @fig-joint, can produce
identical marginals. Reporting only marginal rates, "60% cache hit rate, 70% SLA compliance,"
hides which of these two sharply different stories is playing out in production.
:::

## The Law of Large Numbers

::: {#fig-lln}
```{=html}
<iframe src="../_generated/chapter-03-fig-lln-convergence.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Four independent simulated runs of a rolling conversion-rate average. Early on, the four lines
scatter widely; by a few hundred requests, all four have settled close to the true rate marked
by the dashed line.
:::

@fig-lln runs the same simulated experiment four times: 3,000 simulated requests, each
converting at some fixed true probability, with a running average tracked after every request.
The four runs start scattered (a run of a dozen lucky or unlucky requests can push an early
average far from the true rate) and converge to the same dashed line as more requests
accumulate, regardless of which run is being watched.

The *Law of Large Numbers* (LLN) is the reason an observed rate gets trusted as an estimate of
a true rate at all. Without it, there would be no basis for treating "42 conversions out of
1,000 visitors" as evidence about the underlying conversion probability, since a small
sample's average could sit anywhere.

Every A/B test in this book, and every canary rollout in this chapter, leans on the LLN
without naming it: watching a metric over a growing sample and trusting that the average
settles down is the LLN in practice. A dashboard that reports a conversion rate from only 20
visitors sits in the regime where the LLN has not yet done its work, the same small-sample
instability @fig-lln shows in its first few hundred requests.

Computing with the LLN needs nothing beyond tracking a running average as data accumulates:
sum the outcomes seen so far and divide by the count. That arithmetic is trivial; the
theorem's substance lies in the guarantee that the answer keeps getting closer to the true
value as the count grows, with no upper limit on how close it can get given enough data.

Formally, the LLN states that as the sample size $n$ grows, the sample mean $\bar{X}_n$
converges to the true population mean $\mu$:

$$\bar{X}_n \xrightarrow{n \to \infty} \mu$$

In other words, no matter how volatile any single observation is, averaging enough of them
washes out that volatility and leaves a number that settles on the truth.

::: {.callout-note}
The LLN and the Central Limit Theorem answer different questions about the same sample mean:
the LLN says where the sample mean is heading (the true mean), and the CLT, covered next, says
what shape its distribution takes on the way there (normal, regardless of the population's own
shape).
:::

## The Central Limit Theorem

Roll one die and the result is unpredictable, spread evenly across six numbers.

But roll ten dice, average them, and repeat that experiment many times: those averages start
clustering into a familiar bell shape. The Central Limit Theorem explains why averages behave
this way.

::: {#fig-clt-three-panel}
```{=html}
<iframe src="../_generated/chapter-03-fig-clt-three-panel.html" width="100%" height="420"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Garbage-collection pause durations across 6,000 simulated JVM instances: raw pauses on the
left, means of 8 pauses in the middle, means of 32 pauses on the right. The right-skewed shape
from the left panel is nearly gone by the right panel.
:::

@fig-clt-three-panel makes the same point three different ways side by side, using a new
production scenario: garbage-collection pause durations sampled across a fleet of JVM
instances. The left panel plots 6,000 raw pause durations, heavily right-skewed, most pauses
short with a long tail of rare, much longer ones. The middle panel plots the mean of every
group of 8 of those same pauses; the tail is shorter and the peak has shifted. The right panel
plots the mean of every group of 32; the shape is close to a symmetric bell, even though not
one of the 6,000 raw pauses that fed into it was drawn from a normal distribution.

Nothing about the three panels required naming a formula first. Averaging 8 raw pause
durations, then 32, and watching the histogram's shape change is the entire content of the
theorem below, before any notation gets involved.

The *Central Limit Theorem* (CLT) states that, given a large enough sample, the distribution
of the sample mean approaches normal, regardless of the shape of the population the sample was
drawn from.

::: {#fig-clt}
```{=html}
<iframe src="../_generated/chapter-03-fig-clt-simulation.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Sample means of a skewed parent distribution become normal-shaped as sample size grows, the
same property that makes the t-test in Chapter 2 usable even on non-normal data at moderate
sample sizes.
:::

@fig-clt simulates 4,000 samples drawn from a heavily right-skewed exponential distribution,
at growing sample sizes, and tracks how the distribution of sample means transforms from
skewed to bell-shaped as sample size increases.

This is because averaging cancels out extreme values from both directions. A sample mean
drawn from a right-skewed population is only pulled upward by outliers to the extent those
outliers outweigh the many ordinary observations averaged alongside them, and as the sample
grows, that influence shrinks.

::: {.callout-tip}
This is the mechanism behind Chapter 2's claim that a t-test can tolerate non-normal data once
the sample is large enough. The test does not need the underlying data to be normal, only the
sample mean, and the CLT delivers that for free at scale.
:::

How large "large enough" needs to be depends on how skewed or heavy-tailed the underlying
distribution is. A mildly skewed metric might converge to a usable normal shape by a few
dozen observations.

A latency distribution with rare, extreme spikes (a long tail with much of the variance
concentrated in a small number of outlier requests) can need a sample far larger than the
usual rule-of-thumb size before the sample mean behaves normally enough to trust.

## The birthday problem: collisions in a space that felt safe

In a room of just 23 people, there is better than a fifty-fifty chance two of them share a
birthday, even with 365 possible birthdays to choose from.

That surprises most people, because what matters is not the number of people but the number
of pairs of people to compare.

The *birthday problem* asks how many people need to be in a room before the probability that
two of them share a birthday passes 50%. The answer, 23, is far smaller than most people
guess, because the relevant count is not the number of people but the number of *pairs* of
people, and pairs grow quadratically.

Those 23 people produce 253 distinct pairs, each an independent chance of a match.

::: {#fig-birthday}
```{=html}
<iframe src="../_generated/chapter-03-fig-birthday-collision.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Collision probability for randomly generated identifiers, by identifier length in bits. The
x-axis is on a log scale: an 8-bit ID space collides almost immediately, while a 64-bit space
needs billions of IDs before collisions become likely.
:::

This is not a party trick. It is the same mechanism behind hash and identifier collisions,
formalized as the *birthday bound*: for an identifier space of size $N$, the number of
randomly generated identifiers needed before a collision becomes likely scales with
$\sqrt{N}$, not $N$.

@fig-birthday shows how quickly collision probability climbs as more identifiers are
generated, at a few different identifier lengths.

The birthday bound is why a widely used hash function's security depends on a space far
larger than an intuition about brute-force guessing would suggest.

::: {.callout-important}
In February 2017, researchers at CWI Amsterdam and Google publicly demonstrated the first
collision in SHA-1, a hash function considered cryptographically broken by theory for years
but not yet broken in practice. The attack, named SHAttered, cost roughly 6,500 CPU-years and
100 GPU-years, well within reach of a well-resourced adversary. It exploited the birthday-bound
structure this section describes rather than any flaw specific to SHA-1's internal design
[@cwigoogle2017].
:::

Any system that generates random identifiers, such as trace IDs, session tokens, or cache
keys, needs identifier lengths chosen with the birthday bound in mind, not the naive
assumption that collisions only become likely once the identifier space is nearly exhausted.

## Selecting without replacement: canary cohorts

Drawing colored marbles one at a time from a bag, without putting each one back, changes the
odds for every draw after the first.

Picking a random handful of accounts for a test group works the same way: once an account is
picked, the odds for the remaining picks shift.

Marble-and-urn problems are, structurally, sampling-without-replacement problems: once a
marble is drawn, it is gone, and every subsequent probability is conditioned on that removal.
The same mechanics apply to selecting a canary cohort from a pool of accounts.

Suppose a rollout needs a canary group of 5 accounts drawn at random from a pool of 200, 3 of
which are known to be unusually heavy users whose behavior would skew the canary's results.

The probability that none of the 3 heavy accounts land in the 5-account canary follows the
same hypergeometric logic as the probability of avoiding a specific marble while drawing
several from a bag: each draw shrinks the remaining pool, and the removed accounts (or
marbles) never return to the pool for the next draw.

::: {.callout-note}
Multiplying the shrinking odds across all 5 draws, (197/200)(196/199)(195/198)(194/197)(193/196),
works out to about 92.6%. So a canary assembled by a documented random draw still carries
roughly a 7% chance that at least one heavy account slips in by chance alone, a number worth
knowing before treating any single canary run as decisive.
:::

That calculation has a name: it is a *hypergeometric* one, 5 draws made without replacement
from a finite pool of 200, where 3 of the 200 carry the outcome being tracked. The
*hypergeometric distribution* formalizes this setup and gives the probability of drawing $k$
successes:

$$f(k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}$$

where $N$ is the pool size (200 accounts), $K$ is the number of successes in the pool (3 heavy
accounts), $n$ is the number of draws (5 canary slots), and $k$ is the number of successes
drawn. Setting $k = 0$ in this formula and working through the binomial coefficients reproduces
the same 92.6% the callout above computed one draw at a time; the shrinking-odds multiplication
and the formula are two routes to one number.

The binomial distribution earlier in this chapter is the same idea with one assumption
dropped: binomial trials draw with replacement, or from a pool so large that removing one item
barely moves the odds, while the hypergeometric distribution tracks what happens when the pool
is small enough that each draw meaningfully changes the odds for the next one, the situation
the canary cohort above sits in.

By contrast, choosing a canary cohort by hand, rather than through a documented random
process, is a quiet source of selection bias, the same problem Chapter 1 introduced. A
canary chosen to "look representative" is not a canary chosen at random, and the two are not
interchangeable.

## Uniform outcomes over many categories: sharding

Rolling a fair six-sided die, each number from 1 to 6 has the same one-in-six chance of
coming up.

Assigning requests to servers through a well-designed hash works the same way: every server
should get a fair, equal share of the traffic, just like every face of the die.

Die-roll problems generalize directly to a common infrastructure pattern: assigning a request
to one of $N$ shards based on a hash of its key, where the hash function is designed so the
assignment behaves like a fair $N$-sided die.

The probability a specific shard receives a given request is $1/N$, and the probability a
request lands in a given range of shards follows by adding individual probabilities.

::: {.callout-warning}
All of this rests on one assumption: that the hash function distributes keys uniformly, the
same fairness assumption a die-roll problem takes for granted. When that assumption breaks (a
hash function that clusters certain keys onto the same shard), the practical result is a hot
shard carrying disproportionate load, the sharding equivalent of a loaded die.
:::

Part 3 turns from describing distributions to using them for a specific practical decision:
whether a treatment beats a control, expressed not as a single p-value but as a full
posterior distribution over how much better it is, and how confident that conclusion should be.

## References {.unnumbered}

::: {#refs}
:::
