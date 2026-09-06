# Hypothesis Testing

Recall the checkout API from Chapter 1: an on-call engineer watching a right-skewed latency
distribution, worried that the mean hides a growing slow tail. Suppose that engineer's team
ships a caching layer meant to cut that latency, and the dashboard shows a lower number the
next morning.

Is the cache working, or did traffic happen to be lighter overnight? *Hypothesis testing* is
the machinery for answering that question with a number instead of a guess.

This chapter works through six questions that come up each time an engineer or data
scientist has to decide whether an observed difference means something:

1. *How do I state a claim so a sample of data can test it?*
   Hypothesis statements turn "the cache helped" into something a formula can evaluate.
2. *How do I test a claim about a single average?*
   The Z-test and t-test compare a sample mean against a target value.
3. *What does a p-value tell me, and where does that stop?*
   A p-value is one of the most quoted and most misread numbers in applied statistics.
4. *What are the two ways a test can be wrong, and what does each one cost?*
   Type I and Type II errors trade off against each other, and the trade is a business
   decision, not just a statistical one.
5. *How much data does a decision need before it can be trusted?*
   Statistical power and sample size determine how long a test needs to run.
6. *What happens when there are more than two groups, more than one confounder, or data that
   refuses to look normal?*
   ANOVA, chi-squared tests, ANCOVA, experimental design, and non-parametric alternatives
   cover the cases a simple t-test cannot.
7. *How do I turn a single estimate into a range I can trust, or compare two proportions
   instead of two means?*
   Confidence intervals and the two-proportion z-test extend the machinery above to estimates
   and proportions directly.
8. *What goes wrong when a test gets checked before it is finished, or when many tests run at
   once?*
   Sequential testing and the multiple-comparisons problem cover two of the most common ways a
   correct test still misleads a team in production.

## Stating a hypothesis

Think of a courtroom trial: the defendant is assumed innocent (the null hypothesis) until the
evidence makes that hard to believe.

A hypothesis test works the same way. It never proves a new claim true; it only checks
whether the evidence is strong enough to doubt the default.

A *hypothesis statement* names two competing claims about a population and lets sample data
decide between them. The *null hypothesis*, written $H_0$, is the default: nothing changed.
The *alternative hypothesis*, written $H_1$, is the claim under investigation: something did
change.

For the caching layer, $H_0$ says the mean latency after the rollout equals the mean latency
before it; $H_1$ says the mean latency after the rollout is lower.

$$H_0: \mu_{\text{after}} = \mu_{\text{before}}, \qquad H_1: \mu_{\text{after}} < \mu_{\text{before}}$$

::: {.callout-note}
The test does not try to prove the cache works. It tries to determine whether, assuming the
cache did nothing, the data collected after the rollout is surprising enough to make "it did
nothing" hard to believe.
:::

This is a subtle distinction, and it shapes every result in this chapter: a hypothesis test
never proves $H_1$ true. It only measures how much the evidence strains against $H_0$.

## Testing a single mean: the Z-test and the t-test

Imagine checking whether a coin is rigged by flipping it 30 times. Landing heads far more
than half the time is a clue something is off.

The Z-test and t-test do the same thing for averages: they measure how far a sample's
average sits from what plain chance would predict.

Suppose the team wants to check the new mean latency against a fixed target, 40 ms, the
number the SLA promises.

Before naming a test, a simulation can answer the same question with nothing but arithmetic.
Suppose the team collects 50 requests after the rollout and the sample mean comes out to 34.8
ms, well under target. Years of pre-rollout monitoring pinned the population's spread at a
standard deviation of 20 ms, so a computer can draw thousands of pretend samples from a system
whose average never moved off 40 ms and count how often a mean that low turns up by chance
alone.

```python
import numpy as np
rng = np.random.default_rng(7)
mu0, sigma, n = 40.0, 20.0, 50
sim_means = rng.normal(mu0, sigma, size=(20_000, n)).mean(axis=1)
observed = 34.8
p_simulated = (sim_means <= observed).mean()
# p_simulated = 0.034
```

Out of 20,000 pretend samples drawn from a population that never moved off the 40 ms target,
only about 3.4 percent came in at 34.8 ms or lower. That count answers how surprising the
result would be if the cache changed nothing: rare, not impossible.

The *Z-test* compares a sample mean to a target when the population standard deviation is
known, reaching the same answer without simulating a single sample:

$$z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}}$$

Here $\bar{x}$ is the sample mean, $\mu$ is the target value, $\sigma$ is the population
standard deviation, and $n$ is the sample size.

Roughly speaking, the Z-test asks how many standard errors the sample mean sits from the
target: a $z$ far from zero means the sample mean is unlikely to have occurred by chance if
the target were correct. Plugging in $\bar{x} = 34.8$, $\mu = 40$, $\sigma = 20$, and $n = 50$
gives $z \approx -1.84$ and a one-sided p-value of 0.033, matching the 3.4 percent the
simulation counted directly.

In practice, the population standard deviation is almost never known in advance; it has to
be estimated from the same sample used to estimate the mean. The *t-test* handles that case:

$$t = \frac{\bar{x} - \mu}{S / \sqrt{n}}$$

where $S$ is the sample standard deviation. The t-test uses the t-distribution rather than
the normal distribution to calculate its p-value, because estimating $\sigma$ from the sample
adds extra uncertainty that the normal distribution does not account for.

The t-distribution's shape depends on its *degrees of freedom*, $n - 1$ for this one-sample
test: with few degrees of freedom the distribution has heavier tails than the normal curve.

As $n$ grows large (past roughly 30 observations, a common rule of thumb), those tails thin
out and the t-distribution converges to the normal distribution, so the two tests give nearly
identical results.

::: {.callout-tip}
Default to the t-test unless the population standard deviation is known independently of the
sample being analyzed, a situation that is rare outside long-running quality-control settings.
The t-test accounts for the extra uncertainty of estimating that standard deviation from the
sample, which the Z-test simply ignores.
:::

::: {#fig-t-vs-normal}
```{=html}
<iframe src="../_generated/chapter-02-fig-t-vs-normal.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The t-distribution's tails thin toward the standard normal as degrees of freedom increase.
Below roughly 10 degrees of freedom the extra tail weight is easy to see; above 30 the two
curves are nearly impossible to tell apart.
:::

@fig-t-vs-normal compares the t-distribution against the standard normal distribution as
sample size changes, showing how the t-distribution's heavier tails (and its extra caution
about declaring significance) shrink toward the normal distribution as $n$ grows.

Recall from Chapter 1 that checkout-API latency is right-skewed, not normal. Both tests
assume the underlying data is close to normal, or that the sample is large enough for the
*Central Limit Theorem* to compensate.

The Central Limit Theorem is the result that a sample mean drifts toward a normal
distribution as sample size grows, regardless of how the underlying data is shaped, which is
what lets the sample mean count as approximately normal even when the data behind it is not.

A small sample of skewed latency data is the case where that assumption is worth checking
rather than trusting.

## What a p-value says

If a coin is fair, getting 25 heads out of 30 flips would be surprising, though not
impossible.

A p-value measures how surprising a result would look if nothing unusual were going on: a
small p-value means the result would be a strange coincidence under the assumption nothing
changed.

A *p-value* is the probability of observing a test statistic as extreme as, or more extreme
than, the one calculated from the sample, if the null hypothesis were true.

Suppose the t-test on the latency data returns a p-value of 0.03. That means: if the cache
changed nothing at all, there would be a 3% chance of seeing a latency drop this large or
larger just from sampling variation.

A common threshold, 0.05, marks the line most teams use to call a result statistically
significant, though the threshold itself is a convention, not a law of nature.

Here is where the widely held intuition about p-values breaks down. It is natural to read "p
= 0.03" as "there is a 3% chance the cache did nothing," but that is not what the number
means.

::: {.callout-important}
The p-value is calculated by assuming the null hypothesis is true; it cannot simultaneously
tell you the probability that the same null hypothesis is true. The American Statistical
Association's own guidance is direct: p-values "do not measure the probability that the
studied hypothesis is true, or the probability that the data were produced by random chance
alone" [@asa2016].
:::

A p-value measures how compatible the data is with the null hypothesis, nothing more. It says
nothing about how large the effect is.

A small p-value from a large enough sample can come from an effect too small to matter
operationally, the same way a 0.1 ms latency improvement can be statistically significant
across ten million requests and still not worth shipping.

## Type I and Type II errors: the two ways to be wrong

A smoke detector can fail two ways: it can go off with no fire (a false alarm) or stay silent
during a fire (a missed alarm).

Hypothesis tests share the same two failure modes, and making one kind of mistake rarer tends
to make the other kind more common.

A hypothesis test can fail in two directions. A *Type I error*, or false positive, happens
when the test rejects $H_0$ even though $H_0$ was true: the dashboard says the cache helped,
and it did not.

A *Type II error*, or false negative, happens when the test fails to reject $H_0$ even though
$H_1$ was true: the cache did help, and the test missed it.

The probability of a Type I error is called $\alpha$, conventionally set at 0.05; the
probability of a Type II error is called $\beta$.

::: {.callout-note}
Standard practice fixes $\alpha$ first and only then works out the resulting $\beta$, rather
than weighing the two costs against each other directly. When a missed effect is more
expensive than a false alarm, that convention is worth overriding on purpose, not applying by
default.
:::

::: {#fig-error-tradeoff}
```{=html}
<iframe src="../_generated/chapter-02-fig-type1-type2.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Moving the significance threshold trades Type I error against Type II error. Pulling the
threshold in one direction to catch more true effects always lets in more false positives at
the same time.
:::

@fig-error-tradeoff shows the overlap between the "no effect" and "true effect" sampling
distributions, and how moving the significance threshold shifts the balance between the two
error rates.

These two error rates trade against each other. Tightening $\alpha$ (requiring stronger
evidence before declaring significance) lowers the false-positive rate but raises the
false-negative rate, since true improvements now need to clear a higher bar.

For the caching layer, a Type I error means shipping a cache that does nothing while
believing it works: wasted infrastructure and false confidence. A Type II error means rolling
back a cache that did cut latency, because the test could not tell the improvement apart from
noise.

Neither error is free, and which one costs more depends on what is being tested, not on the
statistics alone.

## Statistical power: how long to run the canary

A smoke detector with a weak sensor can miss a fire that is happening right in front of it.

Statistical power is how good a test is at catching a change that is there, rather than
missing it because too little data was collected to notice.

*Statistical power* is the probability that a test correctly rejects $H_0$ when $H_1$ is
true: $1 - \beta$. A test with high power reliably detects a true effect; a test with low
power can miss meaningful improvements even when they exist.

Power depends on four things: sample size, the significance level $\alpha$, the size of the
effect being tested for, and the variability of the data.

::: {#fig-power-curve}
```{=html}
<iframe src="../_generated/chapter-02-fig-power-curve.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Statistical power rises with sample size, more slowly for smaller effect sizes. A small true
effect can need many times more traffic than a large one before a test reliably notices it.
:::

@fig-power-curve shows how power climbs as sample size grows, for a few different effect
sizes; small effects need dramatically more traffic before the test can reliably see them.

The slider beneath that figure steps through values of *Cohen's d*, a term used but never
defined until now.

*Cohen's d* is a standardized effect size: a way of expressing how far apart two group means
sit that does not depend on the units the underlying metric happens to use.

Comparing a 7.5 ms latency gap against a two-point jump in a conversion-rate percentage is not
a fair comparison on raw numbers alone, since the two metrics sit on different scales with
different natural spreads. Putting both gaps on the same d-scale is what makes it possible to
say a latency change and a conversion-rate change carry a similar amount of evidence, or that
one is a much bigger jump relative to its own noise than the other.

Computing it divides the gap between two means by a pooled standard deviation that blends both
groups' variability into one number:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

A latency gap of 7.5 ms against a pooled standard deviation of 15 ms works out to $d = 0.5$, a
medium-sized effect under the loose convention Cohen proposed: 0.2 small, 0.5 medium, 0.8
large. Those are the same values (plus 0.1 and 0.3) the slider under @fig-power-curve steps
through.

A team running a canary deployment for the caching layer is, whether they call it this or
not, deciding how much statistical power they want before they trust the result.

A commonly used power target is 0.8, an 80% chance of detecting the effect if it exists.

::: {.callout-tip}
Below a power of 0.8, a canary that shows no significant difference is close to
uninformative. It could mean the cache does nothing, or it could mean the test never had a
fair chance to notice a modest improvement that exists.
:::

## Sample size: how many requests before the number can be trusted

Asking 10 people who they will vote for gives a shaky guess about a whole country. Asking
10,000 gives a much steadier one.

Sample size calculations answer a practical question: how many people, or requests, need to
be measured before an estimate can be trusted.

Suppose the team wants to estimate mean latency within a margin of error of 2 ms, at a 95%
confidence level, and prior data suggests a standard deviation of 20 ms. The sample size
needed for a target margin of error $E$ is

$$n = \frac{z^2 \sigma^2}{E^2}$$

where $z$ is the standard normal value for the desired confidence level (1.96 for 95%).
Plugging in the numbers:

$$n = \frac{1.96^2 \times 20^2}{2^2} = 384.16$$

In other words, roughly 385 requests need to be sampled before the estimate of mean latency
can be trusted to within 2 ms, 95% of the time.

This is the same formula whether the thing being measured is fish weight or API latency;
what changes is which numbers describe the population.

::: {.callout-note}
This formula assumes the standard deviation is known ahead of time, usually from a pilot
sample or historical data. If that estimate is wrong, the resulting sample size will be wrong
too, which is why sample size calculations are typically treated as a starting point rather
than a fixed target.
:::

## Confidence intervals for a mean and a proportion

::: {#fig-ci-bootstrap}
```{=html}
<iframe src="../_generated/chapter-02-fig-ci-bootstrap.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Resampling the same 25-request sample thousands of times and recomputing its mean each time
traces out how much that mean would wobble on a re-draw. Demanding more confidence widens the
interval; there is no way to buy both narrowness and certainty at once.
:::

@fig-ci-bootstrap resamples a small sample of 25 post-rollout requests with replacement ten
thousand times, recomputes the mean of each resample, and reads off the middle 80, 90, 95, or
99 percent of that bootstrap distribution as a confidence interval. A thermometer read once
tells you the temperature at that moment; reading it many times under the same conditions
tells you how much a single reading could have wobbled.

A single sample mean is a point estimate: one number, with nothing attached to say how far off
a second sample might land. A *confidence interval* answers that question directly, giving a
range of values likely to contain the population parameter, along with a confidence level
stating how often intervals built this way would capture the truth across repeated sampling.

Shipping a dashboard that reports "mean latency: 45.2 ms" with no interval invites a team to
treat every day-to-day wiggle in that number as meaningful, when part of that wiggle is the
sampling noise the bootstrap distribution above traces out.

### A confidence interval for a mean

The same interval has a closed-form shortcut once the sample is close to normal, or the sample
size is large enough for the Central Limit Theorem to cover for it:

$$\bar{x} \pm t^{*} \frac{S}{\sqrt{n}}$$

where $\bar x$ is the sample mean, $S$ is the sample standard deviation, $n$ is the sample
size, and $t^{*}$ is the critical value from the t-distribution at the chosen confidence level
and $n - 1$ degrees of freedom (the same t-distribution introduced earlier in this chapter).

For the 25-request sample behind @fig-ci-bootstrap, the sample mean is 45.2 ms with a sample
standard deviation of 13.4 ms. Plugging those numbers in gives a 95 percent confidence interval
of [39.7, 50.8] ms, close to the bootstrap's [40.3, 50.5] ms: two different routes landing on
almost the same range.

::: {.callout-note}
A wider interval reports honest uncertainty, not a weaker result. A 25-request sample cannot
pin down the mean as tightly as a 2,500-request sample can, and a properly built confidence
interval says so instead of hiding it behind a single deceptively precise number.
:::

### A confidence interval for a proportion

The same logic applies to a proportion, the fraction of requests that succeed rather than the
average of a continuous metric. Suppose a canary release serves 120 requests and 109 of them
complete successfully:

```python
import numpy as np
rng = np.random.default_rng(41)
successes = rng.binomial(1, 0.91, 120)  # 109 of 120 succeeded
p_hat = successes.mean()  # 0.9083

boot_p = np.array([rng.choice(successes, size=120, replace=True).mean()
                    for _ in range(10_000)])
ci_95 = np.percentile(boot_p, [2.5, 97.5])
# ci_95 = [0.850, 0.958]
```

Bootstrapping the 120 successes the same way as the latency sample gives a 95 percent interval
of roughly [0.850, 0.958] around the observed success rate of 0.908.

A confidence interval for a proportion has its own closed-form shortcut, built the same way as
the interval for a mean but using the proportion's own standard error:

$$\hat{p} \pm z^{*} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$

where $\hat p$ is the sample proportion, $n$ is the sample size, and $z^{*}$ is 1.96 for a 95
percent interval. This gives [0.857, 0.960] here, close to the bootstrap's range but not
centered quite the same way.

::: {.callout-warning}
This formula (the Wald interval) can misbehave when the observed proportion sits close to 0 or
1, sometimes producing bounds above 1.0 or below 0.0 for a small sample with an extreme success
rate. The Wilson score interval corrects for this and is the safer default whenever a canary's
success rate lands above roughly 0.9 or below 0.1.
:::

## Sequential testing: the cost of checking early

::: {#fig-peeking}
```{=html}
<iframe src="../_generated/chapter-02-fig-peeking.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Two thousand simulated A/B tests, each built with no difference between control and treatment,
checked every 100 requests per group instead of once at the end. The share flagged significant
climbs well past the nominal 5 percent as the checking continues.
:::

@fig-peeking simulates 2,000 A/B tests where nothing separates the two groups (the null
hypothesis is true in every one of them), then checks each test's p-value after every 100
requests per group instead of waiting for a single planned sample size. By 2,000 requests per
group, 23.8 percent of these null-true tests had crossed the 0.05 threshold at some earlier
checkpoint, compared with 4.7 percent that would have crossed it under a single planned look
at that same final sample size.

*Sequential testing*, or "peeking," is the practice of checking a hypothesis test's p-value
repeatedly as data accumulates and stopping the moment it crosses the significance threshold,
rather than fixing the sample size in advance and checking once. A dashboard that shows a live
p-value for a running A/B test invites this behavior, whether or not the team checking it
knows the term for what it is doing.

Each additional look at the data is another chance for random noise alone to cross the 0.05
line, even when nothing in the underlying system changed. Stopping the moment a test looks
significant, instead of continuing to the planned sample size, systematically favors declaring
victory on noise: a team using this habit across many launches will ship changes that do
nothing measurable far more often than the 5 percent rate a single, pre-planned look promises.

Two standard fixes exist. **Fixing the sample size in advance** (using the sample size formula
from earlier in this chapter) and checking the result once removes the temptation entirely, at
the cost of waiting for the full sample before learning anything. **Sequential testing
methods**, such as alpha-spending functions or always-valid p-values, allow repeated looks by
shrinking the significance threshold at each check so that the combined false-positive rate
across all the looks still lands at the intended level.

::: {.callout-warning}
A live dashboard showing a running p-value only counts as a sequential test when the
underlying calculation accounts for how many times the result gets checked. Watching that
number and stopping when it dips below 0.05 otherwise carries the inflated false-positive rate
shown above, well past the 5 percent rate the threshold implies.
:::

Recall from earlier in this chapter that statistical power and sample size determine how much
data a decision needs before it can be trusted; peeking is what happens when a team acts on a
decision before that data has finished arriving.

## Confounding variables

Ice cream sales and drowning deaths both rise in summer, but ice cream does not cause
drowning. Hot weather drives both.

A confounding variable is a hidden factor like that: it influences two things at once and
makes them look connected even when one is not driving the other.

A *confounding variable* is a factor that influences both the thing being measured and the
thing suspected of causing it, distorting the apparent relationship between them.

Suppose the caching layer rolled out at 2 a.m., and the "after" latency sample happens to be
drawn mostly from overnight traffic, while the "before" sample was drawn from the daytime
peak. Time of day now affects both which sample a request fell into and how loaded the system
was, which is what a confounder does by definition.

The latency drop could reflect the cache, the lighter overnight load, or some mix of both,
and a simple before-after comparison cannot tell them apart.

Three standard ways to control for a confounder: **stratify** the analysis by the
confounding variable (compare overnight latency to overnight latency, daytime to daytime);
**match** observations on the confounder (pair each after-rollout hour with a before-rollout
hour at the same time of day); or **include the confounder as a covariate** in a regression
model, which Chapter 4 develops further and this chapter's ANCOVA section applies directly.

Recall from Chapter 1 that correlation alone cannot distinguish a true driver from a
confounder. The same caution applies to any before-after comparison that ignores what else
changed at the same time as the thing being tested.

## Paired and unpaired t-tests

Comparing the same ten runners' race times before and after a new training plan is different
from comparing ten runners on the old plan to ten different runners on the new one.

The first setup is paired, the second is not, and each needs its own version of the t-test.

A *paired t-test* applies when two measurements are linked, typically the same unit measured
twice. Comparing the same set of customer accounts' latency before and after the cache
rollout, matched account by account, is a paired design:

$$t = \frac{\bar{d}}{S_d / \sqrt{n}}$$

where $\bar{d}$ is the mean of the paired differences and $S_d$ is the standard deviation of
those differences.

::: {.callout-tip}
Pairing removes account-to-account variation from the comparison entirely, since each account
serves as its own baseline. That is why a paired design generally needs fewer observations to
reach the same statistical power as an unpaired one.
:::

::: {.callout-warning}
Running an unpaired t-test on data that is naturally paired throws away the correlation
between the two measurements, which usually inflates the variance estimate and makes a true
effect harder to detect. Before choosing between a paired and unpaired test, check whether
each unit in the sample was measured twice.
:::

An *unpaired t-test* (also called an independent t-test) applies when the two groups being
compared are not linked, for instance a randomized A/B test where incoming requests are
assigned to a control path or a cached path.

::: {#fig-permutation-null}
```{=html}
<iframe src="../_generated/chapter-02-fig-permutation-null.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Shuffling which latency values belong to "control" and which belong to "cached" 20,000 times,
and recomputing the difference in group means after each shuffle, builds a null distribution
with no formula involved. The observed difference sits far out in that distribution's tail.
:::

@fig-permutation-null takes 60 control-path and 60 cached-path latency measurements and pools
them into one list of 120 numbers, on the logic that if the caching layer changed nothing,
"control" and "cached" are just arbitrary labels stuck onto interchangeable measurements.
Shuffling those labels thousands of times and recomputing the mean difference under each
relabeling shows what differences arise from label-shuffling alone, with no cache effect
anywhere in the data.

The observed difference between the two groups is 9.8 ms. Only 0.56 percent of 20,000 shuffles
produced a difference at least that large in either direction, making a 9.8 ms gap far bigger
than what label-shuffling alone tends to produce.

The slider under the figure also shows what happens with too few shuffles: at 100 shuffles,
none of the reshuffled differences reached 9.8 ms, making the gap look rarer than the fuller
shuffle count later confirms; the estimate settles down only once the shuffle count reaches the
thousands. Running a permutation test on a production dataset carries this same cost: enough
shuffles to trust the resulting p-value can mean tens of thousands of recomputations of a group
mean, a compute cost worth weighing against simply running the closed-form test below.

The *unpaired t-test* answers the same question the permutation test just answered, in one
closed-form step instead of thousands of reshuffles:

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}$$

Welch's version of this test, applied to the same 120 measurements, returns a p-value of
0.0065, close to the permutation test's 0.0056 despite never shuffling a single label.

When the two groups have unequal variances, which is common when comparing a stable control
path to a newly deployed one, Welch's t-test (a variant that does not assume equal variances)
is the safer default over the standard unpaired t-test.

::: {.callout-warning}
Statistics libraries do not agree on a default here. R's `t.test()` runs Welch's version
unless told otherwise, while SciPy's `ttest_ind()` assumes equal variances unless called with
`equal_var=False`. Check which default a codebase is using before trusting an unpaired t-test
result pulled from unfamiliar code.
:::

Both paired and unpaired tests assume the underlying data is close to normal or that the
sample is large; the non-parametric alternatives near the end of this chapter cover what to
do when that assumption fails.

## The two-proportion z-test

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(2030)
old_layer = rng.binomial(1, 0.81, 1500)    # old caching layer: checkout completion
new_layer = rng.binomial(1, 0.855, 1500)   # new caching layer: checkout completion

p1, p2 = old_layer.mean(), new_layer.mean()           # 0.815, 0.864
pooled = (old_layer.sum() + new_layer.sum()) / 3000   # 0.840
se = np.sqrt(pooled * (1 - pooled) * (1/1500 + 1/1500))
z = (p2 - p1) / se                                    # 3.63
p_value = 2 * (1 - stats.norm.cdf(abs(z)))            # 0.0003
```

Everything the two-sample t-test does for a continuous measurement like latency, a
*two-proportion z-test* does for a checkout completion rate: it compares two proportions
against each other rather than each against a fixed target.

Comparing conversion rates, error rates, or completion rates between two variants is at least
as common in production A/B testing as comparing two means, yet the tools built for
categorical data so far in this chapter (the chi-squared test, covered next) are built around
counting categories, not directly comparing two rates the way an engineer thinks about them:
"the new layer completes checkouts 4.9 percentage points more often, is that gap believable."

For the old caching layer, 1,500 checkouts complete at a rate of 81.5 percent; for the new one,
1,500 checkouts complete at 86.4 percent, a 4.9 percentage point gap. Pooling both groups'
completions into one combined rate under the assumption that the two layers perform
identically gives the standard error the test statistic is measured against:

$$z = \frac{\hat{p}_2 - \hat{p}_1}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

where $\hat p_1$ and $\hat p_2$ are the two observed proportions, $\hat p$ is the pooled
proportion across both groups combined, and $n_1$ and $n_2$ are the two sample sizes. Here $z
= 3.63$, giving a two-sided p-value of 0.0003, a result unlikely to come from label-shuffling
noise: a permutation test on the same 3,000 checkout outcomes, shuffling which group each 0/1
outcome belongs to 20,000 times, puts the resampled p-value at 0.00025, in close agreement.

::: {.callout-tip}
The two-proportion z-test needs a pooled proportion under the assumption of no difference,
unlike the two-sample t-test, which estimates each group's variance separately. A proportion's
variance is fixed by the proportion itself ($\hat p(1-\hat p)$), leaving nothing extra to
estimate, which is why Welch's correction, the safer default for comparing two means, has no
equivalent step here.
:::

A small absolute gap in proportions can still be a substantial relative change: an 81.5
percent to 86.4 percent completion rate is a 4.9-point absolute gain, but it also means the
checkout failure rate dropped from 18.5 percent to 13.6 percent, close to a 27 percent cut in
the share of checkouts that fail. Reporting only the absolute gap can undersell a result like
this one.

## Comparing three or more groups: ANOVA

A t-test can compare two classrooms' test scores, but what about five classrooms? Checking
every possible pair one at a time gets messy fast.

ANOVA compares many groups at once and asks one question: does at least one group stand
apart from the rest?

A t-test compares two groups. *Analysis of variance* (ANOVA) extends the same logic to three
or more, for instance comparing mean latency across three deployment regions running the same
caching layer. ANOVA tests the null hypothesis that all group means are equal against the
alternative that at least one differs:

$$H_0: \mu_1 = \mu_2 = \dots = \mu_k, \qquad H_1: \text{at least one } \mu_i \text{ differs}$$

The test statistic, $F$, compares the variance between group means to the variance within
each group:

$$F = \frac{\text{variance between groups}}{\text{variance within groups}}$$

::: {#fig-anova-between-within}
```{=html}
<iframe src="../_generated/chapter-02-fig-anova-between-within.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Three regions' latency distributions as the gap between their means widens. F rises and p
falls even though the spread within each region never changes.
:::

@fig-anova-between-within holds the within-group spread fixed and widens the gap between the
three region means. It shows $F$ climb and the p-value fall as the between-group signal grows
relative to the within-group noise that never changes.

A large $F$ means the groups differ more from each other than individual observations differ
within a group, evidence that region is driving some of the variation in latency rather than
random noise alone.

ANOVA answers only whether at least one region differs, not which one; the pairwise tests
later in this chapter pick up from there.

The same shuffle-based logic from the unpaired t-test extends here without much change:
relabel which of the three regions each latency observation belongs to, recompute the
between-group and within-group variance under that relabeling, and repeat thousands of times
to build a null distribution for $F$ directly. The closed-form F-test above is the fast version
of that process. Running the full permutation version costs more compute for the same reason
the unpaired t-test's permutation check did: the earlier comparison needed thousands of
reshuffles before its estimated p-value stopped moving (0.0040 at 1,000 shuffles versus 0.0056
at 20,000), and a three-group relabeling has more ways to shuffle than a two-group one, so it
settles even more slowly. Production pipelines default to the closed-form F-test for this
reason, reaching for the permutation version mainly when a group's data is too irregular for
the F-test's assumptions to hold with any confidence.

::: {.callout-warning}
Standard ANOVA assumes each group's latency varies by roughly the same amount. When one
region's traffic is far noisier than another's, the F-test's false-positive rate can drift
from its stated level, and a Welch-corrected ANOVA or the Kruskal-Wallis test covered later
in this chapter becomes the safer choice.
:::

## Categorical association: chi-squared and goodness of fit

Suppose a bag is supposed to hold equal numbers of red, blue, and green candies, but a
handful you grab has far more red than expected.

The chi-squared test measures whether counts like that are ordinary luck or a meaningful
mismatch from what the bag was supposed to contain.

ANOVA compares means of continuous data. The *chi-squared test* compares frequencies of
categorical data. Suppose the team wants to know whether error type (timeout, 500, connection
reset) is independent of which region served the request, or whether certain regions produce
disproportionately more of one error type. The test statistic is

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ is the observed count in category $i$ and $E_i$ is the count expected if error
type and region were independent.

A large $\chi^2$ means the observed counts deviate from what independence would predict,
evidence that region and error type are associated rather than independent.

A *goodness of fit test* uses the same statistic to compare observed frequencies against a
specific expected distribution rather than a table of two variables.

Suppose historical data says timeouts, 500s, and connection resets should occur in a 50/30/20
split; a goodness of fit test checks whether this week's observed error counts are still
consistent with that split, or whether the error mix has shifted.

::: {.callout-note}
Both tests assume the sample is large enough that expected counts in each category are not
too small (a common rule of thumb is at least 5). With rarer categories, Fisher's test (built
around permutation probabilities computed directly rather than the chi-squared approximation)
is the safer choice.
:::

The chi-squared statistic also has a permutation version, built the same way as the tests
above: shuffle the error-type label attached to each observed request, recompute $\chi^2$
under each shuffle, and compare the observed statistic against the resulting simulated null.
As a rule of thumb, resolving a permutation p-value near a stricter threshold such as 0.01
reliably needs on the order of ten times as many shuffles as resolving one near 0.05, since the
estimate's own uncertainty shrinks only with the square root of the shuffle count. A pipeline
that reuses a shuffle count tuned for a 0.05 cutoff at a stricter 0.01 cutoff elsewhere is
working with a noisier estimate than the first case.

## Controlling for a continuous confounder: ANCOVA

Comparing two diets' effect on weight loss is unfair if one group started out heavier than
the other.

ANCOVA compares group averages while accounting for a starting difference like that, so the
comparison is not skewed by something uneven before the diets even began.

*Analysis of covariance* (ANCOVA) extends ANOVA to control for a continuous confounding
variable while comparing group means.

Suppose the team wants to compare mean latency across two deployment strategies, blue-green
versus canary, but request payload size (a continuous variable Chapter 1 showed correlates
with latency) differs between the two rollout groups purely by chance.

ANCOVA compares the two strategies' effect on latency while adjusting for payload size as a
covariate, isolating the deployment strategy's effect from the confound:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$$

where $Y$ is latency, $X_1$ is the categorical deployment strategy, and $X_2$ is payload
size; $\beta_0$ is the intercept, $\beta_1$ and $\beta_2$ are coefficients measuring how much
each predictor moves latency, and $\varepsilon$ is the error term capturing whatever the
model does not explain.

ANCOVA is, structurally, a regression model with a categorical predictor and a continuous
covariate together, which is why Chapter 4's treatment of regression assumptions (normality,
constant variance, independent errors) applies here as well.

::: {.callout-warning}
ANCOVA carries one further assumption: that payload size affects latency the same way under
both deployment strategies. If the two strategies respond to payload size differently, this
additive model cannot separate the deployment effect from the confound cleanly, and needs an
interaction term between strategy and payload size instead.
:::

## Designing the experiment before running it

A science-fair project testing plant growth needs a plan for which plants get sunlight,
water, and fertilizer before anyone plants a seed.

Design of experiments is that planning step: deciding how to assign treatments fairly before
any data gets collected, not after.

*Design of experiments* (DOE) is the discipline of deciding, before collecting any data, how
treatments will be assigned so the resulting comparison is fair. Four designs cover most
production experiments:

- **Completely randomized design**: each request is randomly assigned to a treatment with no
  other structure, the simplest and most common design for a feature-flag rollout.
- **Randomized block design**: requests are first grouped by a known source of variation
  (for instance, region or time of day) and randomized within each group, so that source of
  variation cannot masquerade as a treatment effect. This is the formal version of the
  stratification technique introduced earlier for confounding variables.
- **Factorial design**: two or more factors are varied together, for instance testing cache
  TTL and response compression simultaneously, so their combined and individual effects on
  latency can both be estimated in one experiment rather than two separate ones.
- **Response surface methodology**: used to tune several continuous configuration parameters
  at once (cache TTL, connection pool size, retry backoff) to find the combination that
  minimizes a metric like latency, rather than testing one parameter at a time.

A poorly designed experiment cannot be rescued by a more sophisticated test afterward; the
choice of design determines what a p-value from that experiment can and cannot claim to show.

::: {.callout-warning}
Randomization has to happen before anyone who could influence the outcome knows which
treatment an observation will get. Letting engineers pick which users see a new feature, even
with good intentions, breaks the design no matter how careful the statistical test afterward.
:::

## After ANOVA: pairwise post-hoc tests

If five classrooms' test scores differ overall, that does not say which classrooms differ
from which.

Post-hoc tests check every classroom pair against every other, while being careful not to
rack up false alarms simply from checking so many pairs at once.

When ANOVA rejects the null hypothesis that regions have equal mean latency, it does not say
which regions differ. *Pairwise post-hoc tests* answer that question by comparing every pair
of groups while controlling the overall risk of a false positive across all those
comparisons, a risk that grows the more pairs are checked.

- **Tukey's Honestly Significant Difference (HSD)** compares all pairs of means from a
  one-way ANOVA while controlling the *family-wise error rate* (the probability of at least
  one false positive across the entire set of pairwise comparisons, not just any single one
  of them), the standard choice when every pair is of equal interest.
- **Scheffé's test** is more conservative and better suited to comparing arbitrary
  combinations of groups, not just simple pairs.
- **Bonferroni-Dunn** applies the Bonferroni correction, dividing the significance threshold
  by the number of comparisons, a simple and conservative approach that becomes noticeably
  strict as the number of regions being compared grows.

Choosing among these is a trade between power and conservatism: a stricter correction lowers
the chance of a false positive across many comparisons but raises the chance of missing a
difference that exists between any single pair.

## The multiple-comparisons problem

```python
import numpy as np
rng = np.random.default_rng(5)
m_tests, alpha = 20, 0.05
count_any_significant = 0
for _ in range(20_000):
    p_values = rng.uniform(0, 1, m_tests)   # p-values are uniform when every null is true
    if np.any(p_values < alpha):
        count_any_significant += 1
share = count_any_significant / 20_000   # 0.637
```

Running 20,000 simulated dashboards, each checking 20 independent metrics where nothing has
changed on any of them, found that 63.7 percent of those dashboards had at least one metric
cross the 0.05 threshold anyway. The closed-form probability agrees:
$1 - (1 - 0.05)^{20} \approx 0.642$.

The post-hoc tests above solve a narrow version of a wider problem: checking many things at
once inflates the chance that something looks significant by chance alone, whether those "many
things" are pairwise comparisons after ANOVA or twenty unrelated metrics on a monitoring
dashboard. The *multiple-comparisons problem* names this general fact and reaches far beyond
ANOVA follow-up testing.

A team running one hypothesis test accepts a 5 percent chance of a false alarm on that one
test. A team running twenty independent tests, each at the same 5 percent threshold, faces
close to a 64 percent chance that at least one of them flags a change that is not there,
purely from running that many tests. A metrics dashboard checking latency, error rate,
throughput, and a dozen other signals across several regions every morning is running this
same experiment without recognizing it as one.

Two corrections handle this differently. The **Bonferroni correction**, introduced earlier in
this chapter for post-hoc pairwise tests, divides the significance threshold by the number of
tests: checking 20 metrics at a combined 5 percent risk means treating no single metric as
significant unless its own p-value clears 0.0025, a considerably stricter bar. The
**Benjamini-Hochberg procedure** takes a different approach, controlling the *false discovery
rate* (the expected share of flagged results that are false alarms, rather than the chance of
any false alarm at all) instead of the family-wise error rate Bonferroni protects. It is less
conservative than Bonferroni and is the more common choice for a dashboard running dozens of
tests routinely, where missing a regression is often costlier than tolerating an occasional
false alarm among the flagged metrics.

::: {.callout-warning}
Treating a dashboard's individual metric thresholds as independent 5 percent risks, with no
correction at all, is one of the most common ways a monitoring system generates false alarms.
The more metrics a dashboard checks routinely, the more its uncorrected alert rate drifts away
from the 5 percent a single test promises.
:::

## When the data will not cooperate: non-parametric tests

Some group-comparison methods need the data to roughly follow a bell curve.

Non-parametric tests are the backup plan: they compare the order data falls in rather than
its precise values, so they still work when the data is lopsided or full of outliers.

Every test in this chapter, up to this point, assumes the underlying data is close to normal
or that the sample is large enough for the Central Limit Theorem to compensate. Recall from
Chapter 1 that checkout-API latency is right-skewed with a heavy tail; a small sample of that
data can violate the normality assumption badly enough to make a t-test or ANOVA unreliable.

*Non-parametric tests* make no assumption about the shape of the underlying distribution, at
the cost of somewhat less statistical power when the normality assumption holds.

The **Wilcoxon rank-sum test** (also called the Mann-Whitney U test) is the non-parametric
counterpart to the unpaired t-test, comparing the ranks of two groups' observations rather
than their means. The **Kruskal-Wallis test** extends the same idea to three or more groups,
the non-parametric counterpart to ANOVA.

::: {#fig-nonparametric}
```{=html}
<iframe src="../_generated/chapter-02-fig-parametric-vs-nonparametric.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A t-test's p-value moves more than a Wilcoxon rank-sum test's as outliers are added to skewed
data. The rank-based test barely reacts, since a single extreme value can only move a rank by
one position.
:::

@fig-nonparametric compares a t-test and a Wilcoxon rank-sum test on the same simulated
right-skewed latency data as a small number of outlier requests are added, showing the
t-test's p-value swinging with the outliers while the rank-based test stays comparatively
stable.

::: {.callout-tip}
Both non-parametric tests are the safer choice for comparing latency across groups when the
sample is small and the underlying distribution is visibly skewed, the situation Chapter 1
established as the norm for latency data rather than the exception.
:::

Choosing between a parametric and a non-parametric test comes down to what the data will
support: a parametric test is more powerful when its assumptions hold, and a non-parametric
test is the more trustworthy choice when they do not.

Chapter 3 picks up the underlying probability theory these tests rest on. Part 3 shows a
different way to answer the same question the caching layer raised at the start of this
chapter, using Bayesian methods instead of the p-value framework built here.

## References {.unnumbered}

::: {#refs}
:::
