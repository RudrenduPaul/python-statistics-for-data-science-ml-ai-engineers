# Bayesian Experimentation for A/B Testing

At 1,000 visitors per variant, a two-proportion z-test on a checkout-button redesign returns a
p-value of 0.126. That test asks a narrow question: whether two observed conversion rates
differ by more than chance alone would plausibly explain.

Under the conventional 0.05 threshold, the 0.126 result is not significant: the frequentist
framework says stop, you cannot reject the null.

A Bayesian model fit to the same 2,000 visitors says the redesign has a 93.7% chance of beating
the original. If that 93.7% turns out to be wrong, shipping it now carries an average cost of
0.038 percentage points of conversion.

Both statements are correct. They are answers to different questions, and knowing which
question you are asking is most of what this part of the book is about.

Part 1 introduced Bayes' theorem and frequentist hypothesis testing as two separate tools. This
part treats Bayesian inference as a full alternative to the hypothesis-testing machinery from
Chapter 2.

It gives a way to run an experiment, watch it while it runs, and make a shipping decision
without waiting for a p-value to cross a line someone picked in advance.

## From Bayes' theorem to Bayesian updating

Think of updating a guess as new clues arrive in a mystery. A hunch shifts a little with each
clue, and the final guess reflects everything seen so far. Bayesian updating is that same
nudging process, done with numbers instead of hunches.

Recall Bayes' theorem from Chapter 3: it converts a prior belief and new evidence into an
updated belief. Applied to an experiment, that conversion has a name and a fixed shape.

$$P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta) \times P(\theta)$$

In other words, the *posterior* (what you believe about the conversion rate $\theta$ after
seeing the data) is proportional to the *likelihood* (how probable the observed data is, for
each possible value of $\theta$) times the *prior* (what you believed about $\theta$ before
seeing any data).

This is *Bayesian updating*, and every Bayesian analysis in this chapter is one instance of
this same multiplication, applied to a checkout-button experiment instead of an abstract
$\theta$.

## Priors: how much to assume before the data arrives

A prior is just a starting guess, set before any data comes in. A cautious guess moves easily
once observations start arriving; a stubborn guess takes a lot of evidence to budge.

Suppose the checkout team is testing a redesigned button (Variant B) against the existing one
(Variant A). Before a single visitor sees either variant, a *prior* states what the team
believes about the conversion rate.

Recall from Chapter 3 that a probability distribution assigns weight to every possible value a
quantity could take. A prior does the same for the unknown conversion rate itself.

An *uninformative prior* treats every conversion rate between 0% and 100% as equally plausible.
A $\text{Beta}(1, 1)$ distribution is flat, uniform, and equivalent to having observed zero
prior data.

A *weakly informative prior* nudges the distribution toward plausible values without
pretending to certainty. A $\text{Beta}(11, 91)$ distribution behaves like 100
pseudo-observations at a 10% rate, close to what a checkout team would reasonably expect from a
button converting around 10% of visitors.

An *informative prior* goes further still, behaving like thousands of pseudo-observations
concentrated tightly around a known rate.

::: {#fig-ab-prior-shapes}
```{=html}
<iframe src="../_generated/chapter-05-fig-beta-prior-shapes.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Each curve puts the same 10% central belief on the x-axis (conversion rate, 0% to 40%) but
spreads that belief differently: the uninformative prior spans almost the full range, while the
highly informative prior shown here, built from about 10,000 pseudo-observations, concentrates
nearly all its density within a percentage point of 10%. The stronger the prior, the more data
it takes to move it.
:::

@fig-ab-prior-shapes shows the same 10% central belief expressed at five different prior
strengths, from a flat uninformative prior through an increasingly confident one.

Stronger priors are not automatically better. Suppose the checkout team used an informative
prior built from a redesign three years ago that turned out to be unrepresentative of current
traffic.

That prior would actively resist what the new data is trying to say, and it would take a large
sample to overrule it.

::: {.callout-tip}
Default to a weakly informative prior unless there is a specific, defensible reason to bring in
outside information, and say so explicitly whenever a stronger prior is used.
:::

## Beta-Binomial conjugacy

Updating a guess with new evidence usually takes heavy computation. Conjugacy is a shortcut:
for certain pairings of guess and evidence, the update is nothing more than simple addition.

The reason a Beta prior is the standard choice for a conversion-rate problem is not convention
alone. It is a property called *conjugacy*.

When the likelihood of the observed data follows a binomial distribution (a fixed number of
visitors, each either converting or not) and the prior follows a Beta distribution, the
posterior is also a Beta distribution, updated by simple addition:

$$\text{Beta}(\alpha, \beta) \;\xrightarrow{\;k \text{ successes in } n \text{ trials}\;}\; \text{Beta}(\alpha + k, \;\beta + n - k)$$

In other words, starting from a $\text{Beta}(1, 1)$ prior, observing 46 conversions out of 500
visitors updates the posterior to $\text{Beta}(1 + 46, \;1 + 454) = \text{Beta}(47, 455)$.

That posterior's mean, $47 / 502 \approx 9.4\%$, sits close to the observed rate of 9.2%, and
would sit closer still to the prior's 50% mean if the sample were tiny.

No integral, no simulation, and no specialized software are required. This is
*Beta-Binomial conjugacy*, and it is the entire reason a Bayesian A/B test on conversion rates
is tractable by hand.

::: {#fig-posterior-update}
```{=html}
<iframe src="../_generated/chapter-05-fig-posterior-update.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A single variant's posterior narrows and settles as observed visitors accumulate. Early swings
are large; by a few thousand visitors, new data barely moves it.
:::

@fig-posterior-update shows this update happening to Variant A's posterior alone as its sample
size grows from 0 to 5,000 simulated visitors, starting from a flat $\text{Beta}(1, 1)$ prior.

Watch how far the posterior mean swings between $n = 0$ (sitting at the prior's 50%, since
nothing has been observed yet) and $n = 25$. With almost no data, a handful of successes can
still pull the estimate a long way from the eventual answer near 10%.

By $n = 2{,}000$ the curve has narrowed into a tight spike. The posterior has stopped moving
much because a few more visitors can no longer outweigh the thousands that came before.

## Two variants, one decision

Picture two runners who have each finished several practice laps at slightly different times.
Win probability asks: based on those laps, how often would this runner beat the other in a
head-to-head race?

An experiment needs two posteriors, one per variant, computed the same way and compared
directly.

::: {#fig-ab-overlay}
```{=html}
<iframe src="../_generated/chapter-05-fig-posteriors-ab-overlay.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Variant A and Variant B posteriors separate as each group collects data. By a few thousand
visitors per variant, the two curves barely overlap.
:::

@fig-ab-overlay overlays Variant A's and Variant B's posteriors as each collects more simulated
visitors, using the redesign scenario at a 10% true rate for A and a 12.5% true rate for B.

That 25% relative lift was chosen only to make the simulation worth watching. Nothing about
these specific numbers is a claim about any company's product.

At $n = 50$ per variant the two curves overlap heavily; at $n = 5{,}000$ they barely touch.

Once both posteriors exist, the practical question is no longer "is there a difference" but
"how likely is B to be the better variant, and what would it cost to be wrong."

Evan Miller's closed-form solution answers the first half directly: given Beta posteriors
$\text{Beta}(\alpha_A, \beta_A)$ for A and $\text{Beta}(\alpha_B, \beta_B)$ for B, the
probability that B beats A reduces to a finite sum over the whole-number values $\alpha$ and
$\beta$ can take [@miller2014].

That sum can be computed with ordinary loops rather than the numerical integration a less
convenient distribution pairing would require.

In practice, most teams skip the closed form and estimate the same probability by simulation:
draw a large number of samples from each posterior distribution, and count the fraction of
draws where B's sampled rate exceeds A's.

With 200,000 simulated draws per comparison, the two approaches agree to several decimal
places. The simulation approach also generalizes immediately to metrics that do not have a
convenient closed form, which is why this chapter's figures use it throughout.

::: {.callout-tip}
Miller's closed-form sum grows more terms as successes accumulate, so it slows down right when
a team wants it most: after a lot of traffic has arrived. Simulation cost stays flat regardless
of sample size, which is the more practical reason production dashboards default to it.
:::

## Sample ratio mismatch: checking the split before trusting the comparison

A referee checks that both teams took the field with the agreed number of players before a
single point on the scoreboard means anything. A sample ratio check plays the same role for an
experiment: confirm the assignment mechanism did what it was told before reading any result it
produced.

::: {#fig-srm-check}
```{=html}
<iframe src="../_generated/chapter-bayes-ab-testing-fig-srm-check.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Cumulative logged traffic by variant, week over week. Weeks 1 and 2 sit close to the dashed 50%
line; a redirect-timeout bug shipped after Week 2 pulls Variant B's share down every week after,
crossing the flag threshold by Week 5.
:::

@fig-srm-check tracks a checkout-button test across five weeks at 15,000 new visitors per week.
Assignment stays a clean coin flip the whole time.

After Week 2, a redirect-timeout bug on Variant B's page drops roughly 6% of visitors assigned
to it from the logging pipeline before a conversion is ever recorded for them. Variant A's page
is untouched, so its logged count always matches its assigned count.

By the end of Week 4, cumulative logged traffic sits at 29,921 for A against 29,145 for B, a
chi-square statistic of 10.19 and a p-value of 0.0014, a split drifting the wrong way but not
yet flagged. One more week of the same bug pushes it past the line: 37,430 logged for A against
36,243 for B, chi-square 19.12, p-value near 0.00001.

*Sample ratio mismatch* (SRM) is what a test has when the traffic that ends up analyzed splits
differently from the traffic the randomization mechanism was told to produce. It says nothing
yet about which variant converts better. It says the two groups being compared are no longer
the two groups the experiment was designed to compare.

The gap between who was randomized and who got counted is the whole reason SRM deserves a
check of its own, separate from every decision rule earlier in this chapter. Win probability,
expected loss, and the frequentist
p-value all assume the visitors landing in each bucket are a fair, randomized sample. A
redirect bug does not touch who gets randomized; it touches who survives to be counted, and
that survival is not random. Slower connections and older devices are the ones most likely to
time out on a redirect, so the visitors quietly dropped from Variant B's logs are not a random
5% of Variant B; they skew toward whichever population is slower to load a page.

Comparing the two groups that remain, then, is not comparing Variant A to Variant B. It is
comparing Variant A's full population to a filtered slice of Variant B's population, missing
the specific visitors most likely to behave differently. A win probability of 93.7% computed on
that filtered comparison is a precise answer to a question nobody meant to ask.

Chapter 2 introduced the *goodness of fit* test for this shape of question: whether observed
category counts still match an expected distribution, using the same chi-squared statistic
behind the association test in that chapter,

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

with $O_i$ the observed logged count for variant $i$ and $E_i$ the count the intended split
would produce out of the same total. For a 50/50 design with $n$ total logged visitors, $E_i$ is
simply $n / 2$ for each variant.

::: {.callout-warning}
Run the SRM check on cumulative totals throughout the experiment, not once at the end. This
chapter's peeking-problem section warns against optional stopping on the decision metric;
checking the split itself carries no such penalty, since it is a validity check on the
experiment's plumbing, not a test of the effect being measured.
:::

Kohavi, Tang, and Xu's practitioner reference on controlled experiments recommends a stricter
significance threshold than the conventional 0.05 for this specific check, on the order of
$p < 0.001$, because with tens of thousands of visitors even a trivial, harmless imbalance
clears 0.05 routinely; the stricter bar reserves a flag for a split large enough to point to a
pipeline problem worth fixing [@kohavitangxu2020]. A large-scale study of SRM across four
production companies catalogs the usual culprits behind that kind of split: assignment
bucketing bugs, bot traffic filtered asymmetrically after the fact, and, as in this section's
example, telemetry lost somewhere downstream of assignment [@fabijan2019].

::: {.callout-important}
An SRM flag stops the analysis cold. Every other number this chapter computes needs to wait
until the split is fixed: win probability and expected loss are answers to the wrong question
until then.
:::

## Expected loss as a stopping rule

Imagine picking between two job offers without knowing which pays better long-term. Expected
loss asks not just "which looks better" but "if this pick is wrong, how much does that mistake
typically cost?"

Win probability alone can mislead. Two experiments can both report a 95% chance that B wins
while carrying sharply different risk: one because A and B are meaningfully far apart, the
other because the sample is still small and the estimate is noisy.

Chris Stucchio's *expected loss* framework, developed for the Bayesian testing engine behind
the optimization platform VWO, answers a more honest question than "how likely is B to win": if
you choose B and you turn out to be wrong, how much do you lose, on average, in the metric that
matters [@stucchio2015]?

$$\text{Expected loss from choosing B} = E\big[\max(p_A - p_B, \;0)\big]$$

In other words, expected loss averages the size of the mistake across every simulated scenario
where A would have been the better choice, and counts zero everywhere else.

A team can set a tolerable expected-loss threshold before the experiment starts, expressed in
the units that matter to the business (percentage points of conversion, for instance).

It can then stop the experiment the moment expected loss falls below that threshold, rather
than waiting for an arbitrary significance threshold.

:::{.callout-tip}
A high win probability is not the same as low risk. Two experiments can each show a 95% win
probability for B while carrying sharply different expected loss, so set an expected-loss
threshold before an experiment starts and treat a favorable win probability alone as a reason
to keep watching, not to ship.
:::

::: {#fig-decision-metrics}
```{=html}
<iframe src="../_generated/chapter-05-fig-decision-metrics.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Win probability and expected loss from choosing B, as the sample size grows. Expected loss
stays a steadier guide than win probability when the sample is still small.
:::

@fig-decision-metrics tracks both numbers, win probability and expected loss, across the same
growing sample sizes as the checkout-button simulation.

Note the jaggedness at small sample sizes. At $n = 50$, only two of Variant A's visitors and
five of Variant B's had converted.

That is enough for the simulation to show an 86.6% win probability for B, driven by a small
sample that happened to land unusually far apart (4% versus 10% observed, against true rates of
10% and 12.5%).

The expected-loss number at that same point, 0.0037, stays the steadier guide: it prices in how
thin the evidence still is, even while the win probability alone looks fairly persuasive.

## Continuous monitoring and the peeking problem

A common pitch for Bayesian testing is that a team can watch the dashboard at any point and act
on what it shows. A frequentist test, by contrast, inflates its false-positive rate if a team
checks the p-value early and stops the moment the p-value clears the significance threshold.

That pitch needs a qualification. Georgi Georgiev's direct rebuttal to the strong version of
this claim argues that stopping the moment a result looks favorable, and treating that as clean
evidence, still changes the question being answered.

It shifts from "what does this data say" to "what does this data say, given that I stopped
here because it looked good."

Georgiev further argues that the effect applies to frequentist and Bayesian analyses alike:
five rounds of undisciplined peeking can inflate the error rate to roughly three times the
nominal level [@georgiev2017].

::: {.callout-warning}
Checking a test repeatedly and stopping the moment it looks favorable inflates the
false-positive rate, for frequentist and Bayesian analyses alike. Five undisciplined looks can
roughly triple the error rate.
:::

::: {#fig-ab-peeking}
```{=html}
<iframe src="../_generated/chapter-05-fig-peeking-problem.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

False-positive rate under a true null, as the number of times the test is checked grows. A
single fixed-horizon check holds the nominal 5%; five checks push it past 14%.
:::

@fig-ab-peeking simulates this directly: two identical variants (no true difference, so any
rejection is a false positive) checked with a standard significance test at growing numbers of
look points.

Checking once holds the nominal 5% rate, by construction. Checking five times pushes the
false-positive rate to roughly 14%, close to three times the nominal level, matching Georgiev's
figure.

By 40 checks, a team that stops the moment any check looks favorable is wrong roughly three
times in ten, not one time in twenty.

This simulation runs a classical significance test at each check point. A Bayesian posterior
does not inflate this way in the same mechanical sense, since each posterior computed is a
correct statement of belief at that sample size.

But that mechanical safety does not make optional stopping free of risk.

What differs is narrower than the pitch suggests. A Bayesian posterior is a valid statement of
belief at whatever sample size it gets computed, so the number itself stays correct no matter
when a team looks.

What stays risky is optional stopping used as a decision procedure: choosing to stop
specifically because expected loss happened to dip below a comfortable-looking number at one
particular check, then treating that as though the experiment had always been designed to run
to that size.

A loss estimate computed on a whim at a favorable-looking moment is a noisier estimate than the
same loss computed against a sample size decided in advance.

Two habits keep the dashboard-watching convenience of Bayesian testing honest: setting a
minimum sample size before monitoring begins, and treating the expected-loss threshold as a
stopping rule fixed before the experiment starts, not a justification invented after the fact.

## A direct comparison against the frequentist result

The opening of this chapter quoted a two-proportion z-test on a checkout-button experiment at
the same 10% and 12.5% true rates used throughout this chapter, at $n = 1{,}000$ per variant: 95
conversions out of 1,000 for A (9.5%), 116 out of 1,000 for B (11.6%), $z = 1.53$, $p = 0.126$.

Under a standard 0.05 threshold, that result does not reject the null hypothesis that A and B
convert at the same rate.

The Bayesian posterior at that same sample size puts the win probability for B at 93.7%, with
an expected loss from choosing B of 0.038 percentage points.

The disagreement is not a contradiction. Each framework is answering the question it was built
to answer. The frequentist test asks whether the observed gap is larger than chance alone would
plausibly produce, evaluated against a fixed rejection threshold decided in advance.

The Bayesian framework asks how the team's belief about each variant's rate should update given
the data, and what a wrong decision would cost.

A team using only the p-value would keep the experiment running past 1,000 visitors per
variant, waiting for significance that may or may not arrive.

A team using expected loss might reasonably ship Variant B at $n = 500$, where expected loss
had dropped to 0.017 percentage points. The frequentist test at that same sample size had only
just crossed its own threshold ($p = 0.045$), a result fragile enough that it drifted back
above the threshold by $n = 1{,}000$.

::: {.callout-note}
At $n = 500$ the frequentist test had just crossed significance ($p = 0.045$). By
$n = 1{,}000$ it had drifted back above the threshold. A p-value near the edge of significance
can move in either direction as more data arrives.
:::

That drift is itself informative: a p-value computed at an arbitrary stopping point can move in
either direction as more data arrives, which is the motivation behind treating expected loss,
not statistical significance, as the stopping rule.

Neither framework is simply better. The frequentist test is standardized, taught everywhere,
and immediately legible to a stakeholder who has seen a p-value before. It requires a sample
size fixed in advance and a single accept-or-reject answer at the end.

The Bayesian approach gives a direct probability statement about which variant is better and a
decision rule priced in business-relevant units, and it tolerates a team checking the results
early without inflating the false-positive rate the way repeated frequentist peeking does.

It requires choosing and defending a prior, a step a frequentist test never asks for. And a
stakeholder unfamiliar with posterior distributions will need the win-probability-and-expected-loss
framing translated into plain terms before it sticks.

## Beyond conversion rates

Combining a prior guess with new data works like a weighted average: the more trustworthy
source, whichever one carries less uncertainty, pulls the final answer closer to itself.

The checkout-button test almost certainly has a second question behind the first: does the
redesign also change how much a converting visitor spends, not just whether they convert at
all?

Average order value is continuous, not binary, so Beta-Binomial conjugacy does not apply
directly. A different conjugate pairing fills the same role.

For a metric with roughly normal noise, a Normal prior on the unknown mean, combined with an
observed sample mean, produces a Normal posterior, updated by weighting each source by its
precision (the inverse of its variance).

$$\tau_{\text{post}} = \tau_{\text{prior}} + n \tau_{\text{data}}, \qquad
\mu_{\text{post}} = \frac{\tau_{\text{prior}} \mu_{\text{prior}} + n \tau_{\text{data}} \bar{x}}{\tau_{\text{post}}}$$

Here $\tau_{\text{prior}}$ and $\tau_{\text{data}}$ are the precisions of the prior belief and
of a single observation, $n$ is the sample size, $\mu_{\text{prior}}$ is the prior mean, and
$\bar{x}$ is the observed sample mean.

In other words, the posterior precision $\tau_{\text{post}}$ is just the prior's precision plus
$n$ copies of the data's precision, since $n$ independent observations are $n$ times as
informative as one.

The posterior mean $\mu_{\text{post}}$ is a weighted average of the prior mean and the sample
mean, with each one weighted by how much precision it contributes.

A prior with high precision (a narrow, confident guess) barely moves even when the sample mean
disagrees with it. A prior with low precision (a wide, uncertain guess) gets pulled toward the
sample mean almost immediately.

Suppose the team's pre-launch benchmarking suggests an average order value around \$42, with
enough uncertainty to express as a prior standard deviation of \$5.

After 500 converting visitors under Variant B post an observed mean of \$44.10 with a sample
standard deviation of \$18, the posterior mean settles close to \$44.05.

The result is pulled only slightly toward the prior's \$42, because the sample of 500 carries
far more precision than the prior did. This is the same prior-times-likelihood-equals-posterior
logic from earlier in the chapter, applied to a different distribution family.

::: {.callout-warning}
This Normal-Normal update treats $\tau_{\text{data}}$ as known, but in practice it usually comes
from the sample's own standard deviation. Plugging in an estimated precision as though it were
fixed understates the true uncertainty, especially early in the experiment when the sample
standard deviation itself is still unstable.
:::

::: {#fig-normal-normal-update}
```{=html}
<iframe src="../_generated/chapter-05-fig-normal-normal-update.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The average-order-value posterior narrows and slides from the \$42 prior toward the observed
\$44.10 sample mean as converting visitors accumulate. Move the slider.
:::

@fig-normal-normal-update shows the same narrowing-and-settling behavior @fig-posterior-update
showed for a Beta posterior, this time for the Normal-Normal pairing: at $n = 0$ the curve is
just the prior, centered on \$42.

By $n = 500$ the posterior has both narrowed and shifted almost entirely onto the observed
\$44.10, since 500 observations at a precision of $1/18^2$ each outweigh the prior's single
pseudo-observation-equivalent of precision by a wide margin.

That said, plenty of continuous metrics do not stay this well behaved. Order value tends to be
right-skewed in the same way the latency data from Chapter 1 was, carries outliers from bulk
orders, and often needs a model structure the simple Normal-Normal pairing cannot express.

This includes cases such as separate effects per customer segment or a likelihood that is not
normal at all. This is where conjugacy runs out and numerical methods take over.

`pymc` and `arviz`, both listed among this book's dependencies, exist to draw posterior samples
via Markov chain Monte Carlo once a model no longer has a closed-form answer.

Every figure in this chapter needed nothing more than arithmetic, because Beta-Binomial and
Normal-Normal conjugacy both had closed-form solutions. A model with correlated segment-level
effects or a skewed, non-normal outcome usually will not.

## Multiple metrics, multiple chances to be wrong

Buying ten lottery tickets instead of one does not raise any single ticket's odds, but it does
raise the odds that at least one of them pays out. Watching ten metrics on a dashboard works the
same way: each one gets its own chance to look like a win purely by luck.

::: {#fig-multiple-metrics}
```{=html}
<iframe src="../_generated/chapter-bayes-ab-testing-fig-multiple-metrics.html" width="100%"
        height="560" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The probability that at least one of $k$ metrics clears a 95% win-probability bar, with no true
difference behind any of them. The uncorrected rate climbs past 60% by twenty metrics; a
per-metric threshold adjusted for $k$ holds it near 5% throughout.
:::

@fig-multiple-metrics runs the checkout-button test's own setup, 2,000 visitors per variant,
both arms at a true 10% conversion rate, so any flagged metric is by construction a false one,
across a growing number of simultaneously tracked metrics.

At $k = 1$, a metric flagged as a winner (win probability above 97.5% or below 2.5%) by chance
alone happens 4.8% of the time, close to the 5% a single well-calibrated test should produce. At
$k = 5$ metrics, that rises to 22.6%. At $k = 20$, it reaches 64.4%, close to the 64.2% the
closed-form probability $1 - (1 - 0.05)^{20}$ predicts.

The same arithmetic, $1 - (1 - \alpha)^k$, applied to a different threshold rule, produces the
identical 64.2% figure Chapter 2 found for twenty p-value-based dashboard metrics. A
win-probability cutoff and a p-value cutoff are both single-metric decision rules applied
independently across metrics, with no correction for how many chances each experiment gets.

A Bayesian framing changes what is being estimated, a full posterior instead of a point
estimate and a p-value, but it changes nothing about this specific arithmetic. Nineteen metrics
with no true effect and one metric carrying a true 3-point lift still produce, on average,
several false alarms alongside the one true finding, whether the flagging rule is $p < 0.05$
or win probability above 97.5%.

The frequentist fix from Chapter 2, dividing the significance threshold by the number of
metrics tracked, has a direct analogue here: instead of flagging a metric at a 97.5%/2.5%
win-probability cutoff, raise the bar to $1 - \alpha / (2k)$ on the correct side, the same
Bonferroni logic applied to a probability threshold instead of a p-value. @fig-multiple-metrics
shows this correction holding the false-alarm rate near 5% at every value of $k$ tested.

::: {.callout-warning}
Nothing about computing a posterior instead of a p-value removes the multiple-comparisons
problem. A win-probability threshold checked independently across many metrics inflates its
false-alarm rate the same way a p-value threshold does, because the underlying arithmetic is
the same regardless of which framework produced the threshold.
:::

A hierarchical model offers a more natural Bayesian-native answer than a narrower
Bonferroni-style correction. Treating each metric's effect as drawn from a shared distribution
pulls a noisy metric's estimate toward the group's pattern; a Bonferroni correction only raises
the bar the metric has to clear. The next section builds that same kind of partial-pooling
model, applied to segments of an experiment's traffic instead of metrics, and the shrinkage
mechanism carries over directly.

## Hierarchical A/B testing: partial pooling across segments

A retail chain trusts its flagship store's weekly numbers on their own, treats a brand-new
kiosk's first week with more caution, and still lets what it has learned across every other
store inform its guess about that kiosk. Hierarchical modeling formalizes that same instinct:
each segment gets its own estimate, but small or noisy segments lean on the pattern the other
segments share.

Chapter 10 introduced PSIS-LOO and `az.compare()` with a promise attached: this chapter would
use both to check whether adding a segment-level effect to an experiment's model earns its
keep, or only adds machinery without adding predictive accuracy. This section keeps that
promise on the checkout-button test itself, split across four traffic-source segments: organic
search, paid search, referral links, and email.

| Segment | $n_A$ | $n_B$ | Observed lift (pp) |
|---|---:|---:|---:|
| Organic | 6,000 | 6,000 | +0.80 |
| Paid search | 4,000 | 4,000 | +1.88 |
| Referral | 1,200 | 1,200 | +0.08 |
| Email | 350 | 350 | +0.57 |

Three models fit that same table three different ways. *No pooling* estimates each segment's
lift on its own data alone, as if the other three segments did not exist. *Complete pooling*
collapses all four segments into a single shared lift, as if traffic source made no difference
at all. *Hierarchical* partial pooling sits between the two: each segment keeps its own lift
parameter, but those four parameters are drawn from a shared Normal distribution whose own
mean and spread the model also estimates from the data.

```python
import pymc as pm
import arviz as az

def fit_segment_model(n_a, k_a, n_b, k_b, pooling):
    segments = ["organic", "paid_search", "referral", "email"]
    with pm.Model(coords={"segment": segments}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1.5, dims="segment")
        if pooling == "no_pooling":
            delta = pm.Normal("delta", mu=0, sigma=1.5, dims="segment")
        elif pooling == "complete_pooling":
            delta_shared = pm.Normal("delta_shared", mu=0, sigma=1.5)
            delta = pm.Deterministic("delta", delta_shared * pm.math.ones(4), dims="segment")
        else:
            mu_delta = pm.Normal("mu_delta", mu=0, sigma=1)
            sigma_delta = pm.HalfNormal("sigma_delta", sigma=1)
            offset = pm.Normal("offset", mu=0, sigma=1, dims="segment")
            delta = pm.Deterministic("delta", mu_delta + offset * sigma_delta, dims="segment")
        p_a = pm.math.invlogit(alpha)
        p_b = pm.math.invlogit(alpha + delta)
        pm.Binomial("obs_a", n=n_a, p=p_a, observed=k_a, dims="segment")
        pm.Binomial("obs_b", n=n_b, p=p_b, observed=k_b, dims="segment")
        idata = pm.sample(1000, tune=1000, idata_kwargs={"log_likelihood": True})
    return idata

comparison = az.compare({
    "no_pooling": fit_segment_model(n_a, k_a, n_b, k_b, "no_pooling"),
    "complete_pooling": fit_segment_model(n_a, k_a, n_b, k_b, "complete_pooling"),
    "hierarchical": fit_segment_model(n_a, k_a, n_b, k_b, "hierarchical"),
})
```

::: {.callout-tip}
The hierarchical branch above uses the same non-centered parameterization, an offset drawn from
a standard Normal and scaled afterward, that Chapter 9 introduced for the horseshoe prior.
Sampling a segment's lift directly from `Normal(mu_delta, sigma_delta)` couples that segment's
value to `sigma_delta` in a way that produces a narrow, hard-to-sample funnel whenever a
segment's own data is too sparse to pin its lift down; the offset form breaks that coupling.
:::

Fitting this model can still log divergent transitions if `sigma_delta` gets sampled with a step
size too large for how tight the funnel's neck becomes, even with the offset form above.
@sec-divergences-funnel in Chapter 9 builds the diagnostic for this shape of problem: plotting a
population-level scale against a group's deviation, and marking where a fixed step size stops
working. Applied here, that same picture would put `sigma_delta` on one axis and a sparse
segment's `delta_offset`, Referral or Email, the two segments with the least data to pin down
their own lift, on the other; divergences would cluster where `sigma_delta` gets pulled small by
the other segments while these two still carry plenty of individual uncertainty. Running this
model's own trace through `az.plot_pair(idata, var_names=["sigma_delta", "delta_offset"],
coords={"segment": "referral"}, divergences=True)` gives the same diagnostic Chapter 9 builds,
pointed at this chapter's own segment-level parameters instead of a regression coefficient's
population-level scale.

Running the equivalent computation directly from posterior draws, the same reproducibility
choice Chapter 10 made for its own figure, produces the table `az.compare()` would print:

| Model | elpd_loo | SE | d_elpd | dSE |
|---|---:|---:|---:|---:|
| complete_pooling | -33.8 | 2.1 | 0.0 | 0.0 |
| hierarchical | -34.0 | 2.0 | -0.2 | 0.35 |
| no_pooling | -36.8 | 1.6 | -3.0 | 1.15 |

::: {#fig-hierarchical-segments}
```{=html}
<iframe src="../_generated/chapter-bayes-ab-testing-fig-hierarchical-segments.html" width="100%"
        height="480" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Left: the same elpd_loo comparison as the table, best model first. Right: each segment's true
lift against what no pooling and partial pooling recovered from the data. Referral's near-zero
no-pooling estimate, driven by a small sample landing close to even, moves toward the group
pattern under partial pooling and lands close to its true 0.5-point lift; Email, the smallest
segment, gets pulled the same direction even though its own true lift ran the other way.
:::

Read the table the way Chapter 10 recommended: against `d_elpd` and `dSE`, not the plain `SE`
column. The hierarchical model's gap from complete pooling, -0.2, sits well inside its own
0.35-point standard error, so this dataset cannot distinguish the two on predictive accuracy
alone. No pooling's gap, -3.0 against a 1.15-point standard error, is a clearer, roughly
three-standard-error loss: estimating four segments from scratch, with no sharing of
information, costs more than it buys here.

That leaves an honest, unglamorous answer to Chapter 10's question: adding the segment-level
effect does not measurably improve this model's predictions over ignoring segments entirely.
`az.compare()` is not refusing to reward the segment structure out of excess caution; with only
eight aggregated observations, four segments times two arms, several of them carry a Pareto
$\hat{k}$ diagnostic above 0.70, the same warning sign Chapter 10's own diagnostic flags, so
this particular elpd comparison is closer to suggestive than decisive.

What the elpd table cannot show, and the right-hand panel of @fig-hierarchical-segments does, is
what partial pooling buys at the level of a single segment's estimate instead of the model's
aggregate fit. Referral's no-pooling lift came out to essentially zero, an artifact of a small
sample happening to split nearly even; partial pooling pulled it toward the other segments'
generally positive pattern and closer to its true 0.5-point lift. Email tells the harder
version of the same story: partial pooling pulled its estimate toward that same shared pattern
too, but Email's own underlying lift ran negative, so here the shared pattern pulled a small,
noisy segment further from the truth, not closer to it.

::: {.callout-warning}
Partial pooling borrows strength from segments that resemble each other, and most of the time
that borrowing helps a noisy, small segment more than it hurts. It is not free. A segment whose
underlying effect truly differs from its peers, combined with too little data of its own to
prove that difference, can still end up pulled the wrong direction. Treat hierarchical
structure as a default worth reaching for, and still check what each segment's own estimate
says on its own.
:::

## Latent subgroups: what a mixture model finds when the segment isn't logged

A restaurant's average four-star rating can hide two different dining rooms behind it: one
table that loved the meal and rated it five stars, another that sent a dish back and rated it
two. The average is correct and describes neither table. An experiment's average effect can
hide the same split.

The hierarchical section above handles a version of this problem where the split is a column
the dataset provides: four traffic sources, each one countable and nameable ahead of time.
This section handles the harder version, where a split is there but the experiment platform
never logged it.

Consider the same checkout redesign, now measured on a different metric: seconds spent on the
checkout page, from arrival to a completed order, under Variant B. A visitor who has used the
redesigned layout before moves through it quickly, since a new layout only slows someone down
the first time they see it. A first-time visitor, still expecting the old flow, spends longer
finding each field. Whether a given visitor is seeing the redesign for the first time is the
kind of detail an experiment platform often does not capture: the logging pipeline behind this
chapter's own sample-ratio-mismatch section records a conversion and a timestamp. A visitor's
prior exposure to the page never makes it into that log.

::: {#fig-latent-mixture}
```{=html}
<iframe src="../_generated/chapter-bayes-ab-testing-fig-latent-mixture.html" width="100%"
        height="480" style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Left: the pooled checkout-time change fit as one effect, a single Normal curve sitting between
the data's two humps and matching neither well. Right: the same data fit as a two-component
mixture. One cluster, 62% of simulated visitors, checks out about nine seconds faster; the
other, 38%, checks out about six seconds slower.
:::

@fig-latent-mixture fits both views to the same 2,400 simulated Variant B visitors. The left
panel's single-Normal fit reports one number: checkout time changed by 3.4 seconds on average
(positive means faster), with a standard deviation of 8.1 seconds. That standard deviation is
doing more work than a single-effect summary usually admits. A wide spread like this can look
like ordinary noise around one typical visitor's experience. Here it describes something else:
the gap between two different experiences that got averaged together.

**What it is.** A finite mixture model treats an outcome, here each visitor's change in checkout
time, as a blend of $K$ component distributions. Each visitor belongs to one component, but
which one stays unobserved; the model sees only the outcome and infers a weight for each
component (what share of visitors belong to it), alongside that component's own mean and spread.
Fitting a mixture at $K = 2$ asks a specific question of the data: does a single Normal curve
explain this outcome, or does the outcome look like two effects stacked on top of each other?

**Why it matters.** The right panel's two-component fit recovers a weight of 38% for a cluster
centered near -5.8 seconds, checkouts that got slower, and a weight of 62% for a cluster centered
near +9.0 seconds, checkouts that got faster. A team watching only the aggregate 3.4-second
improvement would ship Variant B and call it a clear win. Both numbers matter for what happens
next: close to 900 visitors out of every 2,400 reaching this checkout page spend roughly six
additional seconds every time they check out, and the aggregate number hides that cost entirely.
Whether that 38% lines up with first-time buyers, a specific device class, or something else
unrecorded is a question this mixture model leaves open, since it never observed group
membership. What it can do is tell a team that a second effect is there to go looking for, a
signal the single win-probability number from earlier in this chapter never provides.

**How to compute it.** Fitting a two-component Normal mixture in PyMC needs one piece of care
beyond the models built earlier in this chapter: without a constraint, the two components are
not identifiable from each other. Swap "component 0" and "component 1" everywhere in a set of
posterior draws and the mixture still describes the same distribution, so nothing in the model
favors one labeling over the other. Label switching describes this risk: an MCMC sampler left
alone can flip which component it calls "0" partway through a chain, and averaging posterior
draws across that flip corrupts every summary statistic computed from them, even though each
individual draw is a valid fit [@stephens2000]. The `ordered` transform on `means` below closes
that gap: it forces component 0's mean to sit below component 1's mean at every draw, so the
sampler always commits to one fixed labeling. Left unconstrained, two equally valid labelings
sit side by side, and the chain can wander between them.

```python
import numpy as np
import pymc as pm

with pm.Model() as mixture_model:
    weights = pm.Dirichlet("weights", a=np.ones(2))
    means = pm.Normal(
        "means", mu=0, sigma=10, shape=2,
        transform=pm.distributions.transforms.ordered,
        initval=np.array([-6.0, 6.0]),
    )
    sigmas = pm.HalfNormal("sigmas", sigma=6, shape=2)
    pm.NormalMixture("checkout_time_change", w=weights, mu=means, sigma=sigmas,
                      observed=checkout_time_delta)
    idata = pm.sample(1500, tune=1500, chains=2, target_accept=0.9)
```

Each visitor's checkout-time change $\delta_i$ is modeled as a weighted blend of $K$ Normal
components:

$$p(\delta_i) = \sum_{c=1}^{K} w_c \, \mathcal{N}(\delta_i \mid \mu_c, \sigma_c^2), \qquad
\sum_{c=1}^{K} w_c = 1$$

with the weights $w_c$ drawn from a Dirichlet prior, so every visitor's outcome is a blend of $K$
possible experiences.

Choosing $K$ is itself a model-comparison question. Chapter 10's `az.compare()` machinery, used
earlier in this chapter to check whether a segment-level effect earned its keep, applies here
too: fit the mixture at $K = 1$ (which collapses to the single-Normal fit in the left panel),
$K = 2$, and $K = 3$, and let `elpd_loo` say whether an extra component pays for itself.

::: {.callout-warning}
A mixture model finds a split in the data. Confirming what causes that split takes independent
evidence: a login flag, a device ID, something this experiment left unlogged. The two clusters
recovered here look like they map onto first-time versus returning visitors; treat that mapping
as a hypothesis for that evidence to confirm, and treat the mixture result itself as a reason to
go find the missing column.
:::

The checks earlier in this chapter still apply here. A sample-ratio-mismatch problem still
invalidates every posterior computed from the affected data, mixture or not. Logging the segment
that explains the split, once that becomes cheap enough to add, still beats inferring it from a
mixture fit.

## The limits of Bayesian inference

Nothing in this chapter repairs the confounding problem from Chapter 1's kidney-stone study. If
the checkout-button experiment happened to assign more mobile visitors to Variant B, and mobile
visitors convert differently regardless of button design, a Bayesian analysis of the combined
results is exposed to the identical Simpson's-paradox reversal a frequentist analysis would be.

Conjugate priors, closed-form win probabilities, and expected loss all assume the underlying
experiment was designed well: randomized assignment, a stable population, and no lurking
variable steering which visitors saw which variant.

::: {.callout-important}
Bayesian methods change how uncertainty is quantified. They do not fix a poorly designed
experiment. Randomized assignment and a stable population still come first.
:::

Bayesian methods change how uncertainty about a well-designed experiment gets quantified and
acted on. Designing the experiment correctly still comes first.

## References {.unnumbered}

::: {#refs}
:::
