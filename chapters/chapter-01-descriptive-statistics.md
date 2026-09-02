# Introduction to Descriptive Statistics

Each dataset raises the same four questions before any model gets near it: where is the
typical value, how spread out is everything around it, how do two variables move together,
and can this sample be trusted to speak for the population it was drawn from?

*Descriptive statistics* is the set of tools that answers those four questions. It has a
reputation for being the easy chapter, the one you skim on the way to regression and machine
learning. That reputation is a problem.

Most of the statistical mistakes that make it into production systems are
descriptive-statistics mistakes: a mean reported where a median was needed, a correlation
mistaken for causation, a sample that quietly excluded the group that mattered most.

This chapter works through five questions a data scientist or engineer runs into constantly:

1. *Where is the center of this data, and which "center" should I report?*
   Mean, median, and mode each answer a slightly different question, and picking the wrong
   one is the single most common way a summary statistic misleads a reader.
2. *How spread out is the data, and does that spread follow a shape I can trust?*
   Variance, standard deviation, skewness, and kurtosis describe the shape of a distribution,
   and that shape determines whether a threshold built on the mean will work at all.
3. *Do two variables move together, and how strongly?*
   Covariance and correlation quantify a relationship, but neither one tells you why it
   exists, a distinction that matters more than the formulas themselves.
4. *Does this sample represent the population I care about?*
   Population, sample, and the different ways a sample can go wrong (selection bias,
   nonresponse, missing data) determine whether anything computed from it means something.
5. *Can a pattern reverse itself depending on how the data is grouped?*
   Simpson's paradox shows that the answer is yes, more often than most people expect.

To keep the chapter grounded, each section returns to the same scenario: a backend engineer
looking at response-time logs for a production API endpoint. The dataset is simulated (a
log-normal distribution built to match the shape production latency data almost always has),
and it is labeled as simulated everywhere it appears.

Two historical case studies anchor the sections on selection bias and Simpson's paradox with
verifiable numbers instead of invented ones.

## Measures of central tendency: mean, median, and mode

Picture five friends timing how late the school bus was this week: 2, 3, 3, 4, and 20 minutes
(the bus broke down once). The mean splits the total evenly across all five. The median just
picks the middle number once they are sorted. The mode is whichever number repeats most.

All three answer "what's typical?" A single outlier like that 20 can pull them apart.

Suppose the on-call engineer for a checkout API pulls the last hour of response times and
wants one number to put in a dashboard. The *mean* is the arithmetic average: sum each
observation and divide by the count. For a set of latencies $x_1, \dots, x_n$,

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.$$

In other words, the mean spreads the total time across each request equally, as if each
request had taken the average amount of time.

The *median* is the middle observation once the data is sorted: half the requests were
faster, half were slower. The *mode* is the most frequently occurring value, useful when a
variable clusters around a small number of repeated values (a request that always returns
from cache in 4 ms, for instance).

::: {#fig-mean-median}
```{=html}
<iframe src="../_generated/chapter-01-fig-mean-vs-median.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Mean and median of 5,000 simulated checkout-API requests, as the share of slow, contaminating
requests increases. At 5% contamination, the mean sits noticeably above the median; drag the
slider further and the gap keeps widening while the bulk of the histogram barely moves.
:::

@fig-mean-median shows why the distinction matters more than it looks like it should. A
slider controls what share of requests are contaminated by a slow tail (a batch job, a cold
Lambda start, a downstream retry storm hitting the same endpoint).

At 0% contamination, the mean and median sit close together. Drag the slider to 5%, and the
mean jumps well past the median while the bulk of the histogram barely moves.

::: {.callout-tip}
The mean is sensitive to every value in the dataset, including the extreme ones. The median
only cares about which value sits in the middle. A single request that takes 30 seconds
instead of 40 milliseconds pulls the mean toward it; the median does not notice.
:::

This is why on-call dashboards that show "average response time" are trusted less than
dashboards that show median and tail percentiles side by side. The mean alone hides the
failure mode (a growing slow tail) that engineers most need to catch early.

## Measures of spread: variance and standard deviation

Imagine two students who both average a 75 across their tests. One always scores between 70
and 80. The other bounces between 40 and 100. Same average, quite different consistency.

Variance and standard deviation put a number on that bounciness: how far, on average, each
score strays from the middle.

Knowing the center of a distribution says nothing about how tightly the data clusters around
it. Two endpoints can share a median of 45 ms and behave completely differently in
production, one consistently landing near 45 ms and the other swinging between 10 ms and
300 ms.

*Variance* measures that spread by averaging the squared distance of each point from the
mean:

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2.$$

Here $\sigma^2$ denotes the variance: take each latency's distance from the mean, square it
so that a request 10 ms too slow and one 10 ms too fast count the same instead of canceling
out, then average those squared distances across all $n$ requests.

In other words, variance is the average squared distance between a request and the typical
request.

::: {.callout-note}
NumPy's `np.var` divides by $n$ by default, matching this formula, while pandas' `.var()`
divides by $n - 1$ by default. The same array of numbers can report two different variance
values depending on which library computed it.
:::

The *standard deviation*, $\sigma = \sqrt{\sigma^2}$, brings that number back into the
original units (milliseconds, rather than milliseconds squared). It is easier to read at a
glance than a squared quantity, which is why standard deviation, not variance, is the number
that shows up on dashboards and in alert rules.

Roughly speaking, a common way to build an alert threshold is "mean plus $k$ standard
deviations": flag anything above that line as anomalous. It is natural to reach for this
rule, since it works well for data that is roughly symmetric and bell-shaped.

Latency data almost never is.

::: {#fig-std-percentile}
```{=html}
<iframe src="../_generated/chapter-01-fig-std-vs-percentile.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A mean-plus-k-standard-deviation alert threshold swings across a wide range of positions as k
moves from 1 to 4. The 90th and 99th percentile lines, drawn on the same data, stay right
where they were defined regardless of k.
:::

@fig-std-percentile puts that threshold on simulated latency data next to two percentile
lines, the 90th and 99th (the latency below which 90%, or 99%, of requests finish). Move the
slider through $k = 1, 2, 3, 4$ and watch the standard-deviation-based threshold swing, while
the percentile lines stay fixed at whatever share of traffic they were defined to capture.

This is because a threshold built from the mean plus a multiple of the standard deviation
assumes a shape the data does not have. The Google SRE book makes the same point about
production latency directly: "a simple average can obscure these tail latencies, as well as
changes in them" [@googlesre2017].

::: {.callout-important}
That is why the SRE book's own recommendation is to define service-level indicators on
percentiles rather than the mean or a standard-deviation band. Variance and standard
deviation remain the right tool once a distribution is confirmed close to normal, a check
worth running before trusting the number.
:::

Percentiles carry a cost of their own, though. The 99th percentile and higher need enough
requests in a given window to mean anything.

A low-traffic endpoint or a narrow time window can make a reported p99 swing from one minute
to the next for reasons that have nothing to do with how the service is performing. That is
why teams that lean on tail percentiles also have to watch request volume per window, not
just the percentile line itself.

## Covariance and correlation

Think about height and shoe size. Taller people tend to have bigger feet, though not
perfectly: there are tall people with small feet and short people with large ones.

Covariance and correlation measure how strongly two things tend to move together, without
saying anything about which one, if either, is causing the other.

Suppose the same engineer suspects that larger request payloads are driving the slow tail.
*Covariance* measures whether two variables move together: positive when they rise and fall
together, negative when one rises as the other falls, and near zero when they move
independently.

$$\text{cov}(X, Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}).$$

Covariance's units are the product of the two variables' units (kilobytes times milliseconds
here), which makes the raw number hard to interpret on its own.

*Correlation* fixes that by rescaling covariance into a unitless number between $-1$ and $1$:

$$r = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}.$$

::: {#fig-cov-corr}
```{=html}
<iframe src="../_generated/chapter-01-fig-covariance-correlation.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Simulated payload size vs. latency, redrawn at increasing correlation strength. At $\rho = 0$
the cloud of points has no visible shape at all; by $\rho = 0.85$ the same two variables trace
a tight, almost linear band.
:::

@fig-cov-corr generates that payload-size-versus-latency scatterplot at several different
underlying correlation strengths, so you can see what a given value of $r$ looks like as a
cloud of points.

In other words, $r$ near 1 means payload size and latency move together almost in lockstep,
$r$ near 0 means knowing the payload size tells you nothing about the latency, and $r$ near
$-1$ means they move in opposite directions.

Does it make sense that a strong correlation here would prove payload size causes the
slowdown? It does not, and this is one of the most consequential gaps in applied statistics.

::: {.callout-warning}
A high $r$ says two variables move together; it says nothing about which one, if either,
drives the other. Both could be driven by a third factor, for instance a specific customer
tier that both uploads larger images and hits a slower code path for unrelated reasons.
:::

Part 3 returns to this problem directly, because untangling correlation from causation is the
entire reason controlled experiments and Bayesian A/B testing exist.

## Skewness and kurtosis

Picture a class's test scores: most students land near 80, but a couple score near 20. That
low pair pulls a "tail" down toward the low end.

Skewness measures which way that tail points and how stretched it is. Kurtosis measures
something different: how often extreme scores, high or low, show up compared to a normal
bell-curve class.

A mean, a median, and a standard deviation summarize a distribution with three numbers, but
two distributions can share all three and still look nothing alike.

*Skewness* measures the asymmetry of a distribution: positive skewness means a long tail
stretching to the right (a few sharply slow requests pulling the tail out, as in the checkout
API), negative skewness means the tail stretches left, and a skewness near zero means the
distribution is roughly symmetric.

*Kurtosis* measures how heavy the tails are relative to a normal distribution: high kurtosis
means more of the extreme values than a normal curve would predict, low kurtosis means fewer.

::: {#fig-skew-kurt}
```{=html}
<iframe src="../_generated/chapter-01-fig-skewness-kurtosis.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

A simulated distribution morphing from nearly symmetric to heavily right-skewed. Both
skewness and kurtosis climb together as the tail stretches, since a longer tail is, by
construction, also a heavier one.
:::

@fig-skew-kurt reports both statistics live as the shape changes. Watch the skewness number
climb as the tail stretches out, and watch kurtosis climb with it.

This is because skewness and kurtosis are not two independent knobs; they are both
consequences of the same underlying shape.

Recall the latency data from the first section. It is right-skewed for the same structural
reason the simulation is: a hard floor near zero (a request cannot finish in negative time)
paired with an unbounded upside (nothing stops a request from taking ten seconds).

Any variable with that shape, whether it is latency, income, or wait times, will show
positive skewness for the same reason.

::: {.callout-tip}
A common rule of thumb: skewness between $-0.5$ and $0.5$ counts as roughly symmetric, while
skewness beyond $1$ or $-1$ signals a distribution stretched enough that mean-based summaries
and standard-deviation thresholds deserve a second look before you trust them.
:::

## Population and sample

Measuring the average height of every student in a country would be the population.
Measuring just one classroom and using that as a stand-in for the whole country is a sample.

A sample is faster and cheaper to collect, but it only works if it represents the population
it stands in for.

*Population* refers to each unit you are ultimately interested in; *sample* refers to the
subset you observed and computed statistics from. If a checkout API serves ten million
requests a day, the population is all ten million; the hour of logs an engineer happened to
pull is a sample.

It is natural to want to skip this distinction, since in practice most day-to-day analysis is
done on samples and reported as if it described the population.

This distinction carries weight in a production system. Distributed-tracing tools (Datadog,
Honeycomb, Jaeger, and similar systems) capture only a fraction of production traffic, never
the full request stream.

Recording a full trace for each one of ten million daily requests would be prohibitively
expensive to store, so these systems *sample*, keeping some fraction and discarding the rest.

Whether the statistics computed from that sampled trace data represent the full population of
requests depends entirely on how the sampling was done. The next section addresses that
question directly.

::: {.callout-note}
Notation keeps population and sample straight on the page: Greek letters ($\mu$, $\sigma$)
denote population values, and Latin letters ($\bar{x}$, $s$) denote the sample statistics
computed to estimate them. A sample statistic carries sampling error that the population value
it estimates does not.
:::

## Data types: nominal, ordinal, continuous, and discrete

You can average a list of temperatures, but you cannot average a list of favorite colors.

Statisticians sort data into a few basic types because the type determines which math is
allowed on it: some data can be ranked or averaged, some can only be counted or labeled.

The API log line behind each figure in this chapter carries four different kinds of fields,
and each kind constrains which statistics are meaningful to compute on it.

- *Nominal* data can be sorted into categories with no inherent order: the endpoint name
  (`/checkout`, `/cart`, `/search`) is nominal. Computing a mean endpoint name is meaningless;
  counting how often each one appears is the right operation.
- *Ordinal* data has categories with a meaningful order but no fixed distance between them: a
  log severity level (`debug` < `info` < `warn` < `error`) is ordinal. "Error" is worse than
  "warn," though the gap between them carries no numeric value.
- *Continuous* data can take any value within a range: latency in milliseconds is continuous,
  since 45.372 ms is as valid a value as 45 ms or 46 ms.
- *Discrete* data can only take specific, typically countable values: the number of retries a
  request went through (0, 1, 2, and so on) is discrete, since a request cannot retry 2.5
  times.

Recall that each statistic covered so far, mean, median, variance, correlation, assumes
numeric (continuous or discrete) data.

Applying them to nominal or ordinal fields without first converting those fields into a
numeric encoding produces a number that looks legitimate and means nothing.

::: {.callout-important}
A common mistake: averaging a log severity level (encoding `debug`=0, `info`=1, `warn`=2,
`error`=3, say) and reporting "mean severity 1.7." Ordinal categories have order but no fixed
distance between them, so the average has no defensible interpretation, even though the
computation itself runs without error.
:::

## Selection bias

Imagine judging how popular a school lunch is by only asking the students who complained
about it. You would conclude almost nobody likes it, even if most students were fine with it.

The complainers were never a fair cross-section of the whole cafeteria in the first place.

Suppose an engineering team wants to know the typical latency their users experience, and
pulls that number exclusively from traces flagged as errors, because those are the traces the
on-call system happened to retain in detail.

Each number that comes out of that analysis will be too slow, because the sample was never a
fair draw from the population of requests, regardless of how the service performs under normal
conditions.

::: {.callout-note}
This is *selection bias*: systematic error that occurs when the sample is not drawn randomly
from the population, so some kinds of observations are more or less likely to be included
than others.
:::

One of the starkest illustrations of selection bias in the historical record has nothing to
do with software.

::: {#fig-literary-digest}
```{=html}
<iframe src="../_generated/chapter-01-fig-literary-digest.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Predicted vote share vs. the election result, from a ten-million-person sample: a 24-point
reversal too large to blame on ordinary sampling noise.
:::

@fig-literary-digest tells the story in numbers. In 1936, the magazine *Literary Digest*
mailed ten million presidential-election questionnaires and got about 2.3 million back, an
enormous sample by any standard.

It predicted Alf Landon would beat Franklin Roosevelt with 57% of the vote. Roosevelt won with
almost 61% of the popular vote.

The magazine had built its mailing list from telephone directories, car registrations, and
its own subscriber rolls, all of which skewed toward wealthier households in the middle of
the Great Depression, a group that leaned Republican. On top of that, the people who bothered
to mail back a postcard were not a random subset of the people who received one [@squire1988].

Ten million questionnaires could not fix a sampling frame that excluded most of the country.

That story maps onto the tracing example directly. A monitoring system that samples more
heavily from slow or error-flagged requests (a common design, since those are the traces
worth keeping in detail) will always compute a latency distribution skewed slower than
production traffic.

The reason is the same one behind the 1936 poll's Republican skew: the sample captured only a
narrow slice of the population. Four variants of this problem come up constantly enough to
have their own names:

- **Sampling bias**: the sample itself is not drawn randomly. A trace sampler that keeps each
  error and 1% of the rest is a sampling-bias machine by design.
- **Self-selection bias**: units choose whether to be in the sample. A customer satisfaction
  survey that only frustrated users bother to fill out will look worse than the customer base
  as a whole.
- **Nonresponse bias**: some units are simply less likely to respond or be recorded, the same
  mechanism behind the *Literary Digest* postcards.
- **Recall bias**: participants misremember past events, common in any survey asking people to
  self-report past behavior rather than measuring it directly.

::: {.callout-important}
The fix is rarely "collect more data." A larger biased sample is still biased; the *Literary
Digest* poll is proof of that at a scale of millions. The fix is a sampling process designed
to give each unit in the population a known, nonzero chance of being included.
:::

## Missing values and mean imputation

Suppose a teacher is missing three students' test scores because they were absent on test
day. One fix is to give those three students the class average as a stand-in score.

That is mean imputation: filling a gap with the average of the numbers you do have, so
nothing gets thrown out, even though the fill-in value is a guess.

Production datasets have gaps: an instrumentation library that fails to emit a trace, a
dropped log line, or a schema field added partway through a rollout, so that some requests
were logged under the old schema and some under the new one.

*Imputation* is the practice of filling those gaps with a plausible value so the row can
still be used. The simplest version, *mean imputation*, replaces each missing value with the
mean of the observed values for that field.

::: {#fig-mean-imputation}
```{=html}
<iframe src="../_generated/chapter-01-fig-mean-imputation.html" width="100%" height="540"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Mean imputation narrows the spread of the dataset as the missing share grows. By the time a
third of the values are missing, the imputed dataset's standard deviation sits well below the
original.
:::

@fig-mean-imputation applies mean imputation to the simulated latency data at increasing
shares of missing values and tracks what happens to the spread of the resulting dataset.

At even a modest missing share, the standard deviation after imputation drops well below the
original.

::: {.callout-warning}
Each imputed value lands right on the mean, contributing nothing to the spread. The more
values you impute this way, the more the dataset looks artificially uniform. Mean imputation
is safe mainly when values are missing completely at random and the underlying distribution
is close to symmetric.
:::

Recall from earlier in this chapter that latency data is right-skewed. Imputing missing
latencies with the mean would systematically understate how heavy the tail is.

Data is not always missing at random. A slow request, for instance, is more likely to hit a
client-side timeout before it finishes writing its trace, so slow requests are the ones most
likely to be dropped from the logs in the first place.

When that happens, more careful approaches such as multiple imputation, which fills each gap
with a range of plausible values drawn from the data's own distribution instead of one fixed
number, are worth the extra effort.

## Simpson's paradox

Suppose one baseball player has a higher batting average than another in both the first half
of the season and the second half, yet a lower average for the season combined.

That sounds impossible, but it can happen if the two halves had sharply different numbers of
at-bats.

Each statistic in this chapter has been building toward a single warning: a pattern that
holds within each group can reverse itself when the groups are combined.

::: {#fig-simpsons}
```{=html}
<iframe src="../_generated/chapter-01-fig-simpsons-paradox.html" width="100%" height="520"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Success rate by treatment, split by stone size vs. combined (Charig et al. 1986). Treatment A
wins both subgroups yet loses once the two groups are pooled together.
:::

This is *Simpson's paradox*, and @fig-simpsons shows the clearest documented case of it: a
1986 study in the *British Medical Journal* comparing two treatments for kidney stones
[@charig1986].

Across 700 patients, split evenly between open surgery (Treatment A) and percutaneous
nephrolithotomy (Treatment B), the combined success rate favored Treatment B: 83% versus 78%
for Treatment A. On that number alone, Treatment B looks like the better choice.

But the study also recorded stone size, and once the results are split by stone size, the
ranking flips in both groups. Among patients with small stones, Treatment A succeeded 93% of
the time versus 87% for Treatment B. Among patients with large stones, Treatment A succeeded
73% of the time versus 69% for Treatment B.

Treatment A wins in each subgroup and still loses overall.

::: {.callout-tip}
The mechanism is a *confounding variable*: physicians preferentially assigned Treatment A,
the more invasive option, to patients with large, more serious stones, and assigned the
gentler Treatment B more often to patients with small, easier stones.
:::

Stone size drove both which treatment a patient received and how likely that patient was to
recover, which is what a confounder does by definition. Because Treatment B was tested
disproportionately on the easier cases, it looked better in the combined numbers despite
losing on merit in both subgroups.

The same mechanism applies well beyond kidney stones. Any time a variable influences both
which group an observation falls into and the outcome being measured, combining groups can
reverse the pattern each group shows on its own.

A feature rollout evaluated across "small customers" and "large customers" is exposed to the
identical risk if the rollout was not randomly assigned within each segment.

Chapter 2 introduces the hypothesis-testing machinery for asking whether a difference like
this reflects a lasting effect or ordinary noise. Part 3 builds Bayesian A/B testing directly
on top of that machinery, precisely so a rollout decision does not fall into the same trap the
kidney stone study did.

## References {.unnumbered}

::: {#refs}
:::
