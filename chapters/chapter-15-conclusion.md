# Conclusion

Fourteen chapters ago, a single dataset opened this book: simulated response times for a
checkout API. That dataset never left.

Its mean and median pulled apart when a slow tail crept in. Its distribution turned out to be
right-skewed, which broke the standard-deviation alert rule Chapter 1 tried first.

A regression line predicted it from payload size in Chapter 4. A cross-validated penalty tuned
that regression in Chapter 5. A spline bent it around a saturation point Chapter 6's straight
line could not follow.

A random forest and a boosted model classified whether a deployment touching it would need a
rollback.

Then Part 3 went back through the same ground a second time. It built a credible interval on
that regression's coefficients (Chapter 9), a PSIS-LOO comparison between competing versions of
it (Chapter 10), and a Gaussian process replacing the spline with a full posterior curve
(Chapter 11).

It also built a BART posterior interval next to the random forest's point prediction
(Chapter 12), a Bayesian-optimized search over the boosted model's hyperparameters
(Chapter 13), and finally a Bayesian posterior deciding whether a checkout-button redesign
should ship (Chapter 14).

One dataset, fourteen different questions asked of it, several of them asked twice. That
repetition was deliberate: the same numbers look different depending on which tool is pointed
at them, and knowing which tool to reach for is most of the job.

## The throughline: a number is not an answer until it survives a check

Three moments recur across this book, in different costumes each time.

The first is the gap between what a summary statistic reports and what happened underneath it.
A mean hides a slow tail (Chapter 1).

A p-value gets treated as the probability the null hypothesis is true, which is not what it
measures (Chapter 2, and the American Statistical Association said so directly, in print, in
2016).

An R-squared goes up every time a useless predictor is added, whether or not that predictor
means anything (Chapter 4).

In every case, the fix was the same: ask what the number was built to measure, then check
whether that matches the question being asked.

The second is the confounder. The kidney-stone data in Chapter 1 showed a treatment that won
every subgroup and lost overall, because stone severity influenced both which treatment a
patient got and how likely they were to recover.

Chapter 14 closed on the same warning: a Bayesian posterior is just as vulnerable to that
confound as a frequentist p-value is, because the problem lives in how the data was collected,
not in which framework analyzes it afterward.

No model in Part 2, however flexible, and no posterior in Part 3, however well calibrated,
repairs a comparison that was never fair to begin with. That is a fact about experimental
design, and it sits upstream of every technique this book covers.

The third is the honest limit of a method. Every chapter in Part 2 ended by naming what its
method could not do.

OLS assumes a linearity that a spline has to relax. Cross-validation assumes something close to
independent observations that a time series violates. A random forest's variable importance can
mislead when predictors are correlated. Gradient boosting overfits fast without a validation
set watching it.

::: {.callout-warning}
A common mistake with cross-validation: shuffling time-ordered data before splitting into folds
lets future observations leak into training, which inflates the reported score. Split
chronologically instead, training on earlier periods and validating on later ones.
:::

A book that only listed what each method does well would be a sales brochure. Knowing where a
tool breaks is what makes it safe to use where it does not.

::: {.callout-note}
Three checks worth repeating for any result: does the number measure what you think it
measures, could a confounder explain the pattern, and where does this method's honest limit
sit?
:::

## Two lenses, not two camps

Part 2 built a point estimate and a p-value or held-out error score for each question: what
value should we expect, and how much would we trust that estimate under repeated sampling.

Part 3 revisited every one of those questions with a full posterior distribution instead. It
closed with a Bayesian A/B test of a conversion rate that answered with a probability that one
variant beats another, and an expected cost of being wrong.

Chapter 14 was explicit that this is not a strictly-better replacement: a frequentist test is
simpler to run, more standardized across a team, and better understood by most stakeholders
reading a dashboard.

A Bayesian posterior costs more to compute and more to explain, and buys a direct probability
statement a p-value or a held-out error score cannot give.

::: {.callout-tip}
Reach for a frequentist test when speed and a shared team understanding matter most. Reach for a
Bayesian posterior when the decision needs a direct probability statement, and the extra cost to
compute and explain it is worth paying.
:::

Part 3's five other chapters made the same trade explicit for a specific Part 2 method each.

Ridge regression turned out to be the posterior mode under a Gaussian prior on the coefficients;
the Lasso, under a Laplace prior. Chapter 9 built the full posterior around both, plus a
credible interval that means something different from Chapter 4's confidence interval, even
though the two are often read as interchangeable.

::: {.callout-warning}
A 95% credible interval means there is a 95% probability, given the data and the prior, that
the parameter falls in that range. A 95% confidence interval makes no such claim about the
parameter itself; it describes how often the procedure would capture the true value across
repeated sampling, and the two get used interchangeably far too often.
:::

Chapter 10 built PSIS-LOO and WAIC as the posterior-based counterpart to Chapter 5's
cross-validated error estimate, model comparison without refitting on every held-out fold.

Chapter 11 replaced Chapter 6's smoothing spline with a Gaussian process, trading a single
fitted curve for a full posterior over curves and uncertainty that widens honestly away from
the data.

Chapter 12 replaced Chapter 7's random forest with BART, a point prediction next to a posterior
interval on the same input.

Chapter 13 took on the one method that resists a clean posterior, gradient boosting, and was
honest about that limit: Bayesian hyperparameter search and NGBoost's predictive distributions
are useful, practical tools, but neither is a full Bayesian boosting model the way BART is a
full Bayesian tree ensemble.

None of these six pairs is a rivalry between Part 2 and Part 3. They are the same question,
asked twice, once for a single best answer and once for a full accounting of how uncertain that
answer is.

## What this book did not cover

This book stops at the boundary of tabular data and single-outcome experiments. It does not
cover multi-armed bandits, sequential experiment designs that adapt while running, causal
inference tools built for observational (non-randomized) data such as instrumental variables or
regression discontinuity, or the deep-learning side of predictive modeling.

::: {.callout-note}
Instrumental variables and regression discontinuity solve a different problem than the A/B
tests in Chapters 2 and 14: they estimate a causal effect when nobody randomized who got the
treatment. Neither technique appears in this book, and neither is a drop-in replacement for
randomization when it is available.
:::

Each of those is a book of its own, and each builds on the foundation laid here: a working grip
on what a distribution looks like, what a confidence interval and a p-value claim, how a
model's flexibility trades off against its variance, and how a posterior differs from a point
estimate.

## Where to go from here

Every chapter's interactive figures are built from the same simulated checkout-API data,
regenerable from the paired `-plots.py` file next to each chapter's prose.

Pull the code apart, change the simulation parameters, and watch the figures respond. That is a
faster way to build intuition for any of these methods than reading about them a second time.

The two case studies that recur throughout, the 1986 kidney-stone study and the 1936 Literary
Digest poll, are both drawn from the historical record and documented in detail well beyond
what this book excerpted from them. The citations are in the bibliography for that reason.
