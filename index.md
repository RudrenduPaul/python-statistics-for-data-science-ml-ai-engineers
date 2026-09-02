# Preface {.unnumbered}

This book is a practical statistics handbook for data scientists and engineers who need to
run, read, and defend an experiment, not just pass a stats exam. Every chapter pairs the
underlying theory with a worked example built around one running dataset: simulated response
times for a checkout API, followed through descriptive statistics, hypothesis testing,
regression, tree-based models, and a Bayesian A/B test.

Every figure in the book is a Plotly chart with a slider. Drag it, and the chart updates in
your browser, no server and no setup required. The source for every chapter is plain Markdown;
the source for every figure is a plain Python script in `chapters/`, so nothing here is a
black box, and every chart is regenerable from the code sitting right next to the prose that
uses it.

This book is free and open source.

## See it in action

Three of the charts that appear later in the book, live and interactive right now. Drag any
slider below before reading a word of Chapter 1.

::: {#fig-index-literary-digest}
```{=html}
<iframe src="_generated/chapter-01-fig-literary-digest.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

The 1936 Literary Digest poll called the election for Landon at 57% to 43%, from a
ten-million-person sample. Roosevelt won. Drag the slider between the poll's prediction and
the result: Chapter 1 walks through why a sample that size still missed the outcome by 24
points.
:::

::: {#fig-index-simpsons-paradox}
```{=html}
<iframe src="_generated/chapter-01-fig-simpsons-paradox.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Charig et al.'s 1986 kidney-stone data: Treatment A wins on both stone sizes taken
separately. Drag the slider to pool the two groups together, and Treatment B wins instead.
Chapter 1 uses this reversal to introduce Simpson's paradox.
:::

::: {#fig-index-bayesian-ridge}
```{=html}
<iframe src="_generated/chapter-04-fig-bayesian-ridge.html" width="100%" height="560"
        style="border:1px solid #ddd; border-radius:6px;" loading="lazy"></iframe>
```

Shrink the prior variance with the slider and watch the posterior over a regression
coefficient narrow and slide toward zero. Chapter 4 shows this is the same shrinkage Ridge
regression's penalty term produces, derived instead from a Gaussian prior.
:::
