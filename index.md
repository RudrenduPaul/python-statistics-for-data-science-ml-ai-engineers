# Preface {.unnumbered}

```{=html}
<p style="margin-top:-0.5rem;">
  <a href="https://github.com/RudrenduPaul" style="display:inline-flex;align-items:center;gap:0.4rem;text-decoration:none;color:inherit;">
    <svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
    </svg>
    <span>github.com/RudrenduPaul</span>
  </a>
</p>
```

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
