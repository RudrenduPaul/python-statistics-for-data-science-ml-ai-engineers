"""
Interactive figures for Chapter 14: Bayesian Experimentation for A/B Testing (Part 3).

Same pattern as prior chapters: each function builds one self-contained, standalone Plotly
HTML page and writes it to ../_generated/. Run directly to regenerate every figure:
    python3 chapter-14-bayesian-ab-testing-plots.py

Every FIGURES key below matches an iframe src embedded in chapter-14-bayesian-ab-testing.md.
Keep it that way: a key here with no matching iframe in the chapter is a figure the reader
never sees; an iframe in the chapter with no key here is a figure that quietly disappears the
next time this script runs from a clean _generated/.
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

RNG = np.random.default_rng(101)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_generated")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig: go.Figure, name: str) -> str:
    path = os.path.join(OUT_DIR, f"{name}.html")
    # auto_play=False: plotly.py's default HTML export appends a
    # `Plotly.animate(divid, null)` call right after the initial render to prime the
    # slider/animation machinery. In the CDN-pinned Plotly.js build this book ships
    # (4.0.0), that priming call can clear a trace that sits outside every frame's data
    # and never repaint it (confirmed for a Contour background in Chapter 13's search-
    # trajectory figure). None of this chapter's figures keep a trace outside its
    # frames, but disabling the priming call costs nothing: the slider itself calls
    # Plotly.animate with concrete frame args on every step.
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, auto_play=False)
    return path


# ---------------------------------------------------------------------------
# Figure 1: Beta priors, from uninformative to strongly informative
# ---------------------------------------------------------------------------
def fig_prior_shapes() -> go.Figure:
    priors = [(1, 1), (11, 91), (50, 450), (200, 1800), (1000, 9000)]
    labels = ["Beta(1,1) uninformative", "Beta(11,91) weakly informative (~100 obs.)",
              "Beta(50,450) more confident (~500 obs.)",
              "Beta(200,1800) strongly informative (~2,000 obs.)",
              "Beta(1000,9000) highly informative (~10,000 obs.)"]
    x = np.linspace(0, 0.4, 400)
    x_list = x.tolist()
    frames = []
    for (a, b), label in zip(priors, labels):
        pdf = stats.beta.pdf(x, a, b)
        frames.append(
            go.Frame(
                name=label,
                data=[go.Scatter(x=x_list, y=pdf.tolist(), mode="lines", fill="tozeroy",
                                  line=dict(color="#4C78A8", width=2.5))],
                layout=go.Layout(annotations=[dict(
                    x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                    xanchor="right",
                    text=f"mean = {a / (a + b):.1%}, pseudo-observations = {a + b - 2:.0f}",
                    font=dict(size=13, color="#333"),
                )]),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="A prior expresses the same 10% central belief at five different strengths",
        xaxis_title="conversion rate",
        yaxis_title="density",
        xaxis_range=[0, 0.4],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "prior: "},
            "steps": [
                # Short display labels on purpose: the full frame names (e.g.
                # "Beta(1000,9000) highly informative (~10,000 obs.)") overhang past
                # the plot's right margin at the slider's end positions and get
                # clipped by the plot boundary. Even the shortened "Beta(1000,9000)"
                # form still clipped by a couple of characters at the rightmost step,
                # so the label drops the "Beta" prefix too and the right margin grows
                # to give the last tick's centered text room to breathe. Frame names
                # stay unchanged below since the animate() call targets them by name.
                {"label": short, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f, short in zip(frames, ["(1,1)", "(11,91)", "(50,450)",
                                              "(200,1800)", "(1000,9000)"])
            ],
        }],
        margin=dict(t=60, l=60, r=60, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2: a single variant's posterior narrowing as data accumulates
# ---------------------------------------------------------------------------
def fig_posterior_update() -> go.Figure:
    prior_a, prior_b = 1, 1
    true_rate = 0.10
    sample_sizes = [0, 25, 100, 500, 2000, 8000]
    x = np.linspace(0, 0.4, 400)
    x_list = x.tolist()
    frames = []
    for n in sample_sizes:
        conv = RNG.binomial(n, true_rate) if n else 0
        a, b = prior_a + conv, prior_b + (n - conv)
        mean = a / (a + b)
        frames.append(
            go.Frame(
                name=str(n),
                data=[go.Scatter(x=x_list, y=stats.beta.pdf(x, a, b).tolist(), mode="lines",
                                  fill="tozeroy", line=dict(color="#4C78A8", width=3))],
                layout=go.Layout(annotations=[dict(
                    x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                    xanchor="right",
                    text=(f"n = {n}, conversions = {conv}<br>"
                          f"posterior mean = {mean:.1%}"),
                    font=dict(size=13, color="#333"),
                )]),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="A single variant's Beta(1,1) posterior narrows and settles as visitors accumulate",
        xaxis_title="conversion rate",
        yaxis_title="density",
        xaxis_range=[0, 0.4],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "visitors observed: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Variant A and Variant B posteriors separating as each collects data
# ---------------------------------------------------------------------------
def fig_ab_overlay() -> go.Figure:
    prior_a, prior_b = 1, 1
    true_a_rate, true_b_rate = 0.10, 0.125
    sample_sizes = [50, 200, 1000, 5000]
    x = np.linspace(0, 0.3, 400)
    x_list = x.tolist()
    frames = []
    for n in sample_sizes:
        a_conv = RNG.binomial(n, true_a_rate)
        b_conv = RNG.binomial(n, true_b_rate)
        aa, ab = prior_a + a_conv, prior_b + (n - a_conv)
        ba, bb = prior_a + b_conv, prior_b + (n - b_conv)
        frames.append(
            go.Frame(
                name=str(n),
                data=[
                    go.Scatter(x=x_list, y=stats.beta.pdf(x, aa, ab).tolist(), mode="lines",
                               name="Variant A posterior", line=dict(color="#4C78A8", width=3)),
                    go.Scatter(x=x_list, y=stats.beta.pdf(x, ba, bb).tolist(), mode="lines",
                               name="Variant B posterior", line=dict(color="#E45756", width=3)),
                ],
                layout=go.Layout(annotations=[dict(
                    x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                    xanchor="right",
                    text=f"A: {a_conv}/{n} conversions, B: {b_conv}/{n} conversions",
                    font=dict(size=12, color="#333"),
                )]),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="Variant A and Variant B posteriors separate as each group collects data",
        xaxis_title="conversion rate",
        yaxis_title="density",
        xaxis_range=[0, 0.3],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "visitors per variant: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: win probability and expected loss as sample size grows
# ---------------------------------------------------------------------------
def fig_decision_metrics() -> go.Figure:
    prior_a, prior_b = 1, 1
    true_a_rate, true_b_rate = 0.10, 0.125
    sample_sizes = [50, 200, 500, 1000, 2000, 5000]
    n_samples = 200_000
    x = np.linspace(0, 0.3, 400)
    x_list = x.tolist()
    frames = []
    for n in sample_sizes:
        a_conv = RNG.binomial(n, true_a_rate)
        b_conv = RNG.binomial(n, true_b_rate)
        aa, ab = prior_a + a_conv, prior_b + (n - a_conv)
        ba, bb = prior_a + b_conv, prior_b + (n - b_conv)

        a_draws = RNG.beta(aa, ab, size=n_samples)
        b_draws = RNG.beta(ba, bb, size=n_samples)
        p_b_best = float(np.mean(b_draws > a_draws))
        expected_loss_b = float(np.mean(np.maximum(a_draws - b_draws, 0)))

        frames.append(
            go.Frame(
                name=str(n),
                data=[
                    go.Scatter(x=x_list, y=stats.beta.pdf(x, aa, ab).tolist(), mode="lines",
                               name="Variant A posterior", line=dict(color="#4C78A8", width=2)),
                    go.Scatter(x=x_list, y=stats.beta.pdf(x, ba, bb).tolist(), mode="lines",
                               name="Variant B posterior", line=dict(color="#E45756", width=2)),
                ],
                layout=go.Layout(annotations=[dict(
                    x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                    xanchor="right",
                    text=(f"A: {a_conv}/{n}, B: {b_conv}/{n}<br>"
                          f"P(B beats A) = {p_b_best:.1%}<br>"
                          f"expected loss choosing B = {expected_loss_b:.4f}"),
                    font=dict(size=12, color="#333"),
                )]),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="Win probability and expected loss, computed from the same posteriors, as n grows",
        xaxis_title="conversion rate",
        yaxis_title="density",
        xaxis_range=[0, 0.3],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "visitors per variant: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4b: Normal-Normal conjugacy for a continuous metric (average order value)
# ---------------------------------------------------------------------------
def fig_normal_normal_update() -> go.Figure:
    prior_mean, prior_sd = 42.0, 5.0
    prior_var = prior_sd**2
    sample_mean, sample_sd = 44.10, 18.0
    data_var = sample_sd**2

    sample_sizes = [0, 10, 50, 150, 500]
    x_grid = np.linspace(20, 60, 400)
    x_list = x_grid.tolist()

    prior_density = stats.norm.pdf(x_grid, loc=prior_mean, scale=prior_sd).tolist()

    frames = []
    for n in sample_sizes:
        tau_prior = 1.0 / prior_var
        tau_data = n / data_var
        tau_post = tau_prior + tau_data
        post_mean = (tau_prior * prior_mean + tau_data * sample_mean) / tau_post
        post_sd = np.sqrt(1.0 / tau_post)
        posterior_density = stats.norm.pdf(x_grid, loc=post_mean, scale=post_sd)
        frames.append(
            go.Frame(
                name=str(n),
                data=[
                    go.Scatter(x=x_list, y=prior_density, mode="lines",
                               line=dict(color="#B7C7DB", width=2, dash="dot"),
                               name="prior: N(42, 5²)"),
                    go.Scatter(x=x_list, y=posterior_density.tolist(), mode="lines",
                               fill="tozeroy", line=dict(color="#4C78A8", width=2.5),
                               name="posterior"),
                ],
                layout=go.Layout(annotations=[dict(
                    x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                    xanchor="right",
                    text=(f"n = {n} converting visitors<br>"
                          f"posterior mean = ${post_mean:.2f}, posterior sd = ${post_sd:.2f}"),
                    font=dict(size=13, color="#333"),
                )]),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="A Normal prior on average order value narrows toward the observed "
              "$44.10 sample mean as n grows",
        xaxis_title="average order value ($)",
        yaxis_title="density",
        xaxis_range=[20, 60],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "converting visitors observed: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5: the peeking problem, false-positive rate as the number of looks grows
# ---------------------------------------------------------------------------
def fig_peeking_problem() -> go.Figure:
    n_simulations = 3000
    max_n = 6000
    look_counts = [1, 2, 5, 10, 20, 40]
    rate = 0.10

    control = RNG.binomial(1, rate, size=(n_simulations, max_n))
    treatment = RNG.binomial(1, rate, size=(n_simulations, max_n))  # same rate: null is true
    control_cum = np.cumsum(control, axis=1)
    treatment_cum = np.cumsum(treatment, axis=1)

    frames = []
    for looks in look_counts:
        check_points = np.linspace(max_n // looks, max_n, looks).astype(int) - 1
        rejected = np.zeros(n_simulations, dtype=bool)
        for idx in check_points:
            n = idx + 1
            p1 = control_cum[:, idx] / n
            p2 = treatment_cum[:, idx] / n
            pooled = (control_cum[:, idx] + treatment_cum[:, idx]) / (2 * n)
            se = np.sqrt(pooled * (1 - pooled) * (2 / n)) + 1e-12
            z = (p2 - p1) / se
            rejected |= np.abs(z) > 1.96
        false_positive_rate = float(np.mean(rejected))
        frames.append(
            go.Frame(
                name=str(looks),
                data=[go.Bar(x=["False-positive rate"], y=[false_positive_rate],
                              marker_color="#E45756", width=[0.5],
                              text=[f"{false_positive_rate:.1%}"], textposition="outside")],
                layout=go.Layout(shapes=[dict(
                    type="line", x0=-0.5, x1=0.5, y0=0.05, y1=0.05,
                    line=dict(color="#333", width=1.5, dash="dash"),
                )]),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="Checking a fixed-horizon significance test repeatedly inflates the "
              "false-positive rate above the nominal 5%",
        yaxis_title="false-positive rate under a true null",
        yaxis_range=[0, 0.5],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "number of times the test is checked: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=80, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 6: sample ratio mismatch, a redirect-timeout bug skewing logged traffic
# ---------------------------------------------------------------------------
def fig_srm_check() -> go.Figure:
    rng = np.random.default_rng(303)
    weekly_n = [15000, 15000, 15000, 15000, 15000]
    week_labels = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
    # Weeks 1-2: healthy 50/50 randomization, everything assigned gets logged. After Week 2 a
    # redirect-timeout bug ships on Variant B's path and starts dropping about 6% of assigned-B
    # visitors from the logging pipeline before a conversion is ever recorded for them; Variant
    # A's path is untouched, so its logged count always matches its assigned count.
    drop_rate_b = [0.0, 0.0, 0.06, 0.06, 0.06]

    cum_a, cum_b = 0, 0
    frames = []
    for label, n, drop in zip(week_labels, weekly_n, drop_rate_b):
        assigned_a = rng.binomial(n, 0.5)
        assigned_b = n - assigned_a
        logged_a = assigned_a
        logged_b = rng.binomial(assigned_b, 1 - drop)
        cum_a += logged_a
        cum_b += logged_b
        total = cum_a + cum_b
        expected = total / 2
        chi2 = ((cum_a - expected) ** 2) / expected + ((cum_b - expected) ** 2) / expected
        p_value = float(stats.chi2.sf(chi2, df=1))
        flagged = p_value < 0.001
        pct_a, pct_b = cum_a / total * 100, cum_b / total * 100
        frames.append(
            go.Frame(
                name=label,
                data=[go.Bar(x=["Variant A", "Variant B"], y=[pct_a, pct_b],
                              marker_color=["#4C78A8", "#E45756"],
                              text=[f"{pct_a:.2f}%", f"{pct_b:.2f}%"], textposition="outside")],
                layout=go.Layout(
                    shapes=[dict(type="line", x0=-0.5, x1=1.5, y0=50, y1=50,
                                 line=dict(color="#333", width=1.5, dash="dash"))],
                    annotations=[dict(
                        x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                        xanchor="right",
                        text=(f"cumulative logged: A={cum_a:,}, B={cum_b:,}<br>"
                              f"chi-square = {chi2:.2f}, p = {p_value:.2g}<br>"
                              + ("SRM FLAGGED (p < 0.001)" if flagged else "no SRM detected")),
                        font=dict(size=12, color="#B02A2A" if flagged else "#333"),
                    )],
                ),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="A redirect-timeout bug after Week 2 quietly drops logged Variant B traffic",
        yaxis_title="share of cumulative logged traffic (%)",
        yaxis_range=[45, 55],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "cumulative through: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 7: tracking many metrics at once inflates the chance one flags by luck
# ---------------------------------------------------------------------------
def fig_multiple_metrics() -> go.Figure:
    rng = np.random.default_rng(404)
    n_simulations = 30_000
    max_k = 20
    n_per_group = 2000
    true_rate = 0.10
    alpha = 0.05
    k_values = [1, 2, 3, 5, 8, 12, 16, 20]

    # Every metric shares the same true 10% rate for both variants (a true null on all of
    # them), and each metric's win probability comes from a Normal approximation to the two
    # Beta posteriors instead of a nested per-metric simulation loop: with n=2000 per arm the
    # Beta posterior sits close enough to Normal that the two-proportion z-score converts
    # directly to a win probability through the Normal CDF.
    a_conv = rng.binomial(n_per_group, true_rate, size=(n_simulations, max_k))
    b_conv = rng.binomial(n_per_group, true_rate, size=(n_simulations, max_k))
    p_a, p_b = a_conv / n_per_group, b_conv / n_per_group
    var_sum = p_a * (1 - p_a) / n_per_group + p_b * (1 - p_b) / n_per_group
    z = (p_b - p_a) / np.sqrt(var_sum + 1e-12)
    win_prob = stats.norm.cdf(z)

    frames = []
    for k in k_values:
        wp_k = win_prob[:, :k]
        frac_uncorrected = float(np.mean(np.any((wp_k > 0.975) | (wp_k < 0.025), axis=1)))

        z_crit = stats.norm.ppf(1 - (alpha / k) / 2)
        frac_corrected = float(np.mean(np.any(np.abs(z[:, :k]) > z_crit, axis=1)))

        frames.append(
            go.Frame(
                name=str(k),
                data=[go.Bar(x=["uncorrected, 95% win-prob rule", "Bonferroni-corrected"],
                              y=[frac_uncorrected, frac_corrected],
                              marker_color=["#E45756", "#4C78A8"],
                              text=[f"{frac_uncorrected:.1%}", f"{frac_corrected:.1%}"],
                              textposition="outside")],
                layout=go.Layout(shapes=[dict(
                    type="line", x0=-0.5, x1=1.5, y0=0.05, y1=0.05,
                    line=dict(color="#333", width=1.5, dash="dash"),
                )]),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="At least one of k metrics clears a 95% win-probability bar by chance alone, "
              "with no true difference behind any of them",
        yaxis_title="probability at least one metric is flagged",
        yaxis_range=[0, 0.8],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "metrics tracked (k): "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate",
                                      "frame": {"duration": 300, "redraw": True},
                                      "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=80, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 8: hierarchical A/B testing, partial pooling across traffic-source segments
# ---------------------------------------------------------------------------
def fig_hierarchical_segments() -> go.Figure:
    # Unlike the linear-Gaussian models in Chapters 9 and 10, a hierarchical logistic model has
    # no conjugate posterior, so this figure (the only one in the book) runs PyMC's sampler
    # directly instead of a closed-form approximation. Import locally so the rest of this
    # script's figures, and any environment that only needs those, never pay PyMC's import cost.
    import pymc as pm
    import arviz as az

    segments = ["organic", "paid_search", "referral", "email"]
    n_A = np.array([6000, 4000, 1200, 350])
    n_B = np.array([6000, 4000, 1200, 350])
    true_rate_A = np.array([0.095, 0.110, 0.080, 0.130])
    true_lift_pp = np.array([0.010, 0.030, 0.005, -0.010])
    true_rate_B = true_rate_A + true_lift_pp

    data_rng = np.random.default_rng(707)
    k_A = data_rng.binomial(n_A, true_rate_A)
    k_B = data_rng.binomial(n_B, true_rate_B)

    n_segments = len(segments)
    seg_idx = np.repeat(np.arange(n_segments), 2)
    arm_idx = np.tile(np.array([0, 1]), n_segments)
    n_all = np.empty(2 * n_segments, dtype=int)
    k_all = np.empty(2 * n_segments, dtype=int)
    n_all[0::2], n_all[1::2] = n_A, n_B
    k_all[0::2], k_all[1::2] = k_A, k_B

    def fit(pooling):
        coords = {"segment": segments, "obs_id": np.arange(2 * n_segments)}
        with pm.Model(coords=coords):
            alpha = pm.Normal("alpha", mu=0, sigma=1.5, dims="segment")
            if pooling == "no_pooling":
                delta = pm.Normal("delta", mu=0, sigma=1.5, dims="segment")
            elif pooling == "complete_pooling":
                delta_shared = pm.Normal("delta_shared", mu=0, sigma=1.5)
                delta = pm.Deterministic(
                    "delta", delta_shared * pm.math.ones(n_segments), dims="segment"
                )
            else:
                # Non-centered parameterization, the same funnel-avoiding trick Chapter 9 uses
                # for the horseshoe prior, applied here to the segment-level treatment effect.
                mu_delta = pm.Normal("mu_delta", mu=0, sigma=1)
                sigma_delta = pm.HalfNormal("sigma_delta", sigma=1)
                delta_offset = pm.Normal("delta_offset", mu=0, sigma=1, dims="segment")
                delta = pm.Deterministic(
                    "delta", mu_delta + delta_offset * sigma_delta, dims="segment"
                )
            p_all = pm.math.invlogit(alpha[seg_idx] + delta[seg_idx] * arm_idx)
            pm.Binomial("obs", n=n_all, p=p_all, observed=k_all, dims="obs_id")
            idata = pm.sample(1000, tune=1000, chains=2, cores=1, target_accept=0.9,
                               random_seed=101, progressbar=False)
            pm.compute_log_likelihood(idata)
        return idata

    idata_no_pooling = fit("no_pooling")
    idata_complete_pooling = fit("complete_pooling")
    idata_hierarchical = fit("hierarchical")

    comparison = az.compare({
        "no_pooling": idata_no_pooling,
        "complete_pooling": idata_complete_pooling,
        "hierarchical": idata_hierarchical,
    })

    # arviz's compare() column names have moved between versions (elpd_loo/d_elpd/dSE in the
    # convention Chapter 10 uses, elpd/elpd_diff/dse in the version this book's dependencies
    # pin); read them from whichever set is present so this figure keeps working across an
    # arviz upgrade instead of raising a KeyError.
    elpd_col = "elpd_loo" if "elpd_loo" in comparison.columns else "elpd"
    se_col = "se" if "se" in comparison.columns else "SE"

    order = comparison.index.tolist()
    elpds = comparison.loc[order, elpd_col].astype(float).tolist()
    ses = comparison.loc[order, se_col].astype(float).tolist()

    def alpha_delta_means(idata):
        return (
            idata.posterior["alpha"].mean(dim=["chain", "draw"]).values,
            idata.posterior["delta"].mean(dim=["chain", "draw"]).values,
        )

    alpha_no, delta_no = alpha_delta_means(idata_no_pooling)
    alpha_hier, delta_hier = alpha_delta_means(idata_hierarchical)

    def invlogit(x):
        return 1.0 / (1.0 + np.exp(-x))

    lift_no = (invlogit(alpha_no + delta_no) - invlogit(alpha_no)) * 100
    lift_hier = (invlogit(alpha_hier + delta_hier) - invlogit(alpha_hier)) * 100
    lift_true = true_lift_pp * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Model comparison", "Segment-level lift"),
        column_widths=[0.42, 0.58],
    )
    fig.add_trace(
        go.Scatter(
            x=elpds, y=order, mode="markers",
            error_x=dict(type="data", array=ses, color="#4C78A8", thickness=2, width=6),
            marker=dict(size=12, color="#4C78A8"), showlegend=False,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="elpd_loo (higher is better)", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=lift_true, y=segments, mode="markers", name="true lift",
                    marker=dict(symbol="diamond", size=13, color="#333")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=lift_no, y=segments, mode="markers", name="no pooling",
                    marker=dict(symbol="circle", size=12, color="#E45756")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=lift_hier, y=segments, mode="markers", name="hierarchical (partial pooling)",
                    marker=dict(symbol="square", size=11, color="#4C78A8")),
        row=1, col=2,
    )
    fig.add_vline(x=0, line=dict(color="#999", width=1, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="posterior mean lift (percentage points)", row=1, col=2)
    # No overall figure title here: the surrounding chapter prose and this figure's own
    # caption carry that framing, and stacking a suptitle on top of both subplot titles and
    # the legend left too little vertical room for all three to stay legible.
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.16, xanchor="center", x=0.79),
        margin=dict(t=90, l=110, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 9: latent subgroup heterogeneity, a two-component mixture on checkout-
# time change reveals a helped cluster and a harmed cluster that a single
# aggregate effect hides.
# ---------------------------------------------------------------------------
def fig_latent_mixture() -> go.Figure:
    # Same reasoning as fig_hierarchical_segments: import PyMC locally so every other
    # figure in this script, and any environment that only needs those, never pays its
    # import cost.
    import pymc as pm

    rng = np.random.default_rng(505)
    n_users = 2400
    # Two latent groups the experiment platform never logs: visitors who have seen the
    # redesigned checkout before (it saves them time) and first-time visitors who are
    # still expecting the old flow (it costs them time). Only the pooled outcome is
    # observed; group membership is not.
    true_weight_returning = 0.62
    true_mean_returning, true_sd_returning = 9.0, 3.0
    true_mean_first_time, true_sd_first_time = -6.0, 4.5

    is_returning = rng.random(n_users) < true_weight_returning
    checkout_time_delta = np.where(
        is_returning,
        rng.normal(true_mean_returning, true_sd_returning, n_users),
        rng.normal(true_mean_first_time, true_sd_first_time, n_users),
    )

    naive_mean = float(checkout_time_delta.mean())
    naive_sd = float(checkout_time_delta.std())

    with pm.Model():
        weights = pm.Dirichlet("weights", a=np.ones(2))
        # The `ordered` transform pins component 0 below component 1 at every draw,
        # which is what keeps the sampler from flipping which component it calls "0"
        # partway through a chain (the label-switching problem the chapter text walks
        # through). Without it, two chains, or even two stretches of one chain, could
        # each be a valid fit while disagreeing on which cluster is which.
        means = pm.Normal(
            "means", mu=0, sigma=10, shape=2,
            transform=pm.distributions.transforms.ordered,
            initval=np.array([-6.0, 6.0]),
        )
        sigmas = pm.HalfNormal("sigmas", sigma=6, shape=2)
        pm.NormalMixture("checkout_time_change", w=weights, mu=means, sigma=sigmas,
                          observed=checkout_time_delta)
        idata = pm.sample(1500, tune=1500, chains=2, cores=1, target_accept=0.9,
                           random_seed=101, progressbar=False)

    w = idata.posterior["weights"].mean(dim=["chain", "draw"]).values
    mu = idata.posterior["means"].mean(dim=["chain", "draw"]).values
    sd = idata.posterior["sigmas"].mean(dim=["chain", "draw"]).values

    x = np.linspace(-25, 25, 400)
    x_list = x.tolist()
    counts, edges = np.histogram(checkout_time_delta, bins=40, density=True)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    widths = (edges[1:] - edges[:-1]).tolist()

    naive_fit = stats.norm.pdf(x, naive_mean, naive_sd).tolist()
    comp_harmed = (w[0] * stats.norm.pdf(x, mu[0], sd[0])).tolist()
    comp_helped = (w[1] * stats.norm.pdf(x, mu[1], sd[1])).tolist()
    mixture_fit = (np.array(comp_harmed) + np.array(comp_helped)).tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Aggregate view: one effect, one number",
                         "Two-component mixture: what the aggregate hides"),
    )
    fig.add_trace(go.Bar(x=centers, y=counts.tolist(), width=widths,
                          marker_color="#B7C7DB", opacity=0.7, showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x_list, y=naive_fit, mode="lines",
                              line=dict(color="#333", width=2.5, dash="dash"),
                              name="single-Normal fit"), row=1, col=1)
    fig.add_vline(x=naive_mean, line=dict(color="#4C78A8", width=2), row=1, col=1)

    fig.add_trace(go.Bar(x=centers, y=counts.tolist(), width=widths,
                          marker_color="#B7C7DB", opacity=0.7, showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x_list, y=comp_harmed, mode="lines", fill="tozeroy",
                              line=dict(color="#E45756", width=2),
                              name=f"harmed cluster (weight {w[0]:.0%})"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_list, y=comp_helped, mode="lines", fill="tozeroy",
                              line=dict(color="#4C78A8", width=2),
                              name=f"helped cluster (weight {w[1]:.0%})"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_list, y=mixture_fit, mode="lines",
                              line=dict(color="#333", width=2.5),
                              name="mixture fit"), row=1, col=2)

    fig.update_xaxes(title_text="change in checkout time (seconds, + saved / - added)")
    fig.update_yaxes(title_text="density", row=1, col=1)
    fig.update_layout(
        title="One aggregate effect hides two different visitor experiences",
        legend=dict(orientation="h", yanchor="bottom", y=1.16, xanchor="center", x=0.5),
        margin=dict(t=110, l=60, r=30, b=60),
    )
    return fig


FIGURES = {
    "chapter-05-fig-beta-prior-shapes": fig_prior_shapes,
    "chapter-05-fig-posterior-update": fig_posterior_update,
    "chapter-05-fig-posteriors-ab-overlay": fig_ab_overlay,
    "chapter-05-fig-decision-metrics": fig_decision_metrics,
    "chapter-05-fig-peeking-problem": fig_peeking_problem,
    "chapter-05-fig-normal-normal-update": fig_normal_normal_update,
    "chapter-bayes-ab-testing-fig-srm-check": fig_srm_check,
    "chapter-bayes-ab-testing-fig-multiple-metrics": fig_multiple_metrics,
    "chapter-bayes-ab-testing-fig-hierarchical-segments": fig_hierarchical_segments,
    "chapter-bayes-ab-testing-fig-latent-mixture": fig_latent_mixture,
}


if __name__ == "__main__":
    for name, builder in FIGURES.items():
        out_path = save(builder(), name)
        print(f"wrote {out_path}")
