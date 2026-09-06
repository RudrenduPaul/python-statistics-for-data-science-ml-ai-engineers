"""
Interactive figures for "Bayesian Nonlinear Regression: Gaussian Processes" (Part 3).

Reuses the concurrent-load / checkout-API-latency saturation scenario from Chapter 6
(chapter-06-regression-splines-plots.py) so the two chapters share one dataset. Fits use
scikit-learn's GaussianProcessRegressor, the standard entry point for GP regression in Python.

Run directly to regenerate every figure:
    python chapters/chapter-bayes-gaussian-processes-plots.py
"""

import os
import re

import numpy as np
import plotly.graph_objects as go
import pymc as pm
from plotly.subplots import make_subplots
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(7)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_generated")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig: go.Figure, name: str) -> str:
    path = os.path.join(OUT_DIR, f"{name}.html")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    if fig.frames:
        # The CDN-pinned Plotly.js build mis-syncs a slider's visible frame with its
        # declared "active" step on first load (most visible with three-step sliders,
        # where it opens on the middle step instead of the first). Plotly.py always
        # appends a `Plotly.animate(divid, null)` call after `newPlot` to sync the
        # slider to the active step; pointing that call at the intended frame by name,
        # instead of leaving it null, makes the opening frame match the slider's
        # declared "active" step (and the chapter prose describing it) every load.
        sliders = fig.layout.sliders
        active = sliders[0].active if sliders and sliders[0].active is not None else 0
        init_frame = fig.frames[active].name
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        fixed = re.sub(
            r"Plotly\.animate\('([0-9a-f-]+)', null\);",
            lambda m: (
                f"Plotly.animate('{m.group(1)}', ['{init_frame}'], "
                "{frame: {duration: 0, redraw: true}, transition: {duration: 0}, "
                "mode: 'immediate'});"
            ),
            html,
        )
        if fixed != html:
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed)
    return path


def saturation_curve(rho: np.ndarray, base_ms: float = 20.0) -> np.ndarray:
    """Mean latency under an M/M/1-style saturation curve as utilization rho -> 1.
    Matches chapter-06-regression-splines-plots.py's data-generating process."""
    return base_ms / (1.0 - rho)


def simulated_load_latency(n: int = 30, noise_scale: float = 3.0, rho_max: float = 0.90):
    rho = np.sort(RNG.uniform(0.05, rho_max, size=n))
    mean_latency = saturation_curve(rho)
    latency = mean_latency + RNG.normal(0, noise_scale, size=n)
    return rho, latency


# ---------------------------------------------------------------------------
# Figure 1: sample functions from a GP prior at different length-scales
# ---------------------------------------------------------------------------
def fig_prior_samples() -> go.Figure:
    x_grid = np.linspace(0, 1, 200).reshape(-1, 1)
    length_scales = [0.03, 0.1, 0.3, 1.0]
    frames = []
    for ls in length_scales:
        gp = GaussianProcessRegressor(kernel=RBF(length_scale=ls), random_state=0)
        samples = gp.sample_y(x_grid, n_samples=5, random_state=1)
        traces = [
            go.Scatter(x=x_grid.ravel(), y=samples[:, i], mode="lines",
                       line=dict(width=1.6), showlegend=False)
            for i in range(samples.shape[1])
        ]
        frames.append(go.Frame(name=f"{ls:.2f}", data=traces))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Five functions sampled from an RBF-kernel GP prior, before any data",
        xaxis_title="x (scaled utilization)",
        yaxis_title="sampled function value",
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "length-scale: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate", "frame": {"duration": 300}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2: RBF vs Matern kernel sample comparison
# ---------------------------------------------------------------------------
def fig_kernel_comparison() -> go.Figure:
    x_grid = np.linspace(0, 1, 200).reshape(-1, 1)
    kernels = {
        "RBF (infinitely smooth)": RBF(length_scale=0.15),
        "Matern, nu=1.5 (rougher)": Matern(length_scale=0.15, nu=1.5),
        "Matern, nu=2.5 (in between)": Matern(length_scale=0.15, nu=2.5),
    }
    frames = []
    for name, kernel in kernels.items():
        gp = GaussianProcessRegressor(kernel=kernel, random_state=0)
        samples = gp.sample_y(x_grid, n_samples=4, random_state=2)
        traces = [
            go.Scatter(x=x_grid.ravel(), y=samples[:, i], mode="lines",
                       line=dict(width=1.6), showlegend=False)
            for i in range(samples.shape[1])
        ]
        frames.append(go.Frame(name=name, data=traces))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Same length-scale, different kernel: smoothness assumption changes the samples",
        xaxis_title="x (scaled utilization)",
        yaxis_title="sampled function value",
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "kernel: "},
            "steps": [
                # Short display labels on purpose: the full kernel names ("RBF
                # (infinitely smooth)", "Matern, nu=2.5 (in between)") overhang past the
                # plot margins at the slider's end positions and get clipped by the plot
                # boundary. Frame names stay unchanged below since the animate() call
                # targets them by name; only the tick label and currentvalue text shorten.
                {"label": label, "method": "animate",
                 "args": [[f.name], {"mode": "immediate", "frame": {"duration": 300}}]}
                for f, label in zip(frames, ["RBF", "Matern 1.5", "Matern 2.5"])
            ],
        }],
        margin=dict(t=60, l=60, r=50, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3: posterior mean + credible band as the amount of observed data grows
# ---------------------------------------------------------------------------
def fig_posterior_fit() -> go.Figure:
    rho_full, latency_full = simulated_load_latency(n=60, noise_scale=3.0)
    grid = np.linspace(0.03, 0.92, 250).reshape(-1, 1)
    true_curve = saturation_curve(grid.ravel())
    sample_sizes = [5, 10, 20, 40, 60]
    # rho_full is sorted ascending, so a plain prefix rho_full[:n] would cluster an
    # early, small n entirely among the smallest utilization values (the 5 lowest of
    # 60 uniform draws all land under rho=0.09), leaving the model with no way to show
    # the "closely tracked below ~80% utilization, wide band above it" contrast the
    # chapter text describes. Sampling each n evenly across a growing prefix POOL
    # (85-100% of the sorted range, widening toward the full range as n grows) keeps
    # every subset spread across the covered part of the domain, so the fit stays
    # accurate up to whatever utilization it has seen and only widens beyond that.
    pool_fracs = [0.88, 0.90, 0.93, 0.98, 1.00]
    frames = []
    for n, pool_frac in zip(sample_sizes, pool_fracs):
        pool_hi = max(n - 1, int(round(pool_frac * (len(rho_full) - 1))))
        idx = np.linspace(0, pool_hi, n).astype(int)
        rho_n, lat_n = rho_full[idx], latency_full[idx]
        kernel = RBF(length_scale=0.2, length_scale_bounds=(0.02, 2.0)) + WhiteKernel(1.0)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
        gp.fit(rho_n.reshape(-1, 1), lat_n)
        mean, std = gp.predict(grid, return_std=True)
        frames.append(
            go.Frame(
                name=str(n),
                data=[
                    go.Scatter(x=grid.ravel(), y=true_curve, mode="lines", name="true curve",
                                line=dict(color="#999", dash="dot")),
                    go.Scatter(x=np.concatenate([grid.ravel(), grid.ravel()[::-1]]),
                               y=np.concatenate([mean + 1.96 * std, (mean - 1.96 * std)[::-1]]),
                               fill="toself", fillcolor="rgba(76,120,168,0.25)",
                               line=dict(color="rgba(0,0,0,0)"), name="95% credible band",
                               showlegend=True),
                    go.Scatter(x=grid.ravel(), y=mean, mode="lines", name="posterior mean",
                               line=dict(color="#4C78A8", width=2.5)),
                    go.Scatter(x=rho_n, y=lat_n, mode="markers", name="observed",
                               marker=dict(color="#333", size=6)),
                ],
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="GP posterior mean and 95% credible band as more requests are observed",
        xaxis_title="Utilization (rho)",
        yaxis_title="Latency (ms)",
        yaxis_range=[0, 300],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "observations so far: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate", "frame": {"duration": 300}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: extrapolation uncertainty widening beyond the observed data range
# ---------------------------------------------------------------------------
def fig_extrapolation() -> go.Figure:
    rho, latency = simulated_load_latency(n=40, noise_scale=3.0, rho_max=0.75)
    grid = np.linspace(0.02, 0.97, 300).reshape(-1, 1)
    kernel = RBF(length_scale=0.2, length_scale_bounds=(0.02, 2.0)) + WhiteKernel(1.0)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(rho.reshape(-1, 1), latency)
    mean, std = gp.predict(grid, return_std=True)

    fig = go.Figure()
    fig.add_vrect(x0=rho.max(), x1=0.97, fillcolor="#F58518", opacity=0.12, line_width=0,
                   annotation_text="beyond observed load range", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid.ravel(), grid.ravel()[::-1]]),
        y=np.concatenate([mean + 1.96 * std, (mean - 1.96 * std)[::-1]]),
        fill="toself", fillcolor="rgba(76,120,168,0.25)",
        line=dict(color="rgba(0,0,0,0)"), name="95% credible band"))
    fig.add_trace(go.Scatter(x=grid.ravel(), y=mean, mode="lines", name="posterior mean",
                              line=dict(color="#4C78A8", width=2.5)))
    fig.add_trace(go.Scatter(x=rho, y=latency, mode="markers", name="observed",
                              marker=dict(color="#333", size=6)))
    fig.update_layout(
        title="Credible band widens sharply past the highest observed utilization",
        xaxis_title="Utilization (rho)",
        yaxis_title="Latency (ms)",
        yaxis_range=[-50, 500],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5: length-scale under/over-smoothing on the fitted posterior
# ---------------------------------------------------------------------------
def fig_length_scale_fit() -> go.Figure:
    rho, latency = simulated_load_latency(n=45, noise_scale=3.0)
    grid = np.linspace(0.03, 0.92, 250).reshape(-1, 1)
    true_curve = saturation_curve(grid.ravel())
    length_scales = [0.02, 0.08, 0.2, 0.6]
    frames = []
    for ls in length_scales:
        kernel = RBF(length_scale=ls, length_scale_bounds="fixed") + WhiteKernel(1.0)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
        gp.fit(rho.reshape(-1, 1), latency)
        mean, std = gp.predict(grid, return_std=True)
        frames.append(
            go.Frame(
                name=f"{ls:.2f}",
                data=[
                    go.Scatter(x=grid.ravel(), y=true_curve, mode="lines", name="true curve",
                                line=dict(color="#999", dash="dot")),
                    go.Scatter(x=np.concatenate([grid.ravel(), grid.ravel()[::-1]]),
                               y=np.concatenate([mean + 1.96 * std, (mean - 1.96 * std)[::-1]]),
                               fill="toself", fillcolor="rgba(229,134,25,0.20)",
                               line=dict(color="rgba(0,0,0,0)"), name="95% credible band"),
                    go.Scatter(x=grid.ravel(), y=mean, mode="lines", name="posterior mean",
                               line=dict(color="#F58518", width=2.5)),
                    go.Scatter(x=rho, y=latency, mode="markers", name="observed",
                               marker=dict(color="#333", size=6)),
                ],
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="A fixed length-scale can under-fit (too long) or chase noise (too short)",
        xaxis_title="Utilization (rho)",
        yaxis_title="Latency (ms)",
        yaxis_range=[0, 300],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "length-scale (held fixed, not optimized): "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate", "frame": {"duration": 300}}]}
                for f in frames
            ],
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 6: per-dimension ARD length-scales as a variable-importance signal
# ---------------------------------------------------------------------------
def fig_ard_length_scales() -> go.Figure:
    """"More than one input at a time" explains that a fitted length-scale per
    dimension is itself a variable-importance signal, but the section had no chart to
    show it. This fits a two-predictor GP (payload size and concurrent-request
    utilization predicting latency, the pairing @fig-posterior-fit and this chapter's
    Chapter-4 callback both use) with an automatic-relevance-determination kernel and
    plots the two fitted length-scales side by side, on inputs standardized to the same
    scale so the two numbers are directly comparable."""
    n = 80
    payload_kb = RNG.uniform(2, 20, n)
    concurrent_util = RNG.uniform(0.05, 0.85, n)
    latency = saturation_curve(concurrent_util) + 0.3 * payload_kb + RNG.normal(0, 3, n)

    X = np.column_stack([payload_kb, concurrent_util])
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    kernel = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.05, 100.0)) + WhiteKernel(1.0)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=8,
                                   random_state=0)
    gp.fit(X_norm, latency)
    fitted_length_scales = gp.kernel_.k1.length_scale

    labels = ["payload size", "concurrent-request<br>utilization"]
    fig = go.Figure(go.Bar(
        x=labels, y=fitted_length_scales, marker_color=["#4C78A8", "#F58518"],
        text=[f"{v:.2f}" for v in fitted_length_scales], textposition="outside",
    ))
    fig.update_layout(
        title="Fitted ARD length-scale per predictor (standardized inputs)",
        yaxis_title="Fitted length-scale",
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 7: RBF-only vs. RBF+periodic composite kernel forecasting an
# unobserved daily cycle
# ---------------------------------------------------------------------------
def hourly_latency_with_daily_cycle(n_hours: int = 96, noise_scale: float = 2.5):
    """Four days of hourly latency: a slow upward drift (growing baseline load)
    plus a daily business-hours peak around 2pm, matching the kind of metric
    Chapter 7/8's rollback classifier treats "hour of day" as a feature for."""
    hours = np.arange(n_hours)
    hour_of_day = hours % 24
    daily_peak = 18.0 * np.exp(-0.5 * ((hour_of_day - 14.0) / 3.5) ** 2)
    drift = 25.0 + 0.25 * hours
    mean_latency = drift + daily_peak
    latency = mean_latency + RNG.normal(0, noise_scale, size=n_hours)
    return hours, latency, mean_latency


def fig_periodic_kernel() -> go.Figure:
    hours, latency, true_mean = hourly_latency_with_daily_cycle()
    n_obs = 72  # first three days observed; the fourth day is held out
    X_obs = hours[:n_obs].reshape(-1, 1).astype(float)
    y_obs = latency[:n_obs]
    X_grid = hours.reshape(-1, 1).astype(float)

    kernel_trend = RBF(length_scale=24.0, length_scale_bounds=(5.0, 100.0)) + \
        WhiteKernel(1.0, noise_level_bounds=(1e-3, 50.0))
    gp_trend = GaussianProcessRegressor(kernel=kernel_trend, normalize_y=True,
                                         n_restarts_optimizer=6, random_state=0)
    gp_trend.fit(X_obs, y_obs)
    mean_trend, std_trend = gp_trend.predict(X_grid, return_std=True)

    kernel_composite = (
        RBF(length_scale=48.0, length_scale_bounds=(10.0, 200.0))
        + ExpSineSquared(length_scale=1.0, length_scale_bounds=(0.5, 10.0),
                          periodicity=24.0, periodicity_bounds="fixed")
        + WhiteKernel(1.0, noise_level_bounds=(1e-3, 50.0))
    )
    gp_composite = GaussianProcessRegressor(kernel=kernel_composite, normalize_y=True,
                                             n_restarts_optimizer=6, random_state=0)
    gp_composite.fit(X_obs, y_obs)
    mean_composite, std_composite = gp_composite.predict(X_grid, return_std=True)

    fig = go.Figure()
    fig.add_vrect(x0=n_obs, x1=hours[-1], fillcolor="#F58518", opacity=0.10, line_width=0,
                   annotation_text="unobserved fourth day", annotation_position="top left")
    fig.add_trace(go.Scatter(x=hours, y=true_mean, mode="lines", name="true mean latency",
                              line=dict(color="#999", dash="dot")))
    fig.add_trace(go.Scatter(
        x=np.concatenate([hours, hours[::-1]]),
        y=np.concatenate([mean_trend + 1.96 * std_trend, (mean_trend - 1.96 * std_trend)[::-1]]),
        fill="toself", fillcolor="rgba(245,133,24,0.18)", line=dict(color="rgba(0,0,0,0)"),
        name="RBF-only 95% band"))
    fig.add_trace(go.Scatter(x=hours, y=mean_trend, mode="lines", name="RBF-only mean",
                              line=dict(color="#F58518", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=np.concatenate([hours, hours[::-1]]),
        y=np.concatenate([mean_composite + 1.96 * std_composite,
                           (mean_composite - 1.96 * std_composite)[::-1]]),
        fill="toself", fillcolor="rgba(76,120,168,0.22)", line=dict(color="rgba(0,0,0,0)"),
        name="RBF+periodic 95% band"))
    fig.add_trace(go.Scatter(x=hours, y=mean_composite, mode="lines",
                              name="RBF+periodic mean", line=dict(color="#4C78A8", width=2.5)))
    fig.add_trace(go.Scatter(x=hours[:n_obs], y=y_obs, mode="markers", name="observed",
                              marker=dict(color="#333", size=5)))
    fig.update_layout(
        title="A periodic kernel forecasts the next daily peak; RBF alone does not",
        xaxis_title="Hour (three observed days, one held out)",
        yaxis_title="Latency (ms)",
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 8: point-estimate vs. full-Bayesian (MCMC) kernel hyperparameters
# ---------------------------------------------------------------------------
def gp_hyperparameter_posterior(n: int = 8):
    """Eight concurrent-load observations, small enough that a marginal-likelihood
    point estimate of the length-scale and a full posterior over it can meaningfully
    disagree, the situation the marginal-likelihood section names as the case where
    the difference matters most. A dedicated RNG keeps this dataset independent of
    every other figure's draws."""
    rng = np.random.default_rng(107)
    rho = np.sort(rng.uniform(0.05, 0.85, size=n))
    latency = saturation_curve(rho) + rng.normal(0, 3.0, size=n)
    return rho, latency


def fig_full_bayes_hyperparams() -> go.Figure:
    rho, latency = gp_hyperparameter_posterior()
    grid = np.linspace(0.03, 0.92, 250)

    point_kernel = RBF(length_scale=0.2, length_scale_bounds=(0.02, 2.0)) + WhiteKernel(1.0)
    gp_point = GaussianProcessRegressor(kernel=point_kernel, normalize_y=True,
                                         n_restarts_optimizer=8, random_state=0)
    gp_point.fit(rho.reshape(-1, 1), latency)
    mean_point, std_point = gp_point.predict(grid.reshape(-1, 1), return_std=True)
    ell_point = float(gp_point.kernel_.k1.length_scale)

    coords = {"obs_id": np.arange(len(rho))}
    with pm.Model(coords=coords) as full_bayes_model:
        rho_data = pm.Data("rho_data", rho, dims="obs_id")
        ell = pm.Gamma("ell", alpha=2.0, beta=8.0)
        eta = pm.HalfNormal("eta", sigma=50.0)
        sigma_n = pm.HalfNormal("sigma_n", sigma=10.0)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ls=ell)
        gp_full = pm.gp.Marginal(cov_func=cov)
        gp_full.marginal_likelihood("latency_obs", X=rho_data[:, None], y=latency, sigma=sigma_n)
        trace = pm.sample(1000, tune=1000, chains=4, random_seed=11, target_accept=0.95,
                           progressbar=False)
        f_pred = gp_full.conditional("f_pred", Xnew=grid[:, None])
        pred = pm.sample_posterior_predictive(trace, var_names=["f_pred"], random_seed=11,
                                               progressbar=False)

    f_samples = pred.posterior_predictive["f_pred"].values.reshape(-1, len(grid))
    mean_full = f_samples.mean(axis=0)
    std_full = f_samples.std(axis=0)
    ell_samples = trace.posterior["ell"].values.ravel()

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.12,
        subplot_titles=("Predictive fit comparison", "Length-scale posterior"),
    )
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid, grid[::-1]]),
        y=np.concatenate([mean_point + 1.96 * std_point, (mean_point - 1.96 * std_point)[::-1]]),
        fill="toself", fillcolor="rgba(245,133,24,0.18)", line=dict(color="rgba(0,0,0,0)"),
        name="point-estimate 95% band"), row=1, col=1)
    fig.add_trace(go.Scatter(x=grid, y=mean_point, mode="lines", name="point-estimate mean",
                              line=dict(color="#F58518", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid, grid[::-1]]),
        y=np.concatenate([mean_full + 1.96 * std_full, (mean_full - 1.96 * std_full)[::-1]]),
        fill="toself", fillcolor="rgba(76,120,168,0.22)", line=dict(color="rgba(0,0,0,0)"),
        name="full-Bayesian 95% band"), row=1, col=1)
    fig.add_trace(go.Scatter(x=grid, y=mean_full, mode="lines", name="full-Bayesian mean",
                              line=dict(color="#4C78A8", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rho, y=latency, mode="markers", name="observed",
                              marker=dict(color="#333", size=7)), row=1, col=1)
    fig.add_trace(go.Histogram(x=ell_samples, nbinsx=40, marker_color="#4C78A8",
                                name="length-scale posterior", showlegend=False), row=1, col=2)
    fig.add_vline(x=ell_point, line=dict(color="#F58518", width=2, dash="dash"), row=1, col=2)
    fig.update_xaxes(title_text="Utilization (rho)", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", range=[0, 250], row=1, col=1)
    fig.update_xaxes(title_text="length-scale", row=1, col=2)
    fig.update_yaxes(title_text="posterior draws", row=1, col=2)
    fig.update_layout(
        title="Eight observations: a point-estimate length-scale vs. a full posterior over it",
        margin=dict(t=110, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 9: Gaussian process classification on a non-monotonic anomaly-flag rate
# ---------------------------------------------------------------------------
def anomalous_payload_labels(n: int = 90):
    """Request payload sizes with a U-shaped anomaly-flag probability: both small
    (malformed or probe-like) and large (oversized or attack-like) payloads get
    flagged, while a broad middle range does not. A single logistic-regression
    coefficient can only make the flag rate rise everywhere or fall everywhere as
    payload grows, so it cannot represent two separate flagged regions on opposite
    sides of a safe middle range. A dedicated RNG keeps this dataset independent of
    every other figure's draws."""
    rng = np.random.default_rng(207)
    payload_kb = np.sort(rng.uniform(2.0, 20.0, size=n))
    true_logit = (
        -3.5
        + 5.0 * np.exp(-0.5 * ((payload_kb - 3.0) / 2.5) ** 2)
        + 5.0 * np.exp(-0.5 * ((payload_kb - 19.0) / 2.5) ** 2)
    )
    flagged = rng.binomial(1, expit(true_logit))
    return payload_kb, flagged


def fig_gp_classification_boundary() -> go.Figure:
    payload_kb, flagged = anomalous_payload_labels()
    mid, half_range = 11.0, 9.0
    payload_scaled = (payload_kb - mid) / half_range
    grid_kb = np.linspace(2.0, 20.0, 120)
    grid_scaled = (grid_kb - mid) / half_range

    coords = {"obs_id": np.arange(len(payload_kb))}
    with pm.Model(coords=coords) as gp_classifier:
        payload_data = pm.Data("payload_data", payload_scaled, dims="obs_id")
        ell = pm.Gamma("ell", alpha=2.0, beta=3.0)
        cov = pm.gp.cov.ExpQuad(1, ls=ell)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=payload_data[:, None], dims="obs_id")
        p = pm.Deterministic("p", pm.math.invlogit(f), dims="obs_id")
        pm.Bernoulli("flagged_obs", p=p, observed=flagged, dims="obs_id")
        trace = pm.sample(1000, tune=1000, chains=4, random_seed=11, target_accept=0.9,
                           progressbar=False)
        f_pred = gp.conditional("f_pred", Xnew=grid_scaled[:, None])
        pred = pm.sample_posterior_predictive(trace, var_names=["f_pred"], random_seed=11,
                                               progressbar=False)

    f_samples = pred.posterior_predictive["f_pred"].values.reshape(-1, len(grid_kb))
    p_samples = expit(f_samples)
    p_mean = p_samples.mean(axis=0)
    p_lo = np.percentile(p_samples, 2.5, axis=0)
    p_hi = np.percentile(p_samples, 97.5, axis=0)

    true_logit_grid = (
        -3.5
        + 5.0 * np.exp(-0.5 * ((grid_kb - 3.0) / 2.5) ** 2)
        + 5.0 * np.exp(-0.5 * ((grid_kb - 19.0) / 2.5) ** 2)
    )
    true_prob_grid = expit(true_logit_grid)

    logreg = LogisticRegression()
    logreg.fit(payload_kb.reshape(-1, 1), flagged)
    p_logreg = logreg.predict_proba(grid_kb.reshape(-1, 1))[:, 1]

    jitter_rng = np.random.default_rng(208)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid_kb, y=true_prob_grid, mode="lines", name="true probability",
                              line=dict(color="#999", dash="dot")))
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid_kb, grid_kb[::-1]]),
        y=np.concatenate([p_hi, p_lo[::-1]]),
        fill="toself", fillcolor="rgba(76,120,168,0.22)", line=dict(color="rgba(0,0,0,0)"),
        name="95% credible band"))
    fig.add_trace(go.Scatter(x=grid_kb, y=p_mean, mode="lines", name="GP posterior mean",
                              line=dict(color="#4C78A8", width=2.5)))
    fig.add_trace(go.Scatter(x=grid_kb, y=p_logreg, mode="lines", name="logistic regression",
                              line=dict(color="#F58518", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=payload_kb, y=flagged + jitter_rng.normal(0, 0.015, len(flagged)), mode="markers",
        name="observed (jittered)", marker=dict(color="#333", size=5, opacity=0.6)))
    fig.update_layout(
        title="A GP classifier bends twice to follow a U-shaped anomaly-flag rate",
        xaxis_title="Request payload size (KB)",
        yaxis_title="P(flagged as anomalous)",
        yaxis_range=[-0.05, 1.05],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


FIGURES = {
    "chapter-bayes-gp-fig-prior-samples": fig_prior_samples,
    "chapter-bayes-gp-fig-kernel-comparison": fig_kernel_comparison,
    "chapter-bayes-gp-fig-posterior-fit": fig_posterior_fit,
    "chapter-bayes-gp-fig-extrapolation": fig_extrapolation,
    "chapter-bayes-gp-fig-length-scale-fit": fig_length_scale_fit,
    "chapter-bayes-gp-fig-ard-length-scales": fig_ard_length_scales,
    "chapter-bayes-gp-fig-periodic-kernel": fig_periodic_kernel,
    "chapter-bayes-gp-fig-full-bayes-hyperparams": fig_full_bayes_hyperparams,
    "chapter-bayes-gp-fig-classification-boundary": fig_gp_classification_boundary,
}


if __name__ == "__main__":
    for name, builder in FIGURES.items():
        out_path = save(builder(), name)
        print(f"wrote {out_path}")
