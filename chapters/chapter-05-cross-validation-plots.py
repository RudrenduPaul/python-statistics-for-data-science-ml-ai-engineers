"""
Interactive figures for "Cross-Validation and Model Selection".

Every function builds one self-contained, standalone Plotly HTML page and writes it
to ../_generated/. The matching chapter-cv-cross-validation.md file embeds each page
in an <iframe>, so the chapter never depends on a live Python kernel to render.

Run directly to regenerate every figure:
    python chapter-cv-cross-validation-plots.py
"""

import os

import numpy as np
import plotly.graph_objects as go
from scipy import stats

RNG = np.random.default_rng(42)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_generated")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig: go.Figure, name: str) -> str:
    path = os.path.join(OUT_DIR, f"{name}.html")
    # write_html()'s auto_play defaults to True: for a figure built from animation
    # frames, Plotly auto-plays through every frame once the page loads, so a reader
    # lands on whichever frame the playback reaches rather than the slider's configured
    # "active" step, and the slider handle and label can be caught mid-playback. Setting
    # auto_play=False leaves the figure showing the active step, matching the slider.
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, auto_play=False)
    return path


def true_function(x):
    return 40 + 0.9 * x + 0.004 * x**2


def make_dataset(n=60, noise=25, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = true_function(x) + rng.normal(0, noise, n)
    return x, y


def fit_poly_train_test_error(x_train, y_train, x_test, y_test, degree):
    coeffs = np.polyfit(x_train, y_train, degree)
    pred_train = np.polyval(coeffs, x_train)
    pred_test = np.polyval(coeffs, x_test)
    train_mse = np.mean((pred_train - y_train) ** 2)
    test_mse = np.mean((pred_test - y_test) ** 2)
    return train_mse, test_mse


# ---------------------------------------------------------------------------
# Figure 1: training error keeps falling, test error is U-shaped
# ---------------------------------------------------------------------------
def fig_train_vs_test_error() -> go.Figure:
    x_train, y_train = make_dataset(n=40, seed=1)
    x_test, y_test = make_dataset(n=200, seed=99)
    degrees = list(range(1, 13))
    train_errs, test_errs = [], []
    for d in degrees:
        tr, te = fit_poly_train_test_error(x_train, y_train, x_test, y_test, d)
        train_errs.append(tr)
        test_errs.append(te)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=train_errs, mode="lines+markers",
                              name="Training error", line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=degrees, y=test_errs, mode="lines+markers",
                              name="Test error (held-out data)", line=dict(color="#E45756")))
    best_d = degrees[int(np.argmin(test_errs))]
    fig.add_vline(x=best_d, line=dict(color="#54A24B", dash="dot"),
                  annotation_text=f"lowest test error: degree {best_d}")
    fig.update_layout(
        title="Training error keeps falling; test error is U-shaped",
        xaxis_title="Model flexibility (polynomial degree)",
        yaxis_title="Mean squared error",
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2: variance of the test-error estimate across validation strategies
# ---------------------------------------------------------------------------
def fig_validation_strategy_variance() -> go.Figure:
    strategies = ["Single 50/50\nvalidation split", "5-fold CV", "10-fold CV", "Leave-one-out CV"]
    n_repeats = 60
    n = 40
    degree = 3
    results = {s: [] for s in strategies}

    for rep in range(n_repeats):
        x, y = make_dataset(n=n, seed=1000 + rep)

        # single validation split (a fresh random split each repeat)
        idx = RNG.permutation(n)
        half = n // 2
        tr_idx, va_idx = idx[:half], idx[half:]
        _, mse = fit_poly_train_test_error(x[tr_idx], y[tr_idx], x[va_idx], y[va_idx], degree)
        results["Single 50/50\nvalidation split"].append(mse)

        for k, label in [(5, "5-fold CV"), (10, "10-fold CV"), (n, "Leave-one-out CV")]:
            folds = np.array_split(RNG.permutation(n), k)
            fold_mses = []
            for fold in folds:
                mask = np.ones(n, dtype=bool)
                mask[fold] = False
                if mask.sum() < degree + 2 or len(fold) == 0:
                    continue
                _, mse = fit_poly_train_test_error(x[mask], y[mask], x[fold], y[fold], degree)
                fold_mses.append(mse)
            results[label].append(np.mean(fold_mses))

    fig = go.Figure()
    colors = ["#F58518", "#4C78A8", "#54A24B", "#B279A2"]
    for strat, color in zip(strategies, colors):
        fig.add_trace(go.Box(y=results[strat], name=strat, marker_color=color, boxmean=True))
    fig.update_layout(
        title="Estimated test error across 60 repeats, by validation strategy",
        yaxis_title="Estimated mean squared error",
        margin=dict(t=60, l=60, r=30, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3: choosing regularization strength (lambda) by k-fold CV
# ---------------------------------------------------------------------------
def ridge_fit_predict(x_train, y_train, x_test, lam, degree=3):
    Xtr = np.vander(x_train / 50.0, degree + 1, increasing=True)
    Xte = np.vander(x_test / 50.0, degree + 1, increasing=True)
    p = Xtr.shape[1]
    penalty = lam * np.eye(p)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Xtr.T @ Xtr + penalty, Xtr.T @ y_train)
    return Xte @ beta


def fig_lambda_selection() -> go.Figure:
    x, y = make_dataset(n=50, noise=30, seed=7)
    lambdas = np.logspace(-2, 3, 40)
    k_values = [3, 10]
    frames = []
    for k in k_values:
        folds = np.array_split(RNG.permutation(len(x)), k)
        cv_curve = []
        for lam in lambdas:
            fold_mses = []
            for fold in folds:
                mask = np.ones(len(x), dtype=bool)
                mask[fold] = False
                pred = ridge_fit_predict(x[mask], y[mask], x[fold], lam)
                fold_mses.append(np.mean((pred - y[fold]) ** 2))
            cv_curve.append(np.mean(fold_mses))
        best_lam = lambdas[int(np.argmin(cv_curve))]
        frames.append(
            go.Frame(
                name=f"k={k}",
                data=[go.Scatter(x=lambdas, y=cv_curve, mode="lines+markers",
                                  line=dict(color="#4C78A8"))],
                layout=go.Layout(
                    shapes=[dict(type="line", x0=best_lam, x1=best_lam, y0=0, y1=1,
                                 yref="paper", line=dict(color="#E45756", dash="dot"))],
                    annotations=[dict(x=np.log10(best_lam), y=1.05, yref="paper",
                                       xref="x", showarrow=False,
                                       text=f"CV-selected lambda = {best_lam:.2f}",
                                       font=dict(color="#E45756"))],
                ),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        title="k-fold CV error as regularization strength (lambda) changes",
        xaxis_title="lambda",
        xaxis_type="log",
        yaxis_title="Cross-validated mean squared error",
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Number of folds: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate", "frame": {"duration": 300}}]}
                for f in frames
            ],
        }],
        margin=dict(t=90, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Bayesian analog, expected log predictive density (ELPD) via PSIS-LOO
# ---------------------------------------------------------------------------
def fig_elpd_by_complexity() -> go.Figure:
    x_train, y_train = make_dataset(n=40, seed=1)
    x_test, y_test = make_dataset(n=300, seed=321)
    degrees = list(range(1, 11))
    elpd_approx = []
    for d in degrees:
        coeffs = np.polyfit(x_train, y_train, d)
        resid_train = y_train - np.polyval(coeffs, x_train)
        sigma = np.std(resid_train) + 1e-6
        pred_test = np.polyval(coeffs, x_test)
        # approximate held-out log predictive density under a Gaussian likelihood,
        # standing in for what PSIS-LOO estimates from posterior draws
        log_dens = stats.norm.logpdf(y_test, loc=pred_test, scale=sigma)
        elpd_approx.append(np.sum(log_dens) / len(x_test) * len(x_train))

    best_d = degrees[int(np.argmax(elpd_approx))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=elpd_approx, mode="lines+markers",
                              line=dict(color="#72B7B2")))
    fig.add_vline(x=best_d, line=dict(color="#E45756", dash="dot"),
                  annotation_text=f"highest ELPD: degree {best_d}")
    fig.update_layout(
        title="Expected log predictive density (ELPD) by model complexity",
        xaxis_title="Model flexibility (polynomial degree)",
        yaxis_title="Approximate ELPD (higher is better)",
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5: from-scratch k-fold grid search over two KNN hyperparameters
# (app-crash telemetry: predicting crashes per 1,000 sessions from heap
# pressure and hours since the app process last restarted)
# ---------------------------------------------------------------------------
def true_crash_rate(heap_pressure, hours_since_restart):
    base = 1.2 + 0.0018 * heap_pressure ** 2
    uptime_effect = 0.028 * hours_since_restart
    interaction = 0.00045 * heap_pressure * hours_since_restart
    return base + uptime_effect + interaction


def make_crash_dataset(n=240, seed=39):
    rng = np.random.default_rng(seed)
    heap_pressure = rng.uniform(15, 90, n)
    hours_since_restart = rng.uniform(0, 96, n)
    true_rate = true_crash_rate(heap_pressure, hours_since_restart)
    observed = true_rate + rng.normal(0, 3.0, n)
    observed = np.clip(observed, 0, None)
    X = np.column_stack([heap_pressure, hours_since_restart])
    return X, observed


def standardize_train_other(X_train, X_other):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_other - mean) / std


def knn_predict(X_train, y_train, X_query, n_neighbors, weighting):
    preds = np.empty(len(X_query))
    for i, q in enumerate(X_query):
        dists = np.sqrt(((X_train - q) ** 2).sum(axis=1))
        nearest = np.argsort(dists)[:n_neighbors]
        if weighting == "uniform":
            preds[i] = y_train[nearest].mean()
        else:
            w = 1.0 / np.maximum(dists[nearest], 1e-6)
            preds[i] = np.average(y_train[nearest], weights=w)
    return preds


def knn_grid_search_cv(k_folds=5, fold_seed=19):
    """Manual fold assignment, nested hyperparameter grid search, and a
    pivot-table summary, mirroring what GridSearchCV/cross_val_score
    automate. Returns (neighbor_grid, weight_grid, cv_error matrix).

    Uses its own seeded generator (independent of the shared module-level
    RNG) so the fold assignment, and therefore every number in this
    section's figure and prose, stays fixed regardless of which other
    figures ran first.
    """
    X, y = make_crash_dataset()
    n = len(y)
    neighbor_grid = [1, 3, 5, 10, 15, 25, 40]
    weight_grid = ["uniform", "distance"]
    fold_rng = np.random.default_rng(fold_seed)

    # manual fold-assignment loop: every observation gets a fold id 0..k_folds-1
    fold_id = np.empty(n, dtype=int)
    shuffled_positions = fold_rng.permutation(n)
    for position, row in enumerate(shuffled_positions):
        fold_id[row] = position % k_folds

    # nested grid-search loop: score every (n_neighbors, weighting)
    # combination by hand across all k_folds folds
    cv_error = np.empty((len(neighbor_grid), len(weight_grid)))
    for i, n_neighbors in enumerate(neighbor_grid):
        for j, weighting in enumerate(weight_grid):
            fold_errors = []
            for fold in range(k_folds):
                train_mask = fold_id != fold
                val_mask = fold_id == fold
                X_train_std, X_val_std = standardize_train_other(X[train_mask], X[val_mask])
                preds = knn_predict(X_train_std, y[train_mask], X_val_std, n_neighbors, weighting)
                fold_errors.append(np.mean((preds - y[val_mask]) ** 2))
            cv_error[i, j] = np.mean(fold_errors)

    return neighbor_grid, weight_grid, cv_error


def fig_knn_grid_search() -> go.Figure:
    neighbor_grid, weight_grid, cv_error = knn_grid_search_cv()

    fig = go.Figure(data=go.Heatmap(
        z=cv_error,
        x=weight_grid,
        y=[str(k) for k in neighbor_grid],
        colorscale="Blues_r",
        text=np.round(cv_error, 2),
        texttemplate="%{text}",
        colorbar=dict(title="CV MSE"),
    ))
    fig.update_layout(
        title="Cross-validated error across neighbor count and weighting scheme",
        xaxis_title="Weighting scheme",
        yaxis_title="Number of neighbors (k)",
        margin=dict(t=60, l=70, r=30, b=50),
    )
    return fig


FIGURES = {
    "chapter-cv-fig-train-vs-test-error": fig_train_vs_test_error,
    "chapter-cv-fig-validation-strategy-variance": fig_validation_strategy_variance,
    "chapter-cv-fig-lambda-selection": fig_lambda_selection,
    "chapter-cv-fig-elpd-by-complexity": fig_elpd_by_complexity,
    "chapter-cv-fig-knn-grid-search": fig_knn_grid_search,
}


if __name__ == "__main__":
    for name, builder in FIGURES.items():
        out_path = save(builder(), name)
        print(f"wrote {out_path}")
