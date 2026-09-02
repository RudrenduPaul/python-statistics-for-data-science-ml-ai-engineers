"""
Interactive figures for the Gradient Boosting chapter (XGBoost and LightGBM).

Every function builds one self-contained, standalone Plotly HTML page and writes it
to ../_generated/. The matching chapter-boosting-gradient-boosting.md file embeds each
page in an <iframe>, so the chapter never depends on a live Python kernel to render.

Run directly to regenerate every figure:
    python chapter-boosting-gradient-boosting-plots.py

Requires scikit-learn in addition to the packages in requirements.txt (used here to fit
shallow-tree gradient boosting models on simulated data via scikit-learn's own boosting
implementation, not to hand-wave the curves from a formula).
"""

import os

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RNG = np.random.default_rng(7)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_generated")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig: go.Figure, name: str) -> str:
    path = os.path.join(OUT_DIR, f"{name}.html")
    # auto_play=False matters for the two frame-animated figures in this chapter
    # (shrinkage/early-stopping and the GP hyperparameter search): without it, Plotly
    # auto-plays through every slider step on page load instead of waiting for the
    # reader to drag the slider, which both wastes the interaction and can leave the
    # chart mid-transition (overlapping frames) if the reader drags while it is still
    # auto-advancing.
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, auto_play=False)
    return path


def rollback_risk_data(n=1200):
    """Simulated deploy records: payload_size, canary_error_rate, dependency_count ->
    a continuous rollback-risk score with nonlinear structure and added noise."""
    payload_kb = RNG.uniform(1, 40, n)
    canary_err = RNG.beta(1.3, 25, n) * 100
    deps = RNG.integers(1, 15, n)
    hour = RNG.integers(0, 24, n)
    risk = (
        0.6 * canary_err
        + 0.15 * np.maximum(payload_kb - 20, 0)
        + 0.4 * deps
        + 3.0 * ((hour >= 0) & (hour <= 5))
        + RNG.normal(0, 3, n)
    )
    X = np.column_stack([payload_kb, canary_err, deps, hour])
    return X, risk


# ---------------------------------------------------------------------------
# Figure 1: training/validation loss vs. boosting rounds, at three learning rates
# ---------------------------------------------------------------------------
def fig_shrinkage_early_stopping() -> go.Figure:
    X, y = rollback_risk_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=7)

    learning_rates = [0.5, 0.1, 0.02]
    frames = []
    for lr in learning_rates:
        n_rounds = 300
        model = GradientBoostingRegressor(
            n_estimators=n_rounds, learning_rate=lr, max_depth=2, random_state=7
        )
        model.fit(X_train, y_train)
        train_loss = np.zeros(n_rounds)
        val_loss = np.zeros(n_rounds)
        for i, pred_train in enumerate(model.staged_predict(X_train)):
            train_loss[i] = np.mean((pred_train - y_train) ** 2)
        for i, pred_val in enumerate(model.staged_predict(X_val)):
            val_loss[i] = np.mean((pred_val - y_val) ** 2)
        rounds = np.arange(1, n_rounds + 1)
        best_round = int(np.argmin(val_loss)) + 1
        frames.append(
            go.Frame(
                name=f"{lr}",
                data=[
                    go.Scatter(x=rounds, y=train_loss, mode="lines", name="training loss",
                               line=dict(color="#4C78A8")),
                    go.Scatter(x=rounds, y=val_loss, mode="lines", name="validation loss",
                               line=dict(color="#E45756")),
                ],
                layout=go.Layout(
                    shapes=[dict(type="line", x0=best_round, x1=best_round, y0=0, y1=1,
                                 yref="paper", line=dict(color="#54A24B", width=2, dash="dot"))],
                    annotations=[dict(
                        x=best_round, y=1.05, yref="paper", showarrow=False,
                        text=f"best round: {best_round}", font=dict(color="#54A24B"),
                    )],
                ),
            )
        )

    fig = go.Figure(data=frames[1].data, frames=frames, layout=frames[1].layout)
    fig.update_layout(
        title="Training vs. validation loss across boosting rounds, by learning rate",
        xaxis_title="Boosting round (number of trees added)",
        yaxis_title="Mean squared error",
        sliders=[{
            "active": 1,
            "currentvalue": {"prefix": "Learning rate: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=90, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2: leaf-wise vs. level-wise growth, loss reduction per leaf added
# ---------------------------------------------------------------------------
def fig_leafwise_vs_levelwise() -> go.Figure:
    X, y = rollback_risk_data(n=2000)

    def level_wise_loss_curve(max_leaves_seq):
        losses = []
        for depth in range(1, 9):
            tree = DecisionTreeRegressor(max_depth=depth, random_state=7)
            tree.fit(X, y)
            pred = tree.predict(X)
            losses.append((tree.get_n_leaves(), np.mean((pred - y) ** 2)))
        return losses

    def leaf_wise_loss_curve():
        losses = []
        for leaves in range(2, 130, 4):
            tree = DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=7)
            tree.fit(X, y)
            pred = tree.predict(X)
            losses.append((leaves, np.mean((pred - y) ** 2)))
        return losses

    level = level_wise_loss_curve(None)
    leaf = leaf_wise_loss_curve()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p[0] for p in level], y=[p[1] for p in level], mode="lines+markers",
        name="level-wise growth (depth-limited)", line=dict(color="#4C78A8"),
    ))
    fig.add_trace(go.Scatter(
        x=[p[0] for p in leaf], y=[p[1] for p in leaf], mode="lines+markers",
        name="leaf-wise growth (leaf-limited)", line=dict(color="#F58518"),
    ))
    fig.update_layout(
        title="Training loss for the same leaf budget: leaf-wise reaches it faster",
        xaxis_title="Number of leaves in the tree",
        yaxis_title="Training mean squared error",
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Gaussian-process-guided hyperparameter search surface
# ---------------------------------------------------------------------------
def fig_gp_hyperparameter_search() -> go.Figure:
    X, y = rollback_risk_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=11)

    lr_grid = np.linspace(0.01, 0.5, 12)
    depth_grid = np.arange(1, 7)
    val_loss_grid = np.zeros((len(depth_grid), len(lr_grid)))
    for i, depth in enumerate(depth_grid):
        for j, lr in enumerate(lr_grid):
            model = GradientBoostingRegressor(
                n_estimators=80, learning_rate=lr, max_depth=int(depth), random_state=11
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            val_loss_grid[i, j] = np.mean((pred - y_val) ** 2)

    rng_order = RNG.permutation(len(lr_grid) * len(depth_grid))
    n_shown_options = [4, 8, 16, 24, len(rng_order)]
    frames = []
    flat_i, flat_j = np.meshgrid(np.arange(len(depth_grid)), np.arange(len(lr_grid)), indexing="ij")
    flat_i, flat_j = flat_i.ravel(), flat_j.ravel()
    for n_shown in n_shown_options:
        chosen = rng_order[:n_shown]
        best_idx = chosen[np.argmin(val_loss_grid[flat_i[chosen], flat_j[chosen]])]
        frames.append(
            go.Frame(
                name=str(n_shown),
                data=[
                    go.Contour(
                        z=val_loss_grid, x=lr_grid, y=depth_grid,
                        colorscale="Blues_r", contours=dict(showlabels=True),
                        opacity=0.55, showscale=False,
                    ),
                    go.Scatter(
                        x=lr_grid[flat_j[chosen]], y=depth_grid[flat_i[chosen]],
                        mode="markers", marker=dict(color="#E45756", size=9),
                        name="evaluated so far",
                    ),
                    go.Scatter(
                        x=[lr_grid[flat_j[best_idx]]], y=[depth_grid[flat_i[best_idx]]],
                        mode="markers", marker=dict(color="#54A24B", size=14, symbol="star"),
                        name="best found",
                    ),
                ],
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=go.Layout(
        title="Validation-loss surface over (learning rate, max depth); "
              "a search only sees the marked points",
        xaxis_title="Learning rate",
        yaxis_title="Max tree depth",
    ))
    fig.update_layout(
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Hyperparameter combinations evaluated: "},
            "steps": [
                {"label": f.name, "method": "animate",
                 "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for f in frames
            ],
        }],
        margin=dict(t=70, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: gradient boosting vs. random forest, validation error at increasing dataset size
# ---------------------------------------------------------------------------
def fig_model_comparison_by_dataset_size() -> go.Figure:
    from sklearn.ensemble import RandomForestRegressor

    sizes = [200, 500, 1000, 3000, 8000]
    frames = []
    results = {"Random forest": [], "Gradient boosting (shallow trees)": []}
    for n in sizes:
        X, y = rollback_risk_data(n=n)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=3)
        rf = RandomForestRegressor(n_estimators=200, random_state=3).fit(X_train, y_train)
        gb = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=3
        ).fit(X_train, y_train)
        results["Random forest"].append(np.mean((rf.predict(X_val) - y_val) ** 2))
        results["Gradient boosting (shallow trees)"].append(
            np.mean((gb.predict(X_val) - y_val) ** 2)
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=results["Random forest"], mode="lines+markers",
                              name="Random forest", line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=sizes, y=results["Gradient boosting (shallow trees)"],
                              mode="lines+markers", name="Gradient boosting (shallow trees)",
                              line=dict(color="#F58518")))
    fig.update_layout(
        title="Validation error vs. training-set size: random forest and gradient boosting",
        xaxis_title="Training examples",
        yaxis_title="Validation mean squared error",
        xaxis_type="log",
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5: the running prediction converging as boosting rounds accumulate
# ---------------------------------------------------------------------------
def fig_boosting_rounds() -> go.Figure:
    x = np.sort(RNG.uniform(0, 10, size=80))
    y_true = 2 + 3 * np.sin(x / 2)
    y = y_true + RNG.normal(0, 0.5, size=x.shape[0])
    X = x.reshape(-1, 1)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.15, random_state=3
    )
    model.fit(X, y)
    x_grid = np.linspace(0, 10, 200).reshape(-1, 1)
    staged = list(model.staged_predict(x_grid))

    round_checkpoints = [1, 5, 20, 100]
    colors = ["#B7C7DB", "#F58518", "#4C78A8", "#E45756"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                              marker=dict(color="#DDDDDD", size=6), name="training data",
                              visible=True))
    for i, r in enumerate(round_checkpoints):
        fig.add_trace(go.Scatter(
            x=x_grid.ravel(), y=staged[r - 1], mode="lines",
            line=dict(color=colors[i], width=3), visible=(r == round_checkpoints[0]),
            name=f"after {r} round(s)",
        ))

    steps = []
    for i, r in enumerate(round_checkpoints):
        visible = [True] + [j == i for j in range(len(round_checkpoints))]
        steps.append({"label": str(r), "method": "update", "args": [{"visible": visible}]})

    fig.update_layout(
        title="Each boosting round adds one small tree fit to the current residual",
        xaxis_title="Feature value",
        yaxis_title="Rollback duration (minutes, simulated)",
        showlegend=False,
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Boosting rounds: "},
            "steps": steps,
        }],
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 6: XGBoost's L2 leaf-weight penalty, training vs. validation error
# ---------------------------------------------------------------------------
def fig_xgboost_regularization() -> go.Figure:
    X, y = rollback_risk_data(n=1500)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=13)

    reg_lambda_grid = np.logspace(-2, 3, 18)
    train_loss, val_loss = [], []
    for reg_lambda in reg_lambda_grid:
        model = XGBRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=4,
            reg_lambda=float(reg_lambda), random_state=13,
        )
        model.fit(X_train, y_train)
        train_loss.append(np.mean((model.predict(X_train) - y_train) ** 2))
        val_loss.append(np.mean((model.predict(X_val) - y_val) ** 2))

    best_idx = int(np.argmin(val_loss))

    # A star marker for the best point, not fig.add_vline: add_vline's shape geometry
    # does not compose reliably with a log-scaled x-axis (the vertical line's position
    # gets computed before the axis is set to log, which blew the axis range out to
    # 10^162 in an earlier version of this figure). A scatter marker sidesteps that.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reg_lambda_grid, y=train_loss, mode="lines+markers",
                              name="training error", line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=reg_lambda_grid, y=val_loss, mode="lines+markers",
                              name="validation error", line=dict(color="#E45756")))
    fig.add_trace(go.Scatter(
        x=[reg_lambda_grid[best_idx]], y=[val_loss[best_idx]], mode="markers",
        marker=dict(color="#54A24B", size=14, symbol="star"),
        name=f"lowest validation error (reg_lambda={reg_lambda_grid[best_idx]:.2g})",
    ))
    fig.update_layout(
        title="XGBoost's L2 leaf-weight penalty (reg_lambda): training error keeps rising "
              "while validation error bottoms out, then creeps back up",
        xaxis_title="reg_lambda (L2 penalty on leaf weights)",
        xaxis_type="log",
        yaxis_title="Mean squared error",
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


FIGURES = {
    "chapter-boosting-fig-shrinkage-early-stopping": fig_shrinkage_early_stopping,
    "chapter-boosting-fig-leafwise-vs-levelwise": fig_leafwise_vs_levelwise,
    "chapter-boosting-fig-gp-hyperparameter-search": fig_gp_hyperparameter_search,
    "chapter-boosting-fig-model-comparison-size": fig_model_comparison_by_dataset_size,
    "chapter-boosting-fig-boosting-rounds": fig_boosting_rounds,
    "chapter-boosting-fig-xgboost-regularization": fig_xgboost_regularization,
}


if __name__ == "__main__":
    for name, builder in FIGURES.items():
        out_path = save(builder(), name)
        print(f"wrote {out_path}")
