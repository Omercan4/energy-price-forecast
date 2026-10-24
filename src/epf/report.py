"""Figures for the README (saved as PNG files in docs/img)."""

import matplotlib

matplotlib.use("Agg")  # draw into files, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from epf import config, metrics  # noqa: E402

IMG = config.DOCS_DIR / "img"
BLUE, GREY, ORANGE = "#1f6feb", "#8b949e", "#d9822b"


def make_figures(bt: pd.DataFrame) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    local = bt.copy()
    local.index = local.index.tz_convert(config.TIMEZONE)
    week_figure(local)
    hour_figure(local)
    models_figure(local)


def week_figure(local: pd.DataFrame) -> None:
    """One week with the largest price swings in the backtest: actual price, forecast and 80 % interval."""
    daily_range = (local["price"].resample("D").max() - local["price"].resample("D").min())
    start = daily_range.rolling(7).mean().idxmax() - pd.Timedelta(days=6)
    week = local[(local.index >= start) & (local.index < start + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(week.index, week["q10"], week["q90"], color=BLUE, alpha=0.18, label="80 % interval (q10 to q90)")
    ax.plot(week.index, week["q50"], color=BLUE, lw=2, label="forecast (median)")
    ax.plot(week.index, week["price"], color="black", lw=1.4, label="actual day-ahead price")
    ax.plot(week.index, week["naive_168"], color=GREY, lw=1, ls="--", label="same hour last week (baseline)")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Day-ahead price: forecast vs. actual (backtest, one week, model trained only on earlier data)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "week.png", dpi=150)
    plt.close(fig)


def hour_figure(local: pd.DataFrame) -> None:
    errors = pd.DataFrame({"last week": (local["price"] - local["naive_168"]).abs(), "LightGBM": (local["price"] - local["q50"]).abs()})
    by_hour = errors.groupby(local.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(by_hour.index, by_hour["last week"], color=GREY, marker="o", ms=3, label="same hour last week")
    ax.plot(by_hour.index, by_hour["LightGBM"], color=BLUE, marker="o", ms=3, label="LightGBM (safe inputs)")
    ax.set_xlabel("hour of the day (local time)")
    ax.set_ylabel("MAE, EUR/MWh")
    ax.set_title("Error by hour of the day (12-month backtest)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "mae_by_hour.png", dpi=150)
    plt.close(fig)


def models_figure(local: pd.DataFrame) -> None:
    models = {
        "last day": local["naive_24"],
        "last week": local["naive_168"],
        "ridge": local["ridge"],
        "LightGBM\n(safe inputs)": local["q50"],
        "LightGBM\n(+ grid forecasts)": local["lgbm_full"],
    }
    values = [metrics.mae(local["price"], f) for f in models.values()]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(list(models), values, color=[GREY, GREY, GREY, BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.1f", padding=2)
    ax.set_ylabel("MAE, EUR/MWh (lower is better)")
    ax.set_title("Model comparison, 12-month rolling backtest")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "models.png", dpi=150)
    plt.close(fig)
