"""Make docs/img/social-preview.png (1280 x 640): the card GitHub and LinkedIn show when the repo link is shared.

Run: PYTHONPATH=src .venv/bin/python scripts/make_social_preview.py
It reads the backtest (data/processed/backtest.parquet) and docs/results.json, so the numbers on the card are the real ones.
"""

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from epf import config  # noqa: E402

BG, CARD, LINE = "#0d1117", "#161b22", "#30363d"
FG, MUTED, BLUE, GREEN = "#e6edf3", "#8b949e", "#58a6ff", "#3fb950"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def typical_week(bt: pd.DataFrame) -> pd.DataFrame:
    """A week with a typical price swing: not the calmest, not the one with a record spike, so the chart is readable."""
    local = bt.copy()
    local.index = local.index.tz_convert(config.TIMEZONE)
    daily_range = local["price"].resample("D").max() - local["price"].resample("D").min()
    weekly = daily_range.rolling(7).mean().dropna()
    start = (weekly - weekly.quantile(0.6)).abs().idxmin() - pd.Timedelta(days=6)
    return local[(local.index >= start) & (local.index < start + pd.Timedelta(days=7))]


def main():
    results = json.loads((config.DOCS_DIR / "results.json").read_text())
    models = results["models"]
    mae_model = models["LightGBM, safe inputs (released model)"]["MAE"]
    improvement = results["improvement_vs_naive_168_pct"]
    coverage = results["interval_80"]["coverage_calibrated"] * 100
    bt = pd.read_parquet(config.PROCESSED_DIR / "backtest.parquet")

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(640, 0)
    ax.axis("off")

    ax.text(64, 70, "PORTFOLIO PROJECT  |  TIME SERIES FORECASTING", color=BLUE, fontsize=13, fontweight="bold")
    ax.text(64, 135, "Energy Price Forecast", color=FG, fontsize=38, fontweight="bold")
    subtitle = "Day-ahead electricity prices for Germany, with prediction intervals, anomaly detection and daily monitoring."
    ax.text(64, 178, "\n".join(textwrap.wrap(subtitle, 46)), color=MUTED, fontsize=17, va="top", linespacing=1.4)

    stats = [
        (f"-{improvement:.1f} %", "error vs. same hour\nlast week (backtest)", GREEN),
        (f"{coverage:.0f} %", "of prices inside\nthe 80 % interval", BLUE),
        ("daily", "forecast and monitoring\nby GitHub Actions", BLUE),
    ]
    for i, (big, small, color) in enumerate(stats):
        x = 64 + i * 205
        ax.text(x, 372, big, color=color, fontsize=34, fontweight="bold")
        ax.text(x, 392, small, color=MUTED, fontsize=11.5, va="top", linespacing=1.4)

    ax.text(64, 520, f"MAE {mae_model:.1f} EUR/MWh, model always trained on the past only", color=FG, fontsize=13)
    ax.text(64, 590, "LightGBM  |  MLflow  |  FastAPI  |  GitHub Actions  |  GitHub Pages", color=MUTED, fontsize=13)
    ax.text(1216, 590, "github.com/Omercan4/energy-price-forecast", color=MUTED, fontsize=12, ha="right")

    week = typical_week(bt)
    chart = fig.add_axes([0.575, 0.22, 0.395, 0.56], facecolor=CARD)
    chart.fill_between(week.index, week["q10"], week["q90"], color=BLUE, alpha=0.22, lw=0)
    chart.plot(week.index, week["naive_168"], color=MUTED, lw=1, ls="--")
    chart.plot(week.index, week["q50"], color=BLUE, lw=2.2)
    chart.plot(week.index, week["price"], color=FG, lw=1.3)
    chart.set_xticks([])
    chart.tick_params(colors=MUTED, labelsize=9)
    for spine in chart.spines.values():
        spine.set_color(LINE)
    chart.grid(color=LINE, alpha=0.5, lw=0.6)
    fig.text(0.575, 0.80, "One backtest week", color=FG, fontsize=12, fontweight="bold")
    legend = [("actual price", FG, "-"), ("forecast + 80 % interval", BLUE, "-"), ("same hour last week", MUTED, "--")]
    for i, (label, color, style) in enumerate(legend):
        fig.text(0.575 + i * 0.135, 0.185, label, color=color, fontsize=9.5)

    out = config.DOCS_DIR / "img" / "social-preview.png"
    fig.savefig(out, dpi=100, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
