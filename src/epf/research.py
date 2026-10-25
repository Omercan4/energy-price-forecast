"""The workflows behind the CLI: backtest report, final model, and the daily monitoring update."""

import json
from datetime import date, timedelta

import joblib
import pandas as pd

from epf import anomaly, config, metrics, monitor
from epf.backtest import CALIBRATION_DAYS, run_backtest
from epf.data import build, smard
from epf.features import FULL_FEATURES, SAFE_FEATURES, training_table
from epf.models import classical
from epf.models.gbm import QuantileForecaster
from epf.models.linear import RidgeForecaster

DRIFT_FEATURES = ["load_fc", "residual_load_fc", "wind_ns", "temp"]  # inputs that change within a month (not a 7-day mean)
TOP_EXAMPLES = 6


def season(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer"}.get(month, "autumn")


def mlflow_run(name: str):
    """An MLflow run if mlflow is installed, otherwise a do-nothing stand-in."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
        mlflow.set_experiment("energy-price-forecast")
        return mlflow.start_run(run_name=name)
    except ImportError:
        from contextlib import nullcontext

        return nullcontext()


def log_to_mlflow(params=None, scores=None, artifacts=()):
    try:
        import mlflow
    except ImportError:
        return
    mlflow.log_params(params or {})
    mlflow.log_metrics(scores or {})
    for path in artifacts:
        mlflow.log_artifact(str(path))


def backtest_scores(bt: pd.DataFrame) -> dict:
    """All numbers of the backtest in one dict (this becomes docs/results.json and the README table)."""
    actual = bt["price"]
    classical_failures = {name: int(bt[name].isna().sum()) for name in classical.NAMES if name in bt}
    for name in classical.NAMES:  # a fit that failed falls back to the seasonal naive forecast, and is counted above
        if name in bt:
            bt[name] = bt[name].fillna(bt["naive_24"])
    statistical = {
        "seasonal naive, m=24 (same hour yesterday)": bt["naive_24"],
        "seasonal naive, m=168 (same hour last week)": bt["naive_168"],
        "seasonal average, 4 weeks": bt["sma_4w"],
        "Holt-Winters ETS(A,N,A), m=24": bt["ets"],
        "SARIMA(1,0,0)(1,1,0)24 (AR)": bt["sarima_ar"],
        "SARIMA(1,0,1)(0,1,1)24 (ARMA + seasonal MA)": bt["sarima_arma"],
    }
    models = {
        **statistical,
        "ridge regression (AR lags, weather, calendar)": bt["ridge"],
        "LightGBM, safe inputs (released model)": bt["q50"],
        "LightGBM, with grid forecasts (reference)": bt["lgbm_full"],
    }
    table = metrics.score_table(actual, models)
    best_baseline = table.loc[list(statistical), "MAE"].idxmin()
    local = bt.index.tz_convert(config.TIMEZONE)
    by_hour = pd.DataFrame({"naive_168": (actual - bt["naive_168"]).abs(), "lightgbm": (actual - bt["q50"]).abs()}).groupby(local.hour).mean().round(2)
    by_season = pd.DataFrame({"naive_168": (actual - bt["naive_168"]).abs(), "lightgbm": (actual - bt["q50"]).abs()}).groupby([season(m) for m in local.month]).mean().round(2)
    return {
        "rows": len(bt),
        "from": bt.index.min().isoformat(),
        "to": bt.index.max().isoformat(),
        "models": table.to_dict(orient="index"),
        "improvement_vs_naive_168_pct": round((1 - table.loc["LightGBM, safe inputs (released model)", "MAE"] / table.loc["seasonal naive, m=168 (same hour last week)", "MAE"]) * 100, 1),
        "best_statistical_baseline": best_baseline,
        "improvement_vs_best_statistical_baseline_pct": round((1 - table.loc["LightGBM, safe inputs (released model)", "MAE"] / table.loc[best_baseline, "MAE"]) * 100, 1),
        "classical_fit_failures": classical_failures,
        "interval_80": {
            "target": 0.8,
            "coverage_calibrated": round(metrics.coverage(actual, bt["q10"], bt["q90"]), 3),
            "mean_width_calibrated": round(float((bt["q90"] - bt["q10"]).mean()), 2),
            "coverage_raw": round(metrics.coverage(actual, bt["q10_raw"], bt["q90_raw"]), 3),
            "mean_width_raw": round(float((bt["q90_raw"] - bt["q10_raw"]).mean()), 2),
        },
        "pinball": {f"q{int(q * 100)}": round(metrics.pinball(actual, bt[f"q{int(q * 100)}"], q), 2) for q in config.QUANTILES},
        "mae_by_hour": by_hour.to_dict(orient="index"),
        "mae_by_season": by_season.to_dict(orient="index"),
    }


def anomaly_results(bt: pd.DataFrame, table: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    """Residual-based detector vs Isolation Forest, scored against negative-price and spike hours."""
    residual = bt["price"] - bt["q50"]
    stats = anomaly.residual_stats(residual)
    z = anomaly.robust_z(residual, stats)
    events = anomaly.reference_events(bt["price"])
    flags_resid = z.abs() >= 4.0
    iso_features = table.loc[bt.index, ["price", "residual_load_fc", "hour", "load_fc"]]
    flags_iso = anomaly.flag_by_isolation_forest(iso_features, contamination=max(float(flags_resid.mean()), 0.005))
    summary = {
        "events": {"definition": "price below 0, or in the top 1 % of the backtest period", "count": int(events.sum())},
        "residual_robust_z_ge_4": anomaly.precision_recall(flags_resid, events),
        "isolation_forest_same_flag_rate": anomaly.precision_recall(flags_iso, events),
    }
    typical = anomaly.typical_levels(table)
    top = z.abs().sort_values(ascending=False).head(TOP_EXAMPLES).index
    examples = bt.loc[top].join(table[["residual_load_fc", "wind_onshore_fc", "wind_offshore_fc", "solar_fc", "is_holiday", "weekday"]])
    examples["z"] = z.loc[top].round(1)
    examples["why"] = [anomaly.explain(row, typical) for _, row in examples.iterrows()]
    return summary, examples, {"stats": stats, "typical": typical.to_dict()}


def run_all(months: int = 12):
    """Backtest, scores, anomaly detection, figures. Everything is saved under data/processed and docs/."""
    frame = build.load_dataset()
    table = training_table(frame)
    with mlflow_run("backtest"):
        bt = run_backtest(
            frame,
            months=months,
            extra_models={
                "ridge": RidgeForecaster,
                "lgbm_full": lambda: QuantileForecaster(quantiles=(0.5,), features=FULL_FEATURES),
            },
        )
        bt = bt.join(classical.classical_backtest(frame["price"], bt.index))
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        bt.to_parquet(config.PROCESSED_DIR / "backtest.parquet")
        scores = backtest_scores(bt)
        summary, examples, anomaly_state = anomaly_results(bt, table)
        scores["anomaly"] = summary
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (config.DOCS_DIR / "results.json").write_text(json.dumps(scores, indent=1, default=float))
        examples_out = examples.reset_index()
        examples_out["time"] = examples_out["time"].dt.tz_convert(config.TIMEZONE).dt.strftime("%Y-%m-%d %H:%M")
        (config.DOCS_DIR / "anomaly_examples.json").write_text(
            examples_out[["time", "price", "q50", "z", "why"]].round(1).to_json(orient="records", indent=1, force_ascii=False)
        )
        (config.PROCESSED_DIR / "anomaly_state.json").write_text(json.dumps(anomaly_state, default=float))
        log_to_mlflow(
            params={"months": months, "features_safe": len(SAFE_FEATURES), "features_full": len(FULL_FEATURES), "quantiles": str(config.QUANTILES)},
            scores={"mae_lightgbm": scores["models"]["LightGBM, safe inputs (released model)"]["MAE"], "mae_lightgbm_full": scores["models"]["LightGBM, with grid forecasts (reference)"]["MAE"], "mae_naive_168": scores["models"]["seasonal naive, m=168 (same hour last week)"]["MAE"],
                    "coverage_80": scores["interval_80"]["coverage_calibrated"], "coverage_80_raw": scores["interval_80"]["coverage_raw"]},
            artifacts=[config.DOCS_DIR / "results.json"],
        )
    from epf import report

    report.make_figures(bt)
    print(json.dumps({k: v for k, v in scores.items() if k not in ("mae_by_hour", "mae_by_season")}, indent=1, default=float))
    return scores


def train_champion():
    """Fit on all data, save the model, and save what the monitor needs (reference values, thresholds)."""
    frame = build.load_dataset()
    table = training_table(frame)
    with mlflow_run("train-champion"):
        model = QuantileForecaster(calibration_days=CALIBRATION_DAYS).fit(table)
        config.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, config.MODEL_DIR / "champion.joblib")
        results = json.loads((config.DOCS_DIR / "results.json").read_text()) if (config.DOCS_DIR / "results.json").exists() else {}
        state = json.loads((config.PROCESSED_DIR / "anomaly_state.json").read_text()) if (config.PROCESSED_DIR / "anomaly_state.json").exists() else {}
        sample = table[[*DRIFT_FEATURES, "month"]].sample(min(6000, len(table)), random_state=0)
        metadata = {
            "trained_until": table.index.max().isoformat(),
            "rows": len(table),
            "features": model.features,
            "interval_offset": model.offset,
            "backtest_mae": results.get("models", {}).get("LightGBM, safe inputs (released model)", {}).get("MAE"),
            "anomaly_stats": state.get("stats", {}),
            "reference": {name: sample[name].round(3).tolist() for name in [*DRIFT_FEATURES, "month"]},
        }
        (config.MODEL_DIR / "metadata.json").write_text(json.dumps(metadata))
        log_to_mlflow(params={"rows": len(table), "trained_until": metadata["trained_until"]}, artifacts=[config.MODEL_DIR / "metadata.json"])
    print(f"saved model, trained on {len(table)} rows until {metadata['trained_until']}")


def update_monitoring(today: date | None = None):
    """Score finished forecasts against the published prices and refresh the health box and the anomaly list."""
    today = today or date.today()
    metadata = json.loads((config.MODEL_DIR / "metadata.json").read_text())
    history_path = config.DOCS_DIR / "history.json"
    records = monitor.load_history(history_path)
    done = {r["day"] for r in records}

    for path in sorted((config.DOCS_DIR / "forecasts").glob("*.json")):
        day = path.stem
        if day in done or date.fromisoformat(day) > today:
            continue
        payload = json.loads(path.read_text())
        target = date.fromisoformat(day)
        prices = smard.fetch_series("price", target - timedelta(days=9), target + timedelta(days=1))
        start = pd.Timestamp(target, tz=config.TIMEZONE).tz_convert("UTC")
        end = pd.Timestamp(target + timedelta(days=1), tz=config.TIMEZONE).tz_convert("UTC")
        actual = prices[(prices.index >= start) & (prices.index < end)]
        if len(actual) < 23:
            continue  # prices of that day are not published (yet)
        forecast = pd.DataFrame(payload["hours"])
        forecast.index = pd.to_datetime(forecast["time"], utc=True)
        naive = prices.shift(168).reindex(actual.index)
        records.append(monitor.day_record(day, forecast[["q10", "q50", "q90"]], actual, naive))

    monitor.save_history(history_path, records)
    health = monitor.rolling_health(records, metadata["backtest_mae"] or 1.0)

    # drift: compare the last 30 days of inputs with training rows of the SAME calendar months
    # (comparing autumn with a whole year would always look like drift)
    recent = build.build_dataset(today - timedelta(days=40), today, include_weather=True)
    recent_table = training_table(recent).tail(30 * 24)
    reference = pd.DataFrame(metadata["reference"])
    reference = reference[reference["month"].isin(recent_table["month"].unique())]
    drift = {}
    if len(recent_table) >= 24 * 7 and len(reference) >= 200:
        drift = {name: round(monitor.psi(reference[name], recent_table[name], bins=5), 3) for name in DRIFT_FEATURES}
    health["drift_psi"] = drift
    health["drift_alert"] = any(v > 0.25 for v in drift.values())
    health["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    (config.DOCS_DIR / "health.json").write_text(json.dumps(health, indent=1))

    flagged = []
    stats = {int(h): v for h, v in metadata.get("anomaly_stats", {}).items()}
    for record in records[-14:]:
        hourly = pd.DataFrame(record["hourly"])
        if hourly.empty or not stats:
            continue
        hourly.index = pd.to_datetime(hourly["time"], utc=True)
        z = anomaly.robust_z(hourly["price"] - hourly["q50"], stats)
        for t, value in z[z.abs() >= 4.0].items():
            row = hourly.loc[t]
            flagged.append({"time": t.tz_convert(config.TIMEZONE).isoformat(), "price": row["price"], "forecast": row["q50"], "z": round(float(value), 1)})
    (config.DOCS_DIR / "anomalies.json").write_text(json.dumps(flagged, indent=1))
    print(json.dumps(health, indent=1))
    return health
