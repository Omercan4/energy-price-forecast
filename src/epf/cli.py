"""Command line: python -m epf.cli <command>

data       download everything and save data/processed/dataset.parquet
backtest   rolling-origin backtest, models and anomaly detectors, saves results to data/processed and docs/
train      fit the final model on all data and save models/champion.joblib
forecast   forecast tomorrow (or --day YYYY-MM-DD) and write docs/forecast.json
monitor    compare finished forecasts with the published prices and write docs/history.json
explain    SHAP drivers of the forecast, saved to docs/
serve      start the API and the web page on http://127.0.0.1:8000
"""

import argparse
import json
from datetime import date, timedelta

from epf import config


def cmd_data(args):
    from epf.data import build, quality

    frame = build.save_dataset(end=args.end)
    report = quality.quality_report(frame)
    (config.PROCESSED_DIR / "quality.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_backtest(args):
    from epf import research

    research.run_all(months=args.months)


def cmd_train(args):
    from epf import research

    research.train_champion()


def cmd_forecast(args):
    from epf import forecast

    target = date.fromisoformat(args.day) if args.day else date.today() + timedelta(days=1)
    if (config.DOCS_DIR / "forecasts" / f"{target}.json").exists() and not args.force:
        print(f"forecast for {target} exists already (use --force to make it again)")
        return
    metadata = json.loads((config.MODEL_DIR / "metadata.json").read_text())
    result = forecast.forecast_day(target)
    path = forecast.write_payload(forecast.to_payload(target, result, metadata))
    print(f"wrote {path}")
    print(result[["q10", "q50", "q90"]].round(1).to_string())


def cmd_monitor(args):
    from epf import research

    research.update_monitoring()


def cmd_explain(args):
    from epf import explain

    print(json.dumps(explain.explain(), indent=1, default=float))


def cmd_serve(args):
    import uvicorn

    uvicorn.run("epf.serve.app:app", host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(prog="epf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data")
    p.add_argument("--end", default=None, help="last day to download (default: tomorrow)")
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("backtest")
    p.add_argument("--months", type=int, default=12)
    p.set_defaults(func=cmd_backtest)
    sub.add_parser("train").set_defaults(func=cmd_train)
    p = sub.add_parser("forecast")
    p.add_argument("--day", default=None)
    p.add_argument("--force", action="store_true", help="make the forecast again even if it exists")
    p.set_defaults(func=cmd_forecast)
    sub.add_parser("monitor").set_defaults(func=cmd_monitor)
    sub.add_parser("explain").set_defaults(func=cmd_explain)
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
