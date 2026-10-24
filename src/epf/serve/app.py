"""A small API on top of the files the daily job writes, plus the web page.

GET /forecast        tomorrow's 24 hourly prices (q10, q50, q90)
GET /forecast/{day}  an older forecast, for example 2026-10-25
GET /anomalies       unusual hours of the last 14 days
GET /health          model health: rolling error, drift, status
GET /                the web page (the same page runs on GitHub Pages)
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from epf import config

app = FastAPI(title="Energy price forecast", version="0.1.0", description="Day-ahead electricity prices for Germany. Not trading advice.")


def read(name: str):
    path = config.DOCS_DIR / name
    if not path.exists():
        raise HTTPException(404, f"{name} does not exist yet. Run: python -m epf.cli forecast")
    return json.loads(path.read_text())


@app.get("/forecast")
def forecast():
    return read("forecast.json")


@app.get("/forecast/{day}")
def forecast_for(day: str):
    if not (len(day) == 10 and day[4] == "-" and day[7] == "-"):  # also blocks path tricks like ../
        raise HTTPException(400, "Use a date like 2026-10-25")
    return read(f"forecasts/{day}.json")


@app.get("/anomalies")
def anomalies():
    return read("anomalies.json")


@app.get("/health")
def health():
    return read("health.json")


@app.get("/history")
def history():
    return read("history.json")


@app.get("/")
def page():
    return FileResponse(Path(config.DOCS_DIR) / "index.html")


app.mount("/", StaticFiles(directory=config.DOCS_DIR), name="docs")
