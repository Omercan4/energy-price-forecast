"""Paths and constants in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

TIMEZONE = "Europe/Berlin"

# SMARD (Bundesnetzagentur | SMARD.de, CC BY 4.0): our name -> (filter id, region)
SMARD_SERIES = {
    "price": (4169, "DE-LU"),  # day-ahead wholesale price, EUR/MWh
    "load_fc": (411, "DE"),  # forecast of total consumption, MWh
    "wind_onshore_fc": (123, "DE"),  # forecast of wind onshore generation, MWh
    "wind_offshore_fc": (3791, "DE"),  # forecast of wind offshore generation, MWh
    "solar_fc": (125, "DE"),  # forecast of solar generation, MWh
}

# Open-Meteo: a few places that stand for the whole country.
WEATHER_PLACES = {
    "north_sea": (53.55, 8.58),
    "berlin": (52.52, 13.41),
    "cologne": (50.94, 6.96),
    "munich": (48.14, 11.58),
}
WEATHER_VARIABLES = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]

HISTORY_START = "2022-01-01"  # the Open-Meteo historical forecast archive starts here
QUANTILES = (0.1, 0.5, 0.9)
