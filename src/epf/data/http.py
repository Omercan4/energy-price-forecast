"""A tiny HTTP helper with retries and an on-disk cache."""

import json
import time
from pathlib import Path

import requests

_session = requests.Session()


def get_json(url, params=None, retries=3, timeout=60):
    """GET a URL and return its JSON. Retries a few times on network errors or 5xx."""
    last_error = None
    for attempt in range(retries):
        try:
            response = _session.get(url, params=params, timeout=timeout)
            if response.status_code >= 500:
                raise requests.HTTPError(f"{response.status_code} for {url}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def cached_json(url, path: Path, params=None, max_age_hours=None):
    """Like get_json, but keeps a copy in `path`. max_age_hours=None means the file never expires."""
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if max_age_hours is None or age_hours < max_age_hours:
            return json.loads(path.read_text())
    data = get_json(url, params=params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data
