from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from pathlib import Path

import requests
from dateutil import parser as dtparser
import math
import re

# ---------------- CONFIG ----------------
USER_AGENT = "rh-temp-paper-backend/0.1 (contact: you@example.com)"
TIMEOUT = 12

CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
}

# Decision thresholds (match rh_paper.py)
MIN_EDGE = 0.05
ENTRY_BUFFER = 0.01

SIGMA_BY_LEAD_DAYS = {0: 2.5, 1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0, 5: 7.0}
DEFAULT_SIGMA = 7.5

MAX_DAYS_AHEAD = 14
# --------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"
DROUGHT_HTML = BASE_DIR / "drought.html"

app = FastAPI(title="MetApplication Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = requests.Session()


def http_get_json(url: str, headers=None, params=None):
    r = session.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# -------------------- TEMPERATURE MODEL --------------------

def nws_forecast_periods(lat: float, lon: float) -> list[dict]:
    points = http_get_json(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    forecast_url = points["properties"]["forecast"]
    fc = http_get_json(
        forecast_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    return fc["properties"]["periods"]


def daily_daytime_highs(periods: list[dict]) -> dict[date, int]:
    out: dict[date, int] = {}
    for p in periods:
        if not p.get("isDaytime"):
            continue
        d = dtparser.isoparse(p["startTime"]).date()
        t = p.get("temperature")
        if isinstance(t, int):
            out[d] = t
    return out


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_ge(mu: float, sigma: float, threshold: float) -> float:
    z = (threshold - 0.5 - mu) / sigma
    return max(0.0, min(1.0, 1.0 - norm_cdf(z)))


def sigma_for(target_date: date) -> float:
    today = datetime.now().astimezone().date()
    lead = (target_date - today).days
    if lead < 0:
        return SIGMA_BY_LEAD_DAYS.get(0, DEFAULT_SIGMA)
    return SIGMA_BY_LEAD_DAYS.get(lead, DEFAULT_SIGMA)


LINE_RE = re.compile(
    r"""^\s*
    (?P<d>\d{1,2}/\d{1,2}/\d{4})\s*-\s*
    (?P<t>-?\d{1,3}(?:\.\d+)?)\s*-\s*
    yes\s*:\s*(?P<yes>-?\d+(?:\.\d+)?)\s*/\s*
    no\s*:\s*(?P<no>-?\d+(?:\.\d+)?)\s*
    $""",
    re.IGNORECASE | re.VERBOSE
)


def parse_line(line: str):
    m = LINE_RE.match(line)
    if not m:
        return None
    d = dtparser.parse(m.group("d")).date()
    t = float(m.group("t"))
    yes_cents = float(m.group("yes"))
    no_cents = float(m.group("no"))
    return d, t, yes_cents / 100.0, no_cents / 100.0


def decide(p_yes_fair: float, yes_price: float, no_price: float):
    p_no_fair = 1.0 - p_yes_fair
    edge_yes = p_yes_fair - yes_price
    edge_no = p_no_fair - no_price

    buy_yes_ok = (edge_yes >= MIN_EDGE) and (yes_price <= p_yes_fair - ENTRY_BUFFER)
    buy_no_ok  = (edge_no  >= MIN_EDGE) and (no_price  <= p_no_fair  - ENTRY_BUFFER)

    if buy_yes_ok and buy_no_ok:
        return ("BUY YES" if edge_yes >= edge_no else "BUY NO"), edge_yes, edge_no, p_no_fair
    if buy_yes_ok:
        return "BUY YES", edge_yes, edge_no, p_no_fair
    if buy_no_ok:
        return "BUY NO", edge_yes, edge_no, p_no_fair
    return "PASS", edge_yes, edge_no, p_no_fair


class EvalRequest(BaseModel):
    city: str
    lines: List[str]


# -------------------- PAGES --------------------

@app.get("/", response_class=HTMLResponse)
def home():
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>index.html not found next to server.py</h3>")


@app.get("/drought", response_class=HTMLResponse)
def drought_page():
    if DROUGHT_HTML.exists():
        return HTMLResponse(DROUGHT_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>drought.html not found next to server.py</h3>")


@app.get("/drought.html")
def drought_html_file():
    if DROUGHT_HTML.exists():
        return FileResponse(str(DROUGHT_HTML), media_type="text/html")
    return HTMLResponse("<h3>drought.html not found next to server.py</h3>")


# -------------------- TEMPERATURE APIs --------------------

@app.get("/api/highs")
def api_highs(city: str = Query(...)):
    code = city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)

    highs_list = [{"date": d.isoformat(), "high_f": highs[d]} for d in sorted(highs.keys())]
    return {"city": code, "city_name": info["name"], "highs": highs_list}


@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest) -> Dict[str, Any]:
    code = req.city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)

    today = datetime.now().astimezone().date()
    max_date = today + timedelta(days=MAX_DAYS_AHEAD)

    results = []
    for raw in req.lines:
        parsed = parse_line(raw)
        if not parsed:
            results.append({"raw": raw, "error": "invalid_format"})
            continue

        d, t, yes_price, no_price = parsed

        if d < today:
            results.append({"raw": raw, "error": "date_in_past"})
            continue
        if d > max_date:
            results.append({"raw": raw, "error": "too_far_ahead"})
            continue
        if d not in highs:
            results.append({"raw": raw, "error": "nws_high_not_available"})
            continue

        mu = float(highs[d])
        sigma = sigma_for(d)
        fair_yes = prob_ge(mu, sigma, t)

        signal, edge_yes, edge_no, fair_no = decide(fair_yes, yes_price, no_price)

        results.append({
            "date": d.isoformat(),
            "t": t,
            "market_yes": yes_price,
            "market_no": no_price,
            "mu": mu,
            "sigma": sigma,
            "fair_yes": fair_yes,
            "fair_no": fair_no,
            "edge_yes": edge_yes,
            "edge_no": edge_no,
            "signal": signal,
        })

    return {"city": code, "city_name": info["name"], "results": results}


# -------------------- DROUGHT API (FIXES YOUR 404) --------------------

class DroughtRequest(BaseModel):
    city: str


def _drought_fallback(city_code: str) -> Dict[str, Any]:
    # Placeholder so the UI never 404s.
    # We’ll wire real CPC/USDM parsing next.
    return {
        "city": city_code,
        "city_name": CITIES.get(city_code, {}).get("name", city_code),
        "expected_drought_90d": 0.50,
        "drought_outlook": {"type": "unknown", "label": "—"},
        "precip_outlook_3mo": {"dry": 0.33, "normal": 0.34, "wet": 0.33},
        "notes": "Fallback values (real CPC/USDM parsing not active yet).",
    }


def evaluate_drought(city_code: str) -> Dict[str, Any]:
    """
    Tries to call your drought_long.py if present:
      - evaluate_drought(city_code) OR
      - evaluate(city_code)
    Otherwise returns fallback (never errors).
    """
    try:
        import drought_long as dl  # your file: drought_long.py

        if hasattr(dl, "evaluate_drought"):
            data = dl.evaluate_drought(city_code)
        elif hasattr(dl, "evaluate"):
            data = dl.evaluate(city_code)
        else:
            return _drought_fallback(city_code)

        # Ensure minimum keys exist (so front-end doesn’t break)
        if not isinstance(data, dict):
            return _drought_fallback(city_code)

        data.setdefault("city", city_code)
        data.setdefault("city_name", CITIES.get(city_code, {}).get("name", city_code))
        data.setdefault("expected_drought_90d", 0.50)
        data.setdefault("drought_outlook", {"type": "unknown", "label": "—"})
        data.setdefault("precip_outlook_3mo", {"dry": 0.33, "normal": 0.34, "wet": 0.33})
        data.setdefault("notes", "Returned by drought_long.py")
        return data

    except Exception as e:
        out = _drought_fallback(city_code)
        out["notes"] = f"Fallback (error calling drought_long.py: {type(e).__name__}: {e})"
        return out


@app.get("/api/drought")
def api_drought_get(city: str = Query(...)) -> Dict[str, Any]:
    code = city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown city", "allowed": list(CITIES.keys())}
    return evaluate_drought(code)


@app.post("/api/drought")
def api_drought_post(req: DroughtRequest) -> Dict[str, Any]:
    code = req.city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown city", "allowed": list(CITIES.keys())}
    return evaluate_drought(code)


# Aliases so your drought.html won't 404 even if it calls a different path
@app.post("/api/drought_odds")
def api_drought_odds(req: DroughtRequest) -> Dict[str, Any]:
    return api_drought_post(req)


@app.post("/api/drought/evaluate")
def api_drought_evaluate(req: DroughtRequest) -> Dict[str, Any]:
    return api_drought_post(req)
