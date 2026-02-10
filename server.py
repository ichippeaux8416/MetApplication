# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from dateutil import parser as dtparser

import math
import os
import requests

# --- drought helper (used by /api/drought) ---
# Make sure drought_long.py is in the same folder as server.py
from drought_long import evaluate_city  # noqa: E402


# ---------------- CONFIG ----------------
USER_AGENT = "metapplication/1.0 (contact: you@example.com)"
TIMEOUT = 12

# City codes used by the frontend
CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
}

# Decision thresholds
# (Lower MIN_EDGE if you want more signals; 0.03–0.06 is reasonable)
MIN_EDGE = 0.05
ENTRY_BUFFER = 0.01

SIGMA_BY_LEAD_DAYS = {0: 2.5, 1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0, 5: 7.0}
DEFAULT_SIGMA = 7.5

MAX_DAYS_AHEAD = 14
# --------------------------------------


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
    """
    P(High >= threshold) with a continuity correction.
    threshold is in °F.
    """
    if sigma <= 0:
        return 1.0 if mu >= threshold else 0.0
    z = (threshold - 0.5 - mu) / sigma
    return max(0.0, min(1.0, 1.0 - norm_cdf(z)))


def sigma_for(target_date: date) -> float:
    today = datetime.now().astimezone().date()
    lead = (target_date - today).days
    if lead < 0:
        lead = 0
    return SIGMA_BY_LEAD_DAYS.get(lead, DEFAULT_SIGMA)


def decide(fair_yes: float, yes_price: float, no_price: float):
    fair_no = 1.0 - fair_yes
    edge_yes = fair_yes - yes_price
    edge_no = fair_no - no_price

    buy_yes_ok = (edge_yes >= MIN_EDGE) and (yes_price <= fair_yes - ENTRY_BUFFER)
    buy_no_ok = (edge_no >= MIN_EDGE) and (no_price <= fair_no - ENTRY_BUFFER)

    if buy_yes_ok and buy_no_ok:
        signal = "BUY YES" if edge_yes >= edge_no else "BUY NO"
        return signal, edge_yes, edge_no, fair_no

    if buy_yes_ok:
        return "BUY YES", edge_yes, edge_no, fair_no

    if buy_no_ok:
        return "BUY NO", edge_yes, edge_no, fair_no

    return "PASS", edge_yes, edge_no, fair_no


def parse_date_any(s: str) -> date:
    # Accept "MM/DD/YYYY", "YYYY-MM-DD", etc.
    return dtparser.parse(s).date()


def file_if_exists(path: str):
    if os.path.exists(path) and os.path.isfile(path):
        return path
    return None


@app.get("/", response_class=HTMLResponse)
def home():
    # Serve index.html located next to server.py
    p = file_if_exists("index.html")
    if p:
        with open(p, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>index.html not found. Put index.html next to server.py</h3>")


@app.get("/drought.html")
def drought_page():
    # Serve drought.html located next to server.py
    p = file_if_exists("drought.html")
    if p:
        return FileResponse(p, media_type="text/html; charset=utf-8")
    return HTMLResponse("<h3>drought.html not found. Put drought.html next to server.py</h3>", status_code=404)


# Optional: serve a favicon if you add one later
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/highs")
def api_highs(city: str):
    code = city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)

    highs_list = [{"date": d.isoformat(), "high_f": highs[d]} for d in sorted(highs.keys())]
    return {"city": code, "city_name": info["name"], "highs": highs_list}


class EvalOneRequest(BaseModel):
    city: str          # den/nyc/dal/chi
    date: str          # mm/dd/yyyy or yyyy-mm-dd
    threshold_f: float # the "Potential High Temp" (contract threshold to exceed)
    yes_cents: float   # user's entered YES price in cents
    no_cents: float    # user's entered NO price in cents


@app.post("/api/evaluate_one")
def api_evaluate_one(req: EvalOneRequest):
    code = req.city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    try:
        d = parse_date_any(req.date)
    except Exception:
        return {"error": "bad_date"}

    today = datetime.now().astimezone().date()
    max_date = today + timedelta(days=MAX_DAYS_AHEAD)
    if d < today:
        return {"error": "date_in_past"}
    if d > max_date:
        return {"error": "too_far_ahead", "max_date": max_date.isoformat()}

    yes_price = float(req.yes_cents) / 100.0
    no_price = float(req.no_cents) / 100.0

    # clamp input prices to [0,1] (still return the original as you typed)
    yes_price = max(0.0, min(1.0, yes_price))
    no_price = max(0.0, min(1.0, no_price))

    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)
    if d not in highs:
        return {"error": "nws_high_not_available"}

    mu = float(highs[d])
    sigma = sigma_for(d)
    t = float(req.threshold_f)

    fair_yes = prob_ge(mu, sigma, t)
    signal, edge_yes, edge_no, fair_no = decide(fair_yes, yes_price, no_price)

    return {
        "city": code,
        "city_name": info["name"],
        "date": d.isoformat(),
        "threshold_f": t,
        "mu": mu,
        "sigma": sigma,
        "market_yes": yes_price,
        "market_no": no_price,
        "fair_yes": fair_yes,
        "fair_no": fair_no,
        "edge_yes": edge_yes,
        "edge_no": edge_no,
        "signal": signal,
        "thresholds": {"min_edge": MIN_EDGE, "entry_buffer": ENTRY_BUFFER},
    }


# ---------------- DROUGHT API ----------------
class DroughtReq(BaseModel):
    city: str  # den/nyc/dal/chi


@app.post("/api/drought")
def api_drought(req: DroughtReq):
    code = (req.city or "").strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    try:
        out = evaluate_city(info["lat"], info["lon"])
    except Exception as e:
        # Return a readable error body so the frontend can show it
        return {"error": "drought_eval_failed", "detail": str(e)}

    return {
        "city": code,
        "city_name": info["name"],
        "expected_drought_90d": out.get("expected_drought_90d", 0.0),
        "components": out.get("components", {}),
    }
