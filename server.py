# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo

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


def nws_points(lat: float, lon: float) -> dict:
    return http_get_json(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )


def nws_forecast_periods(lat: float, lon: float) -> list[dict]:
    points = nws_points(lat, lon)
    forecast_url = points["properties"]["forecast"]
    fc = http_get_json(
        forecast_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    return fc["properties"]["periods"]


def nws_hourly_periods(lat: float, lon: float) -> list[dict]:
    points = nws_points(lat, lon)
    hourly_url = points["properties"].get("forecastHourly")
    if not hourly_url:
        return []
    fc = http_get_json(
        hourly_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    return fc["properties"]["periods"]


def nws_observation_now(lat: float, lon: float) -> dict:
    """
    Returns:
      {
        "temp_f": float|None,
        "obs_time_utc": "ISO"|None,
        "local_time": "ISO"|None,
        "timezone": "America/...",
        "station": "KXXX"|None
      }
    """
    points = nws_points(lat, lon)
    tz = points["properties"].get("timeZone") or "UTC"

    stations_url = points["properties"].get("observationStations")
    if not stations_url:
        return {"temp_f": None, "obs_time_utc": None, "local_time": None, "timezone": tz, "station": None}

    stations = http_get_json(
        stations_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    feats = stations.get("features") or []
    if not feats:
        return {"temp_f": None, "obs_time_utc": None, "local_time": None, "timezone": tz, "station": None}

    station_id = feats[0].get("properties", {}).get("stationIdentifier") or feats[0].get("id")
    station_url = feats[0].get("id")
    if not station_url:
        return {"temp_f": None, "obs_time_utc": None, "local_time": None, "timezone": tz, "station": station_id}

    obs = http_get_json(
        f"{station_url}/observations/latest",
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
        params={"require_qc": "true"},
    )
    props = obs.get("properties", {}) or {}
    t_c = props.get("temperature", {}).get("value", None)
    ts = props.get("timestamp", None)

    temp_f = None
    if isinstance(t_c, (int, float)) and t_c is not None:
        temp_f = (float(t_c) * 9.0 / 5.0) + 32.0

    local_time = None
    if ts:
        try:
            dt_utc = dtparser.isoparse(ts)
            dt_local = dt_utc.astimezone(ZoneInfo(tz))
            local_time = dt_local.isoformat()
        except Exception:
            local_time = None

    return {"temp_f": temp_f, "obs_time_utc": ts, "local_time": local_time, "timezone": tz, "station": station_id}


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
    return dtparser.parse(s).date()


def file_if_exists(path: str):
    if os.path.exists(path) and os.path.isfile(path):
        return path
    return None


@app.get("/", response_class=HTMLResponse)
def home():
    p = file_if_exists("index.html")
    if p:
        with open(p, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>index.html not found. Put index.html next to server.py</h3>")


@app.get("/drought.html")
def drought_page():
    p = file_if_exists("drought.html")
    if p:
        return FileResponse(p, media_type="text/html; charset=utf-8")
    return HTMLResponse("<h3>drought.html not found. Put drought.html next to server.py</h3>", status_code=404)


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


@app.get("/api/now")
def api_now(city: str):
    code = (city or "").strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    now = nws_observation_now(info["lat"], info["lon"])

    # also return a “server-side” local time (even if obs time missing)
    tz = now.get("timezone") or "UTC"
    server_local = datetime.now(tz=ZoneInfo(tz)).isoformat()

    return {
        "city": code,
        "city_name": info["name"],
        "timezone": tz,
        "server_local_time": server_local,
        "obs_local_time": now.get("local_time"),
        "obs_time_utc": now.get("obs_time_utc"),
        "temp_f": now.get("temp_f"),
        "station": now.get("station"),
    }


def _nowcast_mu_sigma_for_date(city_lat: float, city_lon: float, target_d: date, daily_mu: float) -> tuple[float, float, dict]:
    """
    For today: blend in current obs + hourly forecast remaining max.
    For future dates: fall back to daily_mu + sigma_for.
    """
    info = {"used_nowcast": False}

    today = datetime.now().astimezone().date()
    base_sigma = sigma_for(target_d)

    if target_d != today:
        return float(daily_mu), float(base_sigma), info

    info["used_nowcast"] = True

    # current obs
    obs = nws_observation_now(city_lat, city_lon)
    cur_f = obs.get("temp_f")
    tz = obs.get("timezone") or "UTC"

    # hourly remaining max for today
    hourly = nws_hourly_periods(city_lat, city_lon)
    now_local = datetime.now(tz=ZoneInfo(tz))
    max_rem = None
    hours_left = None

    temps = []
    for p in hourly:
        try:
            st = dtparser.isoparse(p["startTime"]).astimezone(ZoneInfo(tz))
        except Exception:
            continue
        if st.date() != today:
            continue
        if st < now_local:
            continue
        t = p.get("temperature")
        if isinstance(t, (int, float)):
            temps.append(float(t))

    if temps:
        max_rem = max(temps)

    # crude hours left estimate: to 23:59 local
    end_local = datetime(now_local.year, now_local.month, now_local.day, 23, 59, tzinfo=ZoneInfo(tz))
    hours_left = max(0.0, (end_local - now_local).total_seconds() / 3600.0)

    # mu_now = best guess of max: at least current temp, at least forecast daily high, at least remaining hourly max
    mu_now = float(daily_mu)
    if isinstance(cur_f, (int, float)):
        mu_now = max(mu_now, float(cur_f))
    if isinstance(max_rem, (int, float)):
        mu_now = max(mu_now, float(max_rem))

    # tighten sigma late day:
    # early day ~2.5; late day ~1.5 (but never below 1.2)
    if hours_left is None:
        sigma_now = max(1.2, min(2.5, base_sigma))
    else:
        sigma_now = max(1.2, min(2.5, 1.2 + 0.06 * hours_left * 10.0))  # ~1.2..2.5

    info.update({
        "obs_temp_f": cur_f,
        "hourly_max_remaining_f": max_rem,
        "hours_left_today": hours_left,
        "timezone": tz,
    })

    return float(mu_now), float(sigma_now), info


@app.get("/api/curve")
def api_curve(city: str, date_str: str, min_f: float, max_f: float, steps: int = 41):
    """
    Returns curve points for plotting:
      { mu, sigma, points: [{t, p_yes}], debug: {...} }
    """
    code = (city or "").strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    try:
        d = parse_date_any(date_str)
    except Exception:
        return {"error": "bad_date"}

    info_city = CITIES[code]
    periods = nws_forecast_periods(info_city["lat"], info_city["lon"])
    highs = daily_daytime_highs(periods)
    if d not in highs:
        return {"error": "nws_high_not_available"}

    daily_mu = float(highs[d])
    mu, sigma, dbg = _nowcast_mu_sigma_for_date(info_city["lat"], info_city["lon"], d, daily_mu)

    steps = int(max(11, min(201, steps)))
    lo = float(min_f)
    hi = float(max_f)
    if hi <= lo:
        hi = lo + 1.0

    pts = []
    for i in range(steps):
        t = lo + (hi - lo) * (i / (steps - 1))
        p = prob_ge(mu, sigma, t)
        pts.append({"t": t, "p_yes": p})

    return {"city": code, "city_name": info_city["name"], "date": d.isoformat(), "mu": mu, "sigma": sigma, "points": pts, "debug": dbg}


class EvalOneRequest(BaseModel):
    city: str
    date: str
    threshold_f: float
    yes_cents: float
    no_cents: float


@app.post("/api/evaluate_one")
def api_evaluate_one(req: EvalOneRequest):
    code = (req.city or "").strip().lower()
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

    yes_price = max(0.0, min(1.0, float(req.yes_cents) / 100.0))
    no_price = max(0.0, min(1.0, float(req.no_cents) / 100.0))

    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)
    if d not in highs:
        return {"error": "nws_high_not_available"}

    daily_mu = float(highs[d])
    mu, sigma, dbg = _nowcast_mu_sigma_for_date(info["lat"], info["lon"], d, daily_mu)
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
        "nowcast_debug": dbg,
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
        out = evaluate_city(code, info["name"], info["lat"], info["lon"])
    except Exception as e:
        return {"error": "drought_eval_failed", "detail": str(e)}

    return {
        "city": code,
        "city_name": info["name"],
        "expected_drought_90d": out.get("expected_drought_90d", 0.0),
        "components": out.get("components", {}),
        "notes": out.get("notes", ""),
    }
