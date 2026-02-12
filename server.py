from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo
from dateutil import parser as dtparser

import math
import os
import requests

# --- drought helper (used by /api/drought) ---
from drought_long import evaluate_city  # noqa: E402

# --- HRRR helper script (you added this to repo) ---
# MUST be named exactly: hrrr_12h_four_cities.py
try:
    import hrrr_12h_four_cities as hrrr4  # type: ignore
except Exception:
    hrrr4 = None  # endpoint will return hrrr_unavailable

# --- RAP helper script (you added this to repo) ---
# MUST be named exactly: rap.py
try:
    import rap as rap4  # type: ignore
except Exception:
    rap4 = None  # endpoint will return rap_unavailable


# ---------------- CONFIG ----------------
USER_AGENT = "metapplication/1.0 (contact: you@example.com)"
TIMEOUT = 12

CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.006},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.797},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    "lax": {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    "sea": {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    "sfo": {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    "hou": {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    "aus": {"name": "Austin", "lat": 30.2672, "lon": -97.7431},
    "sat": {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936},
    "okc": {"name": "Oklahoma City", "lat": 35.4676, "lon": -97.5164},
    "phx": {"name": "Phoenix", "lat": 33.4484, "lon": -112.074},
    "las": {"name": "Las Vegas", "lat": 36.1699, "lon": -115.1398},
    "msp": {"name": "Minneapolis–Saint Paul", "lat": 44.9778, "lon": -93.265},
    "bna": {"name": "Nashville", "lat": 36.1627, "lon": -86.7816},
}

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


# ---------------- NWS: forecast highs + now ----------------
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


def nws_latest_obs(lat: float, lon: float):
    """
    Returns: (temp_f: float|None, tz_name: str|None, obs_time_iso: str|None)
    """
    points = nws_points(lat, lon)
    tz_name = points["properties"].get("timeZone")

    stations_url = points["properties"].get("observationStations")
    if not stations_url:
        return None, tz_name, None

    stations = http_get_json(
        stations_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    feats = stations.get("features") or []
    if not feats:
        return None, tz_name, None

    station_id = feats[0].get("properties", {}).get("stationIdentifier")
    if not station_id:
        return None, tz_name, None

    obs = http_get_json(
        f"https://api.weather.gov/stations/{station_id}/observations/latest",
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    props = obs.get("properties") or {}
    t_c = (props.get("temperature") or {}).get("value")
    obs_time = props.get("timestamp")

    if isinstance(t_c, (int, float)):
        temp_f = t_c * 9 / 5 + 32
        return float(temp_f), tz_name, obs_time

    return None, tz_name, obs_time


# ---------------- Observations: today's max + post-2pm decline rule ----------------
def nws_primary_station_id(lat: float, lon: float):
    """Return (station_id, tz_name) for the nearest NWS observation station."""
    points = nws_points(lat, lon)
    tz_name = points["properties"].get("timeZone")

    stations_url = points["properties"].get("observationStations")
    if not stations_url:
        return None, tz_name

    stations = http_get_json(
        stations_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    feats = stations.get("features") or []
    if not feats:
        return None, tz_name

    station_id = feats[0].get("properties", {}).get("stationIdentifier")
    return station_id, tz_name


def nws_obs_series_today(lat: float, lon: float):
    """
    Pull observations from local midnight -> now (fallback to last ~24h if needed).

    Returns dict:
      - max_f: max temp observed so far today (F)
      - declined_after_2pm: True if any decrease occurs after 2pm local
      - cap_f: if declined_after_2pm, cap_f == max_f (day's high is capped)
    """
    station_id, tz_name = nws_primary_station_id(lat, lon)
    if not station_id or not tz_name:
        return {
            "tz": tz_name,
            "station_id": station_id,
            "max_f": None,
            "declined_after_2pm": False,
            "cap_f": None,
            "samples": [],
        }

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    now_local = datetime.now(tz)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    end_utc = now_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    url = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {"start": start_utc, "end": end_utc}

    try:
        data = http_get_json(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
            params=params,
        )
    except Exception:
        data = http_get_json(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
            params={"limit": 200},
        )

    feats = data.get("features") or []
    samples = []
    for feat in feats:
        props = feat.get("properties") or {}
        ts = props.get("timestamp")
        t_c = (props.get("temperature") or {}).get("value")
        if not ts or not isinstance(t_c, (int, float)):
            continue
        try:
            dt_utc = dtparser.isoparse(ts)
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            dt_local = dt_utc.astimezone(tz)
            temp_f = float(t_c) * 9 / 5 + 32
            samples.append((dt_local, temp_f))
        except Exception:
            continue

    samples.sort(key=lambda x: x[0])
    if not samples:
        return {
            "tz": tz_name,
            "station_id": station_id,
            "max_f": None,
            "declined_after_2pm": False,
            "cap_f": None,
            "samples": [],
        }

    max_f = max(t for _, t in samples)

    declined_after_2pm = False
    cap_f = None
    if now_local.hour >= 14:
        after2 = [(dt, t) for (dt, t) in samples if dt.hour >= 14]
        if len(after2) >= 2:
            prev = after2[0][1]
            for _, t in after2[1:]:
                if t < prev - 0.05:
                    declined_after_2pm = True
                    cap_f = max_f
                    break
                prev = t

    return {
        "tz": tz_name,
        "station_id": station_id,
        "max_f": float(max_f),
        "declined_after_2pm": declined_after_2pm,
        "cap_f": (float(cap_f) if cap_f is not None else None),
        "samples": samples,
    }


# ---------------- Probability model ----------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_ge(mu: float, sigma: float, threshold: float) -> float:
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


# ---------------- Static pages ----------------
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


# ---------------- API: highs ----------------
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


# ---------------- API: now (temp + local time info) ----------------
@app.get("/api/now")
def api_now(city: str):
    code = (city or "").strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    temp_f, tz_name, obs_time = nws_latest_obs(info["lat"], info["lon"])
    now_utc = datetime.now(timezone.utc).isoformat()

    return {
        "city": code,
        "city_name": info["name"],
        "temp_f": temp_f,
        "tz": tz_name,
        "obs_time": obs_time,
        "server_utc": now_utc,
    }


# ---------------- API: evaluate one (with simple nowcasting floor + "impossible" rule) ----------------
class EvalOneRequest(BaseModel):
    city: str
    date: str
    threshold_f: float
    yes_cents: float
    no_cents: float


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

    yes_price = max(0.0, min(1.0, float(req.yes_cents) / 100.0))
    no_price = max(0.0, min(1.0, float(req.no_cents) / 100.0))

    info = CITIES[code]

    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)
    if d not in highs:
        return {"error": "nws_high_not_available"}

    mu_forecast = float(highs[d])
    sigma = sigma_for(d)
    t = float(req.threshold_f)

    temp_now, tz_name, obs_time = nws_latest_obs(info["lat"], info["lon"])

    obs_cap = None
    obs_max = None
    obs_declined_after_2pm = False
    if d == today:
        obs = nws_obs_series_today(info["lat"], info["lon"])
        obs_max = obs.get("max_f")
        obs_declined_after_2pm = bool(obs.get("declined_after_2pm"))
        obs_cap = obs.get("cap_f") if obs_declined_after_2pm else None

        # "Impossible" rule: if temps have started falling after 2pm local,
        # today's high cannot exceed the max already observed.
        if isinstance(obs_cap, float) and obs_cap < t:
            fair_yes = 0.0
            signal, edge_yes, edge_no, fair_no = decide(fair_yes, yes_price, no_price)
            return {
                "city": code,
                "city_name": info["name"],
                "date": d.isoformat(),
                "threshold_f": t,
                "mu": mu_forecast,
                "mu_forecast_high": mu_forecast,
                "mu_nowcast": temp_now,
                "sigma": sigma,
                "current_temp_f": temp_now,
                "current_tz": tz_name,
                "obs_time": obs_time,
                "market_yes": yes_price,
                "market_no": no_price,
                "fair_yes": fair_yes,
                "fair_no": 1.0,
                "edge_yes": fair_yes - yes_price,
                "edge_no": (1.0 - fair_yes) - no_price,
                "signal": signal,
                "thresholds": {"min_edge": MIN_EDGE, "entry_buffer": ENTRY_BUFFER},
                "obs_max_f": obs_max,
                "obs_declined_after_2pm": obs_declined_after_2pm,
                "obs_cap_f": obs_cap,
                "notes": f"obs-cap: temps fell after 2pm; capped high {obs_cap:.1f}F < threshold → P=0",
            }

    mu_used = mu_forecast
    nowcast_note = "nowcast: none"

    if d == today and isinstance(temp_now, float):
        if temp_now >= t:
            fair_yes = 1.0
            signal, edge_yes, edge_no, fair_no = decide(fair_yes, yes_price, no_price)
            return {
                "city": code,
                "city_name": info["name"],
                "date": d.isoformat(),
                "threshold_f": t,
                "mu": temp_now,
                "mu_forecast_high": mu_forecast,
                "mu_nowcast": temp_now,
                "sigma": sigma,
                "current_temp_f": temp_now,
                "current_tz": tz_name,
                "obs_time": obs_time,
                "market_yes": yes_price,
                "market_no": no_price,
                "fair_yes": fair_yes,
                "fair_no": 0.0,
                "edge_yes": fair_yes - yes_price,
                "edge_no": (0.0 - no_price),
                "signal": signal,
                "thresholds": {"min_edge": MIN_EDGE, "entry_buffer": ENTRY_BUFFER},
                "obs_max_f": obs_max,
                "obs_declined_after_2pm": obs_declined_after_2pm,
                "obs_cap_f": obs_cap,
                "notes": "nowcast: current temp already >= threshold → P=1",
            }

        if temp_now > mu_forecast:
            mu_used = mu_forecast + 0.65 * (temp_now - mu_forecast)
            nowcast_note = f"nowcast: lifted μ toward current temp ({temp_now:.1f}F)"
        else:
            mu_used = max(mu_forecast, temp_now)
            nowcast_note = "nowcast: applied hard floor at current temp"

    fair_yes = prob_ge(mu_used, sigma, t)
    signal, edge_yes, edge_no, fair_no = decide(fair_yes, yes_price, no_price)

    return {
        "city": code,
        "city_name": info["name"],
        "date": d.isoformat(),
        "threshold_f": t,
        "mu": mu_used,
        "mu_forecast_high": mu_forecast,
        "mu_nowcast": temp_now,
        "sigma": sigma,
        "current_temp_f": temp_now,
        "current_tz": tz_name,
        "obs_time": obs_time,
        "market_yes": yes_price,
        "market_no": no_price,
        "fair_yes": fair_yes,
        "fair_no": fair_no,
        "edge_yes": edge_yes,
        "edge_no": edge_no,
        "signal": signal,
        "thresholds": {"min_edge": MIN_EDGE, "entry_buffer": ENTRY_BUFFER},
        "obs_max_f": obs_max,
        "obs_declined_after_2pm": obs_declined_after_2pm,
        "obs_cap_f": obs_cap,
        "notes": nowcast_note,
    }


# ---------------- HRRR (all cities) via YOUR working script ----------------
HRRR_FOUR_CACHE = {"ts": 0.0, "hours": 12, "payload": None}
HRRR_FOUR_CACHE_TTL_SEC = 180  # 3 minutes


@app.get("/api/hrrr_four")
def api_hrrr_four(hours: int = 12):
    """
    Returns HRRR 2m temp series for all configured cities.
    """
    hours = int(hours)
    if hours < 1:
        hours = 1
    if hours > 24:
        hours = 24

    now_ts = datetime.now(timezone.utc).timestamp()
    if (
        HRRR_FOUR_CACHE["payload"] is not None
        and HRRR_FOUR_CACHE["hours"] == hours
        and (now_ts - HRRR_FOUR_CACHE["ts"] < HRRR_FOUR_CACHE_TTL_SEC)
    ):
        return HRRR_FOUR_CACHE["payload"]

    if hrrr4 is None:
        return {"error": "hrrr_unavailable", "detail": "Could not import hrrr_12h_four_cities.py in backend."}

    try:
        from siphon.catalog import get_latest_access_url
        from siphon.ncss import NCSS
    except Exception as e:
        return {"error": "hrrr_unavailable", "detail": f"Missing siphon: {e}"}

    try:
        ncss_url = get_latest_access_url(hrrr4.HRRR_CATALOG, "NetcdfSubset")
        ncss = NCSS(ncss_url)

        payload = {
            "run_ncss": ncss_url,
            "asof_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hours": hours,
            "cities": {},
        }

        for code, info in CITIES.items():
            series = hrrr4.fetch_city_series(ncss, info["lat"], info["lon"], hours)
            payload["cities"][code] = {
                "name": info["name"],
                "lat": info["lat"],
                "lon": info["lon"],
                "series": series,
            }

        HRRR_FOUR_CACHE["ts"] = now_ts
        HRRR_FOUR_CACHE["hours"] = hours
        HRRR_FOUR_CACHE["payload"] = payload

        return payload
    except Exception as e:
        return {"error": "hrrr_unavailable", "detail": str(e)}


# ---------------- RAP (all cities) via rap.py ----------------
RAP_FOUR_CACHE = {"ts": 0.0, "hours": 12, "payload": None}
RAP_FOUR_CACHE_TTL_SEC = 180  # 3 minutes


@app.get("/api/rap_four")
def api_rap_four(hours: int = 12):
    """
    Returns RAP 2m temp series for all configured cities.
    """
    hours = int(hours)
    if hours < 1:
        hours = 1
    if hours > 24:
        hours = 24

    now_ts = datetime.now(timezone.utc).timestamp()
    if (
        RAP_FOUR_CACHE["payload"] is not None
        and RAP_FOUR_CACHE["hours"] == hours
        and (now_ts - RAP_FOUR_CACHE["ts"] < RAP_FOUR_CACHE_TTL_SEC)
    ):
        return RAP_FOUR_CACHE["payload"]

    if rap4 is None:
        return {"error": "rap_unavailable", "detail": "Could not import rap.py in backend."}

    try:
        from siphon.catalog import get_latest_access_url
        from siphon.ncss import NCSS
    except Exception as e:
        return {"error": "rap_unavailable", "detail": f"Missing siphon: {e}"}

    try:
        ncss_url = get_latest_access_url(rap4.RAP_CATALOG, "NetcdfSubset")
        ncss = NCSS(ncss_url)

        payload = {
            "run_ncss": ncss_url,
            "asof_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hours": hours,
            "cities": {},
        }

        for code, info in CITIES.items():
            series = rap4.fetch_city_series(ncss, info["lat"], info["lon"], hours)
            payload["cities"][code] = {
                "name": info["name"],
                "lat": info["lat"],
                "lon": info["lon"],
                "series": series,
            }

        RAP_FOUR_CACHE["ts"] = now_ts
        RAP_FOUR_CACHE["hours"] = hours
        RAP_FOUR_CACHE["payload"] = payload

        return payload
    except Exception as e:
        return {"error": "rap_unavailable", "detail": str(e)}


# ---------------- DROUGHT API ----------------
class DroughtReq(BaseModel):
    city: str


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
