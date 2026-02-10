from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date

import requests
from dateutil import parser as dtparser
import math
import re

# ---------------- CONFIG ----------------
USER_AGENT = "metapplication-backend/0.3 (contact: you@example.com)"
TIMEOUT = 20

CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
}

# High-temp decision thresholds
MIN_EDGE = 0.05
ENTRY_BUFFER = 0.01

SIGMA_BY_LEAD_DAYS = {0: 2.5, 1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0, 5: 7.0}
DEFAULT_SIGMA = 7.5
MAX_DAYS_AHEAD = 14

# Drought calc window
USDM_LOOKBACK_WEEKS = 8
FORECAST_DAYS = 90
# --------------------------------------


app = FastAPI(title="MetApplication Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = requests.Session()


# ---------------- HTTP helpers ----------------
def http_get_json(url: str, headers=None, params=None) -> Any:
    r = session.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------- NWS highs ----------------
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


# Parse: mm/dd/yyyy - t - yes:$$/no:$$
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
    buy_no_ok = (edge_no >= MIN_EDGE) and (no_price <= p_no_fair - ENTRY_BUFFER)

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


class DroughtRequest(BaseModel):
    city: str


# ---------------- Static page serving ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception:
        return HTMLResponse("<h3>index.html not found next to server.py</h3>")


@app.get("/drought", response_class=HTMLResponse)
def drought_page():
    try:
        with open("drought.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception:
        return HTMLResponse("<h3>drought.html not found next to server.py</h3>")


# ---------------- Existing API: highs ----------------
@app.get("/api/highs")
def api_highs(city: str = Query(...)):
    code = city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)

    highs_list = [{"date": d.isoformat(), "high_f": highs[d]} for d in sorted(highs.keys())]
    return {"city": code, "city_name": info["name"], "highs": highs_list}


# ---------------- Existing API: evaluate temp markets ----------------
@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest) -> Dict[str, Any]:
    code = req.city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

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


# ---------------- NEW: drought odds using REAL USDM REST ----------------
def census_county_fips_for_point(lat: float, lon: float) -> Optional[str]:
    """
    Returns 5-digit county FIPS (SSCCC) from US Census geocoder.
    """
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": str(lon),
        "y": str(lat),
        "benchmark": "2020",
        "vintage": "2020",
        "format": "json",
    }
    js = http_get_json(url, headers={"User-Agent": USER_AGENT}, params=params)
    try:
        geos = js["result"]["geographies"]["Counties"][0]
        state = geos["STATE"]
        county = geos["COUNTY"]
        return f"{state}{county}"
    except Exception:
        return None


def usdm_county_series_percent(county_fips5: str, start: date, end: date) -> list[dict]:
    """
    Pull categorical USDM percent area by week (JSON) for a county.
    """
    url = "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent"
    params = {
        "aoi": county_fips5,
        "startdate": f"{start.month}/{start.day}/{start.year}",
        "enddate": f"{end.month}/{end.day}/{end.year}",
        "statisticsType": "2",  # categorical
    }
    r = session.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_expected_drought_90d_from_usdm(series: list[dict]) -> Dict[str, Any]:
    """
    Uses the last ~8 weeks:
      - current D1+ coverage
      - recent change (trend)
    Produces expected probability of being in drought (D1+) sometime/through next 90 days.
    """
    if not series:
        return {
            "expected_drought_90d": 0.5,
            "d1plus_now": None,
            "d2plus_now": None,
            "trend_d1plus": None,
            "outlook_type": "unknown",
            "outlook_label": "—",
            "notes": "USDM series empty.",
        }

    # Normalize keys that might appear
    def get_num(d: dict, keys: list[str]) -> float:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k]) / 100.0 if float(d[k]) > 1.0 else float(d[k])
                except Exception:
                    pass
        return 0.0

    # Sort by date if present
    def parse_dt(row: dict) -> datetime:
        for k in ["ValidStart", "validStart", "Date", "date"]:
            if k in row and row[k]:
                try:
                    return dtparser.parse(str(row[k]))
                except Exception:
                    continue
        return datetime.min

    series_sorted = sorted(series, key=parse_dt)

    # USDM categorical typically includes: None, D0, D1, D2, D3, D4 (percent)
    latest = series_sorted[-1]
    d1 = get_num(latest, ["D1", "d1"])
    d2 = get_num(latest, ["D2", "d2"])
    d3 = get_num(latest, ["D3", "d3"])
    d4 = get_num(latest, ["D4", "d4"])

    d1plus_now = clamp01(d1 + d2 + d3 + d4)
    d2plus_now = clamp01(d2 + d3 + d4)

    # Trend: compare latest vs ~4 weeks ago (or earliest in window)
    idx_past = max(0, len(series_sorted) - 5)
    past = series_sorted[idx_past]
    past_d1plus = clamp01(
        get_num(past, ["D1", "d1"]) +
        get_num(past, ["D2", "d2"]) +
        get_num(past, ["D3", "d3"]) +
        get_num(past, ["D4", "d4"])
    )
    trend = d1plus_now - past_d1plus  # + means worsening coverage

    # Convert to "expected through 90 days"
    # Heuristic: current coverage dominates, trend nudges, and higher D2+ increases persistence.
    # This outputs a probability in [0,1] that varies by location and time.
    base = d1plus_now
    persistence_boost = 0.35 * d2plus_now
    trend_boost = 0.60 * trend  # small trend shifts matter
    expected = clamp01(base + persistence_boost + trend_boost)

    if expected >= 0.66:
        outlook_type, outlook_label = "high", "High"
    elif expected >= 0.40:
        outlook_type, outlook_label = "med", "Med"
    else:
        outlook_type, outlook_label = "low", "Low"

    return {
        "expected_drought_90d": expected,
        "d1plus_now": d1plus_now,
        "d2plus_now": d2plus_now,
        "trend_d1plus": trend,
        "outlook_type": outlook_type,
        "outlook_label": outlook_label,
        "notes": "USDM REST (county) used. CPC layers not yet integrated.",
    }


@app.post("/api/drought")
def api_drought(req: DroughtRequest) -> Dict[str, Any]:
    code = req.city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    lat, lon = info["lat"], info["lon"]

    try:
        county_fips = census_county_fips_for_point(lat, lon)
        if not county_fips:
            return {
                "city": code,
                "city_name": info["name"],
                "error": "county_fips_lookup_failed",
                "expected_drought_90d": 0.5,
                "drought_outlook": {"type": "unknown", "label": "—"},
                "precip_outlook_3mo": {"dry": None, "normal": None, "wet": None},
                "notes": "Could not determine county FIPS for city point.",
            }

        end = datetime.now().astimezone().date()
        start = end - timedelta(days=7 * USDM_LOOKBACK_WEEKS)

        series = usdm_county_series_percent(county_fips, start, end)
        calc = compute_expected_drought_90d_from_usdm(series)

        # Return schema your drought.html already expects
        return {
            "city": code,
            "city_name": info["name"],
            "expected_drought_90d": float(calc["expected_drought_90d"]),
            "drought_outlook": {
                "type": calc["outlook_type"],
                "label": calc["outlook_label"],
            },
            "precip_outlook_3mo": {
                "dry": None,
                "normal": None,
                "wet": None,
            },
            "notes": calc["notes"],
            "debug": {
                "county_fips": county_fips,
                "d1plus_now": calc["d1plus_now"],
                "d2plus_now": calc["d2plus_now"],
                "trend_d1plus": calc["trend_d1plus"],
            }
        }

    except requests.HTTPError as e:
        # Don’t 500 the whole app — return a readable error payload.
        return {
            "city": code,
            "city_name": info["name"],
            "error": "upstream_http_error",
            "expected_drought_90d": 0.5,
            "drought_outlook": {"type": "unknown", "label": "—"},
            "precip_outlook_3mo": {"dry": None, "normal": None, "wet": None},
            "notes": f"HTTPError while calling USDM/Census services: {str(e)}",
        }
    except Exception as e:
        return {
            "city": code,
            "city_name": info["name"],
            "error": "exception",
            "expected_drought_90d": 0.5,
            "drought_outlook": {"type": "unknown", "label": "—"},
            "precip_outlook_3mo": {"dry": None, "normal": None, "wet": None},
            "notes": f"Exception in /api/drought: {type(e).__name__}: {str(e)}",
        }
