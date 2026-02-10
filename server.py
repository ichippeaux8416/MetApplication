from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date

import requests
from dateutil import parser as dtparser
import math
import re

# NEW: drought odds helper (make sure drought_long.py is in the same folder)
from drought_long import expected_drought_probability

# ---------------- CONFIG ----------------
USER_AGENT = "rh-temp-paper-backend/0.1 (contact: you@example.com)"
TIMEOUT = 12

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


app = FastAPI(title="RH Temperature Paper Trader Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = requests.Session()


# ---------------- HELPERS ----------------
def read_local_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def normalize_city(city_in: str) -> Optional[str]:
    """
    Accepts:
      - code: den/nyc/dal/chi
      - full name: Denver / New York City / Dallas / Chicago (case-insensitive)
    Returns normalized code or None.
    """
    if not city_in:
        return None
    c = city_in.strip().lower()

    if c in CITIES:
        return c

    # match by name
    for code, info in CITIES.items():
        if info["name"].strip().lower() == c:
            return code

    # allow first 3 letters from your earlier scheme (den/nyc/dal/chi)
    if len(c) >= 3 and c[:3] in CITIES:
        return c[:3]

    return None


def http_get_json(url: str, headers=None, params=None):
    r = session.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def nws_forecast_periods(lat: float, lon: float) -> List[dict]:
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


def daily_daytime_highs(periods: List[dict]) -> Dict[date, int]:
    out: Dict[date, int] = {}
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
    # continuity correction: P(X >= threshold) approximated via threshold-0.5
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
    buy_no_ok  = (edge_no  >= MIN_EDGE) and (no_price  <= p_no_fair  - ENTRY_BUFFER)

    if buy_yes_ok and buy_no_ok:
        return ("BUY YES" if edge_yes >= edge_no else "BUY NO"), edge_yes, edge_no, p_no_fair
    if buy_yes_ok:
        return "BUY YES", edge_yes, edge_no, p_no_fair
    if buy_no_ok:
        return "BUY NO", edge_yes, edge_no, p_no_fair
    return "PASS", edge_yes, edge_no, p_no_fair


def evaluate_one(code: str, d: date, t: float, yes_price: float, no_price: float) -> Dict[str, Any]:
    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)

    today = datetime.now().astimezone().date()
    max_date = today + timedelta(days=MAX_DAYS_AHEAD)

    if d < today:
        return {"error": "date_in_past"}
    if d > max_date:
        return {"error": "too_far_ahead"}
    if d not in highs:
        return {"error": "nws_high_not_available"}

    mu = float(highs[d])
    sigma = sigma_for(d)
    fair_yes = prob_ge(mu, sigma, t)
    signal, edge_yes, edge_no, fair_no = decide(fair_yes, yes_price, no_price)

    return {
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
    }


# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = read_local_file("index.html")
    if html is None:
        return HTMLResponse("<h3>index.html not found. Put index.html next to server.py</h3>")
    return HTMLResponse(html)


@app.get("/drought.html", response_class=HTMLResponse)
def drought_page():
    html = read_local_file("drought.html")
    if html is None:
        return HTMLResponse("<h3>drought.html not found. Put drought.html next to server.py</h3>")
    return HTMLResponse(html)


@app.get("/api/highs")
def api_highs(city: str = Query(...)):
    code = normalize_city(city)
    if code is None:
        return {"error": "unknown city", "allowed": list(CITIES.keys()), "allowed_names": [c["name"] for c in CITIES.values()]}

    info = CITIES[code]
    periods = nws_forecast_periods(info["lat"], info["lon"])
    highs = daily_daytime_highs(periods)

    highs_list = [{"date": d.isoformat(), "high_f": highs[d]} for d in sorted(highs.keys())]
    return {"city": code, "city_name": info["name"], "highs": highs_list}


@app.get("/api/drought_odds")
def api_drought_odds(city: str = Query(...)) -> Dict[str, Any]:
    """
    Returns a heuristic "expected drought through ~90 days" probability using CPC layers.
    Accepts city code or full name.
    """
    code = normalize_city(city)
    if code is None:
        return {"error": "unknown city", "allowed": list(CITIES.keys()), "allowed_names": [c["name"] for c in CITIES.values()]}

    city_name = CITIES[code]["name"]

    try:
        data = expected_drought_probability(city_name)
        return data
    except Exception as e:
        return {"error": "failed_to_compute", "detail": str(e), "city": city_name}


@app.post("/api/evaluate")
def api_evaluate(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Supports BOTH formats:

    (A) Batch format (old):
        { "city": "den", "lines": ["mm/dd/yyyy - t - yes:30/no:70", ...] }

    (B) Single format (new UI):
        { "city": "Denver" (or "den"),
          "date": "mm/dd/yyyy",
          "t": 42,
          "yes_cents": 30,
          "no_cents": 70
        }

    Returns:
      - For batch: { city, city_name, results:[...] }
      - For single: flattened keys your index.html expects + also "result" object
    """
    city_in = str(payload.get("city", ""))
    code = normalize_city(city_in)
    if code is None:
        return {"error": "unknown city", "allowed": list(CITIES.keys()), "allowed_names": [c["name"] for c in CITIES.values()]}

    info = CITIES[code]

    # -------- (A) Batch mode --------
    if isinstance(payload.get("lines"), list):
        lines = payload.get("lines") or []
        today = datetime.now().astimezone().date()
        max_date = today + timedelta(days=MAX_DAYS_AHEAD)

        periods = nws_forecast_periods(info["lat"], info["lon"])
        highs = daily_daytime_highs(periods)

        results = []
        for raw in lines:
            parsed = parse_line(str(raw))
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

    # -------- (B) Single mode --------
    date_str = payload.get("date")
    t_val = payload.get("t")
    yes_cents = payload.get("yes_cents")
    no_cents = payload.get("no_cents")

    if date_str is None or t_val is None or yes_cents is None or no_cents is None:
        return {
            "error": "missing_fields",
            "needed_for_single_mode": ["city", "date", "t", "yes_cents", "no_cents"],
            "or_use_batch_mode": ["city", "lines[]"]
        }

    try:
        d = dtparser.parse(str(date_str)).date()
        t = float(t_val)
        yes_price = float(yes_cents) / 100.0
        no_price = float(no_cents) / 100.0
    except Exception:
        return {"error": "bad_input_types"}

    one = evaluate_one(code, d, t, yes_price, no_price)
    if "error" in one:
        return {"error": one["error"], "city": code, "city_name": info["name"]}

    # Flatten for the new index.html
    # (use the names your JS expects)
    action = one["signal"]
    return {
        "city": code,
        "city_name": info["name"],
        "action": action,

        "mu": one["mu"],
        "sigma": one["sigma"],

        "p_yes": one["fair_yes"],
        "p_no": one["fair_no"],

        "mkt_yes": one["market_yes"],
        "mkt_no": one["market_no"],

        "edge_yes": one["edge_yes"],
        "edge_no": one["edge_no"],

        # keep the full object too (handy for debugging)
        "result": one,
    }
