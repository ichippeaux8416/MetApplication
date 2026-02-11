# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
from datetime import datetime, timedelta, date, timezone
from dateutil import parser as dtparser

import math
import os
import io
import requests

# --- drought helper (used by /api/drought) ---
# Make sure drought_long.py is in the same folder as server.py
from drought_long import evaluate_city  # noqa: E402


# ---------------- CONFIG ----------------
USER_AGENT = "metapplication/1.0 (contact: you@example.com)"
TIMEOUT = 18

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

# HRRR (NOMADS filter)
HRRR_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
HRRR_BBOX_DEG = 0.30  # small box around point for tiny subset
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


def nws_current_temp_f(lat: float, lon: float) -> tuple[float | None, str | None]:
    """
    Current observed temp (°F) + ISO time from NWS station obs.
    """
    try:
        points = http_get_json(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
        )
        stations_url = points["properties"].get("observationStations")
        if not stations_url:
            return None, None
        stations = http_get_json(
            stations_url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
        )
        feats = stations.get("features") or []
        if not feats:
            return None, None
        sid = feats[0]["properties"].get("stationIdentifier")
        if not sid:
            return None, None
        obs = http_get_json(
            f"https://api.weather.gov/stations/{sid}/observations/latest",
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
        )
        tC = obs["properties"]["temperature"]["value"]
        tF = (tC * 9 / 5 + 32) if isinstance(tC, (int, float)) else None
        ot = obs["properties"].get("timestamp")
        return tF, ot
    except Exception:
        return None, None


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_ge(mu: float, sigma: float, threshold: float) -> float:
    """
    P(High >= threshold) with continuity correction.
    threshold in °F.
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


# ---------------- HRRR 2m TEMP ----------------
def _utcnow():
    return datetime.now(timezone.utc)


def _hrrr_run_candidates():
    """
    Try last ~4 possible hourly runs (UTC).
    """
    now = _utcnow()
    for back in range(0, 4):
        t = now - timedelta(hours=back)
        ymd = t.strftime("%Y%m%d")
        hh = t.strftime("%H")
        yield ymd, hh


def _hrrr_filter_download(ymd: str, hh: str, fxx: int, lon: float, lat: float) -> bytes:
    """
    Download a tiny GRIB2 subset for TMP @ 2m in a small bbox around the point.
    """
    leftlon = lon - HRRR_BBOX_DEG
    rightlon = lon + HRRR_BBOX_DEG
    bottomlat = lat - HRRR_BBOX_DEG
    toplat = lat + HRRR_BBOX_DEG

    file = f"hrrr.t{hh}z.wrfsfcf{fxx:02d}.grib2"
    # NOMADS filter expects dir like /hrrr.YYYYMMDD/conus
    params = {
        "file": file,
        "dir": f"/hrrr.{ymd}/conus",
        "var_TMP": "on",
        "lev_2_m_above_ground": "on",
        "subregion": "",
        "leftlon": f"{leftlon:.4f}",
        "rightlon": f"{rightlon:.4f}",
        "toplat": f"{toplat:.4f}",
        "bottomlat": f"{bottomlat:.4f}",
    }
    r = session.get(HRRR_FILTER, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.content


def _parse_grib2_point_temp_f(grib_bytes: bytes, lon: float, lat: float) -> float:
    """
    Parse subset GRIB2 and return nearest-gridpoint TMP @ 2m in °F.
    Requires: xarray + cfgrib + eccodes.
    """
    import xarray as xr  # type: ignore

    bio = io.BytesIO(grib_bytes)

    # cfgrib needs a file-like path in some environments; BytesIO works in many,
    # but to be safest we write to a temp file.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tf:
        tf.write(bio.read())
        tf.flush()

        ds = xr.open_dataset(tf.name, engine="cfgrib")

    if "t2m" in ds.data_vars:
        vname = "t2m"
    else:
        # some builds name it "t" or "TMP_2maboveground"
        vname = list(ds.data_vars.keys())[0]

    # Identify lat/lon coords
    latc = None
    lonc = None
    for cand in ("latitude", "lat"):
        if cand in ds.coords:
            latc = ds.coords[cand]
            break
    for cand in ("longitude", "lon"):
        if cand in ds.coords:
            lonc = ds.coords[cand]
            break
    if latc is None or lonc is None:
        raise RuntimeError("HRRR parse failed: lat/lon coords not found in subset.")

    # Nearest selection (handles 2D lat/lon grids)
    # xarray nearest works if coords are 1D; for 2D we compute manual nearest.
    import numpy as np  # type: ignore

    latv = latc.values
    lonv = lonc.values

    # normalize lon grids if needed
    lon_target = lon
    if np.nanmax(lonv) > 180 and lon_target < 0:
        lon_target = lon_target + 360

    if latv.ndim == 2 and lonv.ndim == 2:
        d2 = (latv - lat) ** 2 + (lonv - lon_target) ** 2
        iy, ix = np.unravel_index(np.nanargmin(d2), d2.shape)
        tK = float(ds[vname].values[iy, ix])
    else:
        # 1D case
        iy = int(np.nanargmin((latv - lat) ** 2))
        ix = int(np.nanargmin((lonv - lon_target) ** 2))
        tK = float(ds[vname].values[iy, ix])

    # HRRR TMP is Kelvin
    tC = tK - 273.15
    tF = tC * 9 / 5 + 32
    return float(tF)


@app.get("/api/hrrr_t2m")
def api_hrrr_t2m(city: str, hours: int = 18):
    code = (city or "").strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    hours = max(1, min(int(hours), 18))
    info = CITIES[code]
    lat = info["lat"]
    lon = info["lon"]

    last_err = None
    for ymd, hh in _hrrr_run_candidates():
        series = []
        ok = True
        try:
            for fxx in range(0, hours):
                b = _hrrr_filter_download(ymd, hh, fxx, lon, lat)
                tF = _parse_grib2_point_temp_f(b, lon, lat)
                valid = (datetime.strptime(f"{ymd}{hh}", "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=fxx))
                series.append({"valid_utc": valid.isoformat(), "t2m_f": round(tF, 1)})
        except Exception as e:
            ok = False
            last_err = str(e)

        if ok and series:
            return {
                "city": code,
                "city_name": info["name"],
                "run_utc": f"{ymd}T{hh}:00Z",
                "hours": hours,
                "series": series,
                "source": "NOAA NOMADS HRRR filter_hrrr_2d.pl",
            }

    return {"error": "hrrr_unavailable", "detail": last_err or "unknown error"}


# ---------------- TEMPERATURE MARKET ----------------
class EvalOneRequest(BaseModel):
    city: str          # den/nyc/dal/chi
    date: str          # yyyy-mm-dd (from the dropdown)
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

    # NOWCASTING: high cannot be below current temp
    now_tF, now_time = nws_current_temp_f(info["lat"], info["lon"])
    mu_nowcast = mu_forecast
    if isinstance(now_tF, (int, float)):
        mu_nowcast = max(mu_forecast, float(now_tF))

    fair_yes = prob_ge(mu_nowcast, sigma, t)
    signal, edge_yes, edge_no, fair_no = decide(fair_yes, yes_price, no_price)

    # Build a *real* fair curve for display (so the graph is consistent)
    lo = int(round(t - 10))
    hi = int(round(t + 10))
    curve = []
    for thr in range(lo, hi + 1):
        curve.append({"threshold_f": thr, "fair_yes": prob_ge(mu_nowcast, sigma, float(thr))})

    return {
        "city": code,
        "city_name": info["name"],
        "date": d.isoformat(),
        "threshold_f": t,
        "mu_forecast_high": mu_forecast,
        "mu_nowcast_high": mu_nowcast,
        "sigma": sigma,
        "now_temp_f": now_tF,
        "now_time_iso": now_time,
        "market_yes": yes_price,
        "market_no": no_price,
        "fair_yes": fair_yes,
        "fair_no": fair_no,
        "edge_yes": edge_yes,
        "edge_no": edge_no,
        "signal": signal,
        "thresholds": {"min_edge": MIN_EDGE, "entry_buffer": ENTRY_BUFFER},
        "curve": curve,
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
        "expected_severity_0to4": out.get("expected_severity_0to4", None),
        "components": out.get("components", {}),
        "notes": out.get("notes", ""),
    }
