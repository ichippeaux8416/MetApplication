from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import os
import math
import re
import zipfile
import io
import requests
from dateutil import parser as dtparser
import shapefile  # pyshp
import xml.etree.ElementTree as ET

# ---------------- CONFIG ----------------
USER_AGENT = "metapplication-backend/1.0 (contact: you@example.com)"
TIMEOUT = 18

CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
}

# ---- High-temp decision thresholds (unchanged) ----
MIN_EDGE = 0.05
ENTRY_BUFFER = 0.01
SIGMA_BY_LEAD_DAYS = {0: 2.5, 1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0, 5: 7.0}
DEFAULT_SIGMA = 7.5
MAX_DAYS_AHEAD = 14

# ---- Drought data sources ----
# US Drought Monitor (USDM) current shapefile zip (public, widely used)
USDM_ZIP_URL = "https://droughtmonitor.unl.edu/data/shapefiles_m/usdm_current.zip"

# CPC 3-month precip outlook KML (lead1 = current 3-month outlook)
CPC_PRCP_KML_URL = "https://www.cpc.ncep.noaa.gov/products/predictions/90day/lead1_prcp.kml"

# CPC Seasonal Drought Outlook as an ArcGIS service (queryable by point)
# Layer indices can change; we query the service root and then try a few common layers.
CPC_SDO_SERVICE = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_drought_outlk/MapServer"

# Cache directory (Render: /tmp is writable)
CACHE_DIR = os.environ.get("METAPP_CACHE_DIR", "/tmp/metapplication_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

USDM_CACHE_PATH = os.path.join(CACHE_DIR, "usdm_current.zip")
CPC_PRCP_KML_CACHE_PATH = os.path.join(CACHE_DIR, "cpc_lead1_prcp.kml")
CPC_SDO_LAYERS_CACHE_PATH = os.path.join(CACHE_DIR, "cpc_sdo_layers.json")

# Cache TTLs
USDM_TTL_SECONDS = 6 * 3600
CPC_KML_TTL_SECONDS = 6 * 3600
CPC_SDO_TTL_SECONDS = 12 * 3600
# --------------------------------------

app = FastAPI(title="MetApplication Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# ---------------- Small utilities ----------------
def _now_ts() -> float:
    return datetime.now().timestamp()

def _is_fresh(path: str, ttl_seconds: int) -> bool:
    try:
        st = os.stat(path)
        return (_now_ts() - st.st_mtime) < ttl_seconds and st.st_size > 0
    except Exception:
        return False

def http_get_text(url: str) -> str:
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def http_get_bytes(url: str) -> bytes:
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content

def http_get_json(url: str, params=None) -> dict:
    r = session.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---------------- Static file serving ----------------
def serve_local_html(filename: str) -> HTMLResponse:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception:
        return HTMLResponse(f"<h3>{filename} not found. Put {filename} next to server.py</h3>", status_code=404)

@app.get("/", response_class=HTMLResponse)
def home():
    return serve_local_html("index.html")

@app.get("/drought", response_class=HTMLResponse)
def drought_page():
    return serve_local_html("drought.html")

@app.get("/drought.html", response_class=HTMLResponse)
def drought_page_html():
    return serve_local_html("drought.html")

# ---------------- NWS highs (existing logic) ----------------
def nws_forecast_periods(lat: float, lon: float) -> list[dict]:
    points = http_get_json(
        f"https://api.weather.gov/points/{lat},{lon}",
        params=None,
    )
    forecast_url = points["properties"]["forecast"]
    fc = http_get_json(forecast_url)
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

# ---------------- Drought odds (REAL DATA) ----------------

# ---- Geometry helpers: point in polygon (ray casting) ----
def point_in_ring(x: float, y: float, ring: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def point_in_polygon(x: float, y: float, rings: List[List[Tuple[float, float]]]) -> bool:
    # first ring outer, subsequent rings holes (common)
    if not rings:
        return False
    if not point_in_ring(x, y, rings[0]):
        return False
    # if inside any hole => not inside
    for hole in rings[1:]:
        if point_in_ring(x, y, hole):
            return False
    return True

def shp_shape_to_rings(shp) -> List[List[Tuple[float, float]]]:
    # pyshp stores shape.points and shape.parts indices
    pts = shp.points
    parts = list(shp.parts) + [len(pts)]
    rings = []
    for i in range(len(parts) - 1):
        seg = pts[parts[i]:parts[i+1]]
        rings.append([(float(px), float(py)) for (px, py) in seg])
    return rings

# ---- USDM current drought category at point ----
def fetch_usdm_zip() -> bytes:
    if _is_fresh(USDM_CACHE_PATH, USDM_TTL_SECONDS):
        with open(USDM_CACHE_PATH, "rb") as f:
            return f.read()
    data = http_get_bytes(USDM_ZIP_URL)
    with open(USDM_CACHE_PATH, "wb") as f:
        f.write(data)
    return data

def usdm_category_at(lat: float, lon: float) -> Optional[str]:
    """
    Returns drought category string like 'D0','D1','D2','D3','D4' or None if not found / no drought.
    USDM polygons usually include an attribute with the category (often 'DM' or 'DROUGHT' etc).
    """
    zbytes = fetch_usdm_zip()
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    # find .shp base
    shp_names = [n for n in zf.namelist() if n.lower().endswith(".shp")]
    if not shp_names:
        return None
    shp_name = shp_names[0]
    base = shp_name[:-4]

    # read needed components into memory
    shp = io.BytesIO(zf.read(base + ".shp"))
    shx = io.BytesIO(zf.read(base + ".shx"))
    dbf = io.BytesIO(zf.read(base + ".dbf"))

    r = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
    fields = [f[0] for f in r.fields[1:]]  # skip DeletionFlag
    # common field candidates:
    candidates = ["DM", "DROUGHT", "CAT", "CATEGORY", "DMLEVEL", "DM_LVL"]
    field_idx = None
    for c in candidates:
        if c in fields:
            field_idx = fields.index(c)
            break

    x = float(lon)
    y = float(lat)
    found_cat: Optional[str] = None

    for sr in r.iterShapeRecords():
        shp_obj = sr.shape
        if shp_obj.shapeType not in (5, 15, 25, 31):  # polygon types
            continue
        rings = shp_shape_to_rings(shp_obj)
        if not point_in_polygon(x, y, rings):
            continue

        rec = sr.record
        if field_idx is not None:
            val = str(rec[field_idx]).strip()
        else:
            # brute-force scan record for something like D0..D4
            val = ""
            for v in rec:
                sv = str(v).strip()
                if sv in ("D0", "D1", "D2", "D3", "D4"):
                    val = sv
                    break

        if val in ("D0", "D1", "D2", "D3", "D4"):
            found_cat = val
            break

    return found_cat  # None => either "no drought polygon" or mismatch

# ---- CPC 3-month precip outlook probabilities at point ----
def fetch_cpc_prcp_kml() -> str:
    if _is_fresh(CPC_PRCP_KML_CACHE_PATH, CPC_KML_TTL_SECONDS):
        with open(CPC_PRCP_KML_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    text = http_get_text(CPC_PRCP_KML_URL)
    with open(CPC_PRCP_KML_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text

def parse_prob_from_text(s: str) -> Optional[float]:
    """
    CPC legend bins are like 33-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100.
    We parse the first range and use its midpoint.
    """
    m = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})\s*%?", s)
    if not m:
        m2 = re.search(r"(\d{2,3})\s*%?", s)
        if m2:
            p = float(m2.group(1)) / 100.0
            return clamp(p, 0.0, 1.0)
        return None
    a = float(m.group(1))
    b = float(m.group(2))
    p = (a + b) / 2.0 / 100.0
    return clamp(p, 0.0, 1.0)

def kml_coords_to_rings(coord_text: str) -> List[List[Tuple[float, float]]]:
    """
    KML coordinates are "lon,lat,alt lon,lat,alt ..."
    Often one outer ring; sometimes multiple rings. We'll treat each <coordinates> as a ring.
    """
    pts = []
    for token in coord_text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            lon = float(parts[0])
            lat = float(parts[1])
            pts.append((lon, lat))
    return [pts] if pts else []

def cpc_precip_probs_at(lat: float, lon: float) -> Dict[str, float]:
    """
    Returns dict with dry/normal/wet probabilities in [0,1].
    Default = equal chances.
    """
    kml = fetch_cpc_prcp_kml()
    # KML has namespaces. We'll be namespace-tolerant by stripping tags.
    try:
        root = ET.fromstring(kml.encode("utf-8"))
    except Exception:
        return {"dry": 1/3, "normal": 1/3, "wet": 1/3}

    x = float(lon)
    y = float(lat)

    # Iterate Placemarks, try to infer favored category + probability bin
    # We look at Placemark name/description for "Above/Below/Near/Normal/EC" and percentage.
    def text_of(el) -> str:
        if el is None:
            return ""
        return "".join(el.itertext()).strip()

    best = None  # (fav, p)
    for pm in root.iter():
        tag = pm.tag.lower()
        if not tag.endswith("placemark"):
            continue

        name = ""
        desc = ""
        coords_blocks = []

        for child in pm:
            ctag = child.tag.lower()
            if ctag.endswith("name"):
                name = text_of(child)
            elif ctag.endswith("description"):
                desc = text_of(child)
            # Find any Polygon->outerBoundaryIs->LinearRing->coordinates
            # Some KML uses MultiGeometry
        for coords in pm.iter():
            if coords.tag.lower().endswith("coordinates"):
                coords_blocks.append(text_of(coords))

        fav = None
        blob = f"{name} {desc}".lower()

        if "equal" in blob or "ec" in blob:
            # Equal chances region
            # We still need to ensure the point is inside this polygon
            fav = "ec"
        elif "above" in blob or "wet" in blob:
            fav = "wet"
        elif "below" in blob or "dry" in blob:
            fav = "dry"
        elif "near" in blob or "normal" in blob:
            fav = "normal"

        p = parse_prob_from_text(blob)  # may be None

        # Check containment: any coordinates ring containing point
        inside = False
        for cb in coords_blocks:
            rings = kml_coords_to_rings(cb)
            if rings and point_in_polygon(x, y, rings):
                inside = True
                break
        if not inside:
            continue

        if fav == "ec" or p is None:
            return {"dry": 1/3, "normal": 1/3, "wet": 1/3}

        # Favored category has prob p; other two split remainder
        rem = clamp(1.0 - p, 0.0, 1.0)
        other = rem / 2.0
        if fav == "dry":
            return {"dry": p, "normal": other, "wet": other}
        if fav == "wet":
            return {"dry": other, "normal": other, "wet": p}
        # normal favored
        return {"dry": other, "normal": p, "wet": other}

    # Fallback: equal chances if not found
    return {"dry": 1/3, "normal": 1/3, "wet": 1/3}

# ---- CPC Seasonal Drought Outlook category at point ----
def cpc_sdo_layer_candidates() -> List[int]:
    # We avoid brittle dependence on a single layer id; try common ones.
    # (Service can change; we try several indices fast.)
    return [4, 3, 2, 1, 0]

def cpc_sdo_query_layer(layer_id: int, lat: float, lon: float) -> Optional[dict]:
    url = f"{CPC_SDO_SERVICE}/{layer_id}/query"
    params = {
        "f": "pjson",
        "where": "1=1",
        "geometryType": "esriGeometryPoint",
        "geometry": f"{lon},{lat}",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "false",
    }
    try:
        js = http_get_json(url, params=params)
    except Exception:
        return None
    feats = js.get("features") or []
    if not feats:
        return None
    return feats[0].get("attributes") or None

def normalize_sdo_label(attrs: dict) -> Optional[str]:
    """
    CPC SDO commonly uses categories like:
    - Drought persists / continues
    - Drought develops
    - Drought improves
    - Drought removed
    We try to extract something usable from attributes.
    """
    if not attrs:
        return None
    # Try likely attribute keys
    for k in ["Outlook", "OUTLOOK", "CATEGORY", "CAT", "TYPE", "LABEL", "CLASS", "DESCRIPT"]:
        if k in attrs and attrs[k] is not None:
            s = str(attrs[k]).strip()
            if s:
                return s
    # Otherwise scan all values for keywords
    blob = " ".join([str(v) for v in attrs.values() if v is not None]).lower()
    if "develop" in blob:
        return "Develops"
    if "persist" in blob or "contin" in blob:
        return "Persists"
    if "improv" in blob:
        return "Improves"
    if "remov" in blob:
        return "Removed"
    return None

def cpc_sdo_at(lat: float, lon: float) -> Optional[str]:
    for lid in cpc_sdo_layer_candidates():
        attrs = cpc_sdo_query_layer(lid, lat, lon)
        if not attrs:
            continue
        label = normalize_sdo_label(attrs)
        if label:
            return label
    return None

# ---- Convert components into a 90-day drought probability ----
def drought_base_from_usdm(cat: Optional[str]) -> float:
    # If no drought polygon found, base is low but non-zero
    if cat is None:
        return 0.12
    # D0..D4 base probability for "some drought impact persisting/occurring"
    return {
        "D0": 0.25,
        "D1": 0.40,
        "D2": 0.55,
        "D3": 0.70,
        "D4": 0.82,
    }.get(cat, 0.12)

def drought_adjust_from_sdo(label: Optional[str]) -> Tuple[float, str]:
    if not label:
        return 0.0, "—"
    s = label.lower()
    if "develop" in s:
        return +0.18, "Develops"
    if "persist" in s or "contin" in s:
        return +0.10, "Persists"
    if "improv" in s:
        return -0.12, "Improves"
    if "remov" in s:
        return -0.22, "Removed"
    return 0.0, label

def drought_adjust_from_precip(dry: float, normal: float, wet: float) -> float:
    # Stronger dry tilt => higher drought odds, stronger wet tilt => lower odds
    # Baseline equal chances 0.333
    dry_tilt = dry - (1/3)
    wet_tilt = wet - (1/3)
    # weight tilts into adjustment
    return clamp((dry_tilt - wet_tilt) * 0.75, -0.25, 0.25)

class DroughtRequest(BaseModel):
    city: str

@app.post("/api/drought")
def api_drought(req: DroughtRequest) -> Dict[str, Any]:
    code = req.city.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    lat = float(info["lat"])
    lon = float(info["lon"])

    # 1) USDM current drought category
    usdm_cat = usdm_category_at(lat, lon)

    # 2) CPC 3-month precip outlook probabilities
    pr = cpc_precip_probs_at(lat, lon)
    dry = float(pr["dry"])
    normal = float(pr["normal"])
    wet = float(pr["wet"])

    # 3) CPC seasonal drought outlook category (if available at point)
    sdo_label_raw = cpc_sdo_at(lat, lon)
    sdo_adj, sdo_label = drought_adjust_from_sdo(sdo_label_raw)

    # Combine
    base = drought_base_from_usdm(usdm_cat)
    pr_adj = drought_adjust_from_precip(dry, normal, wet)

    expected = clamp(base + sdo_adj + pr_adj, 0.0, 1.0)

    # Also provide a simple "signal" string
    # (not a bet rec, just a quick confidence label)
    if expected >= 0.70:
        lvl = "HIGH"
    elif expected >= 0.45:
        lvl = "MED"
    else:
        lvl = "LOW"

    return {
        "city": code,
        "city_name": info["name"],
        "expected_drought_90d": expected,
        "level": lvl,
        "components": {
            "usdm_category": usdm_cat or "None",
            "precip_probs": {"dry": dry, "normal": normal, "wet": wet},
            "sdo": sdo_label,
            "adjustments": {
                "base": base,
                "sdo_adj": sdo_adj,
                "precip_adj": pr_adj,
            },
        },
        "sources": {
            "usdm_zip": USDM_ZIP_URL,
            "cpc_precip_kml": CPC_PRCP_KML_URL,
            "cpc_sdo_service": CPC_SDO_SERVICE,
        }
    }
