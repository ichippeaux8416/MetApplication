# drought_long.py
"""
Pull CPC/USDM long-range drought/precip signals for a point location and turn them into a
simple "expected drought through ~90 days" probability + expected drought level.

Designed for your MetApplication project (Denver / NYC / Dallas / Chicago).

Data sources (downloaded at runtime):
- US Drought Monitor (USDM) shapefiles (current drought category D0-D4)
  https://droughtmonitor.unl.edu/data/shapefiles_m/
- CPC 3-Month Precipitation Outlook shapefile (probability category + value)
  https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/monthlyupdate/monthupd_prcp_latest.zip
- CPC Seasonal Drought Outlook (SDO) archive (we scrape newest SHP zip link)
  https://www.cpc.ncep.noaa.gov/products/expert_assessment/sdo_archive.php

DEPENDENCIES:
  pip install requests python-dateutil pyshp

Notes:
- CPC precip outlook polygons typically encode a "favored category" (Above/Below/Normal)
  plus a probability value for that category. We convert to (dry/normal/wet) by distributing
  the remaining probability evenly across the other two categories (a reasonable approximation).
- SDO polygons categorize expected drought development/persistence/removal/improvement.
- We convert those signals + current drought category into:
    * expected_drought_90d (0..1)
    * expected_level (0..4) where 0=D0, 1=D1 ... 4=D4 (with "None" treated separately)
  using a transparent heuristic model (not an official CPC product).

This module is intentionally self-contained so server.py can import it without changing anything else.
"""

from __future__ import annotations

import io
import os
import re
import math
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import requests

try:
    import shapefile  # pyshp
except Exception as e:
    raise RuntimeError(
        "Missing dependency: pyshp. Install with: pip install pyshp"
    ) from e

# ---------------- CONFIG ----------------

USER_AGENT = "metapplication/1.0 (contact: you@example.com)"
TIMEOUT = 20

CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
}

# Where to cache downloaded zips (Render: /tmp is fine)
CACHE_DIR = os.environ.get("METAPP_CACHE_DIR", "/tmp/metapplication_cache")

# USDM latest zip is discovered by scraping the directory listing
USDM_DIR = "https://droughtmonitor.unl.edu/data/shapefiles_m/"

# CPC latest 3-month precip outlook zip (monthly update package)
CPC_PRCP_3MO_ZIP = "https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/monthlyupdate/monthupd_prcp_latest.zip"

# CPC SDO archive page; we scrape the newest "SHP (zip)" link
CPC_SDO_ARCHIVE = "https://www.cpc.ncep.noaa.gov/products/expert_assessment/sdo_archive.php"

# Cache TTLs (seconds)
TTL_USDM = 6 * 3600
TTL_CPC = 12 * 3600

# ---------------- HTTP ----------------

_session = requests.Session()


def _http_get(url: str) -> requests.Response:
    r = _session.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cache_path(name: str) -> str:
    _ensure_dir(CACHE_DIR)
    return os.path.join(CACHE_DIR, name)


def _is_fresh(path: str, ttl: int) -> bool:
    try:
        st = os.stat(path)
        return (time.time() - st.st_mtime) < ttl and st.st_size > 0
    except Exception:
        return False


def _download_to_cache(url: str, cache_name: str, ttl: int) -> str:
    path = _cache_path(cache_name)
    if _is_fresh(path, ttl):
        return path

    r = _http_get(url)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


# ---------------- Geometry (point-in-polygon) ----------------
# We avoid geopandas/shapely to keep dependencies light and compatible on Render.

def _point_in_ring(x: float, y: float, ring: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm for a single ring (closed or not).
    Returns True if point is inside ring.
    """
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x0, y0 = ring[0]
    for i in range(1, n + 1):
        x1, y1 = ring[i % n]
        # Check if edge crosses the horizontal ray at y
        if ((y0 > y) != (y1 > y)):
            # Compute x intersection
            x_int = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-15) + x0
            if x < x_int:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def _shape_contains_point(shp, lon: float, lat: float) -> bool:
    """
    Handles multi-part polygons. Treats each part as a ring.
    For CPC/USDM polygons this is usually adequate.
    """
    pts = shp.points
    parts = list(shp.parts) + [len(pts)]
    # If any ring contains point, return True
    for i in range(len(parts) - 1):
        ring = pts[parts[i] : parts[i + 1]]
        if _point_in_ring(lon, lat, ring):
            return True
    return False


# ---------------- Shapefile helpers ----------------

@dataclass
class ShapeMatch:
    record: Dict[str, Any]
    fields: List[str]


def _read_shapefile_from_zip(zip_path: str) -> Tuple[shapefile.Reader, List[str]]:
    """
    Reads the first .shp found in a zip, loading all companion files into memory.
    Returns (Reader, field_names).
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        shp_names = [n for n in z.namelist() if n.lower().endswith(".shp")]
        if not shp_names:
            raise RuntimeError(f"No .shp found in zip: {zip_path}")
        shp_name = shp_names[0]
        base = shp_name[:-4]

        def read_member(ext: str) -> io.BytesIO:
            name = base + ext
            data = z.read(name)
            return io.BytesIO(data)

        shx = read_member(".shx")
        shp = read_member(".shp")
        dbf = read_member(".dbf")
        # .prj optional
        reader = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
        field_names = [f[0] for f in reader.fields[1:]]  # skip DeletionFlag
        return reader, field_names


def _match_polygon_record(zip_path: str, lon: float, lat: float) -> Optional[ShapeMatch]:
    """
    Returns the first polygon record containing the point (lon,lat).
    """
    reader, fields = _read_shapefile_from_zip(zip_path)

    # records are list-like; convert to dict with field names
    for shp_rec in reader.iterShapeRecords():
        shp = shp_rec.shape
        if shp.shapeType not in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM):
            continue
        if not _shape_contains_point(shp, lon, lat):
            continue
        rec = {fields[i]: shp_rec.record[i] for i in range(len(fields))}
        return ShapeMatch(record=rec, fields=fields)
    return None


# ---------------- USDM (current drought) ----------------

_USDM_ZIP_RE = re.compile(r'USDM_(\d{8})_M\.zip', re.IGNORECASE)


def _find_latest_usdm_zip_url() -> str:
    """
    Scrape USDM directory listing and return URL to the latest USDM_YYYYMMDD_M.zip.
    """
    html = _http_get(USDM_DIR).text
    dates = _USDM_ZIP_RE.findall(html)
    if not dates:
        raise RuntimeError("Could not find USDM zip files in directory listing.")
    latest = max(dates)
    return f"{USDM_DIR}USDM_{latest}_M.zip"


def _parse_usdm_category(rec: Dict[str, Any]) -> Optional[str]:
    """
    USDM shapefile usually has field 'DM' with values like 'D0'...'D4' or 'None'.
    """
    # common candidates
    for key in ("DM", "dm", "DROUGHT", "D0_D4", "CAT"):
        if key in rec:
            v = str(rec[key]).strip()
            if v:
                return v
    # fallback: scan values
    for v in rec.values():
        sv = str(v).strip()
        if sv in ("D0", "D1", "D2", "D3", "D4", "None", "NONE", "No"):
            return sv
    return None


def get_current_drought_usdm(lat: float, lon: float) -> Dict[str, Any]:
    """
    Returns current drought category for a point from USDM polygons.
    """
    url = _find_latest_usdm_zip_url()
    zip_path = _download_to_cache(url, "usdm_latest.zip", TTL_USDM)

    m = _match_polygon_record(zip_path, lon, lat)
    if not m:
        return {"found": False, "category": None, "source": "USDM", "note": "No polygon match"}

    cat = _parse_usdm_category(m.record)
    # Normalize
    if cat is not None:
        cat = cat.upper().replace(" ", "")
        if cat == "NONE":
            cat = "None"
    return {"found": True, "category": cat, "source": "USDM", "raw": m.record}


# ---------------- CPC 3-month precip outlook ----------------

def _parse_cpc_precip_category_and_prob(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    CPC outlook shapefiles commonly encode:
      - a category code (A/B/N or Above/Below/Normal)
      - a probability value for the favored category (e.g., 33..60)
    We try to infer both robustly.
    """
    # Try category fields
    cat = None
    for k in ("CAT", "cat", "CATEGORY", "FCST", "FAVCAT", "CLASS"):
        if k in rec:
            cat = str(rec[k]).strip()
            break

    # Try probability fields
    prob = None
    for k in ("PROB", "prob", "PERCENT", "PCT", "VALUE", "VAL"):
        if k in rec:
            try:
                prob = float(rec[k])
                break
            except Exception:
                pass

    # If category isn't obvious, try scanning string values for A/B/N tokens
    if cat is None or cat == "":
        for v in rec.values():
            sv = str(v).strip().upper()
            if sv in ("A", "ABOVE", "AN", "WET"):
                cat = "A"
                break
            if sv in ("B", "BELOW", "BN", "DRY"):
                cat = "B"
                break
            if sv in ("N", "EC", "NORMAL", "NEAR", "EQUAL"):
                cat = "N"
                break

    # Normalize cat to A/B/N if possible
    if cat is not None:
        cu = cat.strip().upper()
        if cu in ("A", "ABOVE", "AN", "WET"):
            cat = "A"
        elif cu in ("B", "BELOW", "BN", "DRY"):
            cat = "B"
        elif cu in ("N", "EC", "NORMAL", "NEAR", "EQUAL"):
            cat = "N"
        else:
            # unknown
            cat = None

    # Normalize prob to 0..100
    if prob is not None:
        if prob <= 1.0:
            prob = prob * 100.0
        prob = max(0.0, min(100.0, prob))

    return cat, prob


def get_precip_outlook_3mo(lat: float, lon: float) -> Dict[str, Any]:
    """
    Returns approximate (dry, normal, wet) probabilities for the next 3 months.
    """
    zip_path = _download_to_cache(CPC_PRCP_3MO_ZIP, "cpc_prcp_3mo_latest.zip", TTL_CPC)

    m = _match_polygon_record(zip_path, lon, lat)
    if not m:
        # If no polygon hit, assume equal chances
        return {
            "found": False,
            "dry": 1 / 3,
            "normal": 1 / 3,
            "wet": 1 / 3,
            "category": "EC",
            "favored_prob": None,
            "source": "CPC_3MO_PRCP",
            "note": "No polygon match (using equal chances).",
        }

    cat, prob = _parse_cpc_precip_category_and_prob(m.record)

    # Convert favored-category probability to a full tri-prob vector
    if cat is None or prob is None:
        # fallback: equal chances
        dry = normal = wet = 1 / 3
        cat_label = "EC"
        favored_prob = None
    else:
        favored = prob / 100.0
        remaining = max(0.0, 1.0 - favored)
        split = remaining / 2.0
        if cat == "A":  # wet favored
            wet, dry, normal = favored, split, split
            cat_label = "Wet favored"
        elif cat == "B":  # dry favored
            dry, wet, normal = favored, split, split
            cat_label = "Dry favored"
        else:  # N or EC
            dry = normal = wet = 1 / 3
            cat_label = "Equal chances"
        favored_prob = favored

    return {
        "found": True,
        "dry": float(dry),
        "normal": float(normal),
        "wet": float(wet),
        "category": cat_label,
        "favored_prob": favored_prob,
        "source": "CPC_3MO_PRCP",
        "raw": m.record,
    }


# ---------------- CPC Seasonal Drought Outlook (SDO) ----------------

_SDO_SHP_ZIP_RE = re.compile(r'href="([^"]+\.zip)"[^>]*>\s*SHP\s*\(zip\)\s*<', re.IGNORECASE)


def _find_latest_sdo_zip_url() -> str:
    """
    Scrape CPC SDO archive page and return the first SHP(zip) link (newest at top).
    The href is sometimes relative.
    """
    html = _http_get(CPC_SDO_ARCHIVE).text
    m = _SDO_SHP_ZIP_RE.search(html)
    if not m:
        # fallback: grab any .zip on page that looks like 'sdo' and 'shp'
        zips = re.findall(r'href="([^"]+\.zip)"', html, re.IGNORECASE)
        for z in zips:
            if "sdo" in z.lower() and ("shp" in z.lower() or "gis" in z.lower()):
                m = re.match(r".*", z)
                return _absolutize(CPC_SDO_ARCHIVE, z)
        raise RuntimeError("Could not find SDO SHP zip link on CPC archive page.")
    href = m.group(1)
    return _absolutize(CPC_SDO_ARCHIVE, href)


def _absolutize(base: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    # simple join
    if href.startswith("/"):
        # derive origin
        origin = re.match(r"^(https?://[^/]+)", base)
        if origin:
            return origin.group(1) + href
        return href
    # relative to base dir
    base_dir = base.rsplit("/", 1)[0] + "/"
    return base_dir + href


def _parse_sdo_type(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    CPC SDO categories vary by field naming; we detect common patterns.
    Return (type, label) where type is one of:
      develop, persist, improve, remove, none, unknown
    """
    # Common fields: "CATEGORY", "CAT", "LABEL", "TYPE"
    text_blobs = []
    for k in ("CATEGORY", "CAT", "LABEL", "TYPE", "DESCRIPT", "DESC"):
        if k in rec and rec[k] is not None:
            text_blobs.append(str(rec[k]))
    # also scan all stringy values
    for v in rec.values():
        if isinstance(v, str) and len(v) <= 80:
            text_blobs.append(v)

    blob = " | ".join(text_blobs).lower()

    # Heuristics
    if "develop" in blob or "dev" in blob:
        return "develop", "Development"
    if "persist" in blob or "remain" in blob:
        return "persist", "Persistence"
    if "improv" in blob:
        return "improve", "Improvement"
    if "remove" in blob or "end" in blob:
        return "remove", "Removal"
    if "no drought" in blob or "none" in blob:
        return "none", "No drought"
    return "unknown", "—"


def get_seasonal_drought_outlook(lat: float, lon: float) -> Dict[str, Any]:
    """
    Returns SDO category for the point.
    """
    url = _find_latest_sdo_zip_url()
    zip_path = _download_to_cache(url, "cpc_sdo_latest.zip", TTL_CPC)

    m = _match_polygon_record(zip_path, lon, lat)
    if not m:
        return {
            "found": False,
            "type": "unknown",
            "label": "—",
            "source": "CPC_SDO",
            "note": "No polygon match",
        }

    t, label = _parse_sdo_type(m.record)
    return {"found": True, "type": t, "label": label, "source": "CPC_SDO", "raw": m.record}


# ---------------- Heuristic model ----------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _drought_level_from_cat(cat: Optional[str]) -> Tuple[Optional[int], str]:
    """
    Map USDM category to integer level, plus a label.
    None -> (None, "None")
    D0..D4 -> (0..4, "D0"..)
    """
    if not cat:
        return None, "—"
    if cat in ("None", "NO", "NODROUGHT"):
        return None, "None"
    m = re.match(r"^D([0-4])$", cat.strip().upper())
    if m:
        return int(m.group(1)), f"D{m.group(1)}"
    return None, str(cat)


def compute_expected_drought_90d(
    current_level: Optional[int],
    sdo_type: str,
    precip: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce:
      - expected_drought_90d probability (0..1)
      - expected_level (0..4) or None
    """
    # Base probability from current drought status
    # (If already in drought, probability of "drought continues sometime in next 90d" is higher.)
    if current_level is None:
        base = 0.35
    else:
        base = [0.55, 0.65, 0.75, 0.85, 0.92][max(0, min(4, current_level))]

    # SDO effect (from CPC)
    sdo_adj = {
        "develop": +0.55,
        "persist": +0.35,
        "improve": -0.35,
        "remove":  -0.55,
        "none":    -0.25,
        "unknown":  0.00,
    }.get(sdo_type, 0.0)

    # Precip outlook effect:
    # Use wet minus dry as a simple signed signal.
    dry = float(precip.get("dry", 1 / 3))
    wet = float(precip.get("wet", 1 / 3))
    precip_index = wet - dry  # positive => wetter bias
    precip_adj = -1.10 * precip_index  # wetter lowers drought odds, drier raises

    # Combine in logit space (transparent)
    score = _logit(base) + sdo_adj + precip_adj
    p = _sigmoid(score)

    # Expected level: shift current level by signal
    # If no current drought, "develop" can lift to D0/D1; "remove" keeps None.
    delta = 0
    if sdo_type == "develop":
        delta += 1
    elif sdo_type == "persist":
        delta += 0
    elif sdo_type == "improve":
        delta -= 1
    elif sdo_type == "remove":
        delta -= 2

    # precip_index negative => drier => increase severity a bit; positive => reduce
    if precip_index < -0.10:
        delta += 1
    elif precip_index > 0.10:
        delta -= 1

    if current_level is None:
        # developing drought: expected goes to D0 or D1 depending on p
        if p < 0.45:
            expected_level = None
        elif p < 0.70:
            expected_level = 0
        else:
            expected_level = 1
    else:
        expected_level = max(0, min(4, current_level + delta))

    return {
        "expected_drought_90d": float(p),
        "expected_level": expected_level,
        "model": {
            "base": base,
            "sdo_adj": sdo_adj,
            "precip_adj": precip_adj,
            "precip_index": precip_index,
            "score": score,
        },
    }


# ---------------- Public API ----------------

def evaluate_city(code: str) -> Dict[str, Any]:
    """
    Evaluate drought odds for one of the configured city codes (den/nyc/dal/chi).
    Returns a JSON-serializable dict.
    """
    code = code.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    lat = float(info["lat"])
    lon = float(info["lon"])

    usdm = get_current_drought_usdm(lat, lon)
    lvl, lvl_label = _drought_level_from_cat(usdm.get("category"))

    precip = get_precip_outlook_3mo(lat, lon)
    sdo = get_seasonal_drought_outlook(lat, lon)

    combined = compute_expected_drought_90d(lvl, sdo.get("type", "unknown"), precip)

    # Pretty labels
    expected_level = combined["expected_level"]
    if expected_level is None:
        expected_level_label = "None"
    else:
        expected_level_label = f"D{expected_level}"

    return {
        "city": code,
        "city_name": info["name"],
        "lat": lat,
        "lon": lon,
        "current_drought": {
            "category": usdm.get("category"),
            "level": lvl,
            "label": lvl_label,
            "found": bool(usdm.get("found")),
        },
        "drought_outlook": {
            "type": sdo.get("type"),
            "label": sdo.get("label"),
            "found": bool(sdo.get("found")),
        },
        "precip_outlook_3mo": {
            "dry": float(precip.get("dry", 1 / 3)),
            "normal": float(precip.get("normal", 1 / 3)),
            "wet": float(precip.get("wet", 1 / 3)),
            "category": precip.get("category"),
            "found": bool(precip.get("found")),
        },
        "expected_drought_90d": float(combined["expected_drought_90d"]),
        "expected_level": expected_level,
        "expected_level_label": expected_level_label,
        "debug": combined["model"],
        "notes": "Computed from USDM current drought + CPC 3-month precip outlook + CPC Seasonal Drought Outlook (heuristic model).",
    }


def evaluate_all_cities() -> Dict[str, Any]:
    """
    Convenience: evaluate all 4 cities.
    """
    out = {}
    for code in CITIES.keys():
        out[code] = evaluate_city(code)
    return out


if __name__ == "__main__":
    # Simple CLI run: python drought_long.py
    import json
    print(json.dumps(evaluate_all_cities(), indent=2))
