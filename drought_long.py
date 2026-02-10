# drought_long.py
"""
Real CPC/USDM pull for "Long Term Drought Odds" used by drought.html.

drought.html expects POST /api/drought -> JSON:
{
  "city": "chi",
  "city_name": "Chicago",
  "expected_drought_90d": 0.42,
  "components": {
    "usdm_category": "D1" | "D0" | "None" | "...",
    "sdo": "Development" | "Persistence" | "Improvement" | "Removal" | "—",
    "precip_probs": {"dry":0.33,"normal":0.34,"wet":0.33}
  },
  "notes": "..."
}

We pull:
- US Drought Monitor (USDM) shapefile (current drought category polygon)
- CPC 3-month precipitation outlook shapefile (favored category + probability)
- CPC Seasonal Drought Outlook (SDO) shapefile (development/persistence/improve/removal)

Dependencies:
  pip install requests python-dateutil pyshp

No geopandas/shapely required.
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
    raise RuntimeError("Missing dependency: pyshp. Install with: pip install pyshp") from e

# ---------------- CONFIG ----------------

USER_AGENT = "metapplication/1.0 (contact: you@example.com)"
TIMEOUT = 25

CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
}

CACHE_DIR = os.environ.get("METAPP_CACHE_DIR", "/tmp/metapplication_cache")

USDM_DIR = "https://droughtmonitor.unl.edu/data/shapefiles_m/"
CPC_PRCP_3MO_ZIP = "https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/monthlyupdate/monthupd_prcp_latest.zip"
CPC_SDO_ARCHIVE = "https://www.cpc.ncep.noaa.gov/products/expert_assessment/sdo_archive.php"

TTL_USDM = 6 * 3600
TTL_CPC = 12 * 3600

_session = requests.Session()

# ---------------- Cache/HTTP ----------------

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

def _http_get(url: str) -> requests.Response:
    r = _session.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r

def _download_to_cache(url: str, cache_name: str, ttl: int) -> str:
    path = _cache_path(cache_name)
    if _is_fresh(path, ttl):
        return path
    r = _http_get(url)
    with open(path, "wb") as f:
        f.write(r.content)
    return path

# ---------------- Minimal point-in-polygon ----------------

def _point_in_ring(x: float, y: float, ring: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x0, y0 = ring[0]
    for i in range(1, n + 1):
        x1, y1 = ring[i % n]
        if ((y0 > y) != (y1 > y)):
            x_int = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-15) + x0
            if x < x_int:
                inside = not inside
        x0, y0 = x1, y1
    return inside

def _shape_contains_point(shp, lon: float, lat: float) -> bool:
    pts = shp.points
    parts = list(shp.parts) + [len(pts)]
    for i in range(len(parts) - 1):
        ring = pts[parts[i] : parts[i + 1]]
        if _point_in_ring(lon, lat, ring):
            return True
    return False

# ---------------- Shapefile from zip ----------------

@dataclass
class ShapeMatch:
    record: Dict[str, Any]
    fields: List[str]

def _read_shapefile_from_zip(zip_path: str) -> Tuple[shapefile.Reader, List[str]]:
    with zipfile.ZipFile(zip_path, "r") as z:
        shp_names = [n for n in z.namelist() if n.lower().endswith(".shp")]
        if not shp_names:
            raise RuntimeError(f"No .shp found in zip: {zip_path}")
        shp_name = shp_names[0]
        base = shp_name[:-4]

        def read_member(ext: str) -> io.BytesIO:
            name = base + ext
            return io.BytesIO(z.read(name))

        reader = shapefile.Reader(shp=read_member(".shp"), shx=read_member(".shx"), dbf=read_member(".dbf"))
        field_names = [f[0] for f in reader.fields[1:]]  # skip DeletionFlag
        return reader, field_names

def _match_polygon_record(zip_path: str, lon: float, lat: float) -> Optional[ShapeMatch]:
    reader, fields = _read_shapefile_from_zip(zip_path)
    for sr in reader.iterShapeRecords():
        shp = sr.shape
        if shp.shapeType not in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM):
            continue
        if not _shape_contains_point(shp, lon, lat):
            continue
        rec = {fields[i]: sr.record[i] for i in range(len(fields))}
        return ShapeMatch(record=rec, fields=fields)
    return None

# ---------------- USDM ----------------

_USDM_ZIP_RE = re.compile(r'USDM_(\d{8})_M\.zip', re.IGNORECASE)

def _find_latest_usdm_zip_url() -> str:
    html = _http_get(USDM_DIR).text
    dates = _USDM_ZIP_RE.findall(html)
    if not dates:
        raise RuntimeError("Could not find USDM zip files.")
    latest = max(dates)
    return f"{USDM_DIR}USDM_{latest}_M.zip"

def _parse_usdm_category(rec: Dict[str, Any]) -> Optional[str]:
    # USDM commonly uses field "DM" with D0..D4 or "None"
    for key in ("DM", "dm", "DROUGHT", "CAT", "CATEGORY"):
        if key in rec:
            v = str(rec[key]).strip()
            if v:
                return v
    # fallback scan
    for v in rec.values():
        sv = str(v).strip()
        if sv in ("D0", "D1", "D2", "D3", "D4", "None", "NONE"):
            return sv
    return None

def get_usdm_category(lat: float, lon: float) -> Tuple[str, str]:
    """
    Returns (category, note)
    category like D0..D4 or None or —
    """
    url = _find_latest_usdm_zip_url()
    zip_path = _download_to_cache(url, "usdm_latest.zip", TTL_USDM)

    m = _match_polygon_record(zip_path, lon, lat)
    if not m:
        return "—", "USDM: no polygon match"
    cat = _parse_usdm_category(m.record)
    if not cat:
        return "—", "USDM: category not found in record"
    cat = cat.strip()
    if cat.upper() == "NONE":
        cat = "None"
    return cat, "USDM: ok"

def _drought_level_from_cat(cat: str) -> Optional[int]:
    if not cat or cat in ("—",):
        return None
    if cat == "None":
        return None
    m = re.match(r"^D([0-4])$", cat.strip().upper())
    if not m:
        return None
    return int(m.group(1))

# ---------------- CPC 3-month precip outlook ----------------

def _parse_cpc_precip_cat_prob(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    Tries to find:
      cat in A/B/N (Above/Wet, Below/Dry, Normal/Equal)
      prob in percent for favored cat
    """
    cat = None
    for k in ("CAT", "cat", "CATEGORY", "FCST", "FAVCAT", "CLASS"):
        if k in rec and rec[k] is not None:
            cat = str(rec[k]).strip()
            break

    prob = None
    for k in ("PROB", "prob", "PERCENT", "PCT", "VALUE", "VAL"):
        if k in rec and rec[k] is not None:
            try:
                prob = float(rec[k])
                break
            except Exception:
                pass

    if cat is not None:
        cu = cat.upper()
        if cu in ("A", "ABOVE", "WET", "AN"):
            cat = "A"
        elif cu in ("B", "BELOW", "DRY", "BN"):
            cat = "B"
        elif cu in ("N", "EC", "NORMAL", "EQUAL"):
            cat = "N"
        else:
            cat = None

    if prob is not None:
        if prob <= 1.0:
            prob *= 100.0
        prob = max(0.0, min(100.0, prob))

    return cat, prob

def get_cpc_precip_probs(lat: float, lon: float) -> Tuple[Dict[str, float], str]:
    """
    Returns (probs, note) where probs has dry/normal/wet in 0..1
    """
    zip_path = _download_to_cache(CPC_PRCP_3MO_ZIP, "cpc_prcp_3mo_latest.zip", TTL_CPC)
    m = _match_polygon_record(zip_path, lon, lat)

    if not m:
        return {"dry": 1/3, "normal": 1/3, "wet": 1/3}, "CPC precip: no polygon match (equal chances)"

    cat, prob = _parse_cpc_precip_cat_prob(m.record)
    if cat is None or prob is None:
        return {"dry": 1/3, "normal": 1/3, "wet": 1/3}, "CPC precip: fields not detected (equal chances)"

    favored = prob / 100.0
    rem = max(0.0, 1.0 - favored)
    split = rem / 2.0

    if cat == "B":  # dry favored
        probs = {"dry": favored, "normal": split, "wet": split}
        note = f"CPC precip: dry favored {prob:.0f}%"
    elif cat == "A":  # wet favored
        probs = {"wet": favored, "normal": split, "dry": split}
        note = f"CPC precip: wet favored {prob:.0f}%"
    else:
        probs = {"dry": 1/3, "normal": 1/3, "wet": 1/3}
        note = "CPC precip: equal chances"
    return probs, note

# ---------------- CPC Seasonal Drought Outlook (SDO) ----------------

_SDO_SHP_ZIP_RE = re.compile(r'href="([^"]+\.zip)"[^>]*>\s*SHP\s*\(zip\)\s*<', re.IGNORECASE)

def _absolutize(base: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        origin = re.match(r"^(https?://[^/]+)", base)
        return (origin.group(1) + href) if origin else href
    base_dir = base.rsplit("/", 1)[0] + "/"
    return base_dir + href

def _find_latest_sdo_zip_url() -> str:
    html = _http_get(CPC_SDO_ARCHIVE).text
    m = _SDO_SHP_ZIP_RE.search(html)
    if m:
        return _absolutize(CPC_SDO_ARCHIVE, m.group(1))

    # fallback: first zip containing "sdo"
    zips = re.findall(r'href="([^"]+\.zip)"', html, re.IGNORECASE)
    for z in zips:
        if "sdo" in z.lower():
            return _absolutize(CPC_SDO_ARCHIVE, z)
    raise RuntimeError("Could not find SDO zip link.")

def _parse_sdo_label(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (type, label)
    type: develop/persist/improve/remove/unknown
    """
    blobs: List[str] = []
    for k in ("CATEGORY", "CAT", "LABEL", "TYPE", "DESC", "DESCRIPT"):
        if k in rec and rec[k] is not None:
            blobs.append(str(rec[k]))
    for v in rec.values():
        if isinstance(v, str) and 0 < len(v) <= 80:
            blobs.append(v)

    blob = " | ".join(blobs).lower()

    if "develop" in blob or "dev" in blob:
        return "develop", "Development"
    if "persist" in blob or "remain" in blob:
        return "persist", "Persistence"
    if "improv" in blob:
        return "improve", "Improvement"
    if "remove" in blob or "end" in blob:
        return "remove", "Removal"
    return "unknown", "—"

def get_cpc_sdo(lat: float, lon: float) -> Tuple[str, str]:
    """
    Returns (label, note) where label is one of:
    Development / Persistence / Improvement / Removal / —
    """
    url = _find_latest_sdo_zip_url()
    zip_path = _download_to_cache(url, "cpc_sdo_latest.zip", TTL_CPC)
    m = _match_polygon_record(zip_path, lon, lat)
    if not m:
        return "—", "CPC SDO: no polygon match"
    _, label = _parse_sdo_label(m.record)
    return label, "CPC SDO: ok"

# ---------------- Heuristic drought probability model ----------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))

def expected_drought_probability_90d(
    usdm_cat: str,
    sdo_label: str,
    precip_probs: Dict[str, float],
) -> float:
    """
    Transparent heuristic:
    - base from current drought category
    - adjust using SDO label
    - adjust using precip wet-dry tilt
    """
    lvl = _drought_level_from_cat(usdm_cat)

    if lvl is None:
        base = 0.35
    else:
        base = [0.55, 0.65, 0.75, 0.85, 0.92][max(0, min(4, lvl))]

    sdo_adj = {
        "Development": +0.55,
        "Persistence": +0.35,
        "Improvement": -0.35,
        "Removal": -0.55,
        "—": 0.00,
    }.get(sdo_label, 0.0)

    dry = float(precip_probs.get("dry", 1/3))
    wet = float(precip_probs.get("wet", 1/3))
    precip_index = wet - dry  # + => wetter
    precip_adj = -1.10 * precip_index

    score = _logit(base) + sdo_adj + precip_adj
    return float(_sigmoid(score))

# ---------------- Main callable used by server.py ----------------

def evaluate_city(city_code: str) -> Dict[str, Any]:
    code = city_code.strip().lower()
    if code not in CITIES:
        return {"error": "unknown_city", "allowed": list(CITIES.keys())}

    info = CITIES[code]
    lat = float(info["lat"])
    lon = float(info["lon"])

    usdm_cat, usdm_note = get_usdm_category(lat, lon)
    sdo_label, sdo_note = get_cpc_sdo(lat, lon)
    precip_probs, pr_note = get_cpc_precip_probs(lat, lon)

    p = expected_drought_probability_90d(usdm_cat, sdo_label, precip_probs)

    return {
        "city": code,
        "city_name": info["name"],
        "expected_drought_90d": p,
        "components": {
            "usdm_category": usdm_cat,
            "sdo": sdo_label,
            "precip_probs": {
                "dry": float(precip_probs.get("dry", 1/3)),
                "normal": float(precip_probs.get("normal", 1/3)),
                "wet": float(precip_probs.get("wet", 1/3)),
            },
        },
        "notes": f"{usdm_note} | {sdo_note} | {pr_note}",
    }


if __name__ == "__main__":
    import json
    out = {k: evaluate_city(k) for k in CITIES.keys()}
    print(json.dumps(out, indent=2))
