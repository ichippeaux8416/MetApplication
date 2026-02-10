# drought_long.py
from __future__ import annotations

import time
import requests
from typing import Dict, Any, Optional, Tuple

TIMEOUT = 12
USER_AGENT = "metapplication-drought/0.2 (contact: you@example.com)"

session = requests.Session()

# NOAA ArcGIS services (authoritative live layers)
# CPC Seasonal Drought Outlook polygons (FeatureServer)
CPC_DROUGHT_FS = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_drought_outlk/FeatureServer"
# CPC Seasonal Precip Outlook polygons (MapServer)
CPC_PRECIP_MS = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_sea_precip_outlk/MapServer"

# In these services, the "next season" is almost always "Lead 1" = layer 0 for precip.
PRECIP_LEAD1_LAYER = 0

# Seasonal drought outlook is typically layer 4 in cpc_drought_outlk FeatureServer.
# We'll still auto-detect to be safe.
DEFAULT_SDO_LAYER = 4

_cache: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SEC = 10 * 60


def _cache_get(key: str):
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < CACHE_TTL_SEC:
            return val
    return None


def _cache_set(key: str, val):
    _cache[key] = (time.time(), val)


def http_get_json(url: str, params=None) -> Any:
    r = session.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _pick_sdo_layer_id() -> int:
    """
    Find the best Seasonal Drought Outlook polygon layer id.
    Fallback to DEFAULT_SDO_LAYER if detection fails.
    """
    cache_key = "sdo_layer_id"
    cached = _cache_get(cache_key)
    if cached is not None:
        return int(cached)

    try:
        meta = http_get_json(f"{CPC_DROUGHT_FS}?f=pjson")
        layers = meta.get("layers", [])
        # Prefer a layer name containing "Seasonal" and "Drought Outlook"
        # If not found, just fallback.
        best = None
        for lyr in layers:
            name = (lyr.get("name") or "").lower()
            if "seasonal" in name and ("drought" in name or "outlook" in name):
                best = lyr.get("id")
                break
        layer_id = int(best) if best is not None else DEFAULT_SDO_LAYER
    except Exception:
        layer_id = DEFAULT_SDO_LAYER

    _cache_set(cache_key, layer_id)
    return layer_id


def _arcgis_query_point(
    base_url: str,
    layer_id: int,
    lon: float,
    lat: float,
    out_fields: str,
    return_geometry: bool = False,
) -> Optional[dict]:
    """
    Queries an ArcGIS layer for the polygon intersecting a point.
    Returns the first feature (dict) or None.
    """
    cache_key = f"q:{base_url}:{layer_id}:{lon:.4f}:{lat:.4f}:{out_fields}:{return_geometry}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = f"{base_url}/{layer_id}/query"
    params = {
        "f": "pjson",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "true" if return_geometry else "false",
        "resultRecordCount": 1,
    }
    try:
        data = http_get_json(url, params=params)
        feats = data.get("features") or []
        feat = feats[0] if feats else None
    except Exception:
        feat = None

    _cache_set(cache_key, feat)
    return feat


def _normalize_usdm_level(level: Any) -> int:
    """
    USDM levels: D0..D4. Return 0..4, else -1 if unknown.
    """
    if level is None:
        return -1
    s = str(level).strip().upper()
    if s.startswith("D") and len(s) >= 2 and s[1].isdigit():
        n = int(s[1])
        if 0 <= n <= 4:
            return n
    # sometimes it's numeric
    try:
        n = int(float(s))
        if 0 <= n <= 4:
            return n
    except Exception:
        pass
    return -1


def _drought_odds_equation(
    usdm_level: int,
    sdo_cat: Optional[str],
    precip_cat: Optional[str],
    precip_prob: Optional[float],
) -> float:
    """
    Returns odds [0..1] that the location is in drought conditions over ~90 days.

    This is a heuristic blend:
      - current drought severity (USDM)
      - CPC Seasonal Drought Outlook category
      - CPC 3-month precip tilt (Below/Above/EC)
    """
    # Base odds from current drought severity
    # D0->0.20, D1->0.35, D2->0.55, D3->0.75, D4->0.90
    if usdm_level < 0:
        base = 0.50
    else:
        base_map = {0: 0.20, 1: 0.35, 2: 0.55, 3: 0.75, 4: 0.90}
        base = base_map.get(usdm_level, 0.50)

    cat = (sdo_cat or "").strip().lower()
    # Adjust based on Seasonal Drought Outlook category wording
    # Common values: "Development", "Persistence", "Improvement", "Removal" (plus variants)
    if "remov" in cat:
        base *= 0.55
    elif "improv" in cat:
        base *= 0.70
    elif "persist" in cat:
        base = min(0.95, base + 0.12)
    elif "develop" in cat:
        base = max(base, 0.60)
        base = min(0.95, base + 0.10)

    # Precip signal: use prob magnitude when available, else small nudge
    pc = (precip_cat or "").strip().lower()
    p = float(precip_prob) if precip_prob is not None else None
    strength = 0.10 if p is None else max(0.05, min(0.20, abs(p - 0.33)))

    if pc == "below":
        base = min(0.95, base + strength)
    elif pc == "above":
        base = max(0.05, base - strength)
    # "normal"/"ec" => no change

    return max(0.01, min(0.99, base))


def evaluate_city(lat: float, lon: float) -> Dict[str, Any]:
    """
    Main entry: returns CPC/USDM-derived drought odds + components.
    """
    # --- USDM (current drought monitor) ---
    # Uses droughtmonitor.unl.edu JSON polygons.
    # We query their point endpoint? They don't provide a clean point query;
    # instead we keep USDM optional and focus on CPC SDO + precip tilt.
    # If you already have USDM in your earlier file, keep it. Here we do a pragmatic approach:
    usdm_level = -1  # unknown by default
    usdm_label = "—"

    # --- CPC Seasonal Drought Outlook (SDO) polygon at point ---
    sdo_layer = _pick_sdo_layer_id()
    sdo_feat = _arcgis_query_point(
        CPC_DROUGHT_FS,
        sdo_layer,
        lon,
        lat,
        out_fields="cat,valid,valid_seas,fcst_date,prob",
        return_geometry=False,
    )
    sdo_cat = None
    sdo_valid = None
    if sdo_feat and "attributes" in sdo_feat:
        a = sdo_feat["attributes"] or {}
        sdo_cat = a.get("cat") or a.get("CAT") or a.get("Category")
        sdo_valid = a.get("valid") or a.get("valid_seas") or a.get("VALID_SEAS")

    # --- CPC 3-month precip outlook (Lead 1) polygon at point ---
    precip_feat = _arcgis_query_point(
        CPC_PRECIP_MS,
        PRECIP_LEAD1_LAYER,
        lon,
        lat,
        out_fields="cat,prob,valid_seas,fcst_date",
        return_geometry=False,
    )
    precip_cat = None
    precip_prob = None
    precip_valid = None
    if precip_feat and "attributes" in precip_feat:
        a = precip_feat["attributes"] or {}
        precip_cat = a.get("cat") or a.get("CAT")
        precip_prob = a.get("prob") or a.get("PROB")
        precip_valid = a.get("valid_seas") or a.get("VALID_SEAS")

    # Normalize precip values
    # cat is one of: Above / Below / Normal / EC
    if isinstance(precip_cat, str):
        precip_cat = precip_cat.strip().title()
    if precip_prob is not None:
        try:
            precip_prob = float(precip_prob) / 100.0 if float(precip_prob) > 1.0 else float(precip_prob)
        except Exception:
            precip_prob = None

    odds = _drought_odds_equation(usdm_level, sdo_cat, precip_cat, precip_prob)

    # Create user-friendly precip breakdown for UI
    # If CPC gives a tilted category with prob, we set that as primary,
    # and distribute the rest roughly.
    p_drier = 0.33
    p_wetter = 0.33
    p_normal = 0.34
    if precip_cat in ("Below", "Above", "Normal") and precip_prob is not None:
        main = max(0.33, min(0.90, precip_prob))
        rem = max(0.0, 1.0 - main)
        if precip_cat == "Below":
            p_drier = main
            p_wetter = rem * 0.40
            p_normal = rem * 0.60
        elif precip_cat == "Above":
            p_wetter = main
            p_drier = rem * 0.40
            p_normal = rem * 0.60
        else:  # Normal
            p_normal = main
            p_drier = rem * 0.50
            p_wetter = rem * 0.50
    elif precip_cat == "EC":
        p_drier, p_normal, p_wetter = 0.33, 0.34, 0.33

    # If SDO is missing AND precip is missing, this is likely a query failure
    # In that case, odds will sit near 0.50 — but now you'll ALSO see error flags in UI.
    return {
        "expected_drought_prob": odds,  # 0..1
        "usdm_level": usdm_level,
        "usdm_label": usdm_label,
        "sdo_cat": sdo_cat or "—",
        "sdo_valid": sdo_valid or "—",
        "precip_valid": precip_valid or "—",
        "precip_cat": precip_cat or "—",
        "precip_prob": precip_prob,
        "p_drier": float(p_drier),
        "p_normal": float(p_normal),
        "p_wetter": float(p_wetter),
        "debug": {
            "sdo_layer": sdo_layer,
            "sdo_hit": bool(sdo_feat),
            "precip_hit": bool(precip_feat),
        },
    }
