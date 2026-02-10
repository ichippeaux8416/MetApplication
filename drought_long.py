# drought_long.py
# Long-term drought odds engine for MetApplication
#
# Uses CPC + USDM ArcGIS REST services (point query by lat/lon).
# No shapefile downloads, no forbidden directory scraping.
#
# Expected output shape (used by drought.html):
# {
#   "expected_drought_90d": float 0..1,
#   "components": {
#       "usdm_category": "None|D0|D1|D2|D3|D4",
#       "sdo": "<label>",
#       "precip_probs": {"dry":0..1,"normal":0..1,"wet":0..1}
#   },
#   "notes": "..."
# }

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


# -------------------------
# Services (ArcGIS REST)
# -------------------------
# CPC seasonal precip outlook (probabilities)
CPC_PRECIP_SERVICE = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_sea_precip_outlk/MapServer"
# CPC seasonal drought outlook
CPC_DROUGHT_SERVICE = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_drought_outlk/MapServer"

# US Drought Monitor current conditions (NDMC ArcGIS)
# This is a commonly-used public ArcGIS endpoint for USDM current layer.
# If NDMC ever changes it, you can update just this string.
USDM_CURRENT_SERVICE_CANDIDATES = [
    "https://gis.droughtmonitor.unl.edu/arcgis/rest/services/USDM/USDM_Current/MapServer",
    "https://gis.droughtmonitor.unl.edu/arcgis/rest/services/USDM/USDM_Current_Conditions/MapServer",
]

USER_AGENT = os.getenv("METAPP_UA", "metapplication/1.0 (contact: you@example.com)")
TIMEOUT = float(os.getenv("METAPP_TIMEOUT", "14"))

# Debug logging (set METAPP_DROUGHT_DEBUG=1 on Render)
DEBUG = os.getenv("METAPP_DROUGHT_DEBUG", "0") == "1"

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _log(msg: str):
    if DEBUG:
        print(f"[drought_long] {msg}", flush=True)


# -------------------------
# Helpers
# -------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        # ArcGIS often returns ints, floats, or numeric strings
        return float(v)
    except Exception:
        return None


def _arcgis_query_point(service_url: str, layer_id: int, lon: float, lat: float) -> Optional[Dict[str, Any]]:
    """
    Query a FeatureLayer by a point and return first feature's attributes.
    """
    url = f"{service_url}/{layer_id}/query"
    params = {
        "f": "json",
        "where": "1=1",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "false",
        "resultRecordCount": "1",
    }
    r = _session.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    feats = js.get("features") or []
    if not feats:
        return None
    attrs = feats[0].get("attributes") or {}
    return attrs


def _normalize_prob_triplet(dry: float, normal: float, wet: float) -> Dict[str, float]:
    dry = max(0.0, dry)
    normal = max(0.0, normal)
    wet = max(0.0, wet)
    s = dry + normal + wet
    if s <= 0:
        return {"dry": 1/3, "normal": 1/3, "wet": 1/3}
    return {"dry": dry / s, "normal": normal / s, "wet": wet / s}


# -------------------------
# CPC: 3-month precip outlook
# -------------------------
def get_cpc_precip_probs(lon: float, lat: float) -> Tuple[Dict[str, float], str]:
    """
    Returns ({dry,normal,wet}, notes)
    Uses Lead 1 layer (0) by default: this is the first upcoming 3-month season.
    """
    # Most deployments: layer 0 = Lead 1. If CPC changes, adjust layer id here.
    layer_id = 0

    attrs = _arcgis_query_point(CPC_PRECIP_SERVICE, layer_id, lon, lat)
    if not attrs:
        return {"dry": 1/3, "normal": 1/3, "wet": 1/3}, "CPC precip: no feature at point (fallback 33/33/33)."

    # Field names vary; we scan for likely keys.
    # Typical keys seen in CPC outlook layers include words like:
    # BELOW / ABOVE / NEAR, or B / A / N, or DRY / WET / NORMAL.
    keys = {k.lower(): k for k in attrs.keys()}

    def pick(*cands: str) -> Optional[float]:
        for c in cands:
            for lk, ok in keys.items():
                if c in lk:
                    v = _safe_float(attrs.get(ok))
                    if v is not None:
                        return v
        return None

    below = pick("below", "dry", "b_prob", "bprob", "prob_b", "pb")
    near = pick("near", "normal", "n_prob", "nprob", "prob_n", "pn")
    above = pick("above", "wet", "a_prob", "aprob", "prob_a", "pa")

    # Sometimes probabilities are stored as whole percents (e.g., 40) not fractions.
    # Detect and convert if needed.
    trip = [below, near, above]
    if all(v is not None for v in trip):
        b, n, a = float(below), float(near), float(above)
        # If values look like 0-100, convert to 0-1
        if max(b, n, a) > 1.5:
            b /= 100.0
            n /= 100.0
            a /= 100.0
        probs = _normalize_prob_triplet(b, n, a)
        return probs, "CPC precip: OK."

    # If we couldn't find explicit prob fields, fallback
    return {"dry": 1/3, "normal": 1/3, "wet": 1/3}, "CPC precip: fields not found (fallback 33/33/33)."


# -------------------------
# CPC: Seasonal Drought Outlook (SDO)
# -------------------------
def get_cpc_drought_outlook_label(lon: float, lat: float) -> Tuple[str, str]:
    """
    Returns (label, notes).
    Queries the CPC drought outlook map service.
    """
    # CPC drought service typically has layer 1 and 2 as feature layers (CONUS vs AK/HI/PR etc).
    # We'll try layer 1 first, then 2.
    for layer_id in (1, 2):
        try:
            attrs = _arcgis_query_point(CPC_DROUGHT_SERVICE, layer_id, lon, lat)
            if not attrs:
                continue

            # Heuristic extraction of label/category fields
            # Look for something like "CATEGORY", "LABEL", "OUTLOOK", "SOMETHING"
            best = None
            for k, v in attrs.items():
                lk = str(k).lower()
                if any(s in lk for s in ("label", "category", "outlook", "class", "cat")):
                    if isinstance(v, str) and v.strip():
                        best = v.strip()
                        break

            if best:
                return best, f"CPC drought outlook: OK (layer {layer_id})."

            # If only numeric code exists, map it
            # Common codes (varies): try to interpret:
            # 0 none, 1 improvement, 2 removal, 3 development, 4 persistence
            code = None
            for k, v in attrs.items():
                lk = str(k).lower()
                if "code" in lk or lk.endswith("cat") or "class" in lk:
                    code = _safe_float(v)
                    if code is not None:
                        break
            if code is not None:
                code_i = int(round(code))
                mapping = {
                    0: "None / No drought signal",
                    1: "Improvement likely",
                    2: "Drought removal likely",
                    3: "Drought development likely",
                    4: "Drought persistence likely",
                }
                return mapping.get(code_i, f"Unknown code {code_i}"), f"CPC drought outlook: OK (coded, layer {layer_id})."

        except Exception as e:
            _log(f"SDO layer {layer_id} failed: {e}")

    return "—", "CPC drought outlook: unavailable (—)."


# -------------------------
# USDM: Current drought category at point
# -------------------------
def get_usdm_current_category(lon: float, lat: float) -> Tuple[str, str]:
    """
    Returns (category, notes) where category in None/D0..D4.
    Tries a couple of known NDMC ArcGIS endpoints.
    """
    for svc in USDM_CURRENT_SERVICE_CANDIDATES:
        # USDM current layer is usually layer 0, but try 0..3 quickly.
        for layer_id in (0, 1, 2, 3):
            try:
                attrs = _arcgis_query_point(svc, layer_id, lon, lat)
                if not attrs:
                    continue

                # Look for a drought category field
                # Common names include: DM, Dm, drought, cat, category, USDM, etc.
                cat_val = None
                for k, v in attrs.items():
                    lk = str(k).lower()
                    if lk in ("dm", "d0", "drought", "category") or "dm" == lk or "drought" in lk or "usdm" in lk:
                        cat_val = v
                        break

                # If string like "D2"
                if isinstance(cat_val, str) and cat_val.strip():
                    s = cat_val.strip().upper()
                    if s in ("NONE", "N", "0"):
                        return "None", f"USDM current: OK ({svc}, layer {layer_id})."
                    if s.startswith("D") and len(s) == 2 and s[1].isdigit():
                        return s, f"USDM current: OK ({svc}, layer {layer_id})."

                # If numeric code, map:
                code = _safe_float(cat_val)
                if code is None:
                    # try any numeric-like field that looks like category
                    for k, v in attrs.items():
                        lk = str(k).lower()
                        if "dm" in lk or "cat" in lk or "class" in lk:
                            code = _safe_float(v)
                            if code is not None:
                                break

                if code is not None:
                    # Many services use: 0=None,1=D0,2=D1,...,5=D4
                    i = int(round(code))
                    mapping = {0: "None", 1: "D0", 2: "D1", 3: "D2", 4: "D3", 5: "D4"}
                    return mapping.get(i, "None"), f"USDM current: OK ({svc}, layer {layer_id})."

            except Exception as e:
                _log(f"USDM {svc} layer {layer_id} failed: {e}")

    return "None", "USDM current: unavailable (None)."


# -------------------------
# Drought odds model (simple, transparent)
# -------------------------
def compute_expected_drought_probability(
    usdm_cat: str,
    sdo_label: str,
    precip_probs: Dict[str, float],
) -> Tuple[float, float, str]:
    """
    Returns (p_drought_90d, expected_severity_0to4, notes)
    """

    # Base severity from current USDM category
    base_map = {"None": 0.0, "D0": 0.5, "D1": 1.0, "D2": 2.0, "D3": 3.0, "D4": 4.0}
    base = base_map.get(usdm_cat, 0.0)

    # Precip signal: wet - dry. Positive => wetter => lower drought risk
    dry = float(precip_probs.get("dry", 1/3))
    wet = float(precip_probs.get("wet", 1/3))
    precip_index = _clamp(wet - dry, -1.0, 1.0)

    # SDO signal mapping (heuristic)
    s = (sdo_label or "").lower()
    sdo_delta = 0.0
    if "development" in s:
        sdo_delta = +0.9
    elif "persistence" in s:
        sdo_delta = +0.5
    elif "improve" in s:
        sdo_delta = -0.4
    elif "removal" in s:
        sdo_delta = -0.8
    elif s.strip() in ("—", "-", "") or "none" in s:
        sdo_delta = 0.0

    # Convert precip_index to a severity delta: wetter => negative delta, drier => positive delta
    precip_delta = -1.2 * precip_index

    expected_sev = _clamp(base + sdo_delta + precip_delta, 0.0, 4.0)

    # Convert expected severity to probability of having drought (>= D1-ish) in ~90 days.
    # Logistic curve: center near 0.8; steeper with k.
    k = 1.35
    center = 0.8
    p = 1.0 / (1.0 + math.exp(-k * (expected_sev - center)))
    p = _clamp(p, 0.0, 1.0)

    notes = f"Model: base={base:.2f} sdoΔ={sdo_delta:+.2f} precipΔ={precip_delta:+.2f} → sev={expected_sev:.2f}."
    return p, expected_sev, notes


# -------------------------
# Public entrypoint used by server.py
# -------------------------
def evaluate_city(city_code: str, city_name: str, lat: float, lon: float) -> Dict[str, Any]:
    """
    Main function your /api/drought handler should call.
    """
    started = time.time()

    precip_probs, precip_note = get_cpc_precip_probs(lon, lat)
    sdo_label, sdo_note = get_cpc_drought_outlook_label(lon, lat)
    usdm_cat, usdm_note = get_usdm_current_category(lon, lat)

    p, expected_sev, model_note = compute_expected_drought_probability(usdm_cat, sdo_label, precip_probs)

    # Never allow NaN through
    if not isinstance(p, float) or math.isnan(p) or math.isinf(p):
        p = 0.5

    elapsed_ms = int((time.time() - started) * 1000)

    return {
        "city": city_code,
        "city_name": city_name,
        "expected_drought_90d": float(p),
        "expected_severity_0to4": float(expected_sev),
        "components": {
            "usdm_category": usdm_cat,
            "sdo": sdo_label,
            "precip_probs": precip_probs,
        },
        "notes": f"{model_note} {usdm_note} {sdo_note} {precip_note} ({elapsed_ms}ms)",
        "sources": {
            "cpc_precip_service": CPC_PRECIP_SERVICE,
            "cpc_drought_service": CPC_DROUGHT_SERVICE,
            "usdm_service_candidates": USDM_CURRENT_SERVICE_CANDIDATES,
        },
    }
