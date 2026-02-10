# drought_long.py
# Long-term drought odds engine for MetApplication
#
# Uses CPC + USDM ArcGIS REST services (point query by lat/lon).
# No shapefile downloads, no forbidden directory scraping.
#
# server.py expects: evaluate_city(lat, lon) -> dict
# drought.html expects JSON shape:
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
USDM_CURRENT_SERVICE_CANDIDATES = [
    "https://gis.droughtmonitor.unl.edu/arcgis/rest/services/USDM/USDM_Current/MapServer",
    "https://gis.droughtmonitor.unl.edu/arcgis/rest/services/USDM/USDM_Current_Conditions/MapServer",
]

USER_AGENT = os.getenv("METAPP_UA", "metapplication/1.0 (contact: you@example.com)")
TIMEOUT = float(os.getenv("METAPP_TIMEOUT", "14"))
DEBUG = os.getenv("METAPP_DROUGHT_DEBUG", "0") == "1"

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _log(msg: str):
    if DEBUG:
        print(f"[drought_long] {msg}", flush=True)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _normalize_prob_triplet(dry: float, normal: float, wet: float) -> Dict[str, float]:
    dry = max(0.0, dry)
    normal = max(0.0, normal)
    wet = max(0.0, wet)
    s = dry + normal + wet
    if s <= 0:
        return {"dry": 1 / 3, "normal": 1 / 3, "wet": 1 / 3}
    return {"dry": dry / s, "normal": normal / s, "wet": wet / s}


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
    return (feats[0].get("attributes") or {}) or None


# -------------------------
# CPC: 3-month precip outlook
# -------------------------
def get_cpc_precip_probs(lon: float, lat: float) -> Tuple[Dict[str, float], str]:
    """
    Returns ({dry,normal,wet}, notes)
    Uses Lead 1 layer (0) by default.
    """
    layer_id = 0

    attrs = _arcgis_query_point(CPC_PRECIP_SERVICE, layer_id, lon, lat)
    if not attrs:
        return {"dry": 1 / 3, "normal": 1 / 3, "wet": 1 / 3}, "CPC precip: no feature at point (fallback 33/33/33)."

    keys = {str(k).lower(): k for k in attrs.keys()}

    def pick_contains(*needles: str) -> Optional[float]:
        for needle in needles:
            for lk, ok in keys.items():
                if needle in lk:
                    v = _safe_float(attrs.get(ok))
                    if v is not None:
                        return v
        return None

    below = pick_contains("below", "dry", "b_prob", "bprob", "prob_b", "pb")
    near = pick_contains("near", "normal", "n_prob", "nprob", "prob_n", "pn")
    above = pick_contains("above", "wet", "a_prob", "aprob", "prob_a", "pa")

    if below is not None and near is not None and above is not None:
        b, n, a = float(below), float(near), float(above)
        # convert 0–100 to 0–1 if needed
        if max(b, n, a) > 1.5:
            b /= 100.0
            n /= 100.0
            a /= 100.0
        probs = _normalize_prob_triplet(b, n, a)
        return probs, "CPC precip: OK."

    return {"dry": 1 / 3, "normal": 1 / 3, "wet": 1 / 3}, "CPC precip: fields not found (fallback 33/33/33)."


# -------------------------
# CPC: Seasonal Drought Outlook (SDO)
# -------------------------
def get_cpc_drought_outlook_label(lon: float, lat: float) -> Tuple[str, str]:
    """
    Returns (label, notes).
    """
    for layer_id in (1, 2, 0, 3):
        try:
            attrs = _arcgis_query_point(CPC_DROUGHT_SERVICE, layer_id, lon, lat)
            if not attrs:
                continue

            # Try obvious label-like fields first
            for k, v in attrs.items():
                lk = str(k).lower()
                if any(s in lk for s in ("label", "category", "outlook", "class", "cat", "desc")):
                    if isinstance(v, str) and v.strip():
                        return v.strip(), f"CPC drought outlook: OK (layer {layer_id})."

            # Try coded mapping
            code = None
            for k, v in attrs.items():
                lk = str(k).lower()
                if any(s in lk for s in ("code", "cat", "class")):
                    code = _safe_float(v)
                    if code is not None:
                        break

            if code is not None:
                ci = int(round(code))
                mapping = {
                    0: "None / No drought signal",
                    1: "Improvement likely",
                    2: "Drought removal likely",
                    3: "Drought development likely",
                    4: "Drought persistence likely",
                }
                return mapping.get(ci, f"Unknown code {ci}"), f"CPC drought outlook: OK (coded, layer {layer_id})."

        except Exception as e:
            _log(f"SDO layer {layer_id} failed: {e}")

    return "—", "CPC drought outlook: unavailable (—)."


# -------------------------
# USDM: Current drought category at point
# -------------------------
def get_usdm_current_category(lon: float, lat: float) -> Tuple[str, str]:
    """
    Returns (category, notes) where category in None/D0..D4.
    """
    for svc in USDM_CURRENT_SERVICE_CANDIDATES:
        for layer_id in (0, 1, 2, 3):
            try:
                attrs = _arcgis_query_point(svc, layer_id, lon, lat)
                if not attrs:
                    continue

                # scan likely fields
                cat_val = None
                for k, v in attrs.items():
                    lk = str(k).lower()
                    if lk in ("dm", "category", "drought") or ("usdm" in lk) or ("dm" == lk) or ("dm" in lk and "date" not in lk):
                        cat_val = v
                        break

                if isinstance(cat_val, str) and cat_val.strip():
                    s = cat_val.strip().upper()
                    if s in ("NONE", "N", "0"):
                        return "None", f"USDM current: OK ({svc}, layer {layer_id})."
                    if s.startswith("D") and len(s) == 2 and s[1].isdigit():
                        return s, f"USDM current: OK ({svc}, layer {layer_id})."

                code = _safe_float(cat_val)
                if code is None:
                    for k, v in attrs.items():
                        lk = str(k).lower()
                        if "dm" in lk or "cat" in lk or "class" in lk:
                            code = _safe_float(v)
                            if code is not None:
                                break

                if code is not None:
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
    base_map = {"None": 0.0, "D0": 0.5, "D1": 1.0, "D2": 2.0, "D3": 3.0, "D4": 4.0}
    base = base_map.get(usdm_cat, 0.0)

    dry = float(precip_probs.get("dry", 1 / 3))
    wet = float(precip_probs.get("wet", 1 / 3))
    precip_index = _clamp(wet - dry, -1.0, 1.0)

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

    precip_delta = -1.2 * precip_index  # wetter -> negative, drier -> positive
    expected_sev = _clamp(base + sdo_delta + precip_delta, 0.0, 4.0)

    # logistic: center around 0.8 severity
    k = 1.35
    center = 0.8
    p = 1.0 / (1.0 + math.exp(-k * (expected_sev - center)))
    p = _clamp(p, 0.0, 1.0)

    notes = f"Model: base={base:.2f} sdoΔ={sdo_delta:+.2f} precipΔ={precip_delta:+.2f} → sev={expected_sev:.2f}."
    return p, expected_sev, notes


# -------------------------
# Public entrypoint used by server.py
# -------------------------
def evaluate_city(lat: float, lon: float) -> Dict[str, Any]:
    """
    IMPORTANT: This signature matches your current server.py:
        out = evaluate_city(info["lat"], info["lon"])
z
    Returns dict consumed by /api/drought.
    """
    started = time.time()

    # Hard fallback defaults (never let NaN/None hit the frontend)
    precip_probs: Dict[str, float] = {"dry": 1 / 3, "normal": 1 / 3, "wet": 1 / 3}
    sdo_label: str = "—"
    usdm_cat: str = "None"

    notes_parts = []

    try:
        precip_probs, precip_note = get_cpc_precip_probs(lon, lat)
        notes_parts.append(precip_note)
    except Exception as e:
        notes_parts.append(f"CPC precip: error ({e}) fallback 33/33/33.")

    try:
        sdo_label, sdo_note = get_cpc_drought_outlook_label(lon, lat)
        notes_parts.append(sdo_note)
    except Exception as e:
        notes_parts.append(f"CPC drought outlook: error ({e}) (—).")

    try:
        usdm_cat, usdm_note = get_usdm_current_category(lon, lat)
        notes_parts.append(usdm_note)
    except Exception as e:
        notes_parts.append(f"USDM current: error ({e}) (None).")

    try:
        p, expected_sev, model_note = compute_expected_drought_probability(usdm_cat, sdo_label, precip_probs)
        notes_parts.insert(0, model_note)
    except Exception as e:
        p, expected_sev = 0.5, 0.0
        notes_parts.insert(0, f"Model: error ({e}) fallback p=0.5.")

    if not isinstance(p, (int, float)) or math.isnan(float(p)) or math.isinf(float(p)):
        p = 0.5

    elapsed_ms = int((time.time() - started) * 1000)

    return {
        "expected_drought_90d": float(p),
        "expected_severity_0to4": float(expected_sev),
        "components": {
            "usdm_category": usdm_cat,
            "sdo": sdo_label,
            "precip_probs": precip_probs,
        },
        "notes": " ".join(notes_parts) + f" ({elapsed_ms}ms)",
        "sources": {
            "cpc_precip_service": CPC_PRECIP_SERVICE,
            "cpc_drought_service": CPC_DROUGHT_SERVICE,
            "usdm_service_candidates": USDM_CURRENT_SERVICE_CANDIDATES,
        },
    }
