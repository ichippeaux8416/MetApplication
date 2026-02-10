import math
import requests
from typing import Dict, Any, Optional

# CPC ArcGIS REST layers (NOAA mapservices)
# Seasonal Drought Outlook (US & PR): layer id 4
CPC_DROUGHT_LAYER = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_drought_outlk/MapServer/4/query"

# 3-month precip outlook probabilities: "Lead 1" layer id 0
CPC_PRECIP_LAYER = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_sea_precip_outlk/MapServer/0/query"

CITIES = {
    "Denver": {"lat": 39.7392, "lon": -104.9903},
    "New York City": {"lat": 40.7128, "lon": -74.0060},
    "Dallas": {"lat": 32.7767, "lon": -96.7970},
    "Chicago": {"lat": 41.8781, "lon": -87.6298},
}

def _query_point(layer_url: str, lat: float, lon: float, out_fields: str) -> Optional[Dict[str, Any]]:
    """
    Query a polygon layer at a point using ArcGIS REST /query.
    Returns attributes dict for the first intersecting feature.
    """
    params = {
        "f": "json",
        "where": "1=1",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "false",
        "resultRecordCount": "1",
    }
    r = requests.get(layer_url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    feats = data.get("features") or []
    if not feats:
        return None
    return (feats[0].get("attributes") or None)

def get_cpc_signals(city: str) -> Dict[str, Any]:
    if city not in CITIES:
        raise ValueError(f"Unknown city '{city}'. Choose from: {', '.join(CITIES.keys())}")

    lat = CITIES[city]["lat"]
    lon = CITIES[city]["lon"]

    drought = _query_point(CPC_DROUGHT_LAYER, lat, lon, out_fields="outlook,valid")
    precip = _query_point(CPC_PRECIP_LAYER, lat, lon, out_fields="cat,prob,valid_seas")

    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "drought": drought,
        "precip": precip,
    }

def _base_prob_from_outlook(outlook: Optional[str]) -> float:
    """
    Heuristic base probability of drought issues through ~90 days based on the CPC drought outlook category.
    (This is not an official probability product; it's a scoring model for your app.)
    """
    if not outlook:
        return 0.50

    o = outlook.strip().lower()
    # Common CPC drought outlook categories include:
    # Persistence, Improvement, Removal, Development, No Drought (and sometimes "No Data"/blank)
    if "development" in o:
        return 0.70
    if "persistence" in o:
        return 0.62
    if "improvement" in o:
        return 0.42
    if "removal" in o:
        return 0.28
    if "no drought" in o:
        return 0.15
    return 0.50

def _parse_precip_probs(cat: Optional[str], prob: Optional[float]) -> Dict[str, float]:
    """
    CPC precip outlook polygons encode a favored category (Above/Below/Normal) and a probability percent
    (typically 33,40,50,60,70,80,90).
    We'll convert that into rough Above/Below probabilities around 33% baseline.
    """
    # Baseline equal-chances (EC) is ~33/33/33. If favored is X%,
    # then the remainder (100 - X) split between the other two.
    if not cat or prob is None:
        return {"above": 0.33, "below": 0.33, "normal": 0.34}

    p = float(prob) / 100.0
    cat_l = cat.strip().lower()

    rem = max(0.0, 1.0 - p)
    other = rem / 2.0

    above = other
    below = other
    normal = other

    if "above" in cat_l:
        above = p
    elif "below" in cat_l:
        below = p
    else:
        normal = p

    # small normalization
    s = above + below + normal
    if s <= 0:
        return {"above": 0.33, "below": 0.33, "normal": 0.34}
    return {"above": above / s, "below": below / s, "normal": normal / s}

def expected_drought_probability(city: str) -> Dict[str, Any]:
    """
    Combine:
      - CPC Seasonal Drought Outlook category at the city point
      - CPC 3-month precip outlook category+prob at the city point
    Into a single 'expected drought through ~90 days' probability (heuristic model).
    """
    sig = get_cpc_signals(city)
    drought = sig.get("drought") or {}
    precip = sig.get("precip") or {}

    outlook = drought.get("outlook")
    base = _base_prob_from_outlook(outlook)

    cat = precip.get("cat")
    prob = precip.get("prob")
    valid_seas = precip.get("valid_seas")

    probs = _parse_precip_probs(cat, prob)

    # Adjustment:
    # drier bias (Below > Above) increases drought probability
    # wetter bias decreases it.
    # scale factor tuned to keep results sane.
    dry_minus_wet = (probs["below"] - probs["above"])  # -1..+1
    adj = 0.35 * dry_minus_wet

    # Convert to logit space for smoother behavior
    def logit(x: float) -> float:
        x = min(0.999, max(0.001, x))
        return math.log(x / (1 - x))

    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    p_final = sigmoid(logit(base) + adj)
    p_final = max(0.01, min(0.99, p_final))

    return {
        "city": sig["city"],
        "lat": sig["lat"],
        "lon": sig["lon"],
        "p_drought": p_final,
        "drought_outlook": outlook,
        "drought_valid": drought.get("valid"),
        "precip": {
            "valid_seas": valid_seas,
            "cat": cat,
            "prob": prob,
            "above_prob": probs["above"],
            "below_prob": probs["below"],
            "normal_prob": probs["normal"],
        },
    }
