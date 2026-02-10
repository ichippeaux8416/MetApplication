# drought_long.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import requests

# --- HTTP ---
TIMEOUT = 18
UA = "metapplication/1.0 (contact: you@example.com)"
S = requests.Session()

def _get_json(url: str, params: dict | None = None) -> dict:
    r = S.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json()

def _post_json(url: str, body: dict) -> dict:
    r = S.post(url, json=body, timeout=TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json()

# --- Census reverse geocode to county FIPS (works without an API key) ---
def county_fips_from_latlon(lat: float, lon: float) -> str:
    # Census Geocoder: returns county FIPS inside "COUNTY"
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": lon,
        "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    js = _get_json(url, params=params)
    geos = js.get("result", {}).get("geographies", {})
    counties = geos.get("Counties", [])
    if not counties:
        raise RuntimeError("Census geocoder: no county found for point")
    # COUNTY is 3-digit within state; STATE is 2-digit; full county FIPS is 5 digits
    c = counties[0]
    return f"{c['STATE']}{c['COUNTY']}"

# --- USDM Data Services (official programmatic way; avoids blocked directories) ---
# Web service documentation: https://droughtmonitor.unl.edu/.../WebServiceInfo.aspx
# We use: CountyStatistics / GetDroughtSeverityStatisticsByAreaPercent (JSON)
USDM_COUNTY_STATS = (
    "https://usdmdataservices.unl.edu/api/CountyStatistics/"
    "GetDroughtSeverityStatisticsByAreaPercent/"
)

@dataclass
class UsdmNow:
    # % area in each category (0..4), plus "None"
    none: float
    d0: float
    d1: float
    d2: float
    d3: float
    d4: float
    # computed
    expected_severity: float  # 0..4
    most_likely: str          # "None", "D0".."D4"

def usdm_latest_for_county(county_fips5: str) -> UsdmNow:
    # Ask for a recent window; service returns weekly entries.
    end = date.today()
    start = end - timedelta(days=90)

    params = {
        "aoi": county_fips5,
        "startdate": start.isoformat(),
        "enddate": end.isoformat(),
        "statisticsType": 1,  # 1 = percent area
        "format": "json",
    }
    js = _get_json(USDM_COUNTY_STATS, params=params)

    # Response shape varies slightly; normalize to a list of rows.
    rows = None
    if isinstance(js, list):
        rows = js
    elif isinstance(js, dict):
        for k in ("data", "Data", "result", "Result"):
            if isinstance(js.get(k), list):
                rows = js[k]
                break
    if not rows:
        raise RuntimeError("USDM service: unexpected response (no rows)")

    last = rows[-1]
    # Common keys seen in this service:
    # None/D0/D1/D2/D3/D4 (percent). Be defensive with fallbacks.
    def g(*keys, default=0.0):
        for kk in keys:
            if kk in last:
                try:
                    return float(last[kk])
                except Exception:
                    pass
        return float(default)

    none = g("None", "NONE", "NoDrought", default=0.0)
    d0 = g("D0", "D0Pct", default=0.0)
    d1 = g("D1", "D1Pct", default=0.0)
    d2 = g("D2", "D2Pct", default=0.0)
    d3 = g("D3", "D3Pct", default=0.0)
    d4 = g("D4", "D4Pct", default=0.0)

    # Expected severity index (0..4) using % weights.
    expected = (0*d0 + 1*d1 + 2*d2 + 3*d3 + 4*d4) / 100.0

    # Most-likely category by max percent (including None)
    cat_map = {"None": none, "D0": d0, "D1": d1, "D2": d2, "D3": d3, "D4": d4}
    most = max(cat_map.items(), key=lambda kv: kv[1])[0]

    return UsdmNow(none=none, d0=d0, d1=d1, d2=d2, d3=d3, d4=d4, expected_severity=expected, most_likely=most)

# --- CPC layers via NOAA ArcGIS REST Identify (point query) ---
# These are NOAA mapservices (ArcGIS REST) and support Identify.
# (Service names can change; this approach is robust if URLs stay stable.)
CPC_SDO_MAPSERVER = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_drought_outlk/MapServer"
CPC_3MO_PRECIP_MAPSERVER = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/cpc_3M_precip_outlk/MapServer"

def arcgis_identify(mapserver_url: str, lat: float, lon: float) -> dict | None:
    """
    Returns the first Identify result's attributes at this point, or None.
    """
    url = mapserver_url.rstrip("/") + "/identify"
    params = {
        "f": "pjson",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": 4326,
        "layers": "all",
        "tolerance": 2,
        "mapExtent": "-180,-90,180,90",
        "imageDisplay": "800,600,96",
        "returnGeometry": "false",
    }
    js = _get_json(url, params=params)
    results = js.get("results") or []
    if not results:
        return None
    return results[0].get("attributes") or None

def parse_sdo_label(attrs: dict | None) -> str:
    if not attrs:
        return "—"
    # Try common attribute names:
    for k in ("Outlook", "CATEGORY", "CAT", "Label", "LABEL", "SDO", "TYPE"):
        v = attrs.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # If nothing obvious, show a compact fallback
    return "—"

def parse_precip_probs(attrs: dict | None) -> dict:
    """
    Return dry/normal/wet probabilities in 0..1.
    CPC precip outlook polygons typically encode which tercile is favored and by how much.
    Attributes vary; we’ll try multiple key patterns.
    """
    if not attrs:
        return {"dry": 1/3, "normal": 1/3, "wet": 1/3}

    # Sometimes services store explicit percentages for BN/NN/AN.
    # We'll check a few likely keys.
    def gf(*keys):
        for k in keys:
            v = attrs.get(k)
            try:
                if v is None:
                    continue
                fv = float(v)
                return fv
            except Exception:
                continue
        return None

    dry = gf("Below", "BN", "BELOW", "P_BN", "PB", "PROB_BN")
    wet = gf("Above", "AN", "ABOVE", "P_AN", "PA", "PROB_AN")
    nor = gf("Near", "NN", "NEAR", "P_NN", "PN", "PROB_NN")

    # If we got explicit probs, normalize.
    got = [x for x in (dry, nor, wet) if x is not None]
    if got:
        # Many services use percent (0..100). Convert if needed.
        def to01(x):
            return x/100.0 if x > 1.01 else x
        d = to01(dry or 0.0)
        n = to01(nor or 0.0)
        w = to01(wet or 0.0)
        s = d + n + w
        if s > 0:
            return {"dry": d/s, "normal": n/s, "wet": w/s}

    # Otherwise, look for a “favored tercile” + “probability” style encoding.
    # Example: attrs might contain something like "Tercile" and "Prob"
    tercile = None
    for k in ("TERCILE", "Tercile", "FAVORED", "FAV", "CAT"):
        v = attrs.get(k)
        if isinstance(v, str):
            tercile = v.upper()
            break
    p = gf("PROB", "Prob", "PROBABILITY", "PCT", "PERCENT", "VALUE")
    if p is not None:
        p01 = p/100.0 if p > 1.01 else p
        p01 = max(0.33, min(0.9, p01))  # keep sane
        # Allocate: favored tercile gets p01, other two split remaining.
        rem = 1.0 - p01
        if tercile and ("BELOW" in tercile or "BN" in tercile or "DRY" in tercile):
            return {"dry": p01, "normal": rem/2, "wet": rem/2}
        if tercile and ("ABOVE" in tercile or "AN" in tercile or "WET" in tercile):
            return {"dry": rem/2, "normal": rem/2, "wet": p01}
        # default to equal
        return {"dry": rem/2, "normal": p01, "wet": rem/2}

    return {"dry": 1/3, "normal": 1/3, "wet": 1/3}

def drought_probability_90d(usdm: UsdmNow, precip_probs: dict, sdo_label: str) -> float:
    """
    Simple, transparent equation:
    - baseline drought risk from current severity (USDM expected severity)
    - adjust by precip outlook (dry vs wet)
    - adjust by SDO category if it clearly implies development/removal
    Output 0..1
    """
    # Baseline: map expected severity 0..4 to probability 0.10..0.85
    base = 0.10 + (usdm.expected_severity / 4.0) * 0.75

    # Precip signal: dry - wet in [-1,1]
    dry = float(precip_probs.get("dry", 1/3))
    wet = float(precip_probs.get("wet", 1/3))
    signal = dry - wet  # positive => drier bias

    # SDO hint
    s = (sdo_label or "").lower()
    sdo_adj = 0.0
    if any(k in s for k in ("develop", "development", "onset")):
        sdo_adj += 0.10
    if any(k in s for k in ("persist", "persistence")):
        sdo_adj += 0.05
    if any(k in s for k in ("improve", "improvement")):
        sdo_adj -= 0.08
    if any(k in s for k in ("remove", "removal")):
        sdo_adj -= 0.12

    # Combine
    p = base + 0.35 * signal + sdo_adj

    # squash to 0..1
    return max(0.0, min(1.0, p))

def evaluate_city(lat: float, lon: float) -> dict:
    fips = county_fips_from_latlon(lat, lon)
    usdm = usdm_latest_for_county(fips)

    sdo_attrs = arcgis_identify(CPC_SDO_MAPSERVER, lat, lon)
    sdo_label = parse_sdo_label(sdo_attrs)

    pr_attrs = arcgis_identify(CPC_3MO_PRECIP_MAPSERVER, lat, lon)
    precip_probs = parse_precip_probs(pr_attrs)

    p90 = drought_probability_90d(usdm, precip_probs, sdo_label)

    return {
        "expected_drought_90d": p90,
        "components": {
            "county_fips": fips,
            "usdm_most_likely": usdm.most_likely,
            "usdm_expected_severity": usdm.expected_severity,
            "usdm_percent": {
                "none": usdm.none,
                "d0": usdm.d0,
                "d1": usdm.d1,
                "d2": usdm.d2,
                "d3": usdm.d3,
                "d4": usdm.d4,
            },
            "sdo": sdo_label,
            "precip_probs": precip_probs,
        },
    }
