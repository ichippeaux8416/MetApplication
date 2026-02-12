from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from siphon.catalog import get_latest_access_url
from siphon.ncss import NCSS
from netCDF4 import Dataset, num2date

HRRR_CATALOG = "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/HRRR/CONUS_2p5km/catalog.xml"
VAR_NAME = "Temperature_height_above_ground"  # 2m temp is this var at ~2m

CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.006},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.797},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    "lax": {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    "sea": {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    "sfo": {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    "hou": {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    "aus": {"name": "Austin", "lat": 30.2672, "lon": -97.7431},
    "bna": {"name": "Nashville", "lat": 36.1627, "lon": -86.7816},
    "okc": {"name": "Oklahoma City", "lat": 35.4676, "lon": -97.5164},
    "phx": {"name": "Phoenix", "lat": 33.4484, "lon": -112.074},
    "las": {"name": "Las Vegas", "lat": 36.1699, "lon": -115.1398},
    "msp": {"name": "Minneapolis-Saint Paul", "lat": 44.9778, "lon": -93.265},
    "sat": {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936},
}


def _find_time_var(ds: Dataset) -> str:
    if "time" in ds.variables:
        return "time"
    for k in ds.variables.keys():
        if "time" in k.lower():
            return k
    raise RuntimeError(f"Could not find time variable. Vars: {list(ds.variables.keys())[:60]}")


def _pick_height_index(ds: Dataset, var) -> int:
    for dn in var.dimensions:
        if "height" in dn.lower():
            if dn in ds.variables:
                h = ds.variables[dn][:]
                best_i, best_d = 0, 1e18
                for i, v in enumerate(h):
                    d = abs(float(v) - 2.0)
                    if d < best_d:
                        best_d, best_i = d, i
                return int(best_i)
            return 0
    return 0


def _k_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def _to_iso_utc(t) -> str:
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        else:
            t = t.astimezone(timezone.utc)
        return t.isoformat().replace("+00:00", "Z")

    try:
        dt = datetime(
            int(t.year),
            int(t.month),
            int(t.day),
            int(getattr(t, "hour", 0)),
            int(getattr(t, "minute", 0)),
            int(getattr(t, "second", 0)),
            tzinfo=timezone.utc,
        )
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return str(t) + "Z"


def fetch_city_series(ncss: NCSS, lat: float, lon: float, hours: int) -> list[dict]:
    q = ncss.query()
    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=hours)

    q.lonlat_point(lon, lat)
    q.time_range(start, end)
    q.variables(VAR_NAME)
    q.accept("netcdf4")

    raw = ncss.get_data_raw(q)
    ds = Dataset("inmemory.nc", mode="r", memory=raw)  # type: ignore

    if VAR_NAME not in ds.variables:
        raise RuntimeError(f"{VAR_NAME} not found. Vars: {list(ds.variables.keys())[:80]}")

    tname = _find_time_var(ds)
    tvar = ds.variables[tname]
    if getattr(tvar, "size", 0) == 0:
        raise RuntimeError("Time variable is empty (no forecast times returned).")

    times = num2date(tvar[:], units=tvar.units)  # type: ignore

    v = ds.variables[VAR_NAME]
    arr = v[:]

    if getattr(arr, "ndim", 0) == 2:
        hi = _pick_height_index(ds, v)
        series_k = arr[:, hi]
    else:
        series_k = arr

    out = []
    for tdt, kval in zip(times, series_k):
        k = float(kval)
        out.append({"valid_utc": _to_iso_utc(tdt), "temp_f": float(_k_to_f(k))})

    return out[:hours]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    hours = max(1, min(24, int(args.hours)))

    ncss_url = get_latest_access_url(HRRR_CATALOG, "NetcdfSubset")
    ncss = NCSS(ncss_url)

    payload = {
        "run_ncss": ncss_url,
        "asof_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hours": hours,
        "cities": {},
    }

    for code, info in CITIES.items():
        series = fetch_city_series(ncss, info["lat"], info["lon"], hours)
        payload["cities"][code] = {
            "name": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "series": series,
        }

        temps = ", ".join(f"{p['temp_f']:.1f}" for p in series)
        print(f"{code.upper()} ({info['name']}): {temps}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON -> {args.json}")


if __name__ == "__main__":
    main()
