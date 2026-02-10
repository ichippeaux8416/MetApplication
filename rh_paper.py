import re
import math
import requests
from datetime import datetime, date, timedelta
from dateutil import parser as dtparser

# ---------------- CONFIG ----------------
USER_AGENT = "rh-temp-paper/0.2 (contact: you@example.com)"

# Supported cities (3-letter codes)
# Note: coords are city centers; NWS points API will map to the proper forecast office/grid.
CITIES = {
    "den": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    "dal": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "chi": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
}

# Decision thresholds
MIN_EDGE = 0.05          # need at least 5% edge to signal
ENTRY_BUFFER = 0.01      # extra margin so we're not razor-thin

# Uncertainty model (starter; calibrate later)
SIGMA_BY_LEAD_DAYS = {
    0: 2.5,
    1: 3.0,
    2: 4.0,
    3: 5.0,
    4: 6.0,
    5: 7.0,
}
DEFAULT_SIGMA = 7.5

# How far ahead we allow (prevents accidental typos)
MAX_DAYS_AHEAD = 14

TIMEOUT = 12
# --------------------------------------


# ---------- HTTP ----------
session = requests.Session()

def http_get_json(url: str, headers=None, params=None):
    r = session.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------- NWS forecast ----------
def nws_points(lat: float, lon: float) -> dict:
    return http_get_json(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )

def nws_forecast_periods(lat: float, lon: float) -> list[dict]:
    points = nws_points(lat, lon)
    forecast_url = points["properties"]["forecast"]
    fc = http_get_json(
        forecast_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    )
    return fc["properties"]["periods"]

def daily_daytime_highs(periods: list[dict]) -> dict[date, int]:
    """Map date -> NWS daytime high temp (F)."""
    out: dict[date, int] = {}
    for p in periods:
        if not p.get("isDaytime"):
            continue
        d = dtparser.isoparse(p["startTime"]).date()
        t = p.get("temperature")
        if isinstance(t, int):
            out[d] = t
    return out


# ---------- Probability model ----------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def prob_ge(mu: float, sigma: float, threshold: float) -> float:
    """
    P(X >= threshold) for Normal(mu, sigma).
    continuity-ish tweak: use threshold-0.5 to represent ">= integer temp".
    """
    z = (threshold - 0.5 - mu) / sigma
    return max(0.0, min(1.0, 1.0 - norm_cdf(z)))

def sigma_for(target_date: date) -> float:
    today = datetime.now().astimezone().date()
    lead = (target_date - today).days
    if lead < 0:
        return SIGMA_BY_LEAD_DAYS.get(0, DEFAULT_SIGMA)
    return SIGMA_BY_LEAD_DAYS.get(lead, DEFAULT_SIGMA)


# ---------- Parsing input lines ----------
# Format: mm/dd/yyyy - t - yes:$$/no:$$
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

    yes_price = yes_cents / 100.0
    no_price = no_cents / 100.0

    return d, t, yes_price, no_price


# ---------- Decision ----------
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


def pick_city() -> dict:
    print("\nChoose city code (first three letters):")
    print("  den = Denver")
    print("  nyc = New York City")
    print("  dal = Dallas")
    print("  chi = Chicago")
    while True:
        code = input("City code> ").strip().lower()
        if code in CITIES:
            return CITIES[code]
        print("Invalid code. Enter one of: den, nyc, dal, chi")


def main():
    city = pick_city()
    name = city["name"]
    lat = city["lat"]
    lon = city["lon"]

    # Fetch NWS highs
    periods = nws_forecast_periods(lat, lon)
    highs = daily_daytime_highs(periods)

    print(f"\nNWS forecast daytime highs ({name}) available dates:")
    for d in sorted(highs.keys())[:20]:
        print(f"- {d.isoformat()}: {highs[d]}F")

    print("\nEnter lines in this format (blank line to run):")
    print("  mm/dd/yyyy - t - yes:$$/no:$$")
    print("Where t is the threshold for: High >= t°F")
    print("Example:")
    print("  02/11/2026 - 66 - yes:34 / no:66\n")

    lines = []
    while True:
        try:
            s = input("> ").strip()
        except EOFError:
            break
        if not s:
            break
        lines.append(s)

    if not lines:
        print("\nNo input lines provided. Exiting.")
        return

    today = datetime.now().astimezone().date()
    max_date = today + timedelta(days=MAX_DAYS_AHEAD)

    print("\nResults:\n")
    for raw in lines:
        parsed = parse_line(raw)
        if not parsed:
            print(f"- INVALID FORMAT: {raw}")
            print("  Expected: mm/dd/yyyy - t - yes:$$/no:$$\n")
            continue

        d, t, yes_price, no_price = parsed

        if d < today:
            print(f"- {d} | >= {t:.0f}F | SKIP (date is in the past)\n")
            continue

        if d > max_date:
            print(f"- {d} | >= {t:.0f}F | SKIP (more than {MAX_DAYS_AHEAD} days ahead)\n")
            continue

        if d not in highs:
            print(f"- {d} | >= {t:.0f}F | SKIP (NWS daytime high not available for this date)\n")
            continue

        mu = float(highs[d])
        sigma = sigma_for(d)
        p_yes_fair = prob_ge(mu, sigma, t)

        signal, edge_yes, edge_no, p_no_fair = decide(p_yes_fair, yes_price, no_price)

        print(f"- {d.isoformat()} | {name} | Contract: High >= {t:.0f}F")
        print(f"  Market: YES={yes_price:.4f}  NO={no_price:.4f}")
        print(f"  NWS:    mu={mu:.1f}F  sigma={sigma:.1f}  => P(YES)={p_yes_fair:.4f}  P(NO)={p_no_fair:.4f}")
        print(f"  Edge:   YES={edge_yes:+.4f}  NO={edge_no:+.4f}")
        print(f"  Signal: {signal}\n")

    print("Note: This is a paper-trading evaluation tool, not financial advice.")


if __name__ == "__main__":
    main()