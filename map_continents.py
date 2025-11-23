#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-map ISO3 country codes in `countries.code` to `continents.continent_id`.

- Creates `continents` rows if missing: Africa, Asia, Europe, North America, South America, Oceania, Antarctica
- Fills/updates `countries.continent_id`
- Prints a summary report

Requires:
  pip install pycountry pycountry-convert
"""

from __future__ import annotations
import argparse
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import pycountry
    import pycountry_convert as pcc
except Exception as e:
    print("✖ Missing dependency. Install with: pip install pycountry pycountry-convert")
    raise

CONTINENT_CANONICAL = [
    "Africa", "Asia", "Europe", "North America", "South America", "Oceania", "Antarctica",
]

# processing hardcoded ISO3 -> continent name mapping
HARDCODED_ISO3_TO_CONTINENT: Dict[str, str] = {
    # special cases
    "XKX": "Europe",          # Kosovo
    "PSE": "Asia",            # Palestine
    "HKG": "Asia",            # Hong Kong
    "MAC": "Asia",            # Macao
    "TWN": "Asia",            # Taiwan 
    "SXM": "North America",  # Sint Maarten (Dutch part)
    "TLS": "Asia",           # Timor-Leste (East Timor)
    # difficult cases
    "CIV": "Africa",          # Côte d'Ivoire
    "LAO": "Asia",            # Lao People's Democratic Republic -> Laos
    "COD": "Africa",          # Congo (Democratic Republic of the)
    "COG": "Africa",          # Congo
    "SYR": "Asia",
    "IRN": "Asia",
    "RUS": "Europe",          # attempt to classify Russia as Europe
}

# pcc mapping：continent code -> full name
CONT_CODE_TO_NAME = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
    "AN": "Antarctica",
}

@dataclass
class CountryRow:
    country_id: int
    name: str
    code: str
    continent_id: Optional[int]

def ensure_continents(conn: sqlite3.Connection) -> Dict[str, int]:
    """Ensure continent rows exist; return name->id mapping."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS continents (
          continent_id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE
        );
    """)
    for name in CONTINENT_CANONICAL:
        cur.execute("INSERT OR IGNORE INTO continents(name) VALUES(?)", (name,))
    conn.commit()

    cur.execute("SELECT continent_id, name FROM continents")
    mapping = {name: cid for cid, name in cur.fetchall()}
    return mapping

def fetch_countries(conn: sqlite3.Connection) -> Dict[str, CountryRow]:
    cur = conn.cursor()
    cur.execute("""
        SELECT country_id, name, code, continent_id
        FROM countries
        WHERE code IS NOT NULL AND TRIM(code) <> ''
    """)
    rows = {}
    for cid, name, code, cont_id in cur.fetchall():
        iso3 = (code or "").strip().upper()
        rows[iso3] = CountryRow(cid, name, iso3, cont_id)
    return rows

def iso3_to_continent_name(iso3: str) -> Optional[str]:
    """Return canonical continent name or None."""
    iso3 = iso3.strip().upper()

    # 1) first try hardcoded mapping
    if iso3 in HARDCODED_ISO3_TO_CONTINENT:
        return HARDCODED_ISO3_TO_CONTINENT[iso3]

    # 2) normal lookup via pycountry-convert
    try:
        # some countries may not be found here
        alpha2 = pcc.country_alpha3_to_country_alpha2(iso3)
        cont_code = pcc.country_alpha2_to_continent_code(alpha2)
        return CONT_CODE_TO_NAME.get(cont_code)
    except Exception:
        # 3) try pycountry directly (some territories may be found here)
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            if country:
                alpha2 = country.alpha_2
                cont_code = pcc.country_alpha2_to_continent_code(alpha2)
                return CONT_CODE_TO_NAME.get(cont_code)
        except Exception:
            pass
    return None

def update_country_continent(conn: sqlite3.Connection, country: CountryRow, cont_name: str, cont_name_to_id: Dict[str, int]) -> bool:
    """Update countries.continent_id; return True if changed."""
    cont_id = cont_name_to_id.get(cont_name)
    if not cont_id:
        return False
    if country.continent_id == cont_id:
        return False
    cur = conn.cursor()
    cur.execute("UPDATE countries SET continent_id=? WHERE country_id=?", (cont_id, country.country_id))
    return cur.rowcount > 0

def main():
    ap = argparse.ArgumentParser(description="Auto-fill countries.continent_id by ISO3 (using pycountry-convert).")
    ap.add_argument("--db", default="working_hours.db", help="Path to SQLite DB (default: working_hours.db)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes; only print plan.")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        conn.row_factory = sqlite3.Row

        # 1) make sure continents exist
        cont_name_to_id = ensure_continents(conn)

        # 2) read countries and map
        by_iso3 = fetch_countries(conn)

        updated, unresolved, total = 0, 0, len(by_iso3)

        for iso3, row in sorted(by_iso3.items(), key=lambda kv: kv[0]):
            cont_name = iso3_to_continent_name(iso3)
            if not cont_name:
                unresolved += 1
                print(f"⚠ Unresolved continent for {iso3} - {row.name}")
                continue

            changed = False
            if not args.dry_run:
                changed = update_country_continent(conn, row, cont_name, cont_name_to_id)
            else:
                changed = (row.continent_id != cont_name_to_id.get(cont_name))

            if changed:
                updated += 1
                print(f"✓ {iso3:>3s} → {cont_name} (country_id={row.country_id})")

        if not args.dry_run:
            conn.commit()

        # 3) summary
        print("\n— Summary —")
        print(f"Total countries: {total}")
        print(f"Updated rows  : {updated}")
        print(f"Unresolved    : {unresolved}")

        # 4) give a hint for verification
        print("\nTry to verify in sqlite3:")
        print("  SELECT c.name, c.code, co.name AS continent"
              " FROM countries c LEFT JOIN continents co ON c.continent_id=co.continent_id"
              " WHERE co.name IS NULL LIMIT 20;")

    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(main())
