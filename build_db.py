# build_db.py
# Usage:
#   python build_db.py \
#     --hours_csv annual-working-hours-per-worker.csv \
#     --gdp_csv gdp-per-capita-worldbank.csv \
#     --out_db working_hours.db
#
# This script reads working hours and GDP data from CSV files,
# processes them, and builds a SQLite database with the relevant tables.

import argparse, sqlite3, re, os
import pandas as pd

ISO3_RE = re.compile(r"^[A-Z]{3}$")

def read_hours(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # find the value column (the one that is not Entity/Code/Year)
    value_cols = [c for c in df.columns if c not in ("Entity","Code","Year")]
    if len(value_cols) != 1:
        # generally should be only one, but if multiple, take the last one
        value_col = value_cols[-1]
    else:
        value_col = value_cols[0]
    df = df.rename(columns={
        "Entity":"country",
        "Code":"code",
        "Year":"year",
        value_col:"hours"
    })[["country","code","year","hours"]]
    # only ISO3 codes
    df = df[df["code"].astype(str).str.fullmatch(ISO3_RE)].copy()
    # convert types
    df["year"] = df["year"].astype(int)
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    df = df.dropna(subset=["hours"])
    # get latest if duplicates
    df = df.sort_values(["code","year"]).drop_duplicates(["code","year"], keep="last")
    return df

def read_gdp(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # GDP usually has multiple columns, find the first one that is not Entity/Code/Year
    metric_cols = [c for c in df.columns if c not in ("Entity","Code","Year")]
    if not metric_cols:
        raise RuntimeError("GDP CSV has no metric columns")
    value_col = metric_cols[0]
    df = df.rename(columns={
        "Entity":"country",
        "Code":"countrycode",
        "Year":"year",
        value_col:"rgdpe"       # rgdpe is often used for "real GDP per capita"
    })[["country","countrycode","year","rgdpe"]]
    df = df[df["countrycode"].astype(str).str.fullmatch(ISO3_RE)].copy()
    df["year"] = df["year"].astype(int)
    df["rgdpe"] = pd.to_numeric(df["rgdpe"], errors="coerce")
    df = df.dropna(subset=["rgdpe"])
    df = df.sort_values(["countrycode","year"]).drop_duplicates(["countrycode","year"], keep="last")
    return df

def build_db(hours_df: pd.DataFrame, gdp_df: pd.DataFrame, out_db: str):
    if os.path.exists(out_db):
        os.remove(out_db)
    con = sqlite3.connect(out_db)
    cur = con.cursor()

    # --- schema ---
    cur.executescript("""
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;

    CREATE TABLE countries(
        country_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL,
        code         TEXT UNIQUE,
        continent_id INTEGER
    );

    CREATE TABLE working_hours_fact(
        country_id INTEGER NOT NULL,
        year       INTEGER NOT NULL,
        hours      REAL,
        PRIMARY KEY(country_id, year),
        FOREIGN KEY(country_id) REFERENCES countries(country_id)
    );

    CREATE TABLE gdp_data(
        countrycode  TEXT,
        country      TEXT NOT NULL,
        year         INTEGER NOT NULL,
        rgdpe        REAL,  -- at this time, only rgdpe is included
        rgdpo        REAL,
        PRIMARY KEY(countrycode, year)
    );

    CREATE INDEX idx_countries_code ON countries(code);
    CREATE INDEX idx_hours_year ON working_hours_fact(year);
    CREATE INDEX idx_gdp_year ON gdp_data(year);
    """)

    # countries
    countries_df = pd.DataFrame({
        "code": sorted(set(hours_df["code"]) | set(gdp_df["countrycode"]))
    })
    # for name, prefer hours_df names
    name_map = dict(hours_df.drop_duplicates("code")[["code","country"]].values)
    for code, country in gdp_df.drop_duplicates("countrycode")[["countrycode","country"]].values:
        name_map.setdefault(code, country)
    countries_df["name"] = countries_df["code"].map(name_map)

    countries_df[["name","code"]].to_sql("countries", con, if_exists="append", index=False)

    # code -> country_id
    code_id = dict(pd.read_sql_query("SELECT country_id, code FROM countries", con).set_index("code")["country_id"])

    # --- working_hours_fact ---
    wh = hours_df.copy()
    wh["country_id"] = wh["code"].map(code_id)
    wh = wh.dropna(subset=["country_id"])
    wh[["country_id","year","hours"]].to_sql("working_hours_fact", con, if_exists="append", index=False)

    # --- gdp_data ---
    gdp_df.to_sql("gdp_data", con, if_exists="append", index=False)

    con.commit()
    con.close()
    print(f"✅ built {out_db} with {len(countries_df)} countries, "
          f"{len(wh)} hours rows, {len(gdp_df)} gdp rows.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours_csv", required=True)
    ap.add_argument("--gdp_csv", required=True)
    ap.add_argument("--out_db", default="working_hours.db")
    args = ap.parse_args()

    hours_df = read_hours(args.hours_csv)
    gdp_df = read_gdp(args.gdp_csv)
    build_db(hours_df, gdp_df, args.out_db)

if __name__ == "__main__":
    main()
