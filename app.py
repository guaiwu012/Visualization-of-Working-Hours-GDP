from __future__ import annotations
import os
import math
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime

# absolute path to the SQLite DB file
DB_PATH = r"E:\AIDM\AIDM7360_big_data\proj\main\working_hours.db"

# admin password
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")  # default password if not set

app = Flask(__name__)

# cors settings
CORS(app,
     resources={r"/api/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-DS-KEY", "X-ADMIN-PASS"],
     methods=["GET", "POST", "OPTIONS"])

@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, X-DS-KEY, X-ADMIN-PASS")
    return resp


# ----------------- DB helpers -----------------
def rows(q: str, params: tuple = ()) -> List[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(q, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def one(q: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    r = rows(q, params)
    return r[0] if r else None


def exec_sql(q: str, params: tuple = ()) -> int:
    # returns affected row count
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute(q, params)
        con.commit()
        return cur.rowcount
    finally:
        con.close()


# init admin tables
def ensure_admin_tables():
    exec_sql("""
        CREATE TABLE IF NOT EXISTS admin_deleted(
            code TEXT PRIMARY KEY,
            deleted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            reason TEXT
        )
    """)
ensure_admin_tables()


# country name / code resolution with alias support
COUNTRY_ALIASES = {
    "COTE DIVOIRE": "CIV", "CÔTE D'IVOIRE": "CIV",
    "IRAN": "IRN", "IRAN ISLAMIC REPUBLIC OF": "IRN", "IRAN (ISLAMIC REPUBLIC OF)": "IRN",
    "VIETNAM": "VNM", "VIET NAM": "VNM",
    "RUSSIA": "RUS", "RUSSIAN FEDERATION": "RUS",
    "SOUTH KOREA": "KOR", "KOREA REPUBLIC OF": "KOR", "KOREA, REPUBLIC OF": "KOR",
    "NORTH KOREA": "PRK", "KOREA DEMOCRATIC PEOPLES REPUBLIC OF": "PRK", "KOREA, DEM. PEOPLE'S REP.": "PRK",
    "DEMOCRATIC REPUBLIC OF THE CONGO": "COD", "CONGO, THE DEM. REP. OF THE": "COD",
    "REPUBLIC OF THE CONGO": "COG", "CONGO": "COG",
    "LAOS": "LAO", "LAO PDR": "LAO",
    "BOLIVIA": "BOL", "BOLIVIA (PLURINATIONAL STATE OF)": "BOL",
    "MOLDOVA": "MDA", "REPUBLIC OF MOLDOVA": "MDA",
    "PALESTINE": "PSE", "STATE OF PALESTINE": "PSE",
    "ESWATINI": "SWZ", "SWAZILAND": "SWZ",
    "CZECHIA": "CZE", "CZECH REPUBLIC": "CZE",
    "UNITED STATES OF AMERICA": "USA", "UNITED STATES": "USA", "US": "USA", "USA": "USA",
    "UNITED KINGDOM": "GBR", "UK": "GBR", "GREAT BRITAIN": "GBR",
    "SYRIA": "SYR",
    "TANZANIA": "TZA",
}

def _norm(s: str) -> str:
    if not s: return ""
    t = s.strip().upper()
    t = t.replace("\u200b","").replace("\u00a0"," ").replace("’","'")
    if "(" in t and ")" in t: t = t[:t.index("(")].strip()
    if "," in t: t = t.split(",")[0].strip()
    for ch in ["-", "–", "—", ".", "_", "/"]:
        t = t.replace(ch, " ")
    t = " ".join(t.split())
    return t

def resolve_country(code_or_name: str) -> Optional[dict]:
    if not code_or_name: return None
    x = code_or_name.strip()
    if x in ("—", "-"):  # avoid confusion with missing data
        return None
    xu = x.upper()
    nx = _norm(x)
    cs = rows("SELECT country_id, name, code, continent_id FROM countries")

    for c in cs:
        if (c["code"] or "").strip().upper() == xu:
            return c
    for c in cs:
        if (c["name"] or "").strip().upper() == xu:
            return c
    alias_iso = COUNTRY_ALIASES.get(xu) or COUNTRY_ALIASES.get(nx)
    if alias_iso:
        for c in cs:
            if (c["code"] or "").strip().upper() == alias_iso:
                return c
    for c in cs:
        if _norm(c["name"]) == nx:
            return c
    for c in cs:
        if nx and nx in _norm(c["name"]):
            return c
    return None


# query helpers
def continent_avg_hours(year: int, continent: str) -> Optional[float]:
    r = one("""
        SELECT AVG(wh.hours) AS avg_hours
        FROM continents co
        JOIN countries c ON c.continent_id = co.continent_id
        LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
        JOIN working_hours_fact wh ON wh.country_id = c.country_id
        WHERE d.code IS NULL AND wh.year = ? AND TRIM(co.name) = TRIM(?)
    """, (year, continent))
    return r["avg_hours"] if r and r["avg_hours"] is not None else None


def country_hours(year: int, code_or_name: str) -> Optional[Tuple[str, str, Optional[float], str]]:
    c = resolve_country(code_or_name)
    if not c:
        return None
    # if deleted, return with None hours
    r_del = one("SELECT code FROM admin_deleted WHERE UPPER(code)=UPPER(?)", (c["code"],))
    if r_del:
        return (c["code"], c["name"], None, one("SELECT name FROM continents WHERE continent_id = ?", (c["continent_id"],))["name"])
    r = one("""
        SELECT c.code AS iso3, c.name AS country, wh.hours AS hours, co.name AS continent
        FROM countries c
        JOIN continents co ON co.continent_id = c.continent_id
        LEFT JOIN working_hours_fact wh ON wh.country_id = c.country_id AND wh.year = ?
        WHERE c.country_id = ?
    """, (year, c["country_id"]))
    if r:
        return r["iso3"], r["country"], (None if r["hours"] is None else float(r["hours"])), r["continent"]
    return None


def fastest_decline_since_1950() -> Optional[Tuple[str, float]]:
    data = rows("""
        WITH A AS (
          SELECT co.name AS continent, wh.year AS year, AVG(wh.hours) AS avg_hours
          FROM continents co
          JOIN countries c ON c.continent_id = co.continent_id
          LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
          JOIN working_hours_fact wh ON wh.country_id = c.country_id
          WHERE d.code IS NULL AND wh.year >= 1950
          GROUP BY co.name, wh.year
        ),
        B AS (
          SELECT continent, MIN(year) AS y0, MAX(year) AS y1
          FROM A GROUP BY continent
        )
        SELECT A.continent,
               (MAX(CASE WHEN A.year = B.y1 THEN A.avg_hours END)
                - MIN(CASE WHEN A.year = B.y0 THEN A.avg_hours END)
               ) * 1.0 / NULLIF(B.y1 - B.y0, 0) AS slope
        FROM A JOIN B ON A.continent = B.continent
        GROUP BY A.continent
    """)
    best = None
    for d in data:
        slope = d["slope"]
        if slope is None:
            continue
        if best is None or slope < best[1]:
            best = (d["continent"], float(slope))
    if best:
        return best[0], -best[1]
    return None


def top_bottom_countries(year: int, n_top: int = 3, n_bottom: int = 2):
    topn = rows("""
        SELECT c.name AS country, co.name AS continent, wh.hours AS hours
        FROM countries c
        JOIN continents co ON co.continent_id = c.continent_id
        LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
        JOIN working_hours_fact wh ON wh.country_id = c.country_id
        WHERE d.code IS NULL AND wh.year = ?
        ORDER BY wh.hours DESC
        LIMIT ?
    """, (year, n_top))
    botn = rows("""
        SELECT c.name AS country, co.name AS continent, wh.hours AS hours
        FROM countries c
        JOIN continents co ON co.continent_id = c.continent_id
        LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
        JOIN working_hours_fact wh ON wh.country_id = c.country_id
        WHERE d.code IS NULL AND wh.year = ?
        ORDER BY wh.hours ASC
        LIMIT ?
    """, (year, n_bottom))
    return topn, botn


def global_avg(year: int) -> Optional[float]:
    r = one("""
        SELECT AVG(wh.hours) AS avg_hours
        FROM working_hours_fact wh
        JOIN countries c ON c.country_id = wh.country_id
        LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
        WHERE d.code IS NULL AND wh.year = ?
    """, (year,))
    return r["avg_hours"] if r and r["avg_hours"] is not None else None


def continent_shift(before_year: int, after_year: int):
    data = rows("""
        WITH A AS (
          SELECT co.name AS continent, wh.year AS year, AVG(wh.hours) AS avg_hours
          FROM continents co
          JOIN countries c ON c.continent_id = co.continent_id
          LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
          JOIN working_hours_fact wh ON wh.country_id = c.country_id
          WHERE d.code IS NULL AND wh.year IN (?, ?)
          GROUP BY co.name, wh.year
        )
        SELECT continent,
               MAX(CASE WHEN year=? THEN avg_hours END) AS after_val,
               MAX(CASE WHEN year=? THEN avg_hours END) AS before_val
        FROM A
        GROUP BY continent
    """, (before_year, after_year, after_year, before_year))
    best = None
    for d in data:
        if d["after_val"] is None or d["before_val"] is None:
            continue
        change = float(d["after_val"] - d["before_val"])
        if best is None or change < best[1]:
            best = (d["continent"], change)
    return best


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    meanx = sum(xs) / n
    meany = sum(ys) / n
    num = sum((x - meanx) * (y - meany) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - meanx)**2 for x in xs))
    deny = math.sqrt(sum((y - meany)**2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


# api endpoints
@app.route("/api/choropleth")
def api_choropleth():
    year = request.args.get("year", 2023, type=int)
    q = """
        SELECT
            c.code AS iso3,
            c.name AS country,
            wh.hours AS hours,
            gd.rgdpe AS gdp_per_capita
        FROM countries c
        LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
        LEFT JOIN working_hours_fact wh
               ON d.code IS NULL AND wh.country_id = c.country_id AND wh.year = ?
        LEFT JOIN gdp_data gd
               ON d.code IS NULL AND gd.countrycode = c.code AND gd.year = ?
        WHERE c.code IS NOT NULL AND TRIM(c.code) <> ''
    """
    data = rows(q, (year, year))
    out = {}
    for d in data:
        iso = (d.get("iso3") or "").strip().upper()
        if iso:
            out[iso] = d
    return jsonify(out)


@app.route("/api/continent_trends")
def api_continent_trends():
    q = """
        WITH H AS (
            SELECT co.name AS continent, wh.year AS year, AVG(wh.hours) AS avg_hours
            FROM continents co
            JOIN countries c ON c.continent_id = co.continent_id
            LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
            JOIN working_hours_fact wh ON wh.country_id = c.country_id
            WHERE d.code IS NULL
            GROUP BY co.name, wh.year
        ),
        G AS (
            SELECT co.name AS continent, gd.year AS year, AVG(gd.rgdpe) AS avg_gdp_per_capita
            FROM continents co
            JOIN countries c ON c.continent_id = co.continent_id
            LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
            JOIN gdp_data gd ON gd.countrycode = c.code
            WHERE d.code IS NULL
            GROUP BY co.name, gd.year
        )
        SELECT
            H.continent,
            H.year,
            H.avg_hours,
            G.avg_gdp_per_capita
        FROM H
        LEFT JOIN G ON G.continent = H.continent AND G.year = H.year
        ORDER BY H.continent, H.year
    """
    data = rows(q)
    series: Dict[str, Dict[str, List[Any]]] = {}
    years = set()
    for d in data:
        cont = d["continent"]
        series.setdefault(cont, {"years": [], "hours": [], "gdp": []})
        series[cont]["years"].append(d["year"])
        series[cont]["hours"].append(d["avg_hours"])
        series[cont]["gdp"].append(d["avg_gdp_per_capita"])
        years.add(d["year"])
    return jsonify({"series": series, "years": sorted(years)})


@app.route("/api/continent_membership")
def api_continent_membership():
    data = rows("""
        SELECT
            UPPER(TRIM(c.code)) AS iso3,
            TRIM(co.name)       AS continent
        FROM countries c
        JOIN continents co ON co.continent_id = c.continent_id
        LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
        WHERE d.code IS NULL AND c.code IS NOT NULL AND TRIM(c.code) <> ''
    """)
    membership = {d["iso3"]: d["continent"] for d in data if d.get("iso3") and d.get("continent")}
    continents = sorted({d["continent"] for d in data if d.get("continent")})
    return jsonify({"membership": membership, "continents": continents})


@app.route("/api/timeseries")
def api_timeseries():
    code = request.args.get("code", type=str)
    country = request.args.get("country", type=str)

    cond = ""
    p = ()
    if code:
        cond = "UPPER(TRIM(c.code)) = UPPER(TRIM(?))"
        p = (code,)
    elif country:
        cond = "UPPER(TRIM(c.name)) = UPPER(TRIM(?))"
        p = (country,)
    else:
        return jsonify({"error": "missing code or country"}), 400

    # if deleted, return 404
    rdel = one(f"SELECT 1 FROM admin_deleted WHERE UPPER(code) = (SELECT UPPER(TRIM(code)) FROM countries c WHERE {cond})", p)
    if rdel:
        return jsonify({"error": "country deleted"}), 404

    q = f"""
        WITH BASE AS (
            SELECT c.country_id, c.name AS country, c.code AS iso3
            FROM countries c
            WHERE {cond}
        ),
        Y AS (
            SELECT DISTINCT year FROM working_hours_fact
            UNION
            SELECT DISTINCT year FROM gdp_data
        )
        SELECT
            B.country,
            B.iso3,
            Y.year,
            (SELECT wh.hours
               FROM working_hours_fact wh
              WHERE wh.country_id = B.country_id AND wh.year = Y.year
            ) AS hours,
            (SELECT gd.rgdpe
               FROM gdp_data gd
              WHERE gd.countrycode = B.iso3 AND gd.year = Y.year
            ) AS gdp_per_capita
        FROM BASE B CROSS JOIN Y
        ORDER BY Y.year
    """
    data = rows(q, p)
    if not data:
        return jsonify({"error": "country not found"}), 404

    years, hours, gdp = [], [], []
    meta_country = data[0]["country"]
    meta_code = data[0]["iso3"]
    for r in data:
        years.append(r["year"])
        hours.append(r["hours"])
        gdp.append(r["gdp_per_capita"])
    return jsonify({
        "country": meta_country,
        "code": meta_code,
        "years": years,
        "hours": hours,
        "gdp_per_capita": gdp,
    })


# report generation
@app.route("/api/report")
def api_report():
    rtype = request.args.get("type", type=int)
    year = request.args.get("year", type=int)
    if not rtype or rtype not in (1, 2, 3, 4, 5):
        return jsonify({"error": "invalid type; must be 1..5"}), 400
    if not year:
        return jsonify({"error": "missing year"}), 400

    text = ""

    if rtype == 1:
        continent = request.args.get("continent", type=str)
        if not continent:
            return jsonify({"error": "type=1 requires continent"}), 400
        avg_now = continent_avg_hours(year, continent)
        prev_year = year - 1
        avg_prev = continent_avg_hours(prev_year, continent)
        if avg_now is None or avg_prev is None:
            return jsonify({"error": "insufficient data for the selected continent/year"}), 400
        change = (avg_now - avg_prev) / avg_prev * 100.0 if avg_prev else 0.0
        incdec = "increase" if change >= 0 else "decrease"
        fastest = fastest_decline_since_1950()
        if fastest:
            fastest_cont, decline_rate = fastest
        else:
            fastest_cont, decline_rate = "N/A", 0.0
        text = (
            f"In {year}, workers in {continent} worked an average of {avg_now:.2f} hours, "
            f"representing a {abs(change):.2f}% {incdec} compared to {prev_year}. "
            f"The continent with the fastest decline since 1950 is {fastest_cont}, "
            f"with an average annual reduction of {decline_rate:.2f} hours."
        )

    elif rtype == 2:
        country_param = request.args.get("country", type=str)
        if not country_param:
            return jsonify({"error": "type=2 requires country (ISO3 or name)"}), 400
        ctx = country_hours(year, country_param)
        if not ctx:
            return jsonify({"error": "country not found"}), 404
        iso, country_name, chours, cont = ctx
        if chours is None:
            return jsonify({"error": "no working hours for this year"}), 404
        cont_avg_now = continent_avg_hours(year, cont)
        if cont_avg_now is None:
            return jsonify({"error": "no continent average for this year"}), 404

        diff = chours - cont_avg_now
        above_below = "above" if diff >= 0 else "below"

        past_year = max(year - 5, year - 1)
        ctx_past = country_hours(past_year, country_param)
        cont_avg_past = continent_avg_hours(past_year, cont)
        if ctx_past and ctx_past[2] is not None and cont_avg_past is not None:
            gap_now = chours - cont_avg_now
            gap_past = float(ctx_past[2]) - cont_avg_past
            delta = gap_now - gap_past
            trend = "widened" if abs(gap_now) > abs(gap_past) else "narrowed"
            delta_abs = abs(delta)
        else:
            trend, delta_abs = "changed", 0.0

        text = (
            f"In {year}, {country_name} recorded {chours:.0f} working hours per worker, which is {abs(diff):.0f} hours "
            f"{above_below} the {cont} average of {cont_avg_now:.0f}. Over the last five years, this gap has "
            f"{trend} by {delta_abs:.0f} hours."
        )

    elif rtype == 3:
        topn, botn = top_bottom_countries(year, 3, 2)
        if len(topn) < 3 or len(botn) < 2:
            return jsonify({"error": "insufficient data for ranking"}), 400
        top1, top2, top3 = topn[0]["country"], topn[1]["country"], topn[2]["country"]
        conts = {topn[0]["continent"], topn[1]["continent"], topn[2]["continent"]}
        continent_group = topn[0]["continent"] if len(conts) == 1 else "multiple continents"
        bottom1, bottom2 = botn[0]["country"], botn[1]["country"]
        text = (
            f"In {year}, the countries with the longest working hours were {top1}, {top2}, and {top3}, "
            f"all located in {continent_group}. Meanwhile, {bottom1} and {bottom2} had the shortest working hours, "
            f"reflecting stronger labor protections and productivity efficiency."
        )

    elif rtype == 4:
        event_name = request.args.get("event_name", type=str) or "the event"
        event_year = request.args.get("event_year", type=int)
        if not event_year:
            return jsonify({"error": "type=4 requires event_year"}), 400
        before_year = event_year - 1
        after_year = event_year + 1
        before_val = global_avg(before_year)
        after_val = global_avg(after_year)
        if before_val is None or after_val is None:
            return jsonify({"error": "insufficient global averages around event year"}), 400
        cont_shift = continent_shift(before_year, after_year)
        if cont_shift:
            cont_max, change = cont_shift
        else:
            cont_max, change = "N/A", 0.0
        text = (
            f"Following the {event_name} in {event_year}, the global average working hours decreased from "
            f"{before_val:.0f} to {after_val:.0f}, with {cont_max} showing the largest shift ({change:.0f} hours)."
        )

    elif rtype == 5:
        pairs = rows("""
            SELECT c.name AS country, c.code AS iso3, gd.rgdpe AS gdp, wh.hours AS hours
            FROM countries c
            LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
            JOIN gdp_data gd ON gd.countrycode = c.code AND gd.year = ?
            JOIN working_hours_fact wh ON wh.country_id = c.country_id AND wh.year = ?
            WHERE d.code IS NULL AND gd.rgdpe IS NOT NULL AND wh.hours IS NOT NULL
        """, (year, year))
        if not pairs:
            return jsonify({"error": "insufficient paired data for correlation"}), 400
        xs = [float(p["gdp"]) for p in pairs]
        ys = [float(p["hours"]) for p in pairs]
        corr = pearson(xs, ys) or 0.0

        country_param = request.args.get("country", type=str)
        example = None
        if country_param:
            for p in pairs:
                if str(p["iso3"]).upper() == str(country_param).upper() or str(p["country"]).lower() == str(country_param).lower():
                    example = p; break
        if not example:
            example = max(pairs, key=lambda x: x["gdp"])

        text = (
            f"Countries with higher GDP per capita tend to have shorter working hours. In {year}, the correlation "
            f"coefficient between GDP per capita and working hours across all countries was {corr:.2f}. "
            f"Notably, {example['country']} had a GDP per capita of {float(example['gdp']):.0f} and "
            f"{float(example['hours']):.0f} annual working hours."
        )

    return jsonify({"text": text})


# report generation with DeepSeek AI
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

def ds_request(key: str, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 600) -> str:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

@app.route("/api/report_ai")
def api_report_ai():
    rtype = request.args.get("type", type=int)
    year = request.args.get("year", type=int)
    if not rtype or rtype not in (1,2,3,4,5) or not year:
        return jsonify({"error": "invalid request"}), 400

    ds_key = request.headers.get("X-DS-KEY") or request.args.get("ds_key")
    if not ds_key:
        return jsonify({"error": "missing DeepSeek key"}), 400

    context: Dict[str, Any] = {"year": year, "type": rtype}

    if rtype == 1:
        continent = request.args.get("continent", type=str)
        if not continent:
            return jsonify({"error": "type=1 requires continent"}), 400
        avg_now = continent_avg_hours(year, continent)
        prev_year = year - 1
        avg_prev = continent_avg_hours(prev_year, continent)
        fastest = fastest_decline_since_1950()
        context.update({
            "continent": continent,
            "avg_hours": avg_now,
            "prev_year": prev_year,
            "avg_prev": avg_prev,
            "fastest_decline": fastest
        })

    elif rtype == 2:
        country_param = request.args.get("country", type=str)
        if not country_param:
            return jsonify({"error": "type=2 requires country"}), 400
        ctx = country_hours(year, country_param)
        if not ctx:
            return jsonify({"error": "country not found"}), 404
        iso, country, chours, cont = ctx
        cont_avg_now = continent_avg_hours(year, cont)
        past_year = max(year-5, year-1)
        ctx_past = country_hours(past_year, country_param)
        cont_avg_past = continent_avg_hours(past_year, cont)
        context.update({
            "country": country, "iso3": iso, "continent": cont,
            "country_hours": chours, "continent_avg": cont_avg_now,
            "past_year": past_year,
            "country_hours_past": (ctx_past[2] if ctx_past else None),
            "continent_avg_past": cont_avg_past
        })

    elif rtype == 3:
        topn, botn = top_bottom_countries(year, 3, 2)
        context.update({"topn": topn, "botn": botn})

    elif rtype == 4:
        event_name = request.args.get("event_name", type=str) or "the event"
        event_year = request.args.get("event_year", type=int)
        if not event_year:
            return jsonify({"error": "type=4 requires event_year"}), 400
        before_year = event_year - 1
        after_year  = event_year + 1
        before_val  = global_avg(before_year)
        after_val   = global_avg(after_year)
        cont_shift  = continent_shift(before_year, after_year)

        def top_growth_decline(event_year: int, limit: int = 3):
            q = """
                SELECT
                    c.name  AS country,
                    c.code  AS iso3,
                    gd_prev.rgdpe AS gdp_prev,
                    gd_curr.rgdpe AS gdp_curr,
                    (gd_curr.rgdpe - gd_prev.rgdpe) AS delta_abs,
                    CASE
                      WHEN gd_prev.rgdpe IS NOT NULL AND gd_prev.rgdpe <> 0
                      THEN (gd_curr.rgdpe - gd_prev.rgdpe) * 100.0 / gd_prev.rgdpe
                      ELSE NULL
                    END AS delta_pct
                FROM countries c
                LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
                JOIN gdp_data gd_prev ON gd_prev.countrycode = c.code AND gd_prev.year = ?
                JOIN gdp_data gd_curr ON gd_curr.countrycode = c.code AND gd_curr.year = ?
                WHERE d.code IS NULL AND gd_prev.rgdpe IS NOT NULL AND gd_curr.rgdpe IS NOT NULL
            """
            data = rows(q, (event_year - 1, event_year))
            data = [d for d in data if d["delta_pct"] is not None]
            top_up   = sorted(data, key=lambda x: x["delta_pct"], reverse=True)[:limit]
            top_down = sorted(data, key=lambda x: x["delta_pct"])[:limit]
            def _pick(z):
                return {
                    "country": z["country"],
                    "iso3": z["iso3"],
                    "delta_pct": float(z["delta_pct"]),
                    "delta_abs": float(z["delta_abs"])
                }
            return list(map(_pick, top_up)), list(map(_pick, top_down))

        top_up, top_down = top_growth_decline(event_year, limit=3)

        context.update({
            "event_name": event_name, "event_year": event_year,
            "before_year": before_year, "after_year": after_year,
            "before_value": before_val, "after_value": after_val,
            "continent_max_shift": cont_shift,
            "top_up": top_up, "top_down": top_down
        })

    elif rtype == 5:
        pairs = rows("""
            SELECT c.name AS country, c.code AS iso3, gd.rgdpe AS gdp, wh.hours AS hours
            FROM countries c
            LEFT JOIN admin_deleted d ON UPPER(c.code)=UPPER(d.code)
            JOIN gdp_data gd ON gd.countrycode = c.code AND gd.year = ?
            JOIN working_hours_fact wh ON wh.country_id = c.country_id AND wh.year = ?
            WHERE d.code IS NULL AND gd.rgdpe IS NOT NULL AND wh.hours IS NOT NULL
        """, (year, year))
        xs = [float(p["gdp"]) for p in pairs]
        ys = [float(p["hours"]) for p in pairs]
        corr = pearson(xs, ys) or 0.0
        example = max(pairs, key=lambda x: x["gdp"]) if pairs else None
        context.update({"pairs": pairs[:50], "corr": corr, "example": example})

    template = request.args.get("template", default="", type=str).strip()
    if not template:
        template = "Write a concise analytical paragraph in English using the provided data context."

    sys = (
        "You are a data journalist. Use ONLY the provided numbers; do not invent data. "
        "If some fields are null, say 'data unavailable'."
    )
    user = f"Template:\n{template}\n\nData context (JSON):\n{context}"

    try:
        content = ds_request(ds_key, [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ])
        return jsonify({"text": content, "context": context})
    except requests.HTTPError as e:
        return jsonify({"error": f"DeepSeek HTTP {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": f"DeepSeek request failed: {e}"}), 502


# admin endpoints
def is_admin(req) -> bool:
    pw = req.headers.get("X-ADMIN-PASS") or req.args.get("password") or ""
    return pw == ADMIN_PASSWORD

@app.route("/api/admin/countries")
def api_admin_countries():
    if not is_admin(request):
        return jsonify({"error": "unauthorized"}), 401

    all_c = rows("""
        SELECT c.code AS iso3, c.name AS country, co.name AS continent
        FROM countries c
        JOIN continents co ON co.continent_id = c.continent_id
        ORDER BY co.name, c.name
    """)
    deleted = rows("""
        SELECT d.code AS iso3,
               (SELECT name FROM countries WHERE UPPER(code)=UPPER(d.code)) AS country,
               (SELECT co.name FROM countries c JOIN continents co ON co.continent_id=c.continent_id
                WHERE UPPER(c.code)=UPPER(d.code)) AS continent,
               d.deleted_at, d.reason
        FROM admin_deleted d
        ORDER BY d.deleted_at DESC
    """)
    return jsonify({"countries": all_c, "deleted": deleted})

@app.route("/api/admin/delete", methods=["POST"])
def api_admin_delete():
    if not is_admin(request):
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    reason = (data.get("reason") or "").strip()
    if not code:
        return jsonify({"error": "missing code"}), 400

    sql_insert = "INSERT OR IGNORE INTO admin_deleted(code, deleted_at, reason) VALUES (?, ?, ?)"
    n = exec_sql(sql_insert, (code, datetime.utcnow().isoformat(timespec="seconds"), reason or None))
    return jsonify({
        "ok": True,
        "affected": n,
        "sql": sql_insert,
        "params": [code, "<utc_now>", reason or None]
    })

@app.route("/api/admin/restore", methods=["POST"])
def api_admin_restore():
    if not is_admin(request):
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    if not code:
        return jsonify({"error": "missing code"}), 400

    sql_delete = "DELETE FROM admin_deleted WHERE UPPER(code)=UPPER(?)"
    n = exec_sql(sql_delete, (code,))
    return jsonify({
        "ok": True,
        "affected": n,
        "sql": sql_delete,
        "params": [code]
    })


@app.route("/")
def root():
    return jsonify({
        "ok": True,
        "message": "Use /api/* endpoints. Admin endpoints: /api/admin/countries, /api/admin/delete, /api/admin/restore"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8010))
    print(f"\n➡ Serving on http://127.0.0.1:{port}\nDB={DB_PATH}\nAdmin password set: {'YES' if ADMIN_PASSWORD else 'NO'}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
