# Working Hours × GDP — Choropleth

An interactive visualization of **annual working hours per worker** and **GDP per capita**, combining a Google Maps choropleth, interactive time-series charts, automated narrative reports (rule-based + AI), and an **Admin mode** for managing countries in the database.

---

## Folder Structure

```text
main/
├─ app.py                          # Flask backend (REST API, reports, admin)
├─ index.html                      # Frontend UI (Google Maps + Chart.js)
├─ config.js                       # Frontend configuration (API base, keys, admin password)
├─ countries.geojson               # World country boundaries (ISO_A3 / ADMIN fields)
├─ build_db.py                     # Utility script to build working_hours.db from CSVs
├─ working_hours.db                # SQLite database used by the backend
├─ annual-working-hours-per-worker.csv   # Source: Our World in Data (raw)
├─ gdp-per-capita-worldbank.csv          # Source: World Bank (raw)
├─ map_continents.py               # Helper script mapping countries to continents
├─ requirements.txt                # Python dependencies for the backend
└─ readme.md                       # Project documentation (this file) 

```

## File Descriptions
app.py

Flask backend that powers all API endpoints:

/api/choropleth — Returns working hours & GDP per capita for map rendering.

/api/timeseries — Returns country-level time series data.

/api/continent_trends — Provides aggregated continent averages.

/api/report — Generates rule-based text reports.

/api/report_ai — Generates AI reports (using DeepSeek API).

/api/admin/* — Admin endpoints for soft-deleting/restoring countries.

Includes CORS support, automatic database initialization, and an in-memory cache for performance.

## index.html

Frontend UI integrating:

Google Maps choropleth visualization.

Country-level and continent-level charts (via Chart.js).

Report export (rule-based or AI-generated).

Admin mode:

View all countries and deleted countries.

Delete / restore records live.

See SQL statements executed.

Deleted countries appear as gray (no data) for visitors.

## config.js

Frontend configuration file:

window.APP_CONFIG = {
  GOOGLE_MAPS_API_KEY: "YOUR_GOOGLE_MAPS_KEY",
  DEEPSEEK_API_KEY: "YOUR_DEEPSEEK_KEY",     // optional, for AI reports
  ADMIN_PASSWORD: "admin123"                 // password for admin mode
};


## countries.geojson

GeoJSON dataset for all world countries.
Used by the map to draw boundaries and match ISO_A3 codes.
If missing, the app automatically loads a remote fallback.

## build_db.py

Utility script that constructs the SQLite database (working_hours.db) from the two CSVs:

Merges datasets.

Cleans and normalizes ISO3 codes.

Assigns each country to a continent (via map_continents.py).

## working_hours.db

The main SQLite database used by the backend, containing:

countries

continents

working_hours_fact

gdp_data

admin_deleted (for soft-deleted countries)

annual-working-hours-per-worker.csv

Raw dataset from Our World in Data, listing average annual working hours per worker.

gdp-per-capita-worldbank.csv

Raw dataset from World Bank, providing GDP per capita per year and country.

## map_continents.py

Script for mapping each country to its corresponding continent.
Used in build_db.py during database creation.

## requirements.txt

Python dependencies:

Flask
flask-cors
requests

## 🚀 How to Run
1. Backend Setup (Flask API)
(1) Install dependencies
### create a virtual environment
python -m venv .venv
### activate it
.venv\Scripts\activate        # on Windows
source .venv/bin/activate     # on macOS/Linux

### install requirements
pip install -r requirements.txt

(2) Option A — Use existing database

Simply use the provided working_hours.db.

(3) Option B — Rebuild database (optional)
python build_db.py

(4) Run the backend
### Optional environment variables
$env:PORT="8010"

### Start the Flask app
python app.py


Expected console output:

➡ Serving on http://127.0.0.1:8010
DB=E:\AIDM\AIDM7360_big_data\proj\main\working_hours.db

1. Frontend Setup
(1) Serve the frontend (required for CORS)

From the main/ folder:

python -m http.server 5500

(2) Open in browser:
http://127.0.0.1:5500/index.html?api=http://127.0.0.1:8010

(3) Set keys in config.js:
window.APP_CONFIG = {
  GOOGLE_MAPS_API_KEY: "YOUR_GOOGLE_KEY",
  DEEPSEEK_API_KEY: "YOUR_DEEPSEEK_KEY", 
  ADMIN_PASSWORD: "admin123"
};

## 🧭 How to Use

Choose year and layer → click “Update”.

Click a country to view its working hours and GDP trends.

Switch to “Continent” tab for multi-line trend charts.

Export reports

“Export” → rule-based text report.

“AI Export” → use DeepSeek API to generate summary text.

Admin mode

Click “Admin” and enter password.

Manage country visibility (delete / restore).

Deleted countries appear gray and are excluded from data queries.

SQL actions shown in the bottom admin log.

## ⚙️ API Reference
Endpoint	Description
/api/choropleth?year=2023	Returns per-country data for that year
/api/timeseries?code=IRN	Returns country-level time series
/api/continent_trends	Returns per-continent averages (hours + GDP)
/api/continent_membership	Returns ISO3 → continent mapping
/api/report?type=1..5	Generates rule-based report text
/api/report_ai?type=1..5	Generates AI-based report (DeepSeek)
/api/admin/countries	List all countries + deleted ones
/api/admin/delete	Soft-delete a country
/api/admin/restore	Restore a deleted country
## 🧠 Troubleshooting
Problem	Possible Cause	Fix
Backend error	Flask not running / wrong port	Start app.py on port 8010
CORS policy block	Opened HTML via file://	Use a local server (http://127.0.0.1:5500)
Blank map	Missing Google Maps API key	Add key to config.js
“country not found”	Country alias mismatch	Use ISO3 code or update alias map in app.py
AI report fails	No DeepSeek API key	Set DEEPSEEK_API_KEY in config.js

## 📊 Data & Attribution

Working hours: Our World in Data – Annual working hours per worker

GDP per capita: World Bank – GDP per capita (constant 2015 US$)

Use the data under their respective licenses.
This repository is for educational and visualization purposes only.
