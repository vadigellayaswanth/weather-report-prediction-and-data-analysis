from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, load_raw_dataset

MODEL_PATH = ROOT / "models" / "rainfall_model.pkl"
RESULTS_DIR = ROOT / "outputs" / "results"
GRAPH_DIR = ROOT / "outputs" / "graphs"

st.set_page_config(
    page_title="WEATHER REPORT PREDICTION & DATA ANALYSIS",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark weather dashboard styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b132b 0%, #1c2541 50%, #0b132b 100%);
        color: #e0e6ed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .hero-header {
        background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.15), transparent 70%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.1);
    }
    .hero-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-highlight {
        color: #38bdf8;
    }
    .prediction-box {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid #38bdf8;
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
    }
    .prediction-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .prediction-category {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 8px;
    }
    .cat-none { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }
    .cat-light { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .cat-moderate { background: rgba(59, 130, 246, 0.25); color: #60a5fa; }
    .cat-heavy { background: rgba(239, 68, 68, 0.25); color: #f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset_summary():
    try:
        raw_df = load_raw_dataset()
        total_records = len(raw_df)
        total_features = len(raw_df.columns)
    except Exception:
        total_records = 970339
        total_features = 15
    return total_records, total_features


@st.cache_data
def load_evaluation_data():
    result_path = RESULTS_DIR / "evaluation_report.csv"
    if not result_path.exists():
        return None
    return pd.read_csv(result_path)


@st.cache_data
def load_prediction_table():
    path = RESULTS_DIR / "rainfall_prediction_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_page():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_rainfall_category(mm: float) -> tuple[str, str, str]:
    if mm <= 0.1:
        return "No Rain / Trace", "cat-none", "Dry weather conditions observed."
    elif mm <= 7.5:
        return "Light Rain", "cat-light", "Light drizzle or intermittent showers."
    elif mm <= 64.4:
        return "Moderate Rain", "cat-moderate", "Typical steady rainfall conditions."
    elif mm <= 124.4:
        return "Heavy Rain", "cat-heavy", "Substantial showers; localized pooling possible."
    else:
        return "Very Heavy to Extreme Rain", "cat-heavy", "Intense downpour alert."


# WMO Weather Interpretation Code mapping
def decode_wmo_weather(code: int) -> tuple[str, str]:
    mapping = {
        0: ("Clear Sky", "☀️"),
        1: ("Mainly Clear", "🌤️"),
        2: ("Partly Cloudy", "🌤️"),
        3: ("Overcast", "☁️"),
        45: ("Fog", "🌫️"),
        48: ("Depositing Rime Fog", "🌫️"),
        51: ("Light Drizzle", "🌦️"),
        53: ("Moderate Drizzle", "🌦️"),
        55: ("Dense Drizzle", "🌦️"),
        61: ("Slight Rain", "🌧️"),
        63: ("Moderate Rain", "🌧️"),
        65: ("Heavy Rain", "🌧️"),
        71: ("Slight Snow", "🌨️"),
        73: ("Moderate Snow", "🌨️"),
        75: ("Heavy Snow", "🌨️"),
        80: ("Slight Rain Showers", "🌧️"),
        81: ("Moderate Rain Showers", "🌧️"),
        82: ("Violent Rain Showers", "🌧️"),
        95: ("Thunderstorm", "⛈️"),
        96: ("Thunderstorm with Slight Hail", "⛈️"),
        99: ("Thunderstorm with Heavy Hail", "⛈️"),
    }
    return mapping.get(code, ("Observed Weather", "🌤️"))


# Live Open-Meteo Weather API Fetcher
@st.cache_data(ttl=300)
def fetch_live_weather(lat: float, lon: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,surface_pressure,wind_speed_10m,wind_direction_10m"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&timezone=auto&forecast_days=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "WeatherPredictionApp/1.0"})
    with urllib.request.urlopen(req, timeout=6) as response:
        return json.loads(response.read().decode("utf-8"))


# Reverse Geocoding helper for browser detected coordinates
@st.cache_data(ttl=3600)
def reverse_geocode(lat: float, lon: float) -> str:
    url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat:.4f}&longitude={lon:.4f}&localityLanguage=en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WeatherPredictionApp/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            d = json.loads(resp.read().decode("utf-8"))
            city = d.get("city") or d.get("locality") or "Detected Area"
            state = d.get("principalSubdivision") or ""
            country = d.get("countryName") or ""
            parts = [p for p in [city, state, country] if p]
            return ", ".join(parts) if parts else f"{lat:.2f}°N, {lon:.2f}°E"
    except Exception:
        return f"{lat:.2f}°N, {lon:.2f}°E"


# Geocoding search function
@st.cache_data(ttl=3600)
def search_locations(query: str):
    if not query or len(query.strip()) < 3:
        return []
    encoded = urllib.parse.quote(query.strip())
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=5&language=en&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "WeatherPredictionApp/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("results", [])
    except Exception:
        return []


# Load assets
total_records, total_features = load_dataset_summary()
model = load_model_page()
report = load_evaluation_data()
prediction_table = load_prediction_table()

# Model and metrics extraction
if report is not None and not report.empty:
    report_row = report.iloc[0]
    selected_model_name = report_row.get("selected_model", "Random Forest Regressor")
    mae_val = float(report_row.get("mae", 10.9217))
    rmse_val = float(report_row.get("rmse", 17.3710))
    r2_val = float(report_row.get("r2", 0.0303))
else:
    selected_model_name = "Random Forest Regressor"
    mae_val = 10.9217
    rmse_val = 17.3710
    r2_val = 0.0303

# Check query params for browser geolocation callback
query_params = st.query_params
if "lat" in query_params and "lon" in query_params:
    try:
        st.session_state["detected_lat"] = float(query_params["lat"])
        st.session_state["detected_lon"] = float(query_params["lon"])
        st.session_state["detected_active"] = True
    except Exception:
        pass

# 1. Project Header
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">WEATHER REPORT PREDICTION & DATA ANALYSIS</div>
        <div class="hero-subtitle">Machine Learning Based Rainfall Prediction</div>
        <span class="hero-badge">Batch-3 | Data Analysis Essentials</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Dataset Overview
st.markdown("### 📊 Dataset & System Overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Dataset Records</div>
            <div class="metric-val metric-highlight">{total_records:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Weather Features</div>
            <div class="metric-val">{len(FEATURE_COLUMNS)} Active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Selected ML Model</div>
            <div class="metric-val" style="font-size: 1.25rem; margin-top: 8px;">{selected_model_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Prediction Target</div>
            <div class="metric-val metric-highlight" style="font-size: 1.25rem; margin-top: 8px;">Rainfall (mm)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# 3. Live Weather Conditions
st.markdown("### 🌤️ LIVE WEATHER CONDITIONS")
st.caption("Real-time weather monitoring using live meteorological data from Open-Meteo API.")

# Browser Geolocation Component
geo_html = """
<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <button onclick="getLocation()" style="background: #0ea5e9; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 6px;">
            📍 Detect My Location
        </button>
        <span id="geo-status" style="color: #94a3b8; font-size: 0.88rem;">Click to request browser location</span>
    </div>
    <span style="color: #64748b; font-size: 0.8rem;">Native Browser Geolocation API</span>
</div>

<script>
function getLocation() {
    const status = document.getElementById('geo-status');
    if (!navigator.geolocation) {
        status.innerText = "❌ Geolocation is not supported by your browser.";
        status.style.color = "#f87171";
        return;
    }
    status.innerText = "⏳ Requesting device GPS / location permission...";
    status.style.color = "#38bdf8";

    navigator.geolocation.getCurrentPosition(
        function(position) {
            const lat = position.coords.latitude.toFixed(4);
            const lon = position.coords.longitude.toFixed(4);
            status.innerText = "✅ Location detected: " + lat + "°N, " + lon + "°E. Updating dashboard...";
            status.style.color = "#4ade80";
            
            // Set query params on top window to update Streamlit state
            const url = new URL(window.top.location);
            url.searchParams.set('lat', lat);
            url.searchParams.set('lon', lon);
            window.top.location.href = url.toString();
        },
        function(error) {
            let msg = "Location error: ";
            if (error.code === 1) msg = "⚠️ Location permission denied. Please allow location access or select your city manually.";
            else if (error.code === 2) msg = "⚠️ Location unavailable. Please select your city manually.";
            else if (error.code === 3) msg = "⚠️ Location request timed out. Please select your city manually.";
            else msg = msg + error.message;
            status.innerText = msg;
            status.style.color = "#f87171";
        },
        { enableHighAccuracy: true, timeout: 8000, maximumAge: 0 }
    );
}
</script>
"""
components.html(geo_html, height=65)

# Location Selection & Detected State Resolution
default_cities = {
    "Rajamahendravaram, Andhra Pradesh, India (Default)": (17.0000, 81.7800),
    "Visakhapatnam, Andhra Pradesh, India": (17.6868, 83.2185),
    "Vijayawada, Andhra Pradesh, India": (16.5062, 80.6480),
    "Hyderabad, Telangana, India": (17.3850, 78.4867),
    "Bengaluru, Karnataka, India": (12.9716, 77.5946),
    "Chennai, Tamil Nadu, India": (13.0827, 80.2707),
    "Mumbai, Maharashtra, India": (19.0760, 72.8777),
    "Delhi, India": (28.6139, 77.2090),
}

loc_col1, loc_col2 = st.columns([2, 1])

# If browser location was detected, prioritize it
if st.session_state.get("detected_active", False):
    target_lat = st.session_state["detected_lat"]
    target_lon = st.session_state["detected_lon"]
    detected_name = reverse_geocode(target_lat, target_lon)
    selected_preset = f"📍 {detected_name} (Detected Location)"
    with loc_col1:
        st.success(f"📍 Active Detected Location: **{detected_name}** ({target_lat:.4f}°N, {target_lon:.4f}°E)")
        if st.button("🔄 Switch Back to City List"):
            st.session_state["detected_active"] = False
            st.query_params.clear()
            st.rerun()
    with loc_col2:
        custom_search = st.text_input("🔍 Or Search Another City:", placeholder="e.g. Tirupati, Pune")
        if custom_search and len(custom_search.strip()) >= 3:
            search_res = search_locations(custom_search)
            if search_res:
                chosen = search_res[0]
                target_lat = float(chosen["latitude"])
                target_lon = float(chosen["longitude"])
                selected_preset = f"{chosen['name']}, {chosen.get('admin1', '')}, {chosen.get('country', '')}"
                st.session_state["detected_active"] = False
else:
    with loc_col1:
        selected_preset = st.selectbox("📍 Select City / Weather Monitoring Station:", list(default_cities.keys()), index=0)
        target_lat, target_lon = default_cities[selected_preset]
    with loc_col2:
        custom_search = st.text_input("🔍 Or Search Another City:", placeholder="e.g. Tirupati, Pune")
        if custom_search and len(custom_search.strip()) >= 3:
            search_res = search_locations(custom_search)
            if search_res:
                chosen = search_res[0]
                target_lat = float(chosen["latitude"])
                target_lon = float(chosen["longitude"])
                selected_preset = f"{chosen['name']}, {chosen.get('admin1', '')}, {chosen.get('country', '')}"
                st.success(f"Located: {selected_preset} ({target_lat:.2f}°N, {target_lon:.2f}°E)")

# Fetch Live Weather
live_weather_data = None
live_err = None
try:
    live_weather_data = fetch_live_weather(target_lat, target_lon)
except Exception as e:
    live_err = str(e)

if live_weather_data and "current" in live_weather_data:
    curr = live_weather_data["current"]
    elevation_val = float(live_weather_data.get("elevation", 29.0))
    daily = live_weather_data.get("daily", {})
    hourly = live_weather_data.get("hourly", {})

    # Compute actual minimum and maximum temperatures from hourly observations of current local day
    if "temperature_2m" in hourly and hourly["temperature_2m"]:
        hourly_temps = hourly["temperature_2m"][:24]
        t_min_live = float(min(hourly_temps))
        t_max_live = float(max(hourly_temps))
    else:
        t_min_live = float(daily.get("temperature_2m_min", [curr["temperature_2m"] - 4.0])[0])
        t_max_live = float(daily.get("temperature_2m_max", [curr["temperature_2m"] + 4.0])[0])

    t_curr = float(curr.get("temperature_2m", 25.0))
    t_apparent = float(curr.get("apparent_temperature", t_curr))
    rh_curr = int(curr.get("relative_humidity_2m", 50))
    w_speed = float(curr.get("wind_speed_10m", 10.0))
    rain_curr = float(curr.get("rain", 0.0))
    press_curr = float(curr.get("surface_pressure", 1010.0))
    code_curr = int(curr.get("weather_code", 0))
    cond_text, cond_icon = decode_wmo_weather(code_curr)
    time_str = curr.get("time", datetime.now().strftime("%Y-%m-%d %H:%M"))

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 12px 18px; margin-bottom: 16px;">
            <span style="color: #38bdf8; font-weight: 700;">📡 Live Observation Source:</span> Open-Meteo Forecast API &bull;
            <strong>Location:</strong> {selected_preset} ({target_lat:.2f}°N, {target_lon:.2f}°E) &bull;
            <strong>Elevation:</strong> {elevation_val:.0f} m &bull;
            <strong>Last Updated:</strong> {time_str}
        </div>
        """,
        unsafe_allow_html=True,
    )

    w1, w2, w3, w4, w5, w6 = st.columns(6)
    with w1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Condition</div>
                <div class="metric-val" style="font-size: 1.3rem;">{cond_icon} {cond_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Temperature</div>
                <div class="metric-val metric-highlight">{t_curr:.1f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Feels Like</div>
                <div class="metric-val">{t_apparent:.1f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Humidity</div>
                <div class="metric-val">{rh_curr} %</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Wind Speed</div>
                <div class="metric-val">{w_speed:.1f} km/h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Observed Rain</div>
                <div class="metric-val metric-highlight">{rain_curr:.1f} mm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Live Weather hourly trend plot
    if "temperature_2m" in hourly and hourly["temperature_2m"]:
        h_df = pd.DataFrame({
            "Time": [t.split("T")[1][:5] for t in hourly["time"][:24]],
            "Temperature (°C)": hourly["temperature_2m"][:24],
            "Humidity (%)": hourly["relative_humidity_2m"][:24],
            "Wind Speed (km/h)": hourly["wind_speed_10m"][:24],
        }).set_index("Time")
        st.caption("📈 Live 24-Hour Diurnal Meteorological Observation Curves (Open-Meteo):")
        st.line_chart(h_df, height=210)

    # ── Auto-store live weather values into session state whenever fresh data loads ──
    # This ensures the prediction form is always pre-populated with real values,
    # without requiring the user to click an extra button.
    _new_loc_key = f"{target_lat:.4f}_{target_lon:.4f}"
    if st.session_state.get("_last_live_loc") != _new_loc_key:
        # Location changed or first load — refresh all stored values
        st.session_state["live_avg_temp"] = round(float(t_curr), 2)
        st.session_state["live_min_temp"] = round(float(t_min_live), 2)
        st.session_state["live_max_temp"] = round(float(t_max_live), 2)
        st.session_state["live_wind_speed"] = round(float(w_speed), 2)
        st.session_state["live_air_pressure"] = round(float(press_curr), 2)
        st.session_state["live_elevation"] = round(float(elevation_val), 1)
        st.session_state["live_latitude"] = round(float(target_lat), 4)
        st.session_state["live_longitude"] = round(float(target_lon), 4)
        st.session_state["_last_live_loc"] = _new_loc_key
        st.session_state["auto_prediction_source"] = selected_preset

    # ── OPTION 1: Automatic Location-Based Instant Prediction ──
    st.markdown(
        """
        <div style="background: rgba(14,165,233,0.08); border: 1px solid rgba(56,189,248,0.3);
                    border-radius: 10px; padding: 14px 18px; margin-top: 12px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">
                📍 OPTION 1: Automatic Location-Based Prediction
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem;">
                Detect Location → Fetch Live Weather → Populate All Inputs → Run Model
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show the exact values that will be used for Option 1 prediction
    v_avg = st.session_state["live_avg_temp"]
    v_min = st.session_state["live_min_temp"]
    v_max = st.session_state["live_max_temp"]
    v_wind = st.session_state["live_wind_speed"]
    v_press = st.session_state["live_air_pressure"]
    v_elev = st.session_state["live_elevation"]
    v_lat = st.session_state["live_latitude"]
    v_lon = st.session_state["live_longitude"]

    # Guard: check for zero/invalid temperature values and warn
    _temps_valid = (v_avg != 0.0 or v_min != 0.0 or v_max != 0.0)
    _temps_consistent = (v_min <= v_avg <= v_max)

    st.markdown(
        f"""
        <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(56,189,248,0.15);
                    border-radius: 8px; padding: 12px 16px; margin-top: 10px;
                    font-size: 0.88rem; color: #cbd5e1;">
            <strong style="color: #38bdf8;">🔎 Prediction Source:</strong>
            Automatic Location Weather Data &nbsp;|&nbsp;
            <strong>Location:</strong> {selected_preset} ({v_lat:.2f}°N, {v_lon:.2f}°E)<br>
            <strong>Avg Temp:</strong> {v_avg:.2f}°C &nbsp;|&nbsp;
            <strong>Min Temp:</strong> {v_min:.2f}°C &nbsp;|&nbsp;
            <strong>Max Temp:</strong> {v_max:.2f}°C &nbsp;|&nbsp;
            <strong>Wind:</strong> {v_wind:.2f} km/h &nbsp;|&nbsp;
            <strong>Pressure:</strong> {v_press:.2f} hPa &nbsp;|&nbsp;
            <strong>Elevation:</strong> {v_elev:.1f} m
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _temps_valid:
        st.warning(
            "⚠️ Temperature values appear to be zero. The live weather API may not have returned valid data. "
            "Please switch to Manual Input (Option 2) below or try a different location."
        )
    elif not _temps_consistent:
        st.warning(
            f"⚠️ Temperature consistency issue: Min ({v_min:.1f}°C) > Avg ({v_avg:.1f}°C) or Avg > Max ({v_max:.1f}°C). "
            "Please use Manual Input to correct values before predicting."
        )

    predict_loc_btn = st.button(
        "🌧️ Predict Rainfall Using My Location",
        disabled=(not _temps_valid or not _temps_consistent),
        help="Uses live weather values shown above. Disabled if temperature values are invalid.",
    )

    # 4. Instant Location-Based Prediction Execution
    if predict_loc_btn and model is not None:
        try:
            from src.predict_rainfall import validate_prediction_input

            validated_loc = validate_prediction_input(
                avg_temp=v_avg,
                min_temp=v_min,
                max_temp=v_max,
                wind_speed=v_wind,
                air_pressure=v_press,
                elevation=v_elev,
                latitude=v_lat,
                longitude=v_lon,
            )
            loc_input_df = pd.DataFrame([validated_loc])
            loc_prediction = float(model.predict(loc_input_df)[0])
            st.session_state["last_prediction"] = loc_prediction
            loc_cat, loc_class, loc_desc = get_rainfall_category(loc_prediction)

            st.markdown(
                f"""
                <div class="prediction-box" style="border: 2px solid #38bdf8;">
                    <div style="font-size: 0.85rem; color: #38bdf8; text-transform: uppercase; font-weight: 700;">
                        🤖 Automatic Location Prediction Result
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                        Prediction Source: <strong style="color:#94a3b8;">Automatic Location Weather Data</strong>
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                        📍 {selected_preset} &bull; Avg {v_avg:.1f}°C | Min {v_min:.1f}°C | Max {v_max:.1f}°C | Wind {v_wind:.1f} km/h | {v_press:.1f} hPa
                    </div>
                    <div class="prediction-val">{loc_prediction:.2f} mm</div>
                    <span class="prediction-category {loc_class}">{loc_cat}</span>
                    <div style="margin-top: 8px; color: #cbd5e1; font-size: 0.9rem;">{loc_desc}</div>
                    <div style="margin-top: 10px; font-size: 0.82rem; color: #94a3b8; border-top: 1px solid rgba(56, 189, 248, 0.2); padding-top: 8px;">
                        <strong>Model:</strong> {selected_model_name} &bull;
                        <em>Statistical ML regression based on live meteorological parameters and historical Indian rainfall patterns.</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"⚠️ Location Prediction Error: {e}")
else:
    st.error(f"Live weather data is currently unavailable: {live_err if live_err else 'API did not return observation data'}. The machine learning dashboard remains fully operational.")
    t_curr = None
    t_min_live = None
    t_max_live = None
    rh_curr = 65
    w_speed = None
    rain_curr = 0.0
    press_curr = None
    elevation_val = None

st.write("")

# 5. Model Performance Section
st.markdown("### 🎯 Model Performance & Evaluation")
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #38bdf8;">
            <div class="metric-label">Mean Absolute Error (MAE)</div>
            <div class="metric-val">{mae_val:.4f} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Average absolute difference between actual & predicted rainfall</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #60a5fa;">
            <div class="metric-label">Root Mean Squared Error (RMSE)</div>
            <div class="metric-val">{rmse_val:.4f} <span style="font-size: 0.9rem; color: #94a3b8;">mm</span></div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Penalizes higher deviation errors on intense rainfall spikes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #818cf8;">
            <div class="metric-label">R² Score (Coefficient of Determination)</div>
            <div class="metric-val">{r2_val:.4f}</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Evaluation on chronologically held-out test data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# 6. Weather AI Voice Assistant
st.markdown("### 🎙️ Weather AI Voice Assistant")
st.markdown(
    """
    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
        <span style="color: #38bdf8; font-weight: 700;">Supported Commands:</span>
        <span style="color: #cbd5e1; font-size: 0.9rem;">
            "Detect my location" &bull; "Show current weather" &bull; "Read temperature" &bull; "Read humidity" &bull; "Read wind speed" &bull; "Read current rainfall" &bull; "Use live weather for prediction" &bull; "Predict rainfall at my location" &bull; "Show model performance" &bull; "Show feature importance" &bull; "Show rainfall distribution" &bull; "Read prediction" &bull; "Help"
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

last_pred_val = st.session_state.get("last_prediction", None)
last_pred_text = f"{last_pred_val:.2f} millimeters" if last_pred_val is not None else "No prediction made yet. Please enter weather values in the form below."
live_avail = "true" if live_weather_data is not None else "false"

voice_assistant_html = f"""
<div style="background: #0f172a; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 18px; color: #f8fafc; font-family: sans-serif;">
    <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;">
        <button id="start-btn" onclick="startListening()" style="background: #0284c7; color: white; border: none; padding: 9px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 6px;">
            🎙️ Start Voice Assistant
        </button>
        <button id="stop-btn" onclick="stopListening()" style="background: #475569; color: white; border: none; padding: 9px 18px; border-radius: 8px; font-weight: 600; cursor: pointer;" disabled>
            ⏹️ Stop Voice Assistant
        </button>
        <span id="status-badge" style="background: rgba(148, 163, 184, 0.2); color: #94a3b8; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem;">
            Idle
        </span>
    </div>
    <div style="margin-top: 8px; font-size: 0.9rem;">
        <strong>Recognized Command:</strong> <span id="recognized-text" style="color: #38bdf8; font-style: italic;">None</span>
    </div>
    <div style="margin-top: 6px; font-size: 0.9rem;">
        <strong>Assistant Response:</strong> <span id="assistant-response" style="color: #e2e8f0;">Ready to assist. Click 'Start Voice Assistant' and speak.</span>
    </div>
</div>

<script>
let recognition = null;
const lastPredictionText = "{last_pred_text}";
const maeVal = "{mae_val:.4f}";
const rmseVal = "{rmse_val:.4f}";
const r2Val = "{r2_val:.4f}";
const liveAvail = {live_avail};
const currentTemp = "{t_curr:.1f}";
const currentHumidity = "{rh_curr}";
const currentWind = "{w_speed:.1f}";
const currentRain = "{rain_curr:.1f}";

function speakText(text) {{
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }}
}}

function handleCommand(transcript) {{
    const cmd = transcript.toLowerCase();
    let response = "";
    
    if (cmd.includes("detect my location") || cmd.includes("detect location")) {{
        response = "Detecting your location. Please allow the browser location prompt.";
        if (navigator.geolocation) {{
            navigator.geolocation.getCurrentPosition(function(pos) {{
                const url = new URL(window.top.location);
                url.searchParams.set('lat', pos.coords.latitude.toFixed(4));
                url.searchParams.set('lon', pos.coords.longitude.toFixed(4));
                window.top.location.href = url.toString();
            }});
        }}
    }} else if (cmd.includes("predict rainfall at my location") || cmd.includes("predict at my location")) {{
        response = "Using your live location weather data to generate a machine learning rainfall prediction.";
        const btn = window.parent.document.querySelector('button[kind="secondaryFormSubmit"], button:has-text("Predict Rainfall Using My Location")');
        if (btn) btn.click();
    }} else if (cmd.includes("current weather") || cmd.includes("weather conditions")) {{
        if (liveAvail) {{
            response = "Current weather temperature is " + currentTemp + " degrees Celsius, humidity is " + currentHumidity + " percent, wind speed is " + currentWind + " kilometers per hour, and observed rainfall is " + currentRain + " millimeters.";
        }} else {{
            response = "Live weather information is currently unavailable.";
        }}
    }} else if (cmd.includes("read temperature") || cmd.includes("temperature")) {{
        response = liveAvail ? "The current live temperature is " + currentTemp + " degrees Celsius." : "Live weather information is currently unavailable.";
    }} else if (cmd.includes("read humidity") || cmd.includes("humidity")) {{
        response = liveAvail ? "The relative humidity is " + currentHumidity + " percent." : "Live weather information is currently unavailable.";
    }} else if (cmd.includes("read wind") || cmd.includes("wind speed")) {{
        response = liveAvail ? "The current wind speed is " + currentWind + " kilometers per hour." : "Live weather information is currently unavailable.";
    }} else if (cmd.includes("current rainfall") || cmd.includes("observed rainfall")) {{
        response = liveAvail ? "The observed rainfall is " + currentRain + " millimeters." : "Live weather information is currently unavailable.";
    }} else if (cmd.includes("use live weather") || cmd.includes("use live")) {{
        response = "Live weather values have been populated into the prediction form.";
        const el = window.parent.document.querySelector('[id*="prediction_form"]');
        if (el) el.scrollIntoView({{behavior: "smooth"}});
    }} else if (cmd.includes("model performance") || cmd.includes("performance") || cmd.includes("metric")) {{
        response = "The model has a Mean Absolute Error of " + maeVal + ", an RMSE of " + rmseVal + ", and an R-squared of " + r2Val + ".";
    }} else if (cmd.includes("feature importance") || cmd.includes("importance")) {{
        response = "Displaying the feature importance graph in the dashboard below.";
        const el = window.parent.document.querySelector('[id*="feature_importance"]');
        if (el) el.scrollIntoView({{behavior: "smooth"}});
    }} else if (cmd.includes("rainfall distribution") || cmd.includes("distribution")) {{
        response = "Displaying the rainfall distribution graph below.";
        const el = window.parent.document.querySelector('[id*="rainfall_distribution"]');
        if (el) el.scrollIntoView({{behavior: "smooth"}});
    }} else if (cmd.includes("read prediction") || cmd.includes("last prediction") || cmd.includes("what is the prediction")) {{
        response = "The latest machine learning rainfall prediction is " + lastPredictionText;
    }} else if (cmd.includes("help") || cmd.includes("command")) {{
        response = "Available voice commands: Detect my location, Predict rainfall at my location, Show current weather, Read temperature, Read humidity, Read wind speed, Read current rainfall, Use live weather for prediction, Show model performance, Show feature importance, Show rainfall distribution, Read prediction, and Help.";
    }} else if (cmd.includes("predict") || cmd.includes("form")) {{
        response = "Please enter the weather parameters in the rainfall prediction form below and click Predict Rainfall.";
        const el = window.parent.document.querySelector('[id*="prediction_form"]');
        if (el) el.scrollIntoView({{behavior: "smooth"}});
    }} else {{
        response = "Command not recognized. Say 'Help' to hear available voice commands.";
    }}
    
    document.getElementById("assistant-response").innerText = response;
    speakText(response);
}}

function startListening() {{
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {{
        document.getElementById("status-badge").innerText = "Speech API Not Supported";
        document.getElementById("status-badge").style.color = "#f87171";
        document.getElementById("assistant-response").innerText = "Web Speech API is not supported in this browser. Please use Chrome or Edge.";
        return;
    }}
    
    try {{
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.continuous = false;
        
        recognition.onstart = function() {{
            document.getElementById("status-badge").innerText = "Listening...";
            document.getElementById("status-badge").style.color = "#38bdf8";
            document.getElementById("start-btn").disabled = true;
            document.getElementById("stop-btn").disabled = false;
        }};
        
        recognition.onresult = function(event) {{
            const transcript = event.results[0][0].transcript;
            document.getElementById("recognized-text").innerText = '"' + transcript + '"';
            handleCommand(transcript);
        }};
        
        recognition.onerror = function(event) {{
            document.getElementById("status-badge").innerText = "Error: " + event.error;
            document.getElementById("status-badge").style.color = "#f87171";
            document.getElementById("start-btn").disabled = false;
            document.getElementById("stop-btn").disabled = true;
        }};
        
        recognition.onend = function() {{
            document.getElementById("status-badge").innerText = "Idle";
            document.getElementById("status-badge").style.color = "#94a3b8";
            document.getElementById("start-btn").disabled = false;
            document.getElementById("stop-btn").disabled = true;
        }};
        
        recognition.start();
    }} catch (e) {{
        document.getElementById("assistant-response").innerText = "Microphone error: " + e.message;
    }}
}}

function stopListening() {{
    if (recognition) {{
        recognition.stop();
    }}
}}
</script>
"""
components.html(voice_assistant_html, height=160)

st.write("")

# 7. Rainfall Prediction Section — OPTION 2: Manual Input
st.markdown("### 🌧️ Rainfall Prediction")

st.markdown(
    """
    <div style="background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.2);
                border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="color: #94a3b8; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">
            ✍️ OPTION 2: Manual Weather Input
        </div>
        <div style="color: #64748b; font-size: 0.85rem;">
            Enter weather values manually → Predict rainfall for the exact inputs you provide.
            Values are pre-filled from live weather when available — you may edit them freely.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pre-fill defaults: use live weather session state values (already set from API above),
# falling back to dataset medians ONLY if no live data was ever stored.
# This guarantees the form never shows 0.0 when real data is available.
_ss = st.session_state
default_avg  = _ss.get("live_avg_temp",  26.7)
default_min  = _ss.get("live_min_temp",  22.4)
default_max  = _ss.get("live_max_temp",  31.6)
default_wind = _ss.get("live_wind_speed", 8.4)
default_press = _ss.get("live_air_pressure", 1009.8)
default_elev = _ss.get("live_elevation",  139.0)
default_lat  = _ss.get("live_latitude",   20.8)
default_lon  = _ss.get("live_longitude",  78.1)

# Safety guard: if any temperature is zero and live weather succeeded, use dataset medians
if default_avg == 0.0 and default_min == 0.0 and default_max == 0.0:
    default_avg, default_min, default_max = 26.7, 22.4, 31.6
if default_wind == 0.0:
    default_wind = 8.4
if default_press == 0.0:
    default_press = 1009.8

if MODEL_PATH.exists():
    # Show where defaults came from
    _source_label = "Live Weather" if _ss.get("_last_live_loc") else "Dataset Median"
    st.caption(
        f"📋 Form pre-filled from: **{_source_label}** — "
        f"Avg {default_avg:.1f}°C | Min {default_min:.1f}°C | Max {default_max:.1f}°C | "
        f"Wind {default_wind:.1f} km/h | Pressure {default_press:.1f} hPa. "
        "You can edit any value before predicting."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            avg_temp   = st.number_input("Average Temperature (°C)",  value=float(default_avg),   min_value=-20.0,  max_value=60.0,   step=0.5,  help="Dataset median: ~26.7°C")
            min_temp   = st.number_input("Minimum Temperature (°C)",  value=float(default_min),   min_value=-25.0,  max_value=50.0,   step=0.5,  help="Dataset median: ~22.4°C")
            max_temp   = st.number_input("Maximum Temperature (°C)",  value=float(default_max),   min_value=-15.0,  max_value=65.0,   step=0.5,  help="Dataset median: ~31.6°C")
            wind_speed = st.number_input("Wind Speed (km/h)",          value=float(default_wind),  min_value=0.0,    max_value=150.0,  step=0.5,  help="Dataset median: ~8.4 km/h")
        with col2:
            air_pressure = st.number_input("Air Pressure (hPa)",       value=float(default_press), min_value=850.0,  max_value=1060.0, step=1.0,  help="Dataset median: ~1009.8 hPa")
            elevation    = st.number_input("Elevation (meters)",        value=float(default_elev),  min_value=-100.0, max_value=6000.0, step=10.0, help="Dataset median: ~139.0 m")
            latitude     = st.number_input("Latitude (°N)",             value=float(default_lat),   min_value=-90.0,  max_value=90.0,   step=0.1,  help="Dataset mean: ~20.8°N")
            longitude    = st.number_input("Longitude (°E)",            value=float(default_lon),   min_value=-180.0, max_value=180.0,  step=0.1,  help="Dataset mean: ~78.1°E")

        submitted = st.form_submit_button("🌧️ Predict Rainfall", use_container_width=True)

    if submitted:
        try:
            from src.predict_rainfall import validate_prediction_input

            validated = validate_prediction_input(
                avg_temp, min_temp, max_temp, wind_speed, air_pressure, elevation, latitude, longitude
            )
            input_df = pd.DataFrame([validated])
            prediction = float(model.predict(input_df)[0])
            st.session_state["last_prediction"] = prediction
            category, cat_class, desc = get_rainfall_category(prediction)

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">
                        ✍️ Manual Input Prediction Result
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                        Prediction Source: <strong style="color:#94a3b8;">Manual Weather Input</strong>
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                        Avg {avg_temp:.1f}°C | Min {min_temp:.1f}°C | Max {max_temp:.1f}°C |
                        Wind {wind_speed:.1f} km/h | {air_pressure:.1f} hPa |
                        Elev {elevation:.0f}m | {latitude:.2f}°N, {longitude:.2f}°E
                    </div>
                    <div class="prediction-val">{prediction:.2f} mm</div>
                    <span class="prediction-category {cat_class}">{category}</span>
                    <div style="margin-top: 8px; color: #cbd5e1; font-size: 0.9rem;">{desc}</div>
                    <div style="margin-top: 10px; font-size: 0.82rem; color: #94a3b8;
                                border-top: 1px solid rgba(56, 189, 248, 0.2); padding-top: 8px;">
                        <strong>Model:</strong> {selected_model_name} &bull;
                        <em>Statistical ML regression using the values you entered above.</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except ValueError as exc:
            st.warning(f"⚠️ Validation Notice: {exc}")
else:
    st.warning("Model file not found. Please verify models/rainfall_model.pkl exists.")

st.write("")

# 8. Weather Analytics & Visualizations
st.markdown("---")
st.markdown("## 📊 Exploratory Data Analysis")
st.markdown("Visualizing meteorological patterns, weather distributions, and historical rainfall relationships:")

# Display 01 as full width since it stacks all weather features
p01 = GRAPH_DIR / "01_numerical_feature_histograms.png"
if p01.exists():
    st.image(str(p01), caption="01. Numerical Feature Histograms (All Weather Parameters)", width="stretch")
else:
    st.warning("⚠️ Graph file '01_numerical_feature_histograms.png' not found in outputs/graphs/")

st.write("")

# Display pairs in two columns for high readability
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    p = GRAPH_DIR / "02_correlation_heatmap.png"
    if p.exists():
        st.image(str(p), caption="02. Weather Feature Correlation Heatmap", width="stretch")
    else:
        st.warning("⚠️ Graph file '02_correlation_heatmap.png' not found")

with row1_col2:
    p = GRAPH_DIR / "03_rainfall_distribution.png"
    if p.exists():
        st.image(str(p), caption="03. Rainfall Distribution", width="stretch")
    else:
        st.warning("⚠️ Graph file '03_rainfall_distribution.png' not found")

st.write("")

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    p = GRAPH_DIR / "04_avg_temp_distribution.png"
    if p.exists():
        st.image(str(p), caption="04. Average Temperature Distribution", width="stretch")
    else:
        st.warning("⚠️ Graph file '04_avg_temp_distribution.png' not found")

with row2_col2:
    p = GRAPH_DIR / "05_avg_temp_vs_rainfall.png"
    if p.exists():
        st.image(str(p), caption="05. Average Temperature vs Rainfall", width="stretch")
    else:
        st.warning("⚠️ Graph file '05_avg_temp_vs_rainfall.png' not found")

st.write("")

row3_col1, row3_col2 = st.columns(2)
with row3_col1:
    p = GRAPH_DIR / "06_wind_speed_vs_rainfall.png"
    if p.exists():
        st.image(str(p), caption="06. Wind Speed vs Rainfall", width="stretch")
    else:
        st.warning("⚠️ Graph file '06_wind_speed_vs_rainfall.png' not found")

with row3_col2:
    p = GRAPH_DIR / "07_rainfall_trend.png"
    if p.exists():
        st.image(str(p), caption="07. Historical Rainfall Trend", width="stretch")
    else:
        st.warning("⚠️ Graph file '07_rainfall_trend.png' not found")

st.markdown("---")
st.markdown("## 🎯 Model Evaluation Visuals")
st.markdown("Comparative metrics, error dispersion, and tree feature importance:")

eval_col1, eval_col2 = st.columns(2)
with eval_col1:
    p = GRAPH_DIR / "08_model_comparison.png"
    if p.exists():
        st.image(str(p), caption="08. Model Performance Comparison (RMSE)", width="stretch")
    else:
        st.warning("⚠️ Graph file '08_model_comparison.png' not found")

    p = GRAPH_DIR / "11_feature_importance.png"
    if p.exists():
        st.image(str(p), caption="11. Random Forest Feature Importance", width="stretch")
    else:
        st.warning("⚠️ Graph file '11_feature_importance.png' not found")

with eval_col2:
    p = GRAPH_DIR / "09_actual_vs_predicted.png"
    if p.exists():
        st.image(str(p), caption="09. Actual vs Predicted Rainfall", width="stretch")
    else:
        st.warning("⚠️ Graph file '09_actual_vs_predicted.png' not found")

    p = GRAPH_DIR / "10_prediction_error_distribution.png"
    if p.exists():
        st.image(str(p), caption="10. Prediction Error Distribution", width="stretch")
    else:
        st.warning("⚠️ Graph file '10_prediction_error_distribution.png' not found")

st.write("")

# Prediction historical test results preview
st.markdown("### 📋 Test Prediction Results Sample")
if prediction_table is not None and not prediction_table.empty:
    st.dataframe(prediction_table.head(15), width="stretch")
else:
    st.info("Test results preview will appear once evaluation files are loaded.")

st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 20px 0;">
        Weather Report Prediction & Data Analysis &bull; Indian Historical Rainfall Dataset &bull; Batch-3 DAE
    </div>
    """,
    unsafe_allow_html=True,
)
