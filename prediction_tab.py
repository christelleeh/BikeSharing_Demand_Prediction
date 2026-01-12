# prediction_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
import datetime

# ========= UI: Page + CSS =========
st.set_page_config(page_title="Bike Demand Prediction", layout="wide")

# --- Global CSS (no extra libs needed) ---
st.markdown("""
<style>
/* Import a clean, legible font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}

/* Header ribbon */
.app-header {
  width: 100%;
  padding: 18px 22px;
  border-radius: 16px;
  background: linear-gradient(90deg, rgba(38,132,255,0.18) 0%, rgba(0,212,170,0.18) 100%);
  border: 1px solid rgba(120,120,120,0.15);
  box-shadow: 0 8px 28px rgba(0,0,0,0.06);
  margin-bottom: 8px;
}
.app-header h1 {
  margin: 0;
  font-weight: 700;
  letter-spacing: 0.2px;
}
.app-caption {
  color: rgba(60,60,67,0.85);
  margin-top: 4px;
}

/* Section titles */
.block-container h2, .block-container h3 {
  letter-spacing: 0.2px;
}

/* Inputs + selectboxes + sliders */
.stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stSlider {
  background: transparent !important;
}
div[data-baseweb="select"] > div, .stDateInput, .stTextInput > div > div, .stNumberInput > div > div {
  border-radius: 12px !important;
  border: 1px solid rgba(120,120,120,0.2) !important;
  box-shadow: none !important;
}
div[data-baseweb="select"] > div:hover,
.stDateInput:hover, .stTextInput > div > div:hover, .stNumberInput > div > div:hover {
  border-color: rgba(38,132,255,0.5) !important;
}

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  border: 1px solid rgba(38,132,255,0.25);
  background: linear-gradient(180deg, rgba(38,132,255,0.9), rgba(38,132,255,0.75));
  color: white;
  font-weight: 600;
  letter-spacing: 0.2px;
  box-shadow: 0 6px 16px rgba(38,132,255,0.25);
}
.stButton>button:hover {
  filter: brightness(1.02);
  transform: translateY(-1px);
  transition: all .15s ease-in-out;
}

/* Metric cards */
.stMetric {
  background: white;
  border: 1px solid rgba(120,120,120,0.12);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
[data-testid="stMetricDelta"] {
  font-weight: 600;
}
[data-testid="stMetricValue"] {
  font-weight: 700 !important;
}

/* Info/success/warning styling tweaks */
.stAlert {
  border-radius: 14px;
  border: 1px solid rgba(120,120,120,0.15);
}

/* Expander as a card */
.streamlit-expanderHeader {
  font-weight: 600 !important;
}
.st-expander {
  border-radius: 14px !important;
  border: 1px solid rgba(120,120,120,0.15) !important;
  background: rgba(245,247,250,0.6) !important;
}

/* Divider spacing */
hr {
  margin: 1.2rem 0 0.6rem 0;
}

/* Subtle containers for the three input columns */
.section-card {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(120,120,120,0.12);
  background: white;
  box-shadow: 0 8px 24px rgba(0,0,0,0.04);
  margin-bottom: 12px;
}

/* Tweak captions / helper text */
.small-muted {
  font-size: 0.92rem;
  color: rgba(60,60,67,0.85);
}

/* Make sliders text labels a bit bolder */
.stSlider label, .stSelectbox label, .stDateInput label {
  font-weight: 600;
}

/* Improve wide layout readability */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 3rem;
}
</style>
""", unsafe_allow_html=True)

# --- Header UI ---
st.markdown(
    """
    <div class="app-header">
      <h1>🚲 Bike Demand Prediction</h1>
      <div class="app-caption">Single-hour forecast for Registered, Casual, and Total demand.</div>
    </div>
    """,
    unsafe_allow_html=True
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ======================
# Feature engineering (KEEP dteday for the preprocessor)
# ======================
def add_features_patched(df_input: pd.DataFrame) -> pd.DataFrame:
    X = df_input.copy()

    def temp_curve(t, b0=0.0, b1=25.0, b2=-0.5):
        t = np.asarray(t); return b0 + b1*t + b2*t**2
    def humidity_quad(h, b0=0.0, b1=15.0, b2=-0.15):
        h = np.asarray(h); return b0 + b1*h + b2*h**2
    def humidity_exp(h, a=300.0, c=50.0, k=0.025):
        h = np.asarray(h); return c + a*np.exp(-k*h)
    def wind_curve(w, b0=150.0, b1=6.0, b2=-0.15):
        w = np.asarray(w); return b0 + b1*w + b2*w**2
    def bimodal_hour(hr, A1=280.0, mu1=8.0, s1=1.4,
                     A2=420.0, mu2=17.5, s2=1.8, c=30.0):
        hr = np.asarray(hr)
        g1 = A1*np.exp(-(hr-mu1)**2/(2*s1**2))
        g2 = A2*np.exp(-(hr-mu2)**2/(2*s2**2))
        return g1 + g2 + c
    def workingday_hour(hr, workday,
                        A1_w=260, mu1_w=8.0,  s1_w=1.3,
                        A2_w=480, mu2_w=17.5, s2_w=1.7, c_w=20,
                        A1_we=120, mu1_we=11.0, s1_we=2.2,
                        A2_we=260, mu2_we=18.5, s2_we=2.0, c_we=30):
        hr = np.asarray(hr); workday = np.asarray(workday).astype(int)
        y_w  = (A1_w*np.exp(-(hr-mu1_w)**2/(2*s1_w**2)) + A2_w*np.exp(-(hr-mu2_w)**2/(2*s2_w**2)) + c_w)
        y_we = (A1_we*np.exp(-(hr-mu1_we)**2/(2*s1_we**2)) + A2_we*np.exp(-(hr-mu2_we)**2/(2*s2_we**2)) + c_we)
        return np.where(workday==1, y_w, y_we)

    def first_harmonic_of_cyclic_features(df: pd.DataFrame, cols: list):
        df = df.copy()
        for col in cols:
            if col not in df.columns: 
                continue
            sin_col, cos_col = f"{col}_sin", f"{col}_cos"
            period = 24 if col=='hr' else 12 if col=='mnth' else 7 if col=='weekday' else 4 if col=='season' else max(df[col].nunique(),1)
            df[sin_col] = np.sin(2*np.pi*df[col]/period)
            df[cos_col] = np.cos(2*np.pi*df[col]/period)
        return df

    def _safe(c): return c in X.columns

    # Nonlinear bases
    if _safe("temp"):      X["temp_invU"]       = temp_curve(X["temp"])
    if _safe("hum"):       X["hum_quad_curve"]  = humidity_quad(X["hum"])
    if _safe("hum"):       X["hum_exp_curve"]   = humidity_exp(X["hum"])
    if _safe("windspeed"): X["wind_quad_curve"] = wind_curve(X["windspeed"])
    if _safe("hr"):
        X["hr_bimodal"]    = bimodal_hour(X["hr"])
        X["hr_gauss_morn"] = np.exp(-(X["hr"]-8.0 )**2/(2*1.4**2))
        X["hr_gauss_even"] = np.exp(-(X["hr"]-17.5)**2/(2*1.8**2))
    if _safe("hr") and _safe("workingday"):
        X["workday_hour_curve"]   = workingday_hour(X["hr"], X["workingday"])
        X["work_x_hr_gauss_morn"] = X["workingday"] * X["hr_gauss_morn"]
        X["work_x_hr_gauss_even"] = X["workingday"] * X["hr_gauss_even"]

    # Flags
    if _safe("weekday"): X["is_weekend"] = X["weekday"].isin([0,6]).astype(int)
    if _safe("hr"):      X["is_evening_or_afternoon"] = X["hr"].between(10, 20).astype(int)
    if _safe("mnth"):    X["is_summer_month"] = X["mnth"].between(5, 9).astype(int)
    if _safe("atemp"):   X["pleasant_temp"] = (X["atemp"]*50.0).between(15, 28).astype(int)
    if _safe("weathersit") and _safe("hum") and _safe("windspeed"):
        X["rideability_score"] = (
            (X["weathersit"] <= 2).astype(int)
            + (X["hum"] < 0.7).astype(int)
            + (X["windspeed"] < 0.35).astype(int)
        )
    if "pleasant_temp" in X and "is_weekend" in X:
        X["pleasant_weekend"] = (X["pleasant_temp"] & X["is_weekend"]).astype(int)
    if _safe("weathersit") and "is_evening_or_afternoon" in X:
        X["nice_evening"] = ((X["weathersit"] == 1) & X["is_evening_or_afternoon"]).astype(int)

    # Daylight from dteday
    if _safe("dteday"):
        d = pd.to_datetime(X["dteday"])
        X["doy"] = d.dt.dayofyear.astype(int)
        lat = np.deg2rad(38.9)  # Washington DC
        decl = 23.44*np.pi/180 * np.sin(2*np.pi*(284 + X["doy"].clip(1,365))/365.0)
        H0 = np.arccos(-np.tan(lat) * np.tan(decl))
        X["daylight_hours"] = (2*H0)*24/(2*np.pi)

    # Cyclic encodings
    X = first_harmonic_of_cyclic_features(X, ["hr", "mnth", "weekday", "season"])

    # Ensure yr from dteday if provided
    if _safe("dteday"):
        X["yr"] = pd.to_datetime(X["dteday"]).dt.year - 2011

    return X  # Keep dteday present for preprocessor expectations

# ======================
# Load artifacts + metadata
# ======================
@st.cache_resource
def load_artifacts():
    pre = None
    try:
        pre = joblib.load(HERE / "preprocessor.pkl")
    except Exception:
        pass

    model_reg = joblib.load(HERE / "model_registered.pkl")
    model_cas = joblib.load(HERE / "model_casual.pkl")

    try:
        lag_means = joblib.load(HERE / "lag_means.pkl")
    except Exception:
        lag_means = {}
        st.warning("lag_means.pkl not found — lag features will default to empty (predictions may fail).")

    # Read target scale flags from meta JSONs if present
    def read_meta(fname):
        path = HERE / fname
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    if str(meta.get("target_scale","")).lower() == "log1p":
                        return True
                    if bool(meta.get("y_log1p", False)):
                        return True
            except Exception:
                pass
        return None  # unknown

    y_reg_is_log = read_meta("model_registered_meta.json")
    y_cas_is_log = read_meta("model_casual_meta.json")

    return pre, model_reg, model_cas, lag_means, y_reg_is_log, y_cas_is_log

preprocessor, model_reg, model_cas, lag_means_dict, y_reg_is_log, y_cas_is_log = load_artifacts()

def _is_pipeline_model(m):
    return hasattr(m, "steps") or hasattr(m, "named_steps")

# ======================
# UI
# ======================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Time & Date**")
    default_date = datetime.date(2012, 11, 1)
    date_input = st.date_input("Target Date", default_date)
    hour_input = st.slider("Hour of Day (0–23)", 0, 23, 17)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Seasonal & Work**")
    season_map = {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}
    m = date_input.month
    idx = 0 if 3<=m<=5 else 1 if 6<=m<=8 else 2 if 9<=m<=11 else 3
    season_input = st.selectbox("Season", options=list(season_map.keys()),
                                format_func=lambda x: season_map[x], index=idx)
    workingday_input = st.selectbox("Is it a Working Day?", [0,1],
                                    format_func=lambda x: "Yes (1)" if x==1 else "No (0)", index=1)
    holiday_input = st.selectbox("Is it a Holiday?", [0,1],
                                 format_func=lambda x: "Yes (1)" if x==1 else "No (0)", index=0)
    st.caption("Tip: Working days + commute hours often produce bimodal peaks.", help=None)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Weather**")
    weather_map = {1:"Clear / Few clouds",2:"Mist / Cloudy",3:"Light Rain / Snow"}
    weathersit_input = st.selectbox("Weather Situation", options=list(weather_map.keys()),
                                    format_func=lambda x: weather_map[x])
    temp_input = st.slider("Temperature (°C)", 0.0, 41.0, 24.6, 0.1) / 41
    atemp_input = st.slider("Feeling Temp (°C)", 0.0, 50.0, 30.0, 0.1) / 50
    hum_input = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 1.0) / 100
    windspeed_input = st.slider("Windspeed (0–67)", 0.0, 67.0, 13.4, 0.1) / 67
    st.caption("Rideability improves with clear skies, moderate temps, lower humidity & wind.", help=None)
    st.markdown('</div>', unsafe_allow_html=True)

# Base input row — matches your notebook
input_row = {
    "dteday": pd.to_datetime(date_input),
    "season": season_input,
    "yr": date_input.year - 2011,
    "mnth": date_input.month,
    "hr": hour_input,
    "holiday": holiday_input,
    "weekday": date_input.weekday(),
    "workingday": workingday_input,
    "weathersit": weathersit_input,
    "temp": temp_input,
    "atemp": atemp_input,
    "hum": hum_input,
    "windspeed": windspeed_input,
}
input_df = pd.DataFrame([input_row])

st.markdown("---")
st.subheader("Predicted Hourly Demand")

# ======================
# Helpers for target inverse-transform
# ======================
def inverse_target(pred_value: float, is_log_flag: bool | None) -> float:
    if is_log_flag is True:
        return np.expm1(pred_value)
    if is_log_flag is False:
        return pred_value
    # Heuristic fallback: typical log1p RF outputs are < 10; huge raw means linear scale
    return pred_value if pred_value >= 15 else np.expm1(pred_value)

def postprocess_count(x: float) -> float:
    return float(np.clip(x, 0.0, 10000.0))

# ======================
# Predict
# ======================
if st.button("Calculate Prediction", type="primary"):
    try:
        # 1) Feature engineering
        X = add_features_patched(input_df.copy())

        # 2) Append lag means (needed to supply lag_* columns)
        if isinstance(lag_means_dict, dict) and lag_means_dict:
            X = X.assign(**lag_means_dict)

        # 3) Decide route: pipeline model or standalone estimator with external preprocessor
        model_is_pipeline = _is_pipeline_model(model_reg) and _is_pipeline_model(model_cas)

        if model_is_pipeline:
            X_for_model = X  # model does its own preprocessing
        else:
            if preprocessor is None:
                raise RuntimeError("preprocessor.pkl missing but models require preprocessed inputs.")
            X_proc = preprocessor.transform(X)
            # Try to drop any dteday that leaked through passthrough
            try:
                cols = list(preprocessor.get_feature_names_out())
                X_proc_df = pd.DataFrame(X_proc, columns=cols)
                drop_cols = [c for c in X_proc_df.columns if c == "dteday" or c.lower().endswith("__dteday") or "dteday" in c.lower()]
                if drop_cols:
                    X_proc_df = X_proc_df.drop(columns=drop_cols, errors="ignore")
                X_for_model = X_proc_df
            except Exception:
                X_for_model = X_proc  # fallback without column names

        # 4) Predict
        pred_reg_raw = float(model_reg.predict(X_for_model)[0])
        pred_cas_raw = float(model_cas.predict(X_for_model)[0])

        # 5) Proper inverse transform (metadata-aware)
        pred_reg = postprocess_count(inverse_target(pred_reg_raw, y_reg_is_log))
        pred_cas = postprocess_count(inverse_target(pred_cas_raw, y_cas_is_log))
        pred_total = postprocess_count(pred_reg + pred_cas)

        st.success("Prediction Complete!")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted Registered Users", f"{pred_reg:,.0f}")
        with c2:
            st.metric("Predicted Casual Users", f"{pred_cas:,.0f}")
        with c3:
            st.metric("TOTAL BIKE PROVISIONING NEED", f"{pred_total:,.0f}")

        # ======================
        # 📋 Contextual Recommendation (NEW)
        # ======================
        recommendation = ""

        # Demand magnitude & day type
        if pred_total > 500 and workingday_input == 1:
            recommendation = ("🚦 **Peak commuter demand expected.** "
                              "Increase fleet at residential ↔ business corridors, especially near metro stations. "
                              "Schedule rebalancing trucks 30–45 min before the peak window.")
        elif pred_total > 300 and workingday_input == 0:
            recommendation = ("🌞 **Strong leisure demand expected.** "
                              "Pre-stock parks, riverside and tourist hubs; offer multi-hour passes to lift ARPU.")
        elif pred_total < 50:
            recommendation = ("🌙 **Low expected demand.** "
                              "Maintain baseline fleet and reduce overnight rebalancing to cut cost.")
        else:
            recommendation = ("🚲 **Moderate demand.** "
                              "Keep standard provisioning and monitor station dashboards for micro-adjustments.")

        # Hour-of-day nuance (commute/leisure)
        if 7 <= hour_input <= 9 and workingday_input == 1:
            recommendation += ("\n🕗 **Morning commute:** focus inflow to business districts; "
                               "pull bikes from nearby residential zones.")
        if 16 <= hour_input <= 19 and workingday_input == 1:
            recommendation += ("\n🕕 **Evening commute:** reverse flow — stock residential zones for returns.")
        if 11 <= hour_input <= 16 and workingday_input == 0:
            recommendation += ("\n🏞️ **Midday leisure window:** prioritize scenic/park stations and tourist corridors.")

        # Weather overlay
        if weathersit_input == 3:
            recommendation += ("\n☔ **Adverse weather:** demand suppression likely. Temporarily downscale active fleet; "
                               "delay non-critical rebalancing.")
        elif weathersit_input == 2 and pred_total > 200:
            recommendation += ("\n🌤️ **Cloudy but serviceable:** demand should hold. Keep trucks on standby for quick top-ups.")

        # Season overlay
        if season_input == 2:
            recommendation += ("\n🏖️ **Summer:** high temp sensitivity — keep extra bikes at leisure hubs.")
        elif season_input == 4:
            recommendation += ("\n❄️ **Winter:** plan maintenance windows; lower base provisioning is usually safe.")

        # Thermal/comfort cues (using atemp, hum, windspeed as normalized)
        feels_c = atemp_input * 50.0
        humidity_pct = hum_input * 100.0
        wind_raw = windspeed_input * 67.0
        if 18 <= feels_c <= 28 and humidity_pct < 70 and wind_raw < 24:
            recommendation += ("\n✅ **Comfortable conditions:** expect stable ride propensity — slight uplift vs. baseline.")
        if humidity_pct >= 80 or wind_raw >= 35:
            recommendation += ("\n⚠️ **High humidity or wind:** anticipate shorter rides and more returns clustering.")

        st.markdown("---")
        st.subheader("📋 Recommendation")
        st.info(recommendation)

        # Optional debug pane
        with st.expander("ℹ️ Debug details (feature set / meta)", expanded=False):
            st.write("Model types:", type(model_reg).__name__, "/", type(model_cas).__name__)
            st.write("Metadata says log1p?", {"registered": y_reg_is_log, "casual": y_cas_is_log})
            st.write("First row (post-engineering) columns:", list(X.columns))

    except Exception as e:
        st.error("Prediction failed. Most common causes: wrong preprocessing route or missing lag columns.")
        st.exception(e)
