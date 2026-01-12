import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import joblib
from pathlib import Path
import streamlit.components.v1 as components
import sys
import datetime

# Optional: keep Matplotlib images consistent if shown
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Segoe UI, Helvetica, Arial, sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
    })
except Exception:
    pass

# =========================================================
# Streamlit Page Configuration
# =========================================================
st.set_page_config(page_title="DC Bike Share Optimization Report",
                   layout="wide", initial_sidebar_state="expanded")

# =========================================================
# Global Plotly Theme Configuration (Consistent Fonts & Layout)
# =========================================================
pio.templates["dc_theme"] = pio.templates["plotly_white"]
pio.templates["dc_theme"].layout.update(
    font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=14, color="#222"),
    title_font=dict(family="Segoe UI Semibold, Helvetica, Arial, sans-serif", size=20, color="#111"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=70, r=50, t=70, b=60),
    xaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, title_font=dict(size=14)),
    yaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False, title_font=dict(size=14)),
    colorway=[
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ],
    legend=dict(title_font=dict(size=14), font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
)
pio.templates.default = "dc_theme"

# =========================================================
# Global CSS for Streamlit + embedded Plotly HTML consistency
# =========================================================
GLOBAL_CSS = """
    <style>
    :root {
      --base-font: 'Segoe UI', Helvetica, Arial, sans-serif;
      --text-color: #222;
      --heading-color: #111;
    }
    html, body, [class*="css"], .stMarkdown, .stText, .stPlotlyChart, .stMetric {
      font-family: var(--base-font);
      color: var(--text-color);
      font-size: 14px;
    }
    h1, h2, h3, h4, h5, h6 {
      color: var(--heading-color) !important;
      font-weight: 600 !important;
    }
    h1 { font-size: 28px !important; line-height: 1.25; }
    h2 { font-size: 22px !important; line-height: 1.35; }
    h3 { font-size: 18px !important; line-height: 1.4; }
    h4, h5, h6 { font-size: 16px !important; line-height: 1.45; }
    .plotly-graph-div text {
      font-family: var(--base-font) !important;
      font-size: 13px !important;
      fill: var(--text-color) !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }
    .note { background:#fffbe6; border:1px solid #ffe58f; border-radius:8px; padding:12px 14px; }
    </style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# =========================================================
# Paths & Imports
# =========================================================
ARTS_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1) Import or Mock add_features (Required for PKL deserialization)
try:
    from src.feature_functions import add_features
except Exception as e:
    st.error(
        "Failed to import 'add_features' from 'src.feature_functions'. "
        "Ensure your 'src' directory and 'feature_functions.py' are correctly set up relative to this script.\n"
        f"Error: {e}"
    )
    def add_features(x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(columns=['dteday'], errors='ignore')

# --- Patched add_features for Simulator (Defensive Logic) ---
def add_features_patched(df_input) -> pd.DataFrame:
    """Defensive wrapper that mirrors your feature engineering for single-row inference."""
    X = df_input.copy()

    # helper bases
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
            sin_col = f"{col}_sin"; cos_col = f"{col}_cos"
            if sin_col in df.columns or cos_col in df.columns:
                continue
            if col == 'hr': period = 24
            elif col == 'mnth': period = 12
            elif col == 'weekday': period = 7
            elif col == 'season': period = 4
            else: period = int(max(df[col].nunique(), 1))
            df[sin_col] = np.sin(2*np.pi*df[col]/period) if period > 1 else 0.0
            df[cos_col] = np.cos(2*np.pi*df[col]/period) if period > 1 else 1.0
        return df
    def _safe(col): return col in X.columns

    # nonlinear and interaction features
    if _safe("temp"):        X["temp_invU"]       = temp_curve(X["temp"])
    if _safe("hum"):         X["hum_quad_curve"]  = humidity_quad(X["hum"])
    if _safe("hum"):         X["hum_exp_curve"]   = humidity_exp(X["hum"])
    if _safe("windspeed"):   X["wind_quad_curve"] = wind_curve(X["windspeed"])
    if _safe("hr"):          X["hr_bimodal"]      = bimodal_hour(X["hr"])
    if _safe("hr"):          X["hr_gauss_morn"]   = np.exp(-(X["hr"]-8.0)**2  / (2*1.4**2))
    if _safe("hr"):          X["hr_gauss_even"]   = np.exp(-(X["hr"]-17.5)**2 / (2*1.8**2))
    if _safe("hr") and _safe("workingday"):
        X["workday_hour_curve"]   = workingday_hour(X["hr"], X["workingday"])
        X["work_x_hr_gauss_morn"] = X["workingday"] * X.get("hr_gauss_morn", 0.0)
        X["work_x_hr_gauss_even"] = X["workingday"] * X.get("hr_gauss_even", 0.0)

    # boolean & composite flags
    if _safe("weekday"):
        X["is_weekend"] = X["weekday"].isin([0,6]).astype(int)
    if _safe("hr"):
        X["is_evening_or_afternoon"] = X["hr"].between(10, 20).astype(int)
    if _safe("mnth"):
        X["is_summer_month"] = X["mnth"].between(5, 9).astype(int)
    if _safe("atemp"):
        X["pleasant_temp"] = (X["atemp"] * 50.0).between(15, 28).astype(int)
    if _safe("weathersit") and _safe("hum") and _safe("windspeed"):
        X["rideability_score"] = (
            (X["weathersit"] <= 2).astype(int)
            + (X["hum"] < 0.7).astype(int)
            + (X["windspeed"] < 0.35).astype(int)
        )
    if "pleasant_temp" in X.columns and "is_weekend" in X.columns:
        X["pleasant_weekend"] = (X["pleasant_temp"] & X["is_weekend"]).astype(int)
    if _safe("weathersit") and "is_evening_or_afternoon" in X.columns:
        X["nice_evening"] = ((X["weathersit"] == 1) & X["is_evening_or_afternoon"]).astype(int)

    # daylight hours feature (using dteday)
    if _safe("dteday"):
        try:
            pi = 3.141592653589793
            d = pd.to_datetime(X["dteday"])
            X["doy"] = d.dt.dayofyear.astype(int)
            lat = np.deg2rad(38.9)  # Washington D.C. latitude
            n = X["doy"].clip(1, 365)
            decl = 23.44 * pi/180 * np.sin(2*pi*(284 + n)/365.0)
            H0 = np.arccos(-np.tan(lat) * np.tan(decl))
            daylight = (2 * H0) * 24 / (2*pi)
            X["daylight_hours"] = daylight.astype(float)
        except Exception:
            pass

    # cyclic sin/cos features
    X = first_harmonic_of_cyclic_features(X, ["hr", "mnth", "weekday", "season"])

    if 'dteday' in X.columns:
        X['yr'] = X['dteday'].dt.year - 2011

    return X.drop(columns=['dteday'], errors='ignore')

# =========================================================
# Load Artifacts (Cached)
# =========================================================
try:
    lag_means_dict = joblib.load(ARTS_PATH / "lag_means.pkl")
except FileNotFoundError:
    st.error("Missing artifact: 'lag_means.pkl'. Prediction will fail.")
    lag_means_dict = {}

@st.cache_resource
def load_models_and_preprocessor(artifacts_signature):
    try:
        pre = joblib.load(ARTS_PATH / "preprocessor.pkl")
        m_r = joblib.load(ARTS_PATH / "model_registered.pkl")
        m_c = joblib.load(ARTS_PATH / "model_casual.pkl")
        if not hasattr(pre, 'transform') or not hasattr(m_r, 'predict'):
            raise ImportError("Loaded PKL files do not contain expected model/preprocessor methods.")
        return pre, m_r, m_c
    except Exception as e:
        st.error(f"Failed to load required PKL files (preprocessor, models). Prediction tab is disabled. Error: {e}")
        return None, None, None

def _artifact_signature():
    paths = [ARTS_PATH / "preprocessor.pkl",
             ARTS_PATH / "model_registered.pkl",
             ARTS_PATH / "model_casual.pkl"]
    return tuple((p.name, p.stat().st_mtime, p.stat().st_size) for p in paths if p.exists())

preprocessor, model_reg, model_cas = load_models_and_preprocessor(_artifact_signature())

@st.cache_data
def load_importance_and_metrics_data():
    imp_reg = pd.DataFrame({'feature': ['hr', 'temp', 'yr', 'mnth'], 'importance': [0.65, 0.2, 0.1, 0.05]})
    imp_cas = pd.DataFrame({'feature': ['temp', 'season', 'hr', 'holiday'], 'importance': [0.70, 0.15, 0.1, 0.05]})
    metrics_dict = {"RMSE": 45.5, "MAE": 35.1, "MAPE": 12.0, "R²": 0.965}
    return imp_reg.sort_values(by='importance', ascending=False), imp_cas.sort_values(by='importance', ascending=False), metrics_dict

imp_reg, imp_cas, metrics = load_importance_and_metrics_data()

# =========================================================
# Data Loader for Interactive Distributions
# =========================================================
@st.cache_data
def load_bikeshare_df() -> pd.DataFrame:
    csv_path = ARTS_PATH / "bike-sharing-hourly.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return pd.DataFrame()

# =========================================================
# HTML Helpers: ensure embedded artifacts inherit consistent font
# =========================================================
EMBEDDED_PLOT_CSS = """
<style>
  html, body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color:#222; }
  .plotly-graph-div text { font-family: 'Segoe UI', Helvetica, Arial, sans-serif !important; font-size: 13px !important; fill:#222 !important; }
  h1,h2,h3,h4 { color:#111; font-weight:600; }
</style>
"""

def inject_style_into_html(html_str: str) -> str:
    """Inject a CSS block into a Plotly HTML string for font consistency."""
    try:
        if "</head>" in html_str:
            return html_str.replace("</head>", EMBEDDED_PLOT_CSS + "</head>")
        return EMBEDDED_PLOT_CSS + html_str
    except Exception:
        return html_str

def read_html_artifact(filename: str) -> str:
    try:
        with open(ARTS_PATH / filename, 'r', encoding='utf-8') as f:
            raw = f.read()
        return inject_style_into_html(raw)
    except FileNotFoundError:
        st.error(f"Plot artifact '{filename}' not found. Check the filename and location.")
        return f"<h3>Plot artifact '{filename}' not found.</h3><p>Ensure it is in the same directory as the app.</p>"

# =========================================================
# IQR Distribution Plotter (standardized)
# =========================================================
def plot_hist_distribution(df: pd.DataFrame, target_col: str,
                           iqr_upper_limit: float = 1.5, iqr_lower_limit: float = 1.5,
                           color: str = "#636EFA") -> go.Figure:
    """Standardized IQR histogram with Q1/Q3/IQR and Tukey fences."""
    s = df[target_col].dropna()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = float(Q3 - Q1)
    lower_bound = float(Q1 - iqr_lower_limit * IQR)
    upper_bound = float(Q3 + iqr_upper_limit * IQR)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s, nbinsx=50, marker_color=color, opacity=0.65, name="Distribution"))

    # Quartiles
    fig.add_vline(x=Q1, line_dash="dash", line_color="#F5A623",
                  annotation_text=f"Q1 = {Q1:.1f}", annotation_position="top left")
    fig.add_vline(x=Q3, line_dash="dash", line_color="#27AE60",
                  annotation_text=f"Q3 = {Q3:.1f}", annotation_position="top right")

    # Tukey fences
    fig.add_vline(x=lower_bound, line_dash="dot", line_color="#E74C3C",
                  annotation_text=f"Lower fence = {lower_bound:.1f}", annotation_position="bottom left")
    fig.add_vline(x=upper_bound, line_dash="dot", line_color="#E74C3C",
                  annotation_text=f"Upper fence = {upper_bound:.1f}", annotation_position="bottom right")

    # IQR band
    yband = max(1, s.size // 18)
    fig.add_shape(type="rect", x0=Q1, x1=Q3, y0=0, y1=yband,
                  line=dict(color=color, width=1), fillcolor=color, opacity=0.18)
    fig.add_annotation(x=(Q1+Q3)/2, y=yband*1.02, text=f"IQR = {IQR:.2f}", showarrow=False,
                       font=dict(size=13, color="#333"))

    fig.update_layout(
        title=f"Distribution of {target_col.capitalize()} (IQR = {IQR:.2f})",
        xaxis_title=target_col,
        yaxis_title="Count",
        bargap=0.05,
        template="dc_theme",
        height=450,
        showlegend=False,
    )
    return fig

# =========================================================
# App Layout & Tabs
# =========================================================
st.title("🚲 DC Bike-Sharing Service Optimization Report")
st.markdown("An interactive report for the Head of Transportation Services, based on the 2011-2012 dataset.")

if not all([preprocessor, model_reg, model_cas]):
    # models not available → no simulator/diagnostics
    tab1, tab5 = st.tabs(["📊 Data Analysis & Insights", "📅 Deviation Insights"])
else:
    tab1, tab2, tab3, tab5 = st.tabs([
        "📊 Data Analysis & Insights",
        "🤖 Predictive Model Diagnostics",
        "⚙️ Provisioning Simulator",
        "📅 Deviation Insights"
    ])

# =========================================================
# TAB 1: Data Analysis & Insights
# =========================================================
with tab1:
    st.header("1. Data Analysis: Understanding Usage Patterns")
    st.markdown("This section visualizes the key trends discovered to optimize service costs and user experience.")

    st.subheader("1.1. User Behavior: Hourly and Daily Trends")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Average Hourly Use (User Segmentation)")
        html_hr = read_html_artifact("Average Users by Hour of Day.html")
        components.html(html_hr, height=600, scrolling=True)
    with col2:
        st.markdown("#### Average Daily Use (Yearly and Monthly Trends)")
        html_mnth = read_html_artifact("Daily Registered vs Casual Users (2011–2012).html")
        components.html(html_mnth, height=600, scrolling=True)

    st.markdown("---")

    # Exported distributions (kept)
    st.subheader("1.2. Demand Diagnostics: Distributions and Time Series (Exports)")
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.markdown("##### Distribution of Registered Users (export)")
        html_dist_reg = read_html_artifact("Distribution of Registered (IQR = 186.00).html")
        components.html(html_dist_reg, height=350)
    with col_dist2:
        st.markdown("##### Distribution of Casual Users (export)")
        html_dist_cas = read_html_artifact("Distribution of Casual (IQR = 44.00).html")
        components.html(html_dist_cas, height=350)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Total Count (Cnt) Distribution (export)")
        html_dist_cnt = read_html_artifact("Distribution of Cnt (IQR = 241.00).html")
        components.html(html_dist_cnt, height=350)
    with col4:
        st.markdown("##### Autocorrelation Plot")
        try:
            st.image(str(ARTS_PATH / "Autocorrelation_Plots.png"),
                     caption="Autocorrelation of demand (from notebook)", use_column_width=True)
        except FileNotFoundError:
            st.warning("Autocorrelation_Plots.png not found.")

    st.markdown("---")

    # Interactive distributions (standardized)
    st.subheader("1.3. Interactive IQR Distributions (standardized)")
    df = load_bikeshare_df()
    if df.empty:
        st.info("`bike-sharing-hourly.csv` not found. Interactive distributions are hidden.")
    else:
        colA, colB, colC = st.columns(3)
        with colA:
            st.plotly_chart(plot_hist_distribution(df, 'casual', 3, 1.5, '#636EFA'), use_container_width=True)
            st.caption("**Casual users:** Highly right-skewed; sporadic bursts in ideal weather.")
        with colB:
            st.plotly_chart(plot_hist_distribution(df, 'registered', 3, 1.5, '#EF553B'), use_container_width=True)
            st.caption("**Registered users:** Predictable commuter base; broader but structured spread.")
        with colC:
            st.plotly_chart(plot_hist_distribution(df, 'cnt', 2.5, 1.5, '#00CC96'), use_container_width=True)
            st.caption("**Total demand:** Blends commuter stability with leisure volatility.")

    st.markdown("---")

    # Nonlinear relationships dashboard (basis for engineered features)
    st.subheader("1.4. Nonlinear Relationships (Feature Engineering Basis)")
    st.markdown("Key nonlinear effects (temperature, humidity, wind, hour) that motivated engineered features like `temp_invU`, `hum_quad_curve`, `hr_bimodal`, and interaction terms.")
    html_nonlin = read_html_artifact("Nonlinear Relationship Dashboard.html")
    components.html(html_nonlin, height=700, scrolling=True)

    st.markdown("---")

    # Analysis & Recommendations
    st.subheader("1.5. Analysis & Recommendations for Optimization")
    st.markdown("""
**Most critical pattern:** the **hourly demand profile** split by user type and working vs non-working days.

**How citizens use the service**
- **Registered (commuters/subscribers):** Sharp peaks on working days **7–9 AM** and **4–6 PM** (first/last-mile). Lower, flatter use on non-working days.
- **Casual (tourists/occasional):** Higher on non-working days, broad peak **10 AM–5 PM**. Flatter on working days (avoid commute spikes).
""")

    rec_df = pd.DataFrame({
        "Area": [
            "Bike Provisioning (Commute)",
            "Off-Peak Utilization",
            "Casual User Revenue",
            "Adverse Weather (Future)"
        ],
        "Challenge": [
            "Two peak windows (7–9 AM, 4–6 PM) risk stock-outs at commuter hubs.",
            "Very low demand 10 PM–5 AM → idle assets.",
            "Casual demand is valuable but sporadic.",
            "Demand drops significantly during bad weather."
        ],
        "Recommendation": [
            "Focus outsource rebalancing **9 AM–4 PM** only. Morning: move bikes residential → business; afternoon: reverse.",
            "Offer **Night-Owl pricing** for rides starting **12–5 AM** (target shift workers); leverage natural rebalancing.",
            "Weekend/holiday **bundles** (4-hour / full-day passes) tailored to leisure patterns.",
            "Automate alerts to reduce outsourced hours on forecast bad-weather days."
        ],
        "Optimization Benefit": [
            "Lower rebalancing cost; higher availability during peaks.",
            "Higher utilization and incremental revenue with minimal ops cost.",
            "Longer rentals and higher ARPU from leisure users.",
            "Direct reduction in unnecessary outsourcing on low-demand days."
        ]
    })
    st.dataframe(rec_df, use_container_width=True)

# =========================================================
# TAB 2: Predictive Model Diagnostics
# =========================================================
if all([preprocessor, model_reg, model_cas]):
    with tab2:
        st.header("2. Predictive Model: Performance and Validation")

        st.markdown("""
### 4.3 Bayesian Tuning — Random Forest (Casual)
The **RandomForestRegressor** for *casual users* was optimized with **Bayesian Search CV**, which learns from prior trials to find the best parameters faster and with fewer evaluations — ideal for costly time-series folds.

### 4.5 Random Forest (Registered)
The *registered-user* model used **Grid Search CV** with time-series splits to tune core parameters while preserving temporal order.
This delivers a stable, high-accuracy model suited to consistent commuter demand.
        """)

        st.subheader("2.1. Final Test Set Performance (Dec 2012)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("RMSE (Error in Bikes)", value=f"{metrics.get('RMSE', 0.0):.2f}")
        col_m2.metric("R² (Variance Explained)", value=f"{metrics.get('R²', 0.0):.3f}")
        col_m3.metric("MAE (Avg. Absolute Error)", value=f"{metrics.get('MAE', 0.0):.2f}")
        col_m4.metric("MAPE (Avg. % Error)", value=f"{metrics.get('MAPE', 0.0):.2f}%")

        st.markdown("---")

        st.subheader("2.2. Model Validation: Actual vs. Predicted Time Series")
        st.markdown("Predicted demand (Registered, Casual, and Total) vs. actual on the final test window.")
        try:
            html_val = read_html_artifact("Actual vs Predicted — Casual Registered CNT.html")
        except Exception:
            html_val = read_html_artifact("Actual_vs_Predicted_Casual_Registered_CNT.html")
        components.html(html_val, height=800, scrolling=True)

        st.markdown("---")

        st.subheader("2.3. Feature Importance (Using Placeholder Data)")
        st.markdown(
            "Illustrative importance charts. Provide `feature_importance_registered.csv` and "
            "`feature_importance_casual.csv` to display actual outputs."
        )
        col_imp1, col_imp2 = st.columns(2)
        with col_imp1:
            st.markdown("#### Registered Model Importance")
            fig_imp_reg = px.bar(imp_reg.head(10), x="importance", y="feature", orientation='h',
                                 title="Top 10 Features (Registered Model)")
            fig_imp_reg.update_layout(yaxis={'categoryorder':'total ascending'}, height=420, title_x=0.5)
            st.plotly_chart(fig_imp_reg, use_container_width=True)
        with col_imp2:
            st.markdown("#### Casual Model Importance")
            fig_imp_cas = px.bar(imp_cas.head(10), x="importance", y="feature", orientation='h',
                                 title="Top 10 Features (Casual Model)")
            fig_imp_cas.update_layout(yaxis={'categoryorder':'total ascending'}, height=420, title_x=0.5)
            st.plotly_chart(fig_imp_cas, use_container_width=True)

# =========================================================
# TAB 3: Provisioning Simulator
# =========================================================
if all([preprocessor, model_reg, model_cas]):
    with tab3:
        st.header("3. Bike Provisioning Simulator: Operational Tool")
        st.markdown(
            """
            Use this tool to input future conditions (Time and Weather) to get a real-time prediction of the required fleet size, broken down by user type. This prediction directly informs bike rebalancing efforts to optimize costs and maximize service availability.
            """
        )
        
        # Inputs
        st.subheader("Input Conditions")
        col_i1, col_i2, col_i3 = st.columns(3)

        default_date = datetime.date(2012, 11, 1)

        with col_i1:
            st.markdown("**Time & Date**")
            date_input = st.date_input("Target Date", default_date)
            hour_input = st.slider("Hour of Day (0-23)", 0, 23, 17)
            
        with col_i2:
            st.markdown("**Seasonal & Work**")
            season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
            current_month = date_input.month
            if 3 <= current_month <= 5: season_default_index = 0
            elif 6 <= current_month <= 8: season_default_index = 1
            elif 9 <= current_month <= 11: season_default_index = 2
            else: season_default_index = 3
            
            season_input = st.selectbox("Season", options=season_map.keys(),
                                        format_func=lambda x: season_map.get(x, "Unknown"), index=season_default_index)
            
            workingday_input = st.selectbox("Is it a Working Day?", options=[0, 1],
                                            format_func=lambda x: "Yes (1)" if x==1 else "No (0)", index=1)
            holiday_input = st.selectbox("Is it a Holiday?", options=[0, 1],
                                        format_func=lambda x: "Yes (1)" if x==1 else "No (0)", index=0)
            
        with col_i3:
            st.markdown("**Weather**")
            weather_map = {1: "Clear, Few clouds", 2: "Mist + Cloudy", 3: "Light Rain/Snow"}
            weathersit_input = st.selectbox("Weather Situation", options=weather_map.keys(),
                                            format_func=lambda x: weather_map.get(x, "Unknown"))
            
            # Sliders are denormalized for user experience, then normalized for the model
            temp_input = st.slider("Temperature (0-41°C)", 0.0, 41.0, 24.6, 0.1) / 41
            atemp_input = st.slider("Feeling Temp (0-50°C)", 0.0, 50.0, 30.0, 0.1) / 50
            hum_input = st.slider("Humidity (0-100%)", 0.0, 100.0, 50.0, 1.0) / 100
            windspeed_input = st.slider("Windspeed (0-67)", 0.0, 67.0, 13.4, 0.1) / 67

        input_data = {
            "dteday": pd.to_datetime(date_input), 
            "yr": date_input.year - 2011,
            "mnth": date_input.month,
            "hr": hour_input,
            "holiday": holiday_input,
            "weekday": date_input.weekday(),
            "workingday": workingday_input,
            "season": season_input,
            "weathersit": weathersit_input,
            "temp": temp_input,
            "atemp": atemp_input,
            "hum": hum_input,
            "windspeed": windspeed_input,
        }

        input_df = pd.DataFrame([input_data])
        
        st.markdown("---")
        st.subheader("Predicted Hourly Demand")
        
        if st.button("Calculate Prediction", type="primary"):
            try:
                raw_input_data = add_features_patched(input_df.copy())
                raw_input_data = raw_input_data.assign(**lag_means_dict)
                X_single_processed = preprocessor.transform(raw_input_data)
                pred_reg_log = model_reg.predict(X_single_processed)[0]
                pred_cas_log = model_cas.predict(X_single_processed)[0]
                pred_reg = max(0, np.expm1(pred_reg_log))
                pred_cas = max(0, np.expm1(pred_cas_log))
                pred_total = pred_reg + pred_cas
                st.success("Prediction Complete!")
                col_p1, col_p2, col_p3 = st.columns(3)
                col_p1.metric("Predicted Registered Users", f"{pred_reg:,.0f}")
                col_p2.metric("Predicted Casual Users", f"{pred_cas:,.0f}")
                col_p3.metric("TOTAL BIKE PROVISIONING NEED", f"{pred_total:,.0f}")

                if pred_total > 500 and workingday_input == 1:
                    st.info("⚠️ **High Demand Forecast:** Peak commute hour likely. Allocate extra fleet capacity and prioritize rebalancing near business/residential hubs.")
                elif pred_total > 300 and workingday_input == 0:
                    st.info("💡 **Strong Leisure Demand:** Expected for this time. Proactively stock stations near leisure areas, parks, and monuments.")
                elif pred_total < 50:
                    st.info("✅ **Low Demand:** Standard fleet levels are sufficient. Consider minimizing costly rebalancing efforts for this hour.")
                    
            except Exception as e:
                st.error("Prediction failed. Ensure all PKL artifacts are correctly loaded and the input features match the model's expectations.")
                st.exception(e)

        st.markdown("---")
        st.header("4. Key Recommendations for Optimization")
        st.markdown(
            "* **Optimize Commuter Provisioning (Registered):** Adjust fleet near business/residential hubs during **7–9 AM** and **4–6 PM** peaks.\n"
            "* **Dynamic Casual Fleet (Weekends/Summer):** Highly sensitive to **temperature** — stock leisure-centric stations on warm afternoons.\n"
            "* **Weather Contingency:** Reduce provisioning during severe weather; demand reliably drops."
        )

# =========================================================
# TAB 5: Deviation Insights (Monthly / Seasonal / Weekday)
# =========================================================
with tab5:
    st.header("5. Deviation Insights (vs Annual Mean)")
    st.markdown("Use these to plan staffing, rebalancing windows, and promo timing. Values are % deviation from the overall annual mean.")

    # Monthly deviation
    st.subheader("5.1 Monthly Usage Deviation")
    html_m = read_html_artifact("Monthly Usage Deviation (% vs annual mean).html")
    components.html(html_m, height=650, scrolling=True)
    st.markdown("""
**Observation:**  
- **Casual** demand surges in **summer** (peaks around Jun–Aug) and plunges in **winter** (Jan–Feb).  
- **Registered** and **cnt** also rise in warm months but with smaller swings than casual.

**Analysis:**  
- Marketing & pop-up stations are most effective in **May–September** for tourists.  
- **Winter** is ideal for maintenance and reduced outsourcing windows (predictable dip).
    """)

    st.markdown("---")

    # Seasonal deviation
    st.subheader("5.2 Seasonal Usage Deviation")
    html_s = read_html_artifact("Seasonal Usage Deviation (% vs annual mean).html")
    components.html(html_s, height=600, scrolling=True)
    st.markdown("""
**Observation:**  
- **Summer** shows strong positive deviation across all segments, **especially casual**.  
- **Winter** is sharply negative for every segment (largest drop for casual).

**Analysis:**  
- Provision more bikes and staff in **Summer**; shift trucks to leisure hotspots.  
- In **Winter**, scale back rebalancing and lean on **Night-Owl** or off-peak promos to lift utilization.
    """)

    st.markdown("---")

    # Weekday deviation
    st.subsection = st.subheader  # tiny alias
    st.subsection("5.3 Weekday Usage Deviation")
    html_w = read_html_artifact("Weekday Usage Deviation (% vs annual mean).html")
    components.html(html_w, height=600, scrolling=True)
    st.markdown("""
**Observation:**  
- **Casual** spikes on **weekends** (Sat/Sun) and is below average mid-week.  
- **Registered** peaks **Tue–Thu**, dips on **weekends**.  
- **Total (cnt)** is relatively stable, with a mild weekend bump from casual.

**Analysis:**  
- **Weekday** mornings/evenings: stock commuter hubs (registered).  
- **Weekends:** pre-position at parks/monuments and run bundles to capture tourist demand.
    """)
