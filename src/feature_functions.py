import pandas as pd
import numpy as np

def add_features(x: pd.DataFrame) -> pd.DataFrame:
    X = x.copy()  # never mutate input

    # --- helper bases ---
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
        y_w  = (A1_w*np.exp(-(hr-mu1_w)**2/(2*s1_w**2))
                + A2_w*np.exp(-(hr-mu2_w)**2/(2*s2_w**2)) + c_w)
        y_we = (A1_we*np.exp(-(hr-mu1_we)**2/(2*s1_we**2))
                + A2_we*np.exp(-(hr-mu2_we)**2/(2*s2_we**2)) + c_we)
        return np.where(workday==1, y_w, y_we)

    def first_harmonic_of_cyclic_features(df: pd.DataFrame, cols: list):
        df = df.copy()
        for col in cols:
            if col not in df.columns:
                continue
            sin_col = f"{col}_sin"
            cos_col = f"{col}_cos"
            if sin_col in df.columns or cos_col in df.columns:
                continue
            period = int(max(df[col].nunique(), 1))
            df[sin_col] = np.sin(2*np.pi*df[col]/period) if period > 1 else 0.0
            df[cos_col] = np.cos(2*np.pi*df[col]/period) if period > 1 else 1.0
        return df

    def _safe(col): return col in X.columns

    # --- nonlinear and interaction features ---
    if _safe("temp"):      X["temp_invU"]       = temp_curve(X["temp"])
    if _safe("hum"):       X["hum_quad_curve"]  = humidity_quad(X["hum"])
    if _safe("hum"):       X["hum_exp_curve"]   = humidity_exp(X["hum"])
    if _safe("windspeed"): X["wind_quad_curve"] = wind_curve(X["windspeed"])
    if _safe("hr"):        X["hr_bimodal"]      = bimodal_hour(X["hr"])
    if _safe("hr"):        X["hr_gauss_morn"]   = np.exp(-(X["hr"]-8.0)**2  / (2*1.4**2))
    if _safe("hr"):        X["hr_gauss_even"]   = np.exp(-(X["hr"]-17.5)**2 / (2*1.8**2))
    if _safe("hr") and _safe("workingday"):
        X["workday_hour_curve"]   = workingday_hour(X["hr"], X["workingday"])
        X["work_x_hr_gauss_morn"] = X["workingday"] * X.get("hr_gauss_morn", 0.0)
        X["work_x_hr_gauss_even"] = X["workingday"] * X.get("hr_gauss_even", 0.0)

    # --- boolean & composite flags ---
    if _safe("weekday"):
        X["is_weekend"] = X["weekday"].isin([0,6]).astype(int)
    if _safe("hr"):
        X["is_evening_or_afternoon"] = X["hr"].between(10, 20).astype(int)
    if _safe("mnth"):
        X["is_summer_month"] = X["mnth"].between(5, 9).astype(int)
    if _safe("atemp"):
        X["pleasant_temp"] = (X["atemp"] * 41.0).between(15, 28).astype(int)
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

    # --- daylight hours feature (using dteday) ---
    if _safe("dteday"):
        try:
            pi = 3.14
            d = pd.to_datetime(X["dteday"])
            X["doy"] = d.dt.dayofyear.astype(int)
            lat = np.deg2rad(38.9)  # Washington D.C. latitude
            n = X["doy"].clip(1, 365)
            decl = 23.44 * pi/180 * np.sin(2*pi*(284 + n)/365.0)
            H0 = np.arccos(-np.tan(lat) * np.tan(decl))
            daylight = (2 * H0) * 24 / (2*pi)
            X["daylight_hours"] = daylight.astype(float)
        except Exception as e:
            print(f"⚠️ Daylight-hours feature skipped: {e}")

    # --- cyclic sin/cos features ---
    X = first_harmonic_of_cyclic_features(X, ["hr", "mnth", "weekday", "season"])

    return X
