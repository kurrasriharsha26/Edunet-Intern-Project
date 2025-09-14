"""
Streamlit App: AI-Driven Climate Risk Prediction & Mitigation Framework
Filename: streamlit_ai_climate_risk_app.py

Features:
- Upload a CSV or use a small sample dataset
- Lightweight preprocessing (dropna, select cols)
- If dataset contains 'risk_score' -> trains RandomForestRegressor (with train/test split)
- Otherwise computes a synthetic risk score (user-tunable weights) or KMeans clustering
- Visualizations: summary stats, time series (if date), map (pydeck), feature importance
- Model download (joblib)
- Export processed small CSV ready for GitHub

Notes:
- Keep dataset small before uploading to GitHub; app includes an export-compress button.
- Dependencies: streamlit, pandas, scikit-learn, numpy, matplotlib, pydeck, joblib

Run: streamlit run streamlit_ai_climate_risk_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import pydeck as pdk

st.set_page_config(page_title="AI Climate Risk: Prediction & Mitigation", layout="wide")

# -----------------
# Helpers
# -----------------
@st.cache_data
def load_sample_data(n=500):
    # Create a small synthetic dataset useful for demo and GitHub upload
    rng = np.random.default_rng(42)
    lats = rng.uniform(-60, 60, size=n)
    lons = rng.uniform(-180, 180, size=n)
    temp_anom = rng.normal(loc=1.0, scale=1.5, size=n)  # degrees C
    precip = rng.exponential(scale=50, size=n)  # mm/month
    pop_density = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    elev = rng.normal(loc=200, scale=400, size=n)
    past_disasters = rng.poisson(lam=1.2, size=n)
    # synthetic risk score (0-100)
    risk_score = np.clip((temp_anom*10 + precip*0.05 + np.log1p(pop_density)*3 + past_disasters*5 - elev*0.01) + rng.normal(0,5,n), 0, 100)

    df = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'temp_anomaly_C': temp_anom,
        'precip_mm': precip,
        'population_density': pop_density,
        'elevation_m': elev,
        'past_disasters_count': past_disasters,
        'risk_score': risk_score
    })
    return df

@st.cache_data
def preprocess(df, selected_features):
    df = df.copy()
    # basic cleaning
    df = df.dropna(subset=selected_features)
    # create any derived features here
    if 'population_density' in df.columns:
        df['pop_log'] = np.log1p(df['population_density'])
        if 'pop_log' not in selected_features:
            selected_features = selected_features + ['pop_log']
    return df, selected_features

@st.cache_data
def train_regressor(X, y, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model

# -----------------
# UI: Sidebar
# -----------------
st.sidebar.title("AI Climate Risk — Controls")
upload = st.sidebar.file_uploader("Upload CSV (small files recommended for GitHub)", type=['csv'])
use_sample = st.sidebar.checkbox("Use sample demo dataset (small, ready-to-upload)", value=True)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ['Explore Data', 'Train Model', 'Predict & Map', 'Mitigation Suggestions'])

# Model params
n_estimators = st.sidebar.slider('RandomForest n_estimators', 10, 300, 100)
test_size = st.sidebar.slider('Test size (%)', 10, 50, 20)

# Export / GitHub options
st.sidebar.markdown("### Export / GitHub helpers")
export_sample = st.sidebar.button("Export processed CSV (gzip)")

# -----------------
# Load Data
# -----------------
if upload is not None:
    try:
        df = pd.read_csv(upload)
        st.sidebar.success(f"Loaded {upload.name} — {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        st.sidebar.error(f"Failed to load CSV: {e}")
        st.stop()
elif use_sample:
    df = load_sample_data(500)
    st.sidebar.info("Using built-in sample dataset (500 rows)")
else:
    st.info("Upload a CSV or enable the sample dataset in the sidebar.")
    st.stop()

st.title("AI-Driven Climate Risk Prediction & Mitigation Framework")
st.write("Lightweight Streamlit app to explore datasets, train a risk model, visualize hotspots, and export small CSVs ready for GitHub.")

# Quick view
with st.expander("Preview dataset (first 10 rows)", expanded=True):
    st.dataframe(df.head(10))

# Feature selection
st.sidebar.markdown("---")
st.sidebar.markdown("### Feature selection")
all_columns = list(df.columns)
selected_features = st.sidebar.multiselect('Choose features to use in model / scoring', options=all_columns, default=[c for c in ['temp_anomaly_C','precip_mm','population_density','elevation_m','past_disasters_count','latitude','longitude'] if c in all_columns])

# -----------------
# Export processed CSV
# -----------------
if export_sample:
    proc_df, proc_feats = preprocess(df, selected_features)
    buf = io.BytesIO()
    # export compressed csv
    proc_df.to_csv(buf, index=False, compression='gzip')
    buf.seek(0)
    st.download_button(label='Download processed CSV (gzip)', data=buf, file_name='climate_processed.csv.gz', mime='application/gzip')

# -----------------
# Mode: Explore Data
# -----------------
if mode == 'Explore Data':
    st.header('Dataset summary & plots')
    st.subheader('Basic summary')
    st.write(df.describe())

    st.subheader('Column types & missing values')
    col_info = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'missing': df.isna().sum()
    })
    st.dataframe(col_info)

    # If there's a date column, show time series sample
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        st.subheader(f'Time series using {date_col}')
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            ts = df.set_index(date_col).resample('M').mean()
            st.line_chart(ts.select_dtypes(include=[np.number]))
        except Exception as e:
            st.warning(f'Could not parse {date_col} as datetime: {e}')

# -----------------
# Mode: Train Model
# -----------------
elif mode == 'Train Model':
    st.header('Train risk prediction model')
    if 'risk_score' in df.columns:
        st.success('Detected label column: risk_score — proceeding with supervised regression')
        proc_df, proc_feats = preprocess(df, selected_features + ['risk_score'])
        X = proc_df[[c for c in proc_feats if c != 'risk_score' and c in proc_df.columns]]
        y = proc_df['risk_score']
        st.write('Training rows:', X.shape[0])

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
        model = train_regressor(X_train, y_train, n_estimators=n_estimators)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        st.subheader('Model results')
        st.write(f'RMSE: {rmse:.3f} | R2: {r2:.3f}')

        # feature importance
        try:
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader('Feature importance')
            st.bar_chart(imp)
        except Exception:
            pass

        # Save model
        model_buf = io.BytesIO()
        joblib.dump(model, 'rf_climate_model.joblib')
        with open('rf_climate_model.joblib','rb') as f:
            st.download_button('Download trained model (.joblib)', data=f, file_name='rf_climate_model.joblib')

    else:
        st.info("No 'risk_score' column found. You can either upload a labeled dataset or switch to 'Predict & Map' to compute synthetic risk scores.")

# -----------------
# Mode: Predict & Map
# -----------------
elif mode == 'Predict & Map':
    st.header('Compute risk (supervised or synthetic) and visualize hotspots')
    proc_df, proc_feats = preprocess(df, selected_features)

    if 'risk_score' in proc_df.columns:
        st.write('Using provided risk_score for mapping')
        map_df = proc_df.dropna(subset=['latitude','longitude','risk_score'])
    else:
        st.write('No label found — computing synthetic risk score')
        # allow user to set weights
        st.sidebar.markdown('### Synthetic risk weights (for heuristic scoring)')
        w_temp = st.sidebar.slider('Temp anomaly weight', 0.0, 5.0, 1.0)
        w_precip = st.sidebar.slider('Precip weight', 0.0, 1.0, 0.05)
        w_pop = st.sidebar.slider('Population weight', 0.0, 5.0, 1.0)
        w_past = st.sidebar.slider('Past disasters weight', 0.0, 10.0, 5.0)
        w_elev = st.sidebar.slider('Elevation weight (negative reduces risk)', -1.0, 1.0, -0.01)

        sdf = proc_df.copy()
        # normalize features safely
        def safe_norm(x):
            if x.std() == 0 or np.isnan(x.std()):
                return np.zeros_like(x)
            return (x - x.mean()) / (x.std())

        score = np.zeros(len(sdf))
        if 'temp_anomaly_C' in sdf.columns:
            score += w_temp * safe_norm(sdf['temp_anomaly_C'])
        if 'precip_mm' in sdf.columns:
            score += w_precip * safe_norm(sdf['precip_mm'])
        if 'population_density' in sdf.columns:
            score += w_pop * safe_norm(np.log1p(sdf['population_density']))
        if 'past_disasters_count' in sdf.columns:
            score += w_past * safe_norm(sdf['past_disasters_count'])
        if 'elevation_m' in sdf.columns:
            score += w_elev * safe_norm(sdf['elevation_m'])

        # scale to 0-100
        score = (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score) + 1e-9) * 100
        sdf['risk_score'] = score
        map_df = sdf.dropna(subset=['latitude','longitude','risk_score'])

        st.subheader('Synthetic risk distribution')
        st.write(map_df['risk_score'].describe())
        st.histogram = plt.figure()
        plt.hist(map_df['risk_score'], bins=30)
        plt.xlabel('Risk score')
        plt.ylabel('Count')
        st.pyplot(plt)

    # Map with pydeck
    if 'latitude' in map_df.columns and 'longitude' in map_df.columns:
        st.subheader('Interactive map: hotspots by risk_score')
        midpoint = (map_df['latitude'].mean(), map_df['longitude'].mean())
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[longitude, latitude]',
            get_fill_color='[255 * (risk_score/100), 255*(1-risk_score/100), 80]',
            get_radius=30000,
            pickable=True
        )
        view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=1)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Risk: {risk_score}"})
        st.pydeck_chart(r)
    else:
        st.warning('latitude/longitude columns required for mapping')

# -----------------
# Mode: Mitigation Suggestions
# -----------------
elif mode == 'Mitigation Suggestions':
    st.header('Automated mitigation & next-steps suggestions')
    st.write('Based on hotspots identified, the app suggests types of mitigation actions. These are high-level and must be adapted by domain experts.')

    st.subheader('1) Early warning & monitoring')
    st.write('- Increase local weather and hydrological monitoring stepping up gauge density.
- Deploy community alert systems and mobile notifications.
- Prioritise high-risk population centers for sensor deployment.')

    st.subheader('2) Infrastructure & planning')
    st.write('- Implement nature-based solutions (mangroves, urban trees).
- Strengthen critical infrastructure and drainage in hotspots.
- Update land-use planning to limit development in high-risk zones.')

    st.subheader('3) Socioeconomic measures')
    st.write('- Develop targeted insurance schemes and microfinance instruments.
- Community preparedness training and evacuation drills.
- Investments in livelihood diversification for vulnerable groups.')

    st.subheader('4) Monitoring & MLOps')
    st.write('- Maintain a model retraining schedule when new labeled events arrive.
- Keep all datasets small and versioned; store heavy raw data externally and push lightweight processed CSVs to GitHub.')

st.sidebar.markdown('---')
st.sidebar.write('Tips: Before uploading data to GitHub, remove unnecessary columns, sample or filter by region, and compress CSV (.gz).')

# End of app

