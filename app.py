# AI-Driven-Climate-Risk-Prediction-and-Mitigation-Framework
# Single-file Streamlit app
# Save this as `app.py` and run with: `streamlit run app.py`

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="AI Climate Risk Predictor", layout="wide")

# -----------------
# Helper functions
# -----------------

def load_model(path: str):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        return None


def save_model(model, path: str):
    try:
        joblib.dump(model, path)
        return True
    except Exception as e:
        return False


def synthetic_mitigation_recommendations(row: pd.Series):
    """Return list of mitigation actions based on input feature values (simple rule-based)."""
    recs = []
    # Example rules — adapt these to your feature set
    if 'mean_temp' in row.index and row['mean_temp'] > 30:
        recs.append('Increase urban tree cover and green roofs to reduce heat exposure')
    if 'precip_mm' in row.index and row['precip_mm'] > 200:
        recs.append('Improve stormwater drainage; install retention ponds')
    if 'sea_level_m' in row.index and row['sea_level_m'] > 1.0:
        recs.append('Strengthen coastal defenses and plan managed retreat in high-risk zones')
    if len(recs) == 0:
        recs.append('Perform local vulnerability assessment and community engagement')
    return recs


def train_model(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    return model, metrics

# -----------------
# UI: Sidebar
# -----------------

st.sidebar.title('AI Climate Risk — Controls')
app_mode = st.sidebar.selectbox('Choose action',
                                ['Overview', 'Upload Data & Train', 'Use Model (Predict)', 'Model Management', 'About'])

# where models and artifacts will be stored
ARTIFACTS_DIR = 'artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'climate_risk_model.joblib')

# -----------------
# Overview
# -----------------

if app_mode == 'Overview':
    st.title('AI-Driven Climate Risk Prediction & Mitigation Framework')
    st.markdown(
        """
        This Streamlit app demonstrates a simple but extensible pipeline for predicting climate-related risk
        indicators (for example: economic loss, flood risk index, heat vulnerability index) from input features
        (temperature, precipitation, elevation, population density, landcover, etc.) and generating mitigation
        recommendations.

        **Main features**
        - Upload CSV dataset and train a RandomForest regression model.
        - Evaluate model and download trained model.
        - Predict climate risk for single locations or batch inputs.
        - Receive simple rule-based mitigation actions tailored to predictions.

        _Notes_: This is a starter template — replace feature engineering and rule logic with domain-appropriate models.
        """
    )

    st.header('Sample workflow')
    st.write('1. Prepare a CSV with numeric features and a numeric target column named `target` (or choose your own).')
    st.write('2. Upload it under "Upload Data & Train" and train a model. Play with hyperparameters if needed.')
    st.write('3. Save the trained model and use "Use Model (Predict)" to run single or batch predictions.')

    st.subheader('Example synthetic dataset generator (click to create)')
    if st.button('Generate small synthetic dataset'):
        np.random.seed(42)
        N = 500
        df = pd.DataFrame({
            'mean_temp': np.random.normal(25, 5, N),
            'precip_mm': np.random.exponential(80, N),
            'elevation_m': np.random.normal(100, 50, N),
            'population_density': np.random.lognormal(3, 1, N),
            'sea_level_m': np.random.uniform(0, 2, N)
        })
        # synthetic target: risk score
        df['target'] = (0.4 * (df['mean_temp'] - 15)
                        + 0.003 * df['precip_mm']
                        - 0.001 * df['elevation_m']
                        + 0.0007 * df['population_density']
                        + 0.6 * df['sea_level_m']
                        + np.random.normal(0, 1, N))
        st.success('Synthetic dataset generated — preview below and download as CSV')
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download synthetic dataset (CSV)', data=csv, file_name='synthetic_climate_data.csv')

# -----------------
# Upload & Train
# -----------------

elif app_mode == 'Upload Data & Train':
    st.title('Upload dataset and train model')
    st.markdown('Upload a CSV where one column is the numeric `target` you want to predict (risk score).')
    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
            st.dataframe(df.head())

            st.sidebar.write('Training settings')
            target_col = st.selectbox('Select target column', options=df.columns.tolist(), index=len(df.columns)-1)
            test_size = st.sidebar.slider('Test set fraction', min_value=0.05, max_value=0.5, value=0.2, step=0.05)
            random_state = st.sidebar.number_input('Random seed', value=42)

            # Simple numeric-only preprocessing
            numeric_df = df.select_dtypes(include=[np.number])
            if target_col not in numeric_df.columns:
                st.warning('Selected target column is not numeric — choose a numeric target or preprocess your data first.')
            else:
                if st.button('Train model'):
                    with st.spinner('Training model — this may take a moment'):
                        model, metrics = train_model(numeric_df, target_col, test_size=test_size, random_state=int(random_state))
                        save_model(model, DEFAULT_MODEL_PATH)
                    st.success('Model trained and saved to artifacts/')
                    st.metric('RMSE', f"{metrics['rmse']:.4f}")
                    st.metric('R^2', f"{metrics['r2']:.4f}")
                    st.write(f"Training rows: {metrics['n_train']}, Test rows: {metrics['n_test']}")
                    st.write('Download trained model:')
                    with open(DEFAULT_MODEL_PATH, 'rb') as f:
                        st.download_button('Download model (.joblib)', data=f, file_name='climate_risk_model.joblib')

        except Exception as e:
            st.error(f'Error reading CSV: {e}')

    else:
        st.info('Upload a CSV to begin. If you do not have one, generate a synthetic dataset in Overview.')

# -----------------
# Predict
# -----------------

elif app_mode == 'Use Model (Predict)':
    st.title('Use trained model for prediction')

    model = load_model(DEFAULT_MODEL_PATH)
    if model is None:
        st.warning('No trained model found in artifacts/. Train a model first under "Upload Data & Train" or upload a model in Model Management.')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader('Single location prediction')
        st.write('Enter feature values to predict risk for a single location (numeric features only).')

        # Provide some common default features; in a real app these should reflect your dataset
        mean_temp = st.number_input('Mean temperature (°C)', value=28.0)
        precip_mm = st.number_input('Annual precipitation (mm)', value=120.0)
        elevation_m = st.number_input('Elevation (m)', value=50.0)
        population_density = st.number_input('Population density (people/km²)', value=1500.0)
        sea_level_m = st.number_input('Sea level / distance to coast (m)', value=0.5)

        input_dict = {
            'mean_temp': mean_temp,
            'precip_mm': precip_mm,
            'elevation_m': elevation_m,
            'population_density': population_density,
            'sea_level_m': sea_level_m
        }

        if st.button('Predict single location'):
            input_df = pd.DataFrame([input_dict])
            if model is None:
                st.info('No model loaded — returning a synthetic estimate based on a simple heuristic')
                # simple baseline heuristic
                estimate = (0.4 * (mean_temp - 15) + 0.003 * precip_mm - 0.001 * elevation_m + 0.0007 * population_density + 0.6 * sea_level_m)
                st.metric('Predicted risk score (heuristic)', f"{estimate:.3f}")
                recs = synthetic_mitigation_recommendations(input_df.iloc[0])
                st.subheader('Mitigation recommendations (heuristic rules)')
                for r in recs:
                    st.write('- ', r)
            else:
                # attempt to align model inputs
                try:
                    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_df.columns
                    X = input_df.reindex(columns=model_features, fill_value=0)
                except Exception:
                    X = input_df
                pred = model.predict(X)[0]
                st.metric('Predicted risk score', f"{pred:.3f}")
                recs = synthetic_mitigation_recommendations(input_df.iloc[0])
                st.subheader('Mitigation recommendations')
                for r in recs:
                    st.write('- ', r)

    with col2:
        st.subheader('Batch prediction (CSV)')
        st.write('Upload a CSV with the same features used during training (numeric only).')
        batch_file = st.file_uploader('Upload batch CSV for prediction', type=['csv'], key='batch')
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.write(f'Batch data shape: {batch_df.shape}')
                st.dataframe(batch_df.head())

                if st.button('Run batch prediction'):
                    if model is None:
                        st.error('No trained model available. Train or upload a model first.')
                    else:
                        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else batch_df.columns
                        X = batch_df.reindex(columns=model_features, fill_value=0)
                        preds = model.predict(X)
                        out = batch_df.copy()
                        out['predicted_risk'] = preds
                        st.success('Batch prediction completed')
                        st.dataframe(out.head())
                        csv = out.to_csv(index=False).encode('utf-8')
                        st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv')

                        # Show distribution
                        fig = px.histogram(out, x='predicted_risk', nbins=40, title='Predicted risk distribution')
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f'Error processing batch CSV: {e}')

# -----------------
# Model Management
# -----------------

elif app_mode == 'Model Management':
    st.title('Model management')
    st.write('Upload or download trained models, or delete the artifact.')

    st.write('Local model path: ', DEFAULT_MODEL_PATH)
    if os.path.exists(DEFAULT_MODEL_PATH):
        st.success('A model exists in artifacts/')
        st.download_button('Download current model', data=open(DEFAULT_MODEL_PATH, 'rb'), file_name='climate_risk_model.joblib')
        if st.button('Delete local model'):
            try:
                os.remove(DEFAULT_MODEL_PATH)
                st.success('Model deleted')
            except Exception as e:
                st.error(f'Could not delete model: {e}')
    else:
        st.info('No saved model found.')

    st.subheader('Upload a pre-trained model (.joblib)')
    model_upload = st.file_uploader('Upload .joblib model file', type=['joblib'], key='model_upload')
    if model_upload is not None:
        try:
            bytes_data = model_upload.read()
            with open(DEFAULT_MODEL_PATH, 'wb') as f:
                f.write(bytes_data)
            st.success('Model uploaded and saved to artifacts/')
        except Exception as e:
            st.error(f'Error saving uploaded model: {e}')

# -----------------
# About
# -----------------

elif app_mode == 'About':
    st.title('About & Next steps')
    st.markdown(
        """
        **This is a starter framework.**

        _Possible extensions:_
        - Add geospatial visualization (folium/pydeck) to map predicted risk by coordinates.
        - Replace RandomForest with XGBoost / LightGBM / Neural Networks for better performance.
        - Add feature engineering: lagged climate variables, indices (SPI, SPEI), satellite-derived features.
        - Integrate with Google Drive / S3 to store models and datasets.
        - Implement user authentication (OAuth/Google Sign-In) for multi-user environments.

        _Run instructions:_
        1. Create and activate a Python environment.
        2. Install dependencies: `pip install streamlit pandas scikit-learn joblib matplotlib plotly`
        3. Run: `streamlit run app.py`

        If you'd like, I can also create a `requirements.txt` or a Dockerfile next.
        """
    )

# -----------------
# Footer: quick debug info
# -----------------

st.sidebar.markdown('---')
st.sidebar.write('App last loaded: ', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ'))

# End of file
