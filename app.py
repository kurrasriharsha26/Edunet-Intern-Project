import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import requests
from streamlit_lottie import st_lottie

# -------------------------------
# Animated Gradient Background
# -------------------------------
def set_animated_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, 
                                        #1e3c72, 
                                        #2a5298, 
                                        #f0f8ff, 
                                        #6497b1, 
                                        #3a5683, 
                                        #1e3c72);
            background-size: 400% 400%;
            animation: gradientBG 20s ease-in-out infinite;
            min-height: 100vh;
        }
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Load Lottie Animation
# -------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Weather/climate Lottie animation
climate_lottie_url = "https://assets10.lottiefiles.com/packages/lf20_jmgekfqz.json"
lottie_climate = load_lottieurl(climate_lottie_url)

# -------------------------------
# Generate Synthetic Dataset
# -------------------------------
def create_synthetic_data(n=500):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    temperature = rng.normal(30, 5, n)        # Celsius
    rainfall = rng.normal(100, 20, n)         # mm
    humidity = rng.uniform(40, 90, n)         # %
    risk_index = 0.4 * temperature + 0.3 * (rainfall / 10) + 0.3 * humidity + rng.normal(0, 2, n)

    df = pd.DataFrame({
        "Datetime": dates,
        "Temperature": temperature,
        "Rainfall": rainfall,
        "Humidity": humidity,
        "ClimateRiskIndex": risk_index
    })
    df.set_index("Datetime", inplace=True)
    return df

# -------------------------------
# Train Model
# -------------------------------
def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# -------------------------------
# Mitigation Recommendations
# -------------------------------
def mitigation_recommendations(value, threshold=60):
    if value > threshold:
        return "⚠️ High climate risk detected. Suggested actions: Reduce emissions, improve drainage, plant more trees."
    else:
        return "✅ Climate risk is under control. Continue monitoring."

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="AI-Driven Climate Risk Prediction")
set_animated_background()

# Header with animation
col1, col2 = st.columns([1, 2])
with col1:
    if lottie_climate:
        st_lottie(lottie_climate, width=250, height=250, loop=True)
with col2:
    st.title("🌍 AI-Driven Climate Risk Prediction & Mitigation Framework")
    #st.markdown("### 🔬 Using Machine Learning to Forecast Climate Risk and Suggest Mitigation Strategies")

# Use synthetic dataset
df = create_synthetic_data()
st.success("✅ Synthetic dataset generated successfully!")
st.write("Preview of dataset:", df.head())

# Select target column
target_col = st.selectbox("📌 Select the target variable for prediction:", df.columns, index=len(df.columns)-1)

# Train model
if st.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        model, mse, r2 = train_model(df, target_col)
        st.session_state["model"] = model
        st.success("🎯 Model trained successfully!")
        st.write(f"📊 Mean Squared Error: {mse:.2f}")
        st.write(f"📈 R² Score: {r2:.2f}")

        # Save model
        joblib.dump(model, "climate_risk_model.joblib")
        st.info("💾 Model saved as `climate_risk_model.joblib`")

# Prediction section
if "model" in st.session_state:
    st.subheader("🔮 Make Predictions")

    input_data = {}
    for col in df.drop(columns=[target_col]).columns:
        input_data[col] = st.number_input(
            f"Enter value for {col}",
            float(df[col].min()), float(df[col].max()), float(df[col].mean())
        )

    if st.button("🔎 Predict Climate Risk"):
        input_df = pd.DataFrame([input_data])
        prediction = st.session_state["model"].predict(input_df)[0]
        st.success(f"🌡 Predicted {target_col}: {prediction:.2f}")
        st.write(mitigation_recommendations(prediction))

# Visualization
st.subheader("📊 Data Visualization")
fig = px.line(df, x=df.index, y=target_col, title=f"{target_col} over Time")
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
