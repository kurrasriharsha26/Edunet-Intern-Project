import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# -------------------------------
# Animated Gradient Background (Light Colors)
# -------------------------------
def set_light_gradient_background():
    st.markdown(
        """
        <style>
        @keyframes gradientAnimation {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .stApp {
            background: linear-gradient(-45deg, #FFDEE9, #B5FFFC, #FFFFC7, #FFF1B5, #FFE4E1, #E0FFFF, #F0FFF0, #FFDAB9);
            background-size: 400% 400%;
            animation: gradientAnimation 25s ease infinite;
            color: black;
        }

        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #45A049;
            transform: scale(1.05);
        }

        .styled-table {
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
            font-family: sans-serif;
            min-width: 400px;
            border-radius: 10px 10px 0 0;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }
        .styled-table thead tr {
            background-color: #A3D2CA;
            color: #000000;
            text-align: left;
            font-weight: bold;
        }
        .styled-table th, .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
            background-color: #FFFFFF;
            color: black;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #F1F1F1;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #A3D2CA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_light_gradient_background()

# -------------------------------
# Generate Synthetic Dataset
# -------------------------------
def create_synthetic_data(n=500, start_date="2025-09-20"):
    rng = np.random.default_rng(42)
    dates = pd.date_range(start_date, periods=n, freq="D")
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
st.set_page_config(page_title="AI-Driven Climate Risk Prediction", layout="wide")
st.title("🌍 AI-Driven Climate Risk Prediction & Mitigation Framework")

# Use synthetic dataset starting from 2025-09-20
df = create_synthetic_data(start_date="2025-09-20")
st.success("Synthetic dataset generated starting from 2025-09-20!")

# Render styled HTML table
st.markdown(
    df.head().to_html(classes="styled-table"),
    unsafe_allow_html=True
)

# Select target column
target_col = st.selectbox("Select the target variable for prediction:", df.columns, index=len(df.columns)-1)

# Train model
if st.button("Train Model"):
    with st.spinner("Training model..."):
        model, mse, r2 = train_model(df, target_col)
        st.session_state["model"] = model
        st.success("Model trained successfully!")
        st.write(f"📊 Mean Squared Error: {mse:.2f}")
        st.write(f"📈 R² Score: {r2:.2f}")

        # Save model
        joblib.dump(model, "climate_risk_model.joblib")
        st.info("Model saved as `climate_risk_model.joblib`")

# Prediction section
if "model" in st.session_state:
    st.subheader("🔮 Make Predictions")

    input_data = {}
    for col in df.drop(columns=[target_col]).columns:
        input_data[col] = st.number_input(
            f"Enter value for {col}",
            float(df[col].min()), float(df[col].max()), float(df[col].mean())
        )

    if st.button("Predict Climate Risk"):
        input_df = pd.DataFrame([input_data])
        prediction = st.session_state["model"].predict(input_df)[0]
        st.success(f"Predicted {target_col}: {prediction:.2f}")
        st.write(mitigation_recommendations(prediction))

# -------------------------------
# Visualization (Light Theme)
# -------------------------------
st.subheader("📊 Data Visualization")

fig = px.line(
    df, 
    x=df.index, 
    y=target_col, 
    title=f"{target_col} over Time",
    markers=True
)

fig.update_layout(
    template="plotly_white",
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    font=dict(color="black"),
    title=dict(font=dict(size=22, color="#333333"), x=0.5),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", color="black"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", color="black")
)

fig.update_traces(line=dict(color="#FF6F61", width=3), marker=dict(color="#4CAF50", size=8))

st.plotly_chart(fig, use_container_width=True)
