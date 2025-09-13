import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ Load dataset
@st.cache_data
df = pd.read_csv(
    'climate.csv',  # Correct the filename if needed
    parse_dates=["Datetime"],
    index_col="Datetime"
)
    return df

# ✅ Classify risks and suggest mitigations
def classify_risks_and_mitigations(row):
    heatwave_risk = "High" if row["TempC"] > 40 else "Medium" if row["TempC"] > 30 else "Low"
    coldwave_risk = "High" if row["TempC"] < 12 else "Medium" if row["TempC"] <= 18 else "Low"
    flood_risk = "High" if row["PrecipMM"] > 5 else "Medium" if row["PrecipMM"] >= 2 else "Low"
    storm_risk = "High" if row["WindspeedKmph"] > 30 else "Medium" if row["WindspeedKmph"] >= 20 else "Low"

    strategies = {}
    if heatwave_risk in ['High', 'Medium']:
        strategies['Heatwave'] = [
            "Stay hydrated",
            "Avoid outdoor activities",
            "Use fans or AC indoors"
        ]
    if coldwave_risk in ['High', 'Medium']:
        strategies['Coldwave'] = [
            "Wear warm clothes",
            "Use heaters",
            "Limit outdoor exposure"
        ]
    if flood_risk in ['High', 'Medium']:
        strategies['Flood'] = [
            "Avoid flood-prone areas",
            "Keep emergency kit ready",
            "Follow alerts"
        ]
    if storm_risk in ['High', 'Medium']:
        strategies['Storm'] = [
            "Stay indoors",
            "Secure outdoor items",
            "Avoid travel during storms"
        ]

    return {
        "Heatwave Risk": heatwave_risk,
        "Coldwave Risk": coldwave_risk,
        "Flood Risk": flood_risk,
        "Storm Risk": storm_risk,
        "Mitigation Strategies": strategies
    }

# ✅ Display mitigation strategies nicely
def display_mitigation_strategies(strategies):
    if not strategies:
        st.success("🌿 No immediate risks detected! Enjoy a safe day!")
    else:
        st.error("⚠️ Climate Risks Detected! Suggested Mitigation Strategies:")
        for risk, actions in strategies.items():
            st.markdown(f"**{risk} Risk**:")
            for action in actions:
                st.markdown(f"- {action}")

# ✅ Streamlit App
def main():
    st.set_page_config(page_title="🌤 Climate Dashboard", layout="wide")
    st.title("🌤 Climate Risk Prediction Dashboard")

    # Load data
    df = load_data()

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Select a datetime
    selected_datetime = st.selectbox(
        "Select a Datetime:",
        df.index.strftime("%Y-%m-%d %H:%M:%S")
    )
    selected_row = df.loc[pd.to_datetime(selected_datetime)]

    # Show selected weather parameters
    st.subheader(f"🌟 Weather Parameters for {selected_datetime}")
    st.write(selected_row.to_frame(name="Value"))

    # Classify risks
    risk_info = classify_risks_and_mitigations(selected_row)
    display_mitigation_strategies(risk_info["Mitigation Strategies"])

    # Visualizations
    st.subheader("🌡️ Temperature Trend")
    temp_fig = px.line(df, x=df.index, y="TempC", title="Temperature Over Time")
    st.plotly_chart(temp_fig, use_container_width=True)

    st.subheader("🌧️ Precipitation Trend")
    precip_fig = px.line(df, x=df.index, y="PrecipMM", title="Precipitation Over Time")
    st.plotly_chart(precip_fig, use_container_width=True)

    st.subheader("🌬️ Wind Speed Trend")
    wind_fig = px.line(df, x=df.index, y="WindspeedKmph", title="Wind Speed Over Time")
    st.plotly_chart(wind_fig, use_container_width=True)

if __name__ == "__main__":
    print("✅ Running the NEW Climate Dashboard app...")
    main()
