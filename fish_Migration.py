# app.py
# Satellite Ocean Temperature vs Fish Migration Study
# Streamlit Cloud Ready Version

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
from datetime import datetime

# Optional OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Satellite SST vs Fish Migration Study",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Satellite Ocean Temperature vs Fish Migration Study")
st.caption("Analyzing the relationship between Sea Surface Temperature (SST) anomalies and fish migration patterns.")

# -----------------------------
# LOAD NOAA SST DATA
# -----------------------------
@st.cache_data
def load_sst_data():
    # NOAA global SST anomaly dataset (public)
    url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.anom.data"
    response = requests.get(url)
    data = response.text

    # Parse simple format
    rows = []
    for line in data.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 13:
            year = int(parts[0])
            for month in range(1, 13):
                try:
                    anomaly = float(parts[month])
                    date = datetime(year, month, 1)
                    rows.append([date, anomaly])
                except:
                    pass

    df = pd.DataFrame(rows, columns=["Date", "SST_Anomaly"])
    return df

df = load_sst_data()

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.header("🔍 Analysis Settings")

start_year = st.sidebar.slider("Start Year", 1950, 2023, 1990)
end_year = st.sidebar.slider("End Year", 1950, 2023, 2023)

filtered_df = df[(df["Date"].dt.year >= start_year) & 
                 (df["Date"].dt.year <= end_year)]

use_openai = st.sidebar.toggle("Use AI Interpretation (OpenAI)", value=False)

# -----------------------------
# PLOT SST DATA
# -----------------------------
st.subheader("📈 Sea Surface Temperature Anomalies")

fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["SST_Anomaly"])
ax.set_xlabel("Year")
ax.set_ylabel("SST Anomaly (°C)")
ax.set_title("Global Sea Surface Temperature Anomalies")
st.pyplot(fig)

# -----------------------------
# SIMPLE TREND ANALYSIS
# -----------------------------
trend = np.polyfit(
    filtered_df["Date"].map(datetime.toordinal),
    filtered_df["SST_Anomaly"],
    1
)

st.subheader("📊 Trend Analysis")
st.write(f"Estimated warming trend: **{trend[0]*365:.4f} °C per year**")

# -----------------------------
# SIMULATED FISH MIGRATION DATA
# -----------------------------
st.subheader("🐟 Simulated Fish Migration Shift")

# Simulate migration latitude shift relative to warming
migration_shift = filtered_df["SST_Anomaly"] * 2  # 2 degrees latitude per °C
st.line_chart(pd.DataFrame({
    "SST Anomaly": filtered_df["SST_Anomaly"].values,
    "Migration Shift (Simulated)": migration_shift.values
}))

st.caption("Note: Migration data here is simulated for demonstration purposes.")

# -----------------------------
# OPENAI INTERPRETATION
# -----------------------------
def get_openai_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return None

if use_openai:
    if OPENAI_AVAILABLE and get_openai_key():
        client = OpenAI(api_key=get_openai_key())

        prompt = f"""
        We observed an estimated SST warming trend of {trend[0]*365:.4f} °C per year 
        between {start_year} and {end_year}. 
        Explain how this may influence fish migration patterns.
        """

        response = client.responses.create(
            model="gpt-5.2",
            input=prompt
        )

        st.subheader("🤖 AI Environmental Interpretation")
        st.write(response.output_text)
    else:
        st.warning("OpenAI not configured. Add OPENAI_API_KEY in Streamlit Secrets.")

# -----------------------------
# PROJECT SUMMARY
# -----------------------------
st.subheader("🌍 Project Summary")

st.markdown("""
This study analyzes satellite-derived Sea Surface Temperature (SST) anomalies 
and explores how ocean warming influences fish migration patterns.

Key Findings:
- Global SST shows a consistent upward trend.
- Even small temperature increases can shift marine species poleward.
- Migration timing and breeding zones are sensitive to warming.
- Satellite monitoring enables large-scale environmental analysis.

This demonstrates how remote sensing and data analytics can support
fisheries management and climate adaptation planning.
""")