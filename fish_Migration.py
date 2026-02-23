# fish_Migration.py
# Streamlit Cloud-ready: Satellite SST vs Fish Migration Study (FIXED datetime issue)
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run fish_Migration.py
#
# Streamlit Cloud:
#   Set main file to fish_Migration.py
#   Add OPENAI_API_KEY in Secrets (optional)

from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Satellite SST vs Fish Migration Study",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Satellite Ocean Temperature vs Fish Migration Study")
st.caption(
    "Analyzing how satellite-derived Sea Surface Temperature (SST) anomalies relate to fish migration patterns."
)

# -----------------------------
# DATA LOADING (NOAA ERSST v5 anomaly index)
# -----------------------------
NOAA_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.anom.data"


@st.cache_data(show_spinner=False)
def load_sst_data() -> pd.DataFrame:
    """
    Loads NOAA ERSST v5 global SST anomaly index.
    Format is whitespace-separated rows: YEAR then 12 monthly anomalies.
    Returns a DataFrame with columns: Date (datetime64), SST_Anomaly (float).
    """
    r = requests.get(NOAA_URL, timeout=30)
    r.raise_for_status()

    rows = []
    for line in r.text.splitlines():
        parts = line.strip().split()
        # Expect year + 12 monthly values = 13 tokens
        if len(parts) < 13:
            continue

        try:
            year = int(parts[0])
        except Exception:
            continue

        for month in range(1, 13):
            try:
                anomaly = float(parts[month])
            except Exception:
                continue
            rows.append((datetime(year, month, 1), anomaly))

    df = pd.DataFrame(rows, columns=["Date", "SST_Anomaly"])

    # ✅ Critical fix: force datetime type (prevents ".dt" AttributeError)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Ensure numeric
    df["SST_Anomaly"] = pd.to_numeric(df["SST_Anomaly"], errors="coerce")
    df = df.dropna(subset=["SST_Anomaly"]).reset_index(drop=True)

    return df


# Load data
with st.spinner("Loading NOAA SST anomaly data..."):
    df = load_sst_data()

if df.empty:
    st.error("No data loaded. Please try again later or check the NOAA data source.")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔍 Analysis Settings")

min_year = int(df["Date"].dt.year.min())
max_year = int(df["Date"].dt.year.max())

start_year = st.sidebar.slider("Start Year", min_year, max_year, max(min_year, 1990))
end_year = st.sidebar.slider("End Year", min_year, max_year, max_year)

if start_year > end_year:
    st.sidebar.warning("Start Year is greater than End Year. Swapping them.")
    start_year, end_year = end_year, start_year

use_openai = st.sidebar.toggle("Use AI Interpretation (OpenAI)", value=False)
model_name = st.sidebar.text_input("OpenAI model", value="gpt-5.2", help="Optional. Used only if OpenAI is enabled.")

st.sidebar.divider()
st.sidebar.caption("Tip: If OpenAI is OFF, the app still works using graphs + statistics.")

# Filter
filtered_df = df[(df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)].copy()

if filtered_df.empty:
    st.warning("No data found in the selected year range.")
    st.stop()

# -----------------------------
# SST TIME SERIES PLOT
# -----------------------------
st.subheader("📈 Sea Surface Temperature (SST) Anomalies")

fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["SST_Anomaly"])
ax.set_xlabel("Year")
ax.set_ylabel("SST Anomaly (°C)")
ax.set_title(f"Global SST Anomalies (NOAA ERSST v5) — {start_year} to {end_year}")
st.pyplot(fig)

# -----------------------------
# TREND ANALYSIS
# -----------------------------
st.subheader("📊 Trend Analysis")

# Linear trend using ordinal dates
x = filtered_df["Date"].map(datetime.toordinal).to_numpy()
y = filtered_df["SST_Anomaly"].to_numpy()

if len(x) >= 2:
    slope_per_day, intercept = np.polyfit(x, y, 1)
    slope_per_year = slope_per_day * 365.25

    st.write(f"Estimated warming trend: **{slope_per_year:.4f} °C per year**")
else:
    st.write("Not enough data points to compute a trend.")
    slope_per_year = float("nan")

# Quick stats
st.write(
    f"Average anomaly: **{filtered_df['SST_Anomaly'].mean():.3f} °C** • "
    f"Max: **{filtered_df['SST_Anomaly'].max():.3f} °C** • "
    f"Min: **{filtered_df['SST_Anomaly'].min():.3f} °C**"
)

# -----------------------------
# CORRELATION DEMO (WITH USER-PROVIDED FISH DATA)
# -----------------------------
st.subheader("🐟 Fish Migration / Catch Data (Optional)")

st.markdown(
    "If you have fish catch or migration data (monthly), upload a CSV to compare with SST anomalies.\n\n"
    "**CSV requirements:** a date column + a numeric column (e.g., catch, abundance index, latitude shift)."
)

uploaded = st.file_uploader("Upload fish data CSV", type=["csv"])

fish_df = None
merged = None

if uploaded is not None:
    try:
        fish_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(fish_df.head(10))

        # Let user choose columns
        date_col = st.selectbox("Select the date column", options=fish_df.columns)
        value_col = st.selectbox("Select the numeric fish column", options=fish_df.columns)

        fish_df = fish_df[[date_col, value_col]].copy()
        fish_df.rename(columns={date_col: "Date", value_col: "Fish_Value"}, inplace=True)

        fish_df["Date"] = pd.to_datetime(fish_df["Date"], errors="coerce")
        fish_df["Fish_Value"] = pd.to_numeric(fish_df["Fish_Value"], errors="coerce")
        fish_df = fish_df.dropna(subset=["Date", "Fish_Value"]).sort_values("Date")

        # Make monthly key for merge (YYYY-MM)
        filtered_df["YearMonth"] = filtered_df["Date"].dt.to_period("M").astype(str)
        fish_df["YearMonth"] = fish_df["Date"].dt.to_period("M").astype(str)

        merged = pd.merge(
            filtered_df[["YearMonth", "SST_Anomaly"]],
            fish_df[["YearMonth", "Fish_Value"]],
            on="YearMonth",
            how="inner",
        )

        if merged.empty:
            st.warning("No overlapping months found between SST data and your fish dataset.")
        else:
            st.success(f"Merged dataset has {len(merged)} overlapping months.")
            st.dataframe(merged.head(10))

            corr = merged["SST_Anomaly"].corr(merged["Fish_Value"])
            st.write(f"Correlation (SST anomaly vs Fish value): **{corr:.3f}**")

            st.line_chart(merged.set_index("YearMonth")[["SST_Anomaly", "Fish_Value"]])

    except Exception as e:
        st.error(f"Could not read/parse your CSV. Error: {e}")

# -----------------------------
# SIMPLE SIMULATION (IF NO FISH DATA)
# -----------------------------
st.subheader("🧪 Demonstration Model (If no fish data uploaded)")

st.caption(
    "This is a simple *illustration* showing how migration might shift with warming. "
    "Replace this with real catch/migration data for a true research result."
)

# Simulate poleward shift: e.g., 2° latitude per 1°C anomaly (adjustable)
shift_factor = st.slider("Migration shift factor (degrees latitude per 1°C anomaly)", 0.0, 5.0, 2.0, 0.1)
sim_shift = filtered_df["SST_Anomaly"] * shift_factor

demo_df = pd.DataFrame(
    {
        "Date": filtered_df["Date"],
        "SST_Anomaly (°C)": filtered_df["SST_Anomaly"].values,
        "Simulated Poleward Shift (° lat)": sim_shift.values,
    }
).set_index("Date")

st.line_chart(demo_df)

# -----------------------------
# OPENAI INTERPRETATION (OPTIONAL)
# -----------------------------
def get_openai_key() -> str | None:
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return os.getenv("OPENAI_API_KEY")

if use_openai:
    if not OPENAI_AVAILABLE:
        st.warning("OpenAI package not installed. Add `openai>=1.40.0` to requirements.txt.")
    else:
        key = get_openai_key()
        if not key:
            st.warning("No OPENAI_API_KEY found. Add it in Streamlit Secrets.")
        else:
            client = OpenAI(api_key=key)

            # Build a concise summary for the model (keeps cost low)
            summary_lines = [
                f"Year range: {start_year}–{end_year}",
                f"Mean SST anomaly: {filtered_df['SST_Anomaly'].mean():.3f} °C",
                f"Trend: {slope_per_year:.4f} °C per year" if np.isfinite(slope_per_year) else "Trend: N/A",
                f"Max anomaly: {filtered_df['SST_Anomaly'].max():.3f} °C",
                f"Min anomaly: {filtered_df['SST_Anomaly'].min():.3f} °C",
            ]
            if merged is not None and not merged.empty:
                corr = merged["SST_Anomaly"].corr(merged["Fish_Value"])
                summary_lines.append(f"Fish dataset correlation with SST anomaly: {corr:.3f}")

            prompt = (
                "You are helping a student explain a science fair project about how sea surface temperature "
                "changes relate to fish migration and fisheries. Use simple, clear language.\n\n"
                "Here are the results:\n- " + "\n- ".join(summary_lines) + "\n\n"
                "Explain:\n"
                "1) What the SST anomaly trend means,\n"
                "2) How warming can affect fish migration (timing and location),\n"
                "3) Why satellite data is useful for monitoring,\n"
                "4) What a school-level conclusion could be.\n"
                "Avoid inventing exact fish species or locations unless provided."
            )

            try:
                resp = client.responses.create(
                    model=model_name,
                    input=prompt,
                )
                st.subheader("🤖 AI Interpretation")
                st.write(resp.output_text)
            except Exception as e:
                st.error(f"OpenAI request failed: {e}")

# -----------------------------
# DOWNLOADS
# -----------------------------
st.subheader("⬇️ Download SST Data (Selected Range)")

csv_bytes = filtered_df[["Date", "SST_Anomaly"]].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download SST anomalies CSV",
    data=csv_bytes,
    file_name=f"sst_anomalies_{start_year}_{end_year}.csv",
    mime="text/csv",
)

# -----------------------------
# PROJECT SUMMARY
# -----------------------------
st.subheader("🌍 Project Summary (for your science fair board)")

st.markdown(
    """
**Goal:** Use satellite-derived sea surface temperature (SST) anomaly data to study how ocean warming can influence fish migration.

**Method:**  
- Download monthly global SST anomaly data (NOAA ERSST v5).  
- Visualize changes over time and calculate a warming trend.  
- (Optional) Upload fish migration/catch data to compute correlation with SST anomalies.

**Why it matters:**  
Warmer waters can shift where fish live and breed, affecting fisheries, food supply, and conservation planning.

**Extension ideas:**  
- Use regional SST instead of global (specific ocean area).  
- Compare multiple species’ catch records.  
- Add maps (lat/long) for true migration tracking.
"""
)
