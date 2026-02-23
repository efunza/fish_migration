# fish_Migration.py
# Streamlit Cloud-ready (FIXED): Satellite Ocean Temperature vs Fish Migration Study
#
# Fixes:
# - Prevents ".dt" errors by forcing datetime conversion
# - Prevents crashes from HTTPError by handling download failures gracefully
# - Adds fallback SST/temperature anomaly source if NOAA ERSST link is blocked on Streamlit Cloud
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run fish_Migration.py
#
# Streamlit Cloud:
#   Main file: fish_Migration.py
#   (Optional) Add OPENAI_API_KEY in Secrets

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Tuple

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
    "Visualize temperature anomalies over time and compare them with fish catch/migration data."
)


# -----------------------------
# DATA SOURCES (with fallback)
# -----------------------------
# Primary (sometimes blocked/unstable from Streamlit Cloud): NOAA ERSST v5 anomaly index text file
NOAA_ERSST_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.anom.data"

# Fallback (usually very reliable): DataHub monthly global temperature anomalies (Land+Ocean)
# This is not pure SST, but it's an excellent proxy dataset for trend/correlation demos.
DATAHUB_GLOBAL_TEMP_MONTHLY_CSV = "https://datahub.io/core/global-temp/r/monthly.csv"


def _http_get(url: str) -> requests.Response:
    """HTTP GET with a user-agent (helps some hosts) and timeouts."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Streamlit; EducationalProject) Python requests"
    }
    return requests.get(url, headers=headers, timeout=30)


@st.cache_data(show_spinner=False)
def load_temperature_data() -> Tuple[pd.DataFrame, str]:
    """
    Try NOAA ERSST anomaly index first.
    If it fails (HTTP error / blocked), fall back to DataHub monthly global temp anomalies.

    Returns:
      df: columns [Date (datetime64), Temp_Anomaly (float)]
      source_name: description string
    """
    # -------- Try NOAA ERSST text format --------
    try:
        r = _http_get(NOAA_ERSST_URL)
        if r.status_code == 200 and r.text.strip():
            rows = []
            for line in r.text.splitlines():
                parts = line.strip().split()
                # Expect: year + 12 monthly anomalies (13+ tokens)
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

            df = pd.DataFrame(rows, columns=["Date", "Temp_Anomaly"])

            # Force datetime + numeric (prevents .dt errors)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Temp_Anomaly"] = pd.to_numeric(df["Temp_Anomaly"], errors="coerce")
            df = df.dropna(subset=["Date", "Temp_Anomaly"]).sort_values("Date").reset_index(drop=True)

            if not df.empty:
                return df, "NOAA ERSST v5 (global SST anomaly index)"
    except Exception:
        pass

    # -------- Fallback: DataHub monthly global temp anomalies CSV --------
    r2 = _http_get(DATAHUB_GLOBAL_TEMP_MONTHLY_CSV)
    if r2.status_code != 200:
        raise RuntimeError(
            f"Could not download temperature data. "
            f"NOAA status: unavailable; DataHub status: {r2.status_code}"
        )

    df2 = pd.read_csv(pd.io.common.StringIO(r2.text))

    # DataHub format: Source, Date (YYYY-MM), Mean
    if "Date" not in df2.columns or "Mean" not in df2.columns:
        raise RuntimeError("Fallback dataset format changed (missing Date/Mean columns).")

    df2 = df2.rename(columns={"Mean": "Temp_Anomaly"})
    df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
    df2["Temp_Anomaly"] = pd.to_numeric(df2["Temp_Anomaly"], errors="coerce")
    df2 = df2.dropna(subset=["Date", "Temp_Anomaly"]).sort_values("Date").reset_index(drop=True)

    # Optional: let user pick which source (GCAG vs GISTEMP) later; for now keep both
    # (Still works fine for graphs/correlation)
    return df2, "DataHub Global Temp (Land+Ocean anomalies; proxy fallback)"


# -----------------------------
# LOAD DATA (safe error handling)
# -----------------------------
try:
    with st.spinner("Loading temperature anomaly data..."):
        df, source_name = load_temperature_data()
except Exception as e:
    st.error("Failed to load temperature data from the internet.")
    st.write("Details (safe):", str(e))
    st.info(
        "If you’re on Streamlit Cloud, some NOAA endpoints can be blocked or temporarily down. "
        "Try again later, or use the 'Upload your own SST CSV' option below."
    )
    df = pd.DataFrame(columns=["Date", "Temp_Anomaly"])
    source_name = "None"

if df.empty:
    st.subheader("⬇️ Upload your own SST / temperature anomaly CSV (backup option)")
    st.markdown(
        "Upload a CSV with a **Date** column and a **Temp_Anomaly** column (or choose columns after upload)."
    )
    up = st.file_uploader("Upload temperature CSV", type=["csv"])
    if up is None:
        st.stop()

    user_df = pd.read_csv(up)
    st.dataframe(user_df.head(10))

    date_col = st.selectbox("Select date column", options=user_df.columns)
    val_col = st.selectbox("Select anomaly/value column", options=user_df.columns)

    df = user_df[[date_col, val_col]].rename(columns={date_col: "Date", val_col: "Temp_Anomaly"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Temp_Anomaly"] = pd.to_numeric(df["Temp_Anomaly"], errors="coerce")
    df = df.dropna(subset=["Date", "Temp_Anomaly"]).sort_values("Date").reset_index(drop=True)
    source_name = "User uploaded dataset"

st.success(f"Loaded data source: **{source_name}**")


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("🔍 Analysis Settings")

min_year = int(df["Date"].dt.year.min())
max_year = int(df["Date"].dt.year.max())

start_year = st.sidebar.slider("Start Year", min_year, max_year, max(min_year, 1990))
end_year = st.sidebar.slider("End Year", min_year, max_year, max_year)

if start_year > end_year:
    start_year, end_year = end_year, start_year

use_openai = st.sidebar.toggle("Use AI Interpretation (OpenAI)", value=False)
model_name = st.sidebar.text_input("OpenAI model", value="gpt-5.2")

filtered_df = df[(df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)].copy()
filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)

if filtered_df.empty:
    st.warning("No data found in the selected year range.")
    st.stop()


# -----------------------------
# PLOT TEMPERATURE ANOMALIES
# -----------------------------
st.subheader("📈 Temperature Anomalies Over Time")

fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["Temp_Anomaly"])
ax.set_xlabel("Year")
ax.set_ylabel("Anomaly (°C)")
ax.set_title(f"Temperature Anomalies — {start_year} to {end_year}")
st.pyplot(fig)


# -----------------------------
# TREND ANALYSIS
# -----------------------------
st.subheader("📊 Trend Analysis")

x = filtered_df["Date"].map(datetime.toordinal).to_numpy()
y = filtered_df["Temp_Anomaly"].to_numpy()

if len(x) >= 2:
    slope_per_day, intercept = np.polyfit(x, y, 1)
    slope_per_year = slope_per_day * 365.25
    st.write(f"Estimated trend: **{slope_per_year:.4f} °C per year**")
else:
    slope_per_year = float("nan")
    st.write("Not enough data points to compute trend.")

st.write(
    f"Mean anomaly: **{filtered_df['Temp_Anomaly'].mean():.3f} °C** • "
    f"Max: **{filtered_df['Temp_Anomaly'].max():.3f} °C** • "
    f"Min: **{filtered_df['Temp_Anomaly'].min():.3f} °C**"
)


# -----------------------------
# FISH DATA UPLOAD + CORRELATION
# -----------------------------
st.subheader("🐟 Compare With Fish Catch / Migration Data")

st.markdown(
    "Upload a CSV with a **date column** (monthly is best) and a **fish metric column** "
    "(catch, abundance index, migration latitude, etc.)."
)

fish_upload = st.file_uploader("Upload fish CSV", type=["csv"], key="fish_csv")

merged = None

if fish_upload is not None:
    try:
        fish_df = pd.read_csv(fish_upload)
        st.write("Preview:")
        st.dataframe(fish_df.head(10))

        fish_date_col = st.selectbox("Fish date column", options=fish_df.columns, key="fish_date")
        fish_value_col = st.selectbox("Fish metric column", options=fish_df.columns, key="fish_val")

        fish_df = fish_df[[fish_date_col, fish_value_col]].copy()
        fish_df.rename(columns={fish_date_col: "Date", fish_value_col: "Fish_Value"}, inplace=True)

        fish_df["Date"] = pd.to_datetime(fish_df["Date"], errors="coerce")
        fish_df["Fish_Value"] = pd.to_numeric(fish_df["Fish_Value"], errors="coerce")
        fish_df = fish_df.dropna(subset=["Date", "Fish_Value"]).sort_values("Date")

        # Merge by month (YYYY-MM)
        filtered_df["YearMonth"] = filtered_df["Date"].dt.to_period("M").astype(str)
        fish_df["YearMonth"] = fish_df["Date"].dt.to_period("M").astype(str)

        merged = pd.merge(
            filtered_df[["YearMonth", "Temp_Anomaly"]],
            fish_df[["YearMonth", "Fish_Value"]],
            on="YearMonth",
            how="inner",
        )

        if merged.empty:
            st.warning("No overlapping months found between temperature data and fish data.")
        else:
            st.success(f"Overlapping months: {len(merged)}")
            corr = merged["Temp_Anomaly"].corr(merged["Fish_Value"])
            st.write(f"Correlation (Temp anomaly vs Fish metric): **{corr:.3f}**")
            st.line_chart(merged.set_index("YearMonth")[["Temp_Anomaly", "Fish_Value"]])

    except Exception as e:
        st.error(f"Could not parse fish CSV: {e}")


# -----------------------------
# DEMONSTRATION MODEL (IF NO FISH DATA)
# -----------------------------
st.subheader("🧪 Demonstration: Simulated Poleward Shift (if no fish data)")

st.caption(
    "This is a simple illustration: warming → poleward shift. Replace with real fish data for real conclusions."
)

shift_factor = st.slider("Shift factor (degrees latitude per 1°C anomaly)", 0.0, 5.0, 2.0, 0.1)
sim_shift = filtered_df["Temp_Anomaly"] * shift_factor

demo = pd.DataFrame(
    {
        "Date": filtered_df["Date"],
        "Temp_Anomaly (°C)": filtered_df["Temp_Anomaly"].values,
        "Simulated Shift (° lat)": sim_shift.values,
    }
).set_index("Date")

st.line_chart(demo)


# -----------------------------
# OPENAI INTERPRETATION (OPTIONAL)
# -----------------------------
def get_openai_key() -> Optional[str]:
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
            st.warning("No OPENAI_API_KEY found. Add it in Streamlit Secrets to enable AI interpretation.")
        else:
            client = OpenAI(api_key=key)

            summary = [
                f"Data source: {source_name}",
                f"Year range: {start_year}–{end_year}",
                f"Mean anomaly: {filtered_df['Temp_Anomaly'].mean():.3f} °C",
                f"Trend: {slope_per_year:.4f} °C/year" if np.isfinite(slope_per_year) else "Trend: N/A",
                f"Max anomaly: {filtered_df['Temp_Anomaly'].max():.3f} °C",
                f"Min anomaly: {filtered_df['Temp_Anomaly'].min():.3f} °C",
            ]
            if merged is not None and not merged.empty:
                corr = merged["Temp_Anomaly"].corr(merged["Fish_Value"])
                summary.append(f"Fish correlation: {corr:.3f}")

            prompt = (
                "Explain a science fair project about how ocean temperature anomalies relate to fish migration. "
                "Use simple, clear language.\n\nResults:\n- " + "\n- ".join(summary) + "\n\n"
                "Explain:\n"
                "1) What the trend means,\n"
                "2) How warming affects fish migration (timing + location),\n"
                "3) Why satellite/large-scale datasets help,\n"
                "4) A short conclusion and limitations.\n"
                "Do not invent species or local catch details unless provided."
            )

            try:
                resp = client.responses.create(model=model_name, input=prompt)
                st.subheader("🤖 AI Interpretation")
                st.write(resp.output_text)
            except Exception as e:
                st.error(f"OpenAI request failed: {e}")


# -----------------------------
# DOWNLOAD
# -----------------------------
st.subheader("⬇️ Download Temperature Data (Selected Range)")

out = filtered_df[["Date", "Temp_Anomaly"]].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=out,
    file_name=f"temperature_anomalies_{start_year}_{end_year}.csv",
    mime="text/csv",
)


# -----------------------------
# PROJECT SUMMARY
# -----------------------------
st.subheader("🌍 Project Summary (for your poster)")

st.markdown(
    """
**Goal:** Study how changes in ocean temperature can influence fish migration patterns.

**Method:**
- Load a public temperature anomaly dataset (preferably SST anomalies; fallback uses a global land+ocean anomaly proxy if needed).
- Plot anomalies over time and compute a warming trend.
- Upload fish catch/migration data and test correlation with temperature anomalies (monthly).

**Why it matters:**
Warming oceans can shift where fish live and breed, affecting fisheries, food security, and conservation planning.

**Extensions:**
- Use regional SST (Indian Ocean / Kenya coast) instead of global.
- Add maps for spatial migration.
- Compare multiple fish species datasets.
"""
)
