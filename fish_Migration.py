# fish_Migration.py
# Streamlit Cloud-ready (MORE ROBUST): Satellite Ocean Temperature vs Fish Migration Study
#
# Fixes included:
# ✅ Forces Date column to datetime (prevents ".dt" accessor errors)
# ✅ Never crashes on HTTP errors (handles blocked/down sources gracefully)
# ✅ Uses multiple fallback URLs (NOAA -> GitHub raw -> DataHub)
# ✅ Parses CSV even if column names change (case/whitespace/BOM tolerant)
#
# Local run:
#   pip install -r requirements.txt
#   streamlit run fish_Migration.py

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
st.caption("Plot temperature anomalies over time and compare them with fish catch/migration data (monthly).")


# -----------------------------
# DATA SOURCES
# -----------------------------
# Primary (true SST anomalies, global index)
NOAA_ERSST_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.anom.data"

# Fallback 1 (very reliable): GitHub raw from datasets/global-temp (Land+Ocean anomalies)
GITHUB_GLOBAL_TEMP_MONTHLY = "https://raw.githubusercontent.com/datasets/global-temp/main/data/monthly.csv"

# Fallback 2: DataHub (may redirect/behave oddly on Streamlit Cloud; kept as last resort)
DATAHUB_GLOBAL_TEMP_MONTHLY = "https://datahub.io/core/global-temp/r/monthly.csv"


def http_get_text(url: str, timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, error). Never raises."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Streamlit Educational App)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        text = r.text
        if not text or not text.strip():
            return None, "Empty response"
        return text, None
    except Exception as e:
        return None, str(e)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and BOM characters from column names."""
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find a column in df matching any candidate (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def parse_noaa_ersst(text: str) -> pd.DataFrame:
    """
    Parse NOAA ERSST v5 anomaly index text:
    rows: YEAR then 12 monthly anomalies.
    Output: Date, Temp_Anomaly
    """
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
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
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Temp_Anomaly"] = pd.to_numeric(df["Temp_Anomaly"], errors="coerce")
    df = df.dropna(subset=["Date", "Temp_Anomaly"]).sort_values("Date").reset_index(drop=True)
    return df


def parse_monthly_csv(text: str) -> pd.DataFrame:
    """
    Parse a monthly anomaly CSV. Tries to auto-detect columns even if names change.
    Expected typical columns: Source, Date, Mean
    Output: Date, Temp_Anomaly
    """
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    df = clean_columns(df)

    # Try common schema first
    date_col = find_col(df, ["Date", "date", "Month", "month", "time", "Time"])
    mean_col = find_col(df, ["Mean", "mean", "Anomaly", "anomaly", "Value", "value", "Temp_Anomaly", "temp_anomaly"])

    # Some datasets may use "Year" + "Month" columns
    year_col = find_col(df, ["Year", "year", "YYYY"])
    mon_col = find_col(df, ["Month", "month", "MM"])

    if date_col and mean_col:
        out = df[[date_col, mean_col]].copy()
        out.columns = ["Date", "Temp_Anomaly"]
    elif year_col and mon_col and mean_col:
        out = df[[year_col, mon_col, mean_col]].copy()
        out["Date"] = pd.to_datetime(
            out[year_col].astype(str) + "-" + out[mon_col].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        out = out.rename(columns={mean_col: "Temp_Anomaly"})[["Date", "Temp_Anomaly"]]
    else:
        # Last-resort: try first date-like column + first numeric column
        possible_date_cols = []
        for c in df.columns:
            try_dates = pd.to_datetime(df[c], errors="coerce")
            if try_dates.notna().mean() > 0.8:  # mostly parseable
                possible_date_cols.append(c)

        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]

        if possible_date_cols and numeric_cols:
            out = df[[possible_date_cols[0], numeric_cols[0]]].copy()
            out.columns = ["Date", "Temp_Anomaly"]
        else:
            raise ValueError(f"Could not detect Date/Value columns. Columns found: {list(df.columns)}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Temp_Anomaly"] = pd.to_numeric(out["Temp_Anomaly"], errors="coerce")
    out = out.dropna(subset=["Date", "Temp_Anomaly"]).sort_values("Date").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def load_temperature_data() -> Tuple[pd.DataFrame, str, list[str]]:
    """
    Returns: (df, source_name, notes)
    df columns: Date (datetime64), Temp_Anomaly (float)
    """
    notes: list[str] = []

    # 1) NOAA ERSST (SST anomalies)
    text, err = http_get_text(NOAA_ERSST_URL)
    if text:
        try:
            df = parse_noaa_ersst(text)
            if not df.empty:
                return df, "NOAA ERSST v5 (global SST anomaly index)", notes
            notes.append("NOAA ERSST parsed but produced empty dataframe.")
        except Exception as e:
            notes.append(f"NOAA parse failed: {e}")
    else:
        notes.append(f"NOAA ERSST download failed: {err}")

    # 2) GitHub raw global-temp monthly (Land+Ocean anomalies; proxy fallback)
    text, err = http_get_text(GITHUB_GLOBAL_TEMP_MONTHLY)
    if text:
        try:
            df = parse_monthly_csv(text)
            if not df.empty:
                notes.append("Using Land+Ocean anomaly dataset as fallback (not pure SST).")
                return df, "GitHub datasets/global-temp monthly (Land+Ocean anomalies; fallback)", notes
            notes.append("GitHub fallback parsed but produced empty dataframe.")
        except Exception as e:
            notes.append(f"GitHub fallback parse failed: {e}")
    else:
        notes.append(f"GitHub fallback download failed: {err}")

    # 3) DataHub monthly.csv (last resort)
    text, err = http_get_text(DATAHUB_GLOBAL_TEMP_MONTHLY)
    if text:
        try:
            df = parse_monthly_csv(text)
            if not df.empty:
                notes.append("Using DataHub fallback (Land+Ocean anomalies; not pure SST).")
                return df, "DataHub global-temp monthly (fallback)", notes
            notes.append("DataHub fallback parsed but produced empty dataframe.")
        except Exception as e:
            notes.append(f"DataHub fallback parse failed: {e}")
    else:
        notes.append(f"DataHub fallback download failed: {err}")

    return pd.DataFrame(columns=["Date", "Temp_Anomaly"]), "None", notes


# -----------------------------
# LOAD TEMPERATURE DATA
# -----------------------------
with st.spinner("Loading temperature anomaly data (with fallbacks)..."):
    df, source_name, notes = load_temperature_data()

if notes:
    with st.expander("Data loading notes (click to view)"):
        for n in notes:
            st.write("•", n)

if df.empty:
    st.error("Could not load any temperature anomaly dataset from the internet.")
    st.info("Upload your own SST/temperature CSV below to continue (Date + anomaly/value column).")

    up = st.file_uploader("Upload temperature CSV", type=["csv"])
    if up is None:
        st.stop()

    user_df = pd.read_csv(up)
    user_df = clean_columns(user_df)
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
model_name = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")

filtered_df = df[(df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)].copy()
filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)

if filtered_df.empty:
    st.warning("No data found in the selected year range.")
    st.stop()


# -----------------------------
# PLOT ANOMALIES
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
    slope_per_day, _ = np.polyfit(x, y, 1)
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
        fish_df = clean_columns(fish_df)

        st.write("Preview of fish data:")
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

st.caption("Illustration only: warming → poleward shift. Replace with real fish data for real conclusions.")

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
- Load a public temperature anomaly dataset (preferably SST anomalies).
- Plot anomalies over time and compute a warming trend.
- Upload fish catch/migration data and test correlation with temperature anomalies (monthly).

**Why it matters:**
Warming oceans can shift where fish live and breed, affecting fisheries, food security, and conservation planning.

**Extensions:**
- Use *regional* SST (Kenya coast / Indian Ocean) instead of global.
- Add maps for spatial migration.
- Compare multiple fish species datasets.
"""
)

