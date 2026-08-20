import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Tony Dataset - Live Milling AI Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric {
        background: linear-gradient(135deg, #1e2638 0%, #151b28 100%);
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 14px; color: #a0aec0; }
    .metric-value { font-size: 28px; font-weight: 700; color: #63b3ed; }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13px;
    }
    .status-live { background-color: #22543d; color: #9ae6b4; border: 1px solid #38a169; }
</style>
""", unsafe_allow_html=True)

MASTER_CSV = r"D:\tony dataset\all_datasets_features_12_master.csv"
TEMP_DIR = r"D:\tony dataset\temp_scratch_extract"

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("⚡ Milling Digital Twin — Live Extraction Monitor")
    st.caption("Live monitoring of physical cut extraction, stability boundaries, and multi-sensor features across Schmitz Datasets 1–155.")
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="status-badge status-live">🟢 PIPELINE ACTIVE & SYNCING</span>', unsafe_allow_html=True)

# Auto-refresh control in sidebar
st.sidebar.header("⚙️ Monitor Settings")
refresh_rate = st.sidebar.slider("Auto-refresh interval (seconds)", min_value=2, max_value=30, value=5)
auto_refresh = st.sidebar.checkbox("Enable Live Auto-Refresh", value=True)

if not os.path.exists(MASTER_CSV):
    st.error(f"Master CSV file not found at `{MASTER_CSV}`. Waiting for pipeline...")
    st.stop()

# Load Data
try:
    df = pd.read_csv(MASTER_CSV)
except Exception as e:
    st.warning(f"Reading file in progress... {e}")
    time.sleep(1)
    df = pd.read_csv(MASTER_CSV)

total_cuts = len(df)
unique_datasets = df["dataset_id"].nunique()
stable_count = (df["label"] == 0).sum() if "label" in df.columns else 0
chatter_count = (df["label"] == 1).sum() if "label" in df.columns else 0
stable_pct = (stable_count / total_cuts * 100) if total_cuts > 0 else 0

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Cuts Extracted", f"{total_cuts:,}", delta=f"{unique_datasets} Datasets")
m2.metric("Unique Datasets Active", f"{unique_datasets} / 155", delta=f"{round(unique_datasets/155*100, 1)}% Target")
m3.metric("Stable Machining Cuts", f"{stable_count:,}", delta=f"{stable_pct:.1f}% of total")
m4.metric("Chatter (Unstable) Cuts", f"{chatter_count:,}", delta=f"{100-stable_pct:.1f}% of total", delta_color="inverse")

st.markdown("---")

# Layout: 2 Columns for Charts
c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("📊 Cuts Extracted per Dataset ID")
    ds_counts = df["dataset_id"].value_counts().reset_index()
    ds_counts.columns = ["Dataset ID", "Cuts Count"]
    ds_counts = ds_counts.sort_values("Dataset ID")
    
    fig_bar = px.bar(
        ds_counts,
        x="Dataset ID",
        y="Cuts Count",
        color="Cuts Count",
        color_continuous_scale="Blues",
        text="Cuts Count",
        title="Distribution of Cuts across Ingested Datasets"
    )
    fig_bar.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(type='category')
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("🎯 Stability Phase Space (RPM vs Depth)")
    # Sample 1500 points for snappy rendering if dataset is huge
    sample_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df
    sample_df["Stability Status"] = sample_df["label"].map({0: "Stable (0)", 1: "Chatter (1)"})
    
    fig_scatter = px.scatter(
        sample_df,
        x="omega_rpm",
        y="axial_depth_m",
        color="Stability Status",
        color_discrete_map={"Stable (0)": "#48bb78", "Chatter (1)": "#f56565"},
        opacity=0.6,
        hover_data=["dataset_id", "kurtosis_1st_derivative"],
        labels={"omega_rpm": "Spindle Speed (RPM)", "axial_depth_m": "Axial Depth (m)"},
        title="Live Operating Regime (Stable vs Chatter)"
    )
    fig_scatter.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Live Table: Latest 15 Cuts Extracted
st.subheader("⚡ Live Feed: Most Recent Cuts Extracted")
latest_cuts = df.tail(15).iloc[::-1]

def color_label(val):
    color = '#22543d' if val == 0 else '#742a2a'
    text = '#9ae6b4' if val == 0 else '#feb2b2'
    label_str = "STABLE" if val == 0 else "CHATTER"
    return f'<span style="background-color: {color}; color: {text}; padding: 3px 8px; border-radius: 4px; font-weight: bold;">{label_str} ({val})</span>'

cols_to_show = ["dataset_id", "file", "omega_rpm", "axial_depth_m", "kurtosis_1st_derivative", "d2_energy", "label"]
available_cols = [c for c in cols_to_show if c in latest_cuts.columns]

st.dataframe(
    latest_cuts[available_cols].style.format({
        "omega_rpm": "{:.0f} RPM",
        "axial_depth_m": "{:.4f} m",
        "kurtosis_1st_derivative": "{:.2f}",
        "d2_energy": "{:.2f}"
    }),
    use_container_width=True,
    height=400
)

# Auto-rerun timer if enabled
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
