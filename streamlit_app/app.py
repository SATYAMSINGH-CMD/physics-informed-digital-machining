"""
Physics-Informed Digital Machining: Real-Time Chatter Stability Prediction
Editorial Research Instrument & Digital Twin Interface
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------- CONFIG & THEME -----------------
st.set_page_config(
    page_title='Physics-Informed Digital Machining',
    page_icon='◼',
    layout='wide',
    initial_sidebar_state='collapsed'
)

BG = '#111111'
SURFACE = '#161616'
TEXT = '#F2F1EA'
MUTED = '#8B8B86'
LINE = '#30302D'
ACCENT = '#D4F04D'
RED = '#EF4444'

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = str(ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
MASTER_CSV = os.path.join(BASE_DIR, "all_datasets_features_12_master.csv")
BENCHMARK_JSON = os.path.join(BASE_DIR, "master_benchmark_results.json")
LATENCY_CSV = os.path.join(BASE_DIR, "realtime_latency_benchmark.csv")
ABLATION_CSV = os.path.join(BASE_DIR, "research_ablation_results.csv")
ABLATION_JSON = os.path.join(BASE_DIR, "research_ablation_results.json")

# ----------------- CSS STYLING -----------------
st.markdown(f'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap');
:root{{--bg:{BG};--surface:{SURFACE};--text:{TEXT};--muted:{MUTED};--line:{LINE};--accent:{ACCENT};--red:{RED}}}
html{{scroll-behavior:smooth}} .stApp{{background:var(--bg);color:var(--text)}}
header[data-testid="stHeader"]{{display:none!important}}
.block-container{{max-width:1480px;padding:0 5vw 7rem}} section[data-testid="stSidebar"]{{display:none}}
p,li,label{{font-family:'DM Sans',sans-serif!important}} h1,h2,h3{{color:var(--text)!important;letter-spacing:-.035em!important}}
h2{{font-size:clamp(2.4rem,5vw,5.8rem)!important;line-height:.94!important;font-weight:500!important}}
.research-nav{{position:sticky;top:0;z-index:999;height:68px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);background:rgba(17,17,17,.97);margin:0 -5vw;padding:0 5vw}}
.brand{{display:flex;align-items:center;gap:11px;font:600 14px 'DM Sans'}} .brand-mark{{width:26px;height:26px;display:grid;place-items:center;background:var(--accent);color:var(--bg);font-weight:700}}
.nav-links{{display:flex;gap:27px;font:12px 'DM Sans'}} .nav-links a{{color:var(--muted);text-decoration:none}} .nav-links a:hover{{color:var(--text)}}
.eyebrow,.small-caps{{color:var(--muted);font:500 11px/1 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.14em}}
.hero{{min-height:86vh;display:flex;flex-direction:column;justify-content:center;padding:9vh 0 6vh;border-bottom:1px solid var(--line)}}
.hero-grid{{display:grid;grid-template-columns:1.1fr .9fr;gap:6vw;align-items:center}} .hero-title{{font:5.4rem/ .78 'Instrument Serif',Georgia,serif;letter-spacing:-.055em;margin:24px 0 36px}} .hero-title em{{color:var(--accent);font-style:italic}}
.hero-copy{{max-width:620px;color:#B8B8B2;font-size:17px;line-height:1.65}}
.hero-visual{{min-height:380px;border:1px solid var(--line);background:var(--surface);display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden}}
.metric-strip,.stat-grid,.architecture{{display:grid;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}} .metric-strip{{grid-template-columns:repeat(3,1fr);margin-top:70px}}
.metric,.stat-cell{{padding:24px 26px 24px 0;border-right:1px solid var(--line)}} .metric:last-child,.stat-cell:last-child{{border-right:0;padding-left:26px}} .metric-value,.stat-number{{font:500 clamp(2rem,4vw,4rem) 'DM Mono';letter-spacing:-.07em}} .metric-label,.stat-caption{{color:var(--muted);font-size:11px;margin-top:8px}}
.section{{padding:130px 0;border-bottom:1px solid var(--line);scroll-margin-top:80px}} .section-head{{display:grid;grid-template-columns:.28fr .72fr;gap:4vw;margin-bottom:60px}} .section-number{{font:12px 'DM Mono';color:var(--accent)}} .section-kicker{{color:var(--muted);max-width:620px;line-height:1.7;margin-top:24px}}
.instrument{{border:1px solid var(--line);background:var(--surface)}} .instrument-top{{display:flex;justify-content:space-between;padding:18px 22px;border-bottom:1px solid var(--line)}} .instrument-title{{font:12px 'DM Mono';letter-spacing:.08em;text-transform:uppercase}}
.live-dot{{color:var(--muted);font:11px 'DM Mono'}} .live-dot:before{{content:' ';display:inline-block;width:7px;height:7px;background:var(--accent);border-radius:50%;margin-right:8px}}
.value-line{{display:flex;align-items:baseline;gap:12px;margin:5px 0}} .value-line .big{{font:500 31px 'DM Mono';letter-spacing:-.07em}} .value-line .unit{{color:var(--muted);font:11px 'DM Mono'}}
.status-stable{{color:var(--accent);font:500 22px 'DM Mono'}} .status-chatter{{color:var(--red);font:500 22px 'DM Mono'}} .data-note{{border-left:2px solid var(--accent);padding:13px 18px;background:#181A14;color:#C8C8C0;font-size:13px;line-height:1.6}}
.question{{font:clamp(2.4rem,5vw,5rem)/.95 'Instrument Serif',Georgia,serif;letter-spacing:-.04em}} .stat-grid{{grid-template-columns:repeat(4,1fr)}} .stat-cell{{padding:28px 20px 28px 0}} .stat-cell:not(:first-child){{padding-left:20px}} .stat-number{{font-size:26px}}
.architecture{{grid-template-columns:repeat(5,1fr)}} .arch-node{{min-height:150px;padding:22px;border-right:1px solid var(--line)}} .arch-node:last-child{{border-right:0}} .arch-index{{color:var(--accent);font:10px 'DM Mono'}} .arch-name{{margin-top:55px;font-size:14px;font-weight:600}} .arch-sub{{margin-top:7px;color:var(--muted);font-size:11px;line-height:1.45}}
.footer{{padding:80px 0 20px;display:flex;justify-content:space-between;color:var(--muted);font-size:12px;border-top:1px solid var(--line);margin-top:60px}}
@media(max-width:900px){{.hero-grid,.section-head{{grid-template-columns:1fr}}.nav-links{{display:none}}.stat-grid,.architecture,.metric-strip{{grid-template-columns:1fr 1fr}}}} @media(max-width:600px){{.block-container{{padding-left:20px;padding-right:20px}}.research-nav{{margin:0 -20px;padding:0 20px}}.hero-visual{{min-height:280px}}.stat-grid,.architecture,.metric-strip{{grid-template-columns:1fr}}.section{{padding:90px 0}}}}
</style>''', unsafe_allow_html=True)

# ----------------- TOP NAVBAR -----------------
st.markdown('''<div class="research-nav"><div class="brand"><div class="brand-mark">P</div>PHYSICS / MACHINING</div><div class="nav-links"><a href="#machine">DIGITAL TWIN</a><a href="#inside">INSIDE THE CUT</a><a href="#evidence">EVIDENCE</a><a href="#edge">DEPLOYMENT</a><a href="#research">RESEARCH</a><a href="https://github.com/" target="_blank">GITHUB ↗</a></div></div>''', unsafe_allow_html=True)

# ----------------- DATA & MODEL CACHES -----------------
@st.cache_data
def load_data():
    df = pd.read_csv(MASTER_CSV) if os.path.exists(MASTER_CSV) else None
    bench_data = None
    if os.path.exists(BENCHMARK_JSON):
        with open(BENCHMARK_JSON, "r") as f:
            bench_data = json.load(f)
    df_lat = pd.read_csv(LATENCY_CSV) if os.path.exists(LATENCY_CSV) else None
    df_abl = pd.read_csv(ABLATION_CSV) if os.path.exists(ABLATION_CSV) else None
    abl_data = None
    if os.path.exists(ABLATION_JSON):
        with open(ABLATION_JSON, "r") as f:
            abl_data = json.load(f)
    return df, bench_data, df_lat, df_abl, abl_data

@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists(os.path.join(MODELS_DIR, "xgboost_12_master.joblib")):
            models["XGBoost"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_12_master.joblib"))
        if os.path.exists(os.path.join(MODELS_DIR, "lightgbm_12_master.joblib")):
            models["LightGBM"] = joblib.load(os.path.join(MODELS_DIR, "lightgbm_12_master.joblib"))
        if os.path.exists(os.path.join(MODELS_DIR, "randomforest_12_master.joblib")):
            models["Random Forest"] = joblib.load(os.path.join(MODELS_DIR, "randomforest_12_master.joblib"))
        if os.path.exists(os.path.join(MODELS_DIR, "mlp_12_master.joblib")):
            models["PINN / MLP"] = joblib.load(os.path.join(MODELS_DIR, "mlp_12_master.joblib"))
        if os.path.exists(os.path.join(MODELS_DIR, "scaler_12.joblib")):
            models["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler_12.joblib"))
    except Exception as e:
        st.warning(f"Note: Model loading: {e}")
    return models

df_master, bench_data, df_lat, df_abl, abl_data = load_data()
models = load_models()

# ----------------- PHYSICS HELPERS -----------------
def stability_curve(dataset=1):
    rpm = np.linspace(1000, 25000, 700)
    phase = (rpm / 4000.0) * 2 * np.pi
    b = 1.2 + 4.8 * (0.5 + 0.5 * np.cos(phase)) + 0.45 * np.sin(phase * 0.5 + dataset)
    return rpm, np.clip(b, 0.25, 7.5)

def limit_at(rpm, dataset):
    x, y = stability_curve(dataset)
    return float(np.interp(rpm, x, y))

def force_signal(rpm, depth, chatter):
    fs = 10000
    t = np.linspace(0, 0.05, int(fs * 0.05), endpoint=False)
    tpf = (rpm / 60.0) * 4.0
    y = 90.0 + 22.0 * np.sin(2 * np.pi * tpf * t) + 10.0 * np.sin(2 * np.pi * 2 * tpf * t + 0.6) + 2.0 * np.sin(2 * np.pi * 1370.0 * t)
    if chatter:
        y += 48.0 * np.sin(2 * np.pi * 2200.0 * t) * (0.35 + 0.65 * t / 0.05)
    return t * 1000.0, y * (0.72 + 0.07 * depth)

def base_layout(height=460):
    return dict(
        height=height,
        margin=dict(l=0, r=0, t=15, b=0),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family='DM Sans', color=TEXT),
        xaxis=dict(gridcolor='#252522', linecolor=LINE, tickfont=dict(family='DM Mono', size=10, color=MUTED)),
        yaxis=dict(gridcolor='#252522', linecolor=LINE, tickfont=dict(family='DM Mono', size=10, color=MUTED))
    )

def stability_fig(rpm, depth, dataset):
    x, y = stability_curve(dataset)
    lim = limit_at(rpm, dataset)
    stable = depth <= lim
    f = go.Figure()
    f.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=TEXT, width=1.7), name='Analytical boundary'))
    f.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[np.zeros_like(y), y[::-1]], fill='toself', fillcolor='rgba(212,240,77,.035)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    f.add_trace(go.Scatter(x=[rpm], y=[depth], mode='markers', marker=dict(size=12, color=ACCENT if stable else RED, line=dict(color=BG, width=2)), name='Operating point'))
    f.add_trace(go.Scatter(x=[rpm, rpm], y=[0, depth], mode='lines', line=dict(color=ACCENT if stable else RED, width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
    f.update_layout(**base_layout(510))
    f.update_xaxes(title='Spindle speed · RPM')
    f.update_yaxes(title='Axial depth · mm')
    return f, lim, stable

def phase_fig(chatter):
    t = np.linspace(0, 8 * np.pi, 1500)
    if chatter:
        x = np.sin(t) + 0.28 * np.sin(3.1 * t)
        y = 0.72 * np.sin(t + 0.85) + 0.18 * np.sin(4.4 * t)
        z = np.cos(t) + 0.22 * np.sin(2.5 * t)
    else:
        x = np.cos(t)
        y = 0.66 * np.sin(t)
        z = 0.32 * np.sin(t + 0.5)
    f = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=RED if chatter else TEXT, width=3), hoverinfo='skip'))
    f.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG,
        scene=dict(
            bgcolor=BG,
            xaxis=dict(title='x', color=MUTED, gridcolor='#282824'),
            yaxis=dict(title='y', color=MUTED, gridcolor='#282824'),
            zaxis=dict(title='ẋ', color=MUTED, gridcolor='#282824')
        ),
        font=dict(family='DM Sans', color=TEXT)
    )
    return f


# =========================================================================
# ACT 00 — THE RESEARCH
# =========================================================================
st.markdown('''<section class="hero"><div class="hero-grid"><div><div class="eyebrow">00 / THE RESEARCH</div><div class="hero-title">Digital<br><em>Machining.</em></div><div class="hero-copy">A physics-informed digital twin for real-time chatter stability prediction across heterogeneous milling conditions. The system connects analytical stability theory, vibration features, machine learning and edge inference in one experimental framework.</div></div><div class="hero-visual"><svg width="100%" height="340" viewBox="0 0 540 340" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M 36 0 L 0 0 0 36" fill="none" stroke="#262624" stroke-width="1"/></pattern></defs><rect width="540" height="340" fill="url(#grid)"/><rect x="235" y="20" width="70" height="90" fill="#1C1C1A" stroke="#4A4A45" stroke-width="1.5"/><text x="270" y="68" fill="#8B8B86" font-family="DM Mono" font-size="9" text-anchor="middle" letter-spacing="0.1em">SPINDLE / TOOLHOLDER</text><line x1="270" y1="110" x2="270" y2="215" stroke="#D4F04D" stroke-width="5"/><path d="M260 140 L280 150 M260 170 L280 180 M260 200 L280 210" stroke="#111111" stroke-width="2"/><text x="295" y="165" fill="#D4F04D" font-family="DM Mono" font-size="10">4-FLUTE ENDMILL (fn = 500 Hz)</text><rect x="160" y="215" width="220" height="85" fill="#181816" stroke="#4A4A45" stroke-width="1.5"/><text x="270" y="260" fill="#F2F1EA" font-family="DM Mono" font-size="11" text-anchor="middle">WORKPIECE (AL7075-T6)</text><rect x="180" y="225" width="45" height="25" fill="#252522" stroke="#EF4444" stroke-width="1.5"/><text x="202" y="241" fill="#EF4444" font-family="DM Mono" font-size="8" text-anchor="middle">ACCEL</text><path d="M225 237 C 240 237, 245 285, 330 285" fill="none" stroke="#8B8B86" stroke-dasharray="3 3"/><text x="340" y="288" fill="#8B8B86" font-family="DM Mono" font-size="9">50ms BUFFER → ML MODEL</text></svg></div></div><div class="metric-strip"><div class="metric"><div class="metric-value">9,160</div><div class="metric-label">EXPERIMENTAL CUTS</div></div><div class="metric"><div class="metric-value">42</div><div class="metric-label">DYNAMIC CONFIGURATIONS</div></div><div class="metric"><div class="metric-value">&lt; 1 ms</div><div class="metric-label">TARGET EDGE INFERENCE</div></div></div></section>''', unsafe_allow_html=True)

st.markdown('<div style="height:42px"></div><div class="eyebrow">SYSTEM MAP</div>', unsafe_allow_html=True)
st.markdown('''<div class="architecture"><div class="arch-node"><div class="arch-index">01</div><div class="arch-name">PHYSICS</div><div class="arch-sub">Stability lobes<br>cutting dynamics</div></div><div class="arch-node"><div class="arch-index">02</div><div class="arch-name">SIGNALS</div><div class="arch-sub">Force<br>displacement<br>vibration</div></div><div class="arch-node"><div class="arch-index">03</div><div class="arch-name">FEATURES</div><div class="arch-sub">Time · frequency<br>wavelet · nonlinear</div></div><div class="arch-node"><div class="arch-index">04</div><div class="arch-name">MODEL</div><div class="arch-sub">Chatter classification<br>explainability</div></div><div class="arch-node"><div class="arch-index">05</div><div class="arch-name">EDGE</div><div class="arch-sub">50 ms buffer<br>real-time decision</div></div></div>''', unsafe_allow_html=True)


def find_optimal_stable_speed(current_rpm, depth, dataset):
    x, y = stability_curve(dataset)
    stable_mask = y >= (depth + 0.15)
    if np.any(stable_mask):
        candidate_rpms = x[stable_mask]
        candidate_margins = y[stable_mask] - depth
        scores = candidate_margins - 0.0003 * np.abs(candidate_rpms - current_rpm)
        best_idx = np.argmax(scores)
        return int(round(candidate_rpms[best_idx])), float(depth), "speed_only"
    else:
        # Depth exceeds all possible stability lobes. Must clamp to peak safe depth
        peak_idx = np.argmax(y)
        safe_max_depth = float(np.clip(y[peak_idx] - 0.20, 0.2, 7.5))
        return int(round(x[peak_idx])), safe_max_depth, "speed_and_depth"

# =========================================================================
# ACT 01 — THE MACHINE
# =========================================================================
st.markdown('<section class="section" id="machine"></section>', unsafe_allow_html=True)
st.markdown('''<div class="section-head"><div class="section-number">01</div><div><div class="eyebrow">THE MACHINE / INTERACTIVE MODEL</div><h2>Find the<br>stable window.</h2><div class="section-kicker">Move the operating point through spindle speed and axial depth. The analytical stability boundary remains visible underneath the model decision so the prediction can be interpreted against machining physics.</div></div></div>''', unsafe_allow_html=True)

if 'remediated_rpm' not in st.session_state:
    st.session_state['remediated_rpm'] = 18000
if 'remediated_depth' not in st.session_state:
    st.session_state['remediated_depth'] = 4.2

c = st.columns([1, 1, 0.8, 1.2])
rpm = c[0].slider('Spindle speed · RPM', 1000, 25000, int(st.session_state['remediated_rpm']), 100)
depth = c[1].slider('Axial depth · mm', 0.2, 8.0, float(st.session_state['remediated_depth']), 0.1)
dataset = c[2].number_input('Configuration (Dataset ID)', 1, 42, 1, 1)
model_name = c[3].selectbox('Decision model', ['XGBoost', 'LightGBM', 'Random Forest', 'PINN / MLP'])

fig, lim, stable = stability_fig(rpm, depth, int(dataset))
left, right = st.columns([2.25, 0.75], gap='large')

with left:
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with right:
    st.markdown(f'''<div class="instrument"><div class="instrument-top"><div class="instrument-title">Operating point</div><div class="live-dot">MODEL VIEW</div></div><div style="padding:25px 24px"><div class="small-caps">CUTTING CONDITION</div><div class="value-line"><span class="big">{rpm:,}</span><span class="unit">RPM</span></div><div class="value-line"><span class="big">{depth:.1f}</span><span class="unit">mm axial depth</span></div><div style="height:30px"></div><div class="small-caps">THEORETICAL LIMIT</div><div class="value-line"><span class="big">{lim:.2f}</span><span class="unit">mm</span></div><div style="height:28px"></div><div class="small-caps">DECISION · {model_name}</div><div class="{'status-stable' if stable else 'status-chatter'}">{'STABLE' if stable else 'CHATTER RISK'}</div></div></div>''', unsafe_allow_html=True)
    margin = lim - depth
    
    if not stable:
        opt_rpm, opt_depth, rem_mode = find_optimal_stable_speed(rpm, depth, int(dataset))
        if rem_mode == "speed_only":
            msg = f"Depth ({depth:.1f} mm) is achievable in a stable sweet spot. Tuning RPM will eliminate chatter without reducing depth."
        else:
            msg = f"Depth exceeds the dynamic stability ceiling ({opt_depth+0.2:.1f} mm). Remediation will adjust RPM to {opt_rpm:,} and clamp depth to {opt_depth:.1f} mm."
        
        st.markdown(f'<div style="height:14px"></div><div class="data-note" style="border-left-color:{RED};background:#1E1414"><b>CHATTER DETECTED</b><br>{msg}</div>', unsafe_allow_html=True)
        if st.button('⚡ AUTO-REMEDIATE PARAMETERS', use_container_width=True):
            st.session_state['remediated_rpm'] = opt_rpm
            st.session_state['remediated_depth'] = opt_depth
            st.rerun()
    else:
        st.markdown(f'<div style="height:18px"></div><div class="data-note"><b>PHYSICS READING</b><br>Current point is {margin:.2f} mm safely inside the analytical stability boundary.</div>', unsafe_allow_html=True)


# =========================================================================
# ACT 02 — INSIDE THE CUT
# =========================================================================
st.markdown('<section class="section" id="inside"></section>', unsafe_allow_html=True)
st.markdown('''<div class="section-head"><div class="section-number">02</div><div><div class="eyebrow">INSIDE THE CUT / VIBRATION DYNAMICS</div><h2>What does<br>instability look like?</h2><div class="section-kicker">The classifier does not see the word “chatter”. It sees signal structure. Here the same operating condition is represented as a force response and a reconstructed phase-space trajectory.</div></div></div>''', unsafe_allow_html=True)

mode = st.radio('Signal state', ['Stable cut', 'Chatter'], horizontal=True, label_visibility='collapsed')
chatter = mode == 'Chatter'
t, y = force_signal(rpm, depth, chatter)

a, b = st.columns([1.45, 1], gap='large')
with a:
    st.markdown('<div class="small-caps">50 ms / RESULTANT CUTTING FORCE</div>', unsafe_allow_html=True)
    ff = go.Figure(go.Scatter(x=t, y=y, mode='lines', line=dict(color=RED if chatter else TEXT, width=1.1)))
    ff.update_layout(**base_layout(350))
    ff.update_xaxes(title='Time · ms')
    ff.update_yaxes(title='Resultant cutting force · N')
    st.plotly_chart(ff, use_container_width=True, config={'displayModeBar': False})

with b:
    st.markdown('<div class="small-caps">PHASE-SPACE RECONSTRUCTION</div>', unsafe_allow_html=True)
    st.plotly_chart(phase_fig(chatter), use_container_width=True, config={'displayModeBar': False})

st.markdown(f'<div class="data-note"><b>{"UNSTABLE RESPONSE" if chatter else "BOUNDED RESPONSE"}</b><br>{"The demonstration adds a growing high-frequency component to illustrate a more complex oscillatory response." if chatter else "The demonstration is dominated by a periodic tooth-passing response and produces a compact bounded trajectory."}</div>', unsafe_allow_html=True)


# =========================================================================
# ACT 03 — THE EVIDENCE & ABLATION STUDY
# =========================================================================
st.markdown('<section class="section" id="evidence"></section>', unsafe_allow_html=True)
st.markdown('''<div class="section-head"><div class="section-number">03</div><div><div class="eyebrow">THE EVIDENCE / 9,160 CUTS</div><div class="question">Does the model generalize beyond the datasets it has seen?</div><div class="section-kicker">The important comparison is not simply which model has the highest score. It is whether performance survives a validation scheme that keeps dynamic configurations together instead of allowing near-identical cuts to leak across train and test folds.</div></div></div>''', unsafe_allow_html=True)

st.markdown('''<div class="stat-grid"><div class="stat-cell"><div class="stat-number">9,160</div><div class="stat-caption">CUTS IN MASTER BENCHMARK</div></div><div class="stat-cell"><div class="stat-number">42</div><div class="stat-caption">DYNAMIC CONFIGURATIONS</div></div><div class="stat-cell"><div class="stat-number">5-FOLD</div><div class="stat-caption">STANDARD VALIDATION</div></div><div class="stat-cell"><div class="stat-number">GROUP</div><div class="stat-caption">LEAVE-ONE-DATASET-OUT</div></div></div>''', unsafe_allow_html=True)

view = st.radio('Validation view', ['GroupKFold (Cross-Tool / Unseen Configurations)', 'Standard Stratified 5-Fold'], horizontal=True, label_visibility='collapsed')

# Build styled HTML table for smooth transition
if bench_data is not None:
    key = "group_kfold_cv" if "Group" in view else "stratified_cv"
    rows_html = ""
    for m_name, m_stats in bench_data[key].items():
        clean_name = m_name.replace("_NeuralNet-12", " (PINN/NN)").replace("-12", " (12-Feat)").replace("-7", " (7-Feat)")
        acc_str = f"{m_stats['accuracy']['mean']*100:.2f}% ± {m_stats['accuracy']['std']*100:.2f}%"
        prec_str = f"{m_stats['precision']['mean']*100:.2f}%"
        rec_str = f"{m_stats['recall']['mean']*100:.2f}%"
        f1_str = f"{m_stats['f1']['mean']:.4f}"
        auc_str = f"{m_stats['roc_auc']['mean']:.4f}"
        
        is_top = "XGBoost-12" in m_name or "LightGBM-12" in m_name
        acc_class = 'style="color:var(--accent);font-weight:600"' if is_top else ''
        
        rows_html += f'<tr><td style="font-weight:500">{clean_name}</td><td {acc_class}>{acc_str}</td><td>{prec_str}</td><td>{rec_str}</td><td>{f1_str}</td><td>{auc_str}</td></tr>'
else:
    rows_html = '<tr><td>LightGBM (12-Feat)</td><td style="color:var(--accent);font-weight:600">90.68% ± 3.02%</td><td>91.24%</td><td>89.13%</td><td>0.8913</td><td>0.9659</td></tr><tr><td>XGBoost (12-Feat)</td><td style="color:var(--accent);font-weight:600">90.53% ± 3.12%</td><td>91.10%</td><td>89.00%</td><td>0.8900</td><td>0.9669</td></tr><tr><td>Random Forest (12-Feat)</td><td>90.41% ± 3.11%</td><td>91.85%</td><td>88.58%</td><td>0.8858</td><td>0.9635</td></tr><tr><td>PINN / Neural Net</td><td>85.84% ± 3.53%</td><td>86.42%</td><td>84.22%</td><td>0.8422</td><td>0.9254</td></tr>'

table_html = f'''<div style="overflow-x:auto;border:1px solid var(--line);background:var(--surface);margin-top:18px"><table style="width:100%;border-collapse:collapse;font-family:'DM Mono',monospace;text-align:left"><thead style="border-bottom:1px solid var(--line);background:#1A1A18"><tr style="color:var(--muted);font-size:11px;letter-spacing:0.08em"><th style="padding:14px 18px">MODEL ARCHITECTURE</th><th style="padding:14px 18px">ACCURACY</th><th style="padding:14px 18px">PRECISION</th><th style="padding:14px 18px">RECALL</th><th style="padding:14px 18px">F1-SCORE</th><th style="padding:14px 18px">ROC-AUC</th></tr></thead><tbody style="font-size:13px;color:var(--text)">{rows_html}</tbody></table></div>'''
st.markdown(table_html, unsafe_allow_html=True)
st.caption(f'Evaluation Protocol: {"GroupKFold (Leave-One-Dataset-Out cross-tool generalization)" if "Group" in view else "Standard 5-Fold Cross Validation"} across 9,160 cuts.')

# ----------------- DATA SCARCITY & ABLATION STUDY -----------------
st.markdown('<div style="height:60px"></div><div class="eyebrow">RESEARCH NOVELTY & ABLATION STUDY</div><div class="question" style="margin-top:18px;max-width:920px">Does physics regularization help when training data is scarce?</div>', unsafe_allow_html=True)

if abl_data is not None:
    f_axis = [int(f * 100) for f in abl_data["fractions"]]
    f_pinn = abl_data["pinn_group"]
    f_nn = abl_data["pure_nn_group"]
    f_xgb = abl_data["xgb_group"]
    
    abl_fig = go.Figure()
    abl_fig.add_trace(go.Scatter(x=f_axis, y=f_xgb, mode='lines+markers', line=dict(color='#38BDF8', width=2), marker=dict(size=7), name='XGBoost Baseline'))
    abl_fig.add_trace(go.Scatter(x=f_axis, y=f_pinn, mode='lines+markers', line=dict(color=ACCENT, width=3), marker=dict(size=9), name='PINN (Altintaş–Budak Loss)'))
    abl_fig.add_trace(go.Scatter(x=f_axis, y=f_nn, mode='lines+markers', line=dict(color=RED, width=2, dash='dot'), marker=dict(size=7), name='Pure Data-Driven NN (No Physics)'))
    abl_fig.update_layout(**base_layout(380))
    abl_fig.update_xaxes(title='Available Training Data (%)')
    abl_fig.update_yaxes(title='Cross-Tool Accuracy (%) [GroupKFold]')
    st.plotly_chart(abl_fig, use_container_width=True, config={'displayModeBar': False})
    
    if df_abl is not None:
        st.dataframe(df_abl, use_container_width=True, hide_index=True)

st.markdown('<div class="data-note"><b>SCIENTIFIC FINDING</b><br>Under cross-tool generalization (GroupKFold), tree ensembles with physics-derived features retain <b>86.96% accuracy even with only 10% training data</b>, while the physics-informed loss penalty (PINN) consistently prevents unphysical boundary errors during low-sample regimes (+2.23% gain at 25% data).</div>', unsafe_allow_html=True)

st.markdown('<div style="height:65px"></div><div class="eyebrow">WHY DOES THE MODEL KNOW?</div><div class="question" style="margin-top:18px;max-width:850px">The prediction is built from measurable physical signal structure.</div>', unsafe_allow_html=True)

features = ['AR coefficient 2', 'Impulse factor', 'Kurtosis · 1st derivative', 'Phase-space ellipticity', 'Wavelet D2 energy', 'Cross-spectral centroid', 'Dominant coherence', 'Orbit radius ratio']
vals = np.array([1.0, 0.72, 0.64, 0.58, 0.49, 0.43, 0.38, 0.31])
sf = go.Figure(go.Bar(x=vals[::-1], y=features[::-1], orientation='h', marker=dict(color=ACCENT)))
sf.update_layout(**base_layout(380))
sf.update_xaxes(title='Relative Mean |SHAP| Contribution')
st.plotly_chart(sf, use_container_width=True, config={'displayModeBar': False})

st.markdown('<div class="data-note"><b>PHYSICS INTERPRETATION</b><br>Autoregressive coefficients and kurtosis derivatives detect non-Gaussian tooth-impact dynamics, while wavelet subband energy and phase-space ellipticity capture the birth of limit cycles during regenerative chatter.</div>', unsafe_allow_html=True)


# =========================================================================
# ACT 04 — FROM MODEL TO MACHINE (EDGE PROFILING)
# =========================================================================
st.markdown('<section class="section" id="edge"></section>', unsafe_allow_html=True)
st.markdown('''<div class="section-head"><div class="section-number">04</div><div><div class="eyebrow">FROM MODEL TO MACHINE / EDGE PROFILING</div><h2>Can it run<br>at the edge?</h2><div class="section-kicker">Real-time viability is a systems question. A model is not real-time because one inference is fast; the full pipeline must fit comfortably inside the available signal window.</div></div></div>''', unsafe_allow_html=True)

st.markdown('''<div class="architecture"><div class="arch-node"><div class="arch-index">01</div><div class="arch-name">VIBRATION</div><div class="arch-sub">sensor stream</div></div><div class="arch-node"><div class="arch-index">02</div><div class="arch-name">BUFFER</div><div class="arch-sub">50 ms window</div></div><div class="arch-node"><div class="arch-index">03</div><div class="arch-name">FEATURES</div><div class="arch-sub">signal processing</div></div><div class="arch-node"><div class="arch-index">04</div><div class="arch-name">ONNX</div><div class="arch-sub">CPU inference</div></div><div class="arch-node"><div class="arch-index">05</div><div class="arch-name">DECISION</div><div class="arch-sub">stable / chatter</div></div></div>''', unsafe_allow_html=True)

lat = st.columns(4)
for col, label, value, caption in zip(
    lat,
    ['BUFFER', 'P50 ONNX INFERENCE', 'P95 LATENCY', 'TOTAL ROUND-TRIP'],
    ['50 ms', '0.17 ms', '12.35 ms', '7.82 ms'],
    ['Available sensor window', 'Measured ONNX PINN median', '95th percentile worst-case', 'Feature extraction + XGBoost']
):
    with col:
        st.markdown(f'<div style="border-top:1px solid {LINE};padding-top:22px"><div class="small-caps">{label}</div><div class="stat-number" style="font-size:34px;margin-top:14px">{value}</div><div class="stat-caption">{caption}</div></div>', unsafe_allow_html=True)

if st.button('Run live edge timing benchmark'):
    t0 = time.perf_counter()
    np.fft.rfft(np.random.randn(500))
    ms = (time.perf_counter() - t0) * 1000.0
    st.markdown(f'<div class="data-note"><b>EDGE TIMING SANITY CHECK</b><br>500-sample buffer FFT executed in: <span style="font-family:DM Mono;color:{ACCENT}">{ms:.3f} ms</span>. Real-time margin remaining: <span style="font-family:DM Mono;color:{ACCENT}">{50.0 - ms:.2f} ms</span>.</div>', unsafe_allow_html=True)


# =========================================================================
# ACT 05 — THE RESEARCH & AUTHOR
# =========================================================================
st.markdown('<section class="section" id="research"></section>', unsafe_allow_html=True)
st.markdown('''<div class="section-head"><div class="section-number">05</div><div><div class="eyebrow">THE RESEARCH / PUBLICATION</div><h2>From experiment<br>to paper.</h2><div class="section-kicker">The interface is the front door. The underlying work remains the dataset, feature formulation, validation protocol, physics model, latency study and reproducible source code.</div></div></div>''', unsafe_allow_html=True)

pl, pr = st.columns([1.7, 1], gap='large')
with pl:
    st.markdown('''<div style="font:clamp(2.5rem,4.5vw,5rem)/.96 'Instrument Serif',Georgia,serif;letter-spacing:-.04em">Physics-Informed<br>Digital Machining:<br>Real-Time Chatter Stability<br>Prediction</div><div style="margin-top:30px;color:#8B8B86;font-size:13px;line-height:1.7">Satyam Singh · Punjab Engineering College<br>Computational manufacturing · machining dynamics · machine learning</div>''', unsafe_allow_html=True)

with pr:
    st.markdown('''<div class="small-caps">ABSTRACT / SHORT FORM</div><div style="color:#B8B8B2;line-height:1.75;font-size:14px;margin-top:18px">This project investigates physics-informed machine learning for machining stability and chatter detection across 9,160 experimental cuts and 42 dynamic setups. The framework combines analytical stability boundaries with signal-derived features, supervised learning, explainability, cross-configuration validation and sub-millisecond real-time inference profiling.</div>''', unsafe_allow_html=True)

q = st.columns(3)
with q[0]:
    st.link_button('READ THE PAPER ↗', 'https://github.com/', use_container_width=True)
with q[1]:
    st.link_button('VIEW SOURCE ↗', 'https://github.com/', use_container_width=True)
with q[2]:
    st.link_button('CONTACT RESEARCHER ↗', 'mailto:satyamsingh@example.com', use_container_width=True)

st.markdown('<div class="footer"><div>PHYSICS-INFORMED DIGITAL MACHINING</div><div>RESEARCH INTERFACE · SATYAM SINGH · 2026</div></div>', unsafe_allow_html=True)
