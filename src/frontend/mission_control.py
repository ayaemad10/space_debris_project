import os, base64, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import tensorflow as tf
import streamlit.components.v1 as components

BASE_DIR     = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src" / "scripts"))
from decision_engine import decide

MODELS_DIR   = BASE_DIR / "models"
DATA_DIR     = BASE_DIR / "data"
OUTPUTS_DIR  = BASE_DIR / "outputs"
ASSETS_DIR   = FRONTEND_DIR / "assets"
MODEL_PATH   = MODELS_DIR / "lstm_model.keras"
DATA_PATH    = DATA_DIR   / "engineered_data.csv"
CMP_PATH     = OUTPUTS_DIR / "model_comparison.csv"
MET_PATH     = OUTPUTS_DIR / "baseline_validation_metrics.json"
FI_PATH      = OUTPUTS_DIR / "shap" / "feature_importance.csv"
SIM_PATH     = FRONTEND_DIR / "earth_simulation.html"
LOGO_PATH    = ASSETS_DIR / "tavra_logo.png"
CSS_PATH     = ASSETS_DIR / "theme.css"
BATCH_SIZE   = 500

DARK = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0e0ff", family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=38, b=10),
)

st.set_page_config(page_title="TAVRA | Mission Control", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def _css(p): return p.read_text(encoding="utf-8") if p.exists() else ""
@st.cache_data
def _b64(p): return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""
@st.cache_resource
def _model(): return tf.keras.models.load_model(MODEL_PATH)
@st.cache_data
def _data():
    df = pd.read_csv(DATA_PATH); df["event_batch"] = df.index // BATCH_SIZE; return df
@st.cache_data
def _sim(): return SIM_PATH.read_text(encoding="utf-8")

def card(label, value, sub="", cls="", icon=""):
    return f"""<div class="tv-metric {cls}">
      <div class="tv-metric-icon">{icon}</div>
      <div class="tv-metric-label">{label}</div>
      <div class="tv-metric-value">{value}</div>
      <div class="tv-metric-sub">{sub}</div></div>"""

# ── Theme ──────────────────────────────────────────────────────────────────
css = _css(CSS_PATH)
if css: st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if not MODEL_PATH.exists(): st.error(f"❌ Model not found: {MODEL_PATH}"); st.stop()
if not DATA_PATH.exists():  st.error(f"❌ Data not found: {DATA_PATH}"); st.stop()

model = _model(); df = _data(); logo = _b64(LOGO_PATH)
feat  = [c for c in df.columns if c not in ("risk","risk_label","event_batch")]

# ── Header (st.markdown — integrates natively, no iframe) ─────────────────
logo_tag = f'<img src="data:image/png;base64,{logo}" style="height:62px;flex-shrink:0;filter:drop-shadow(0 0 18px rgba(0,200,255,.65));"/>' if logo else ""
now_utc  = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');
.tv-hdr {{
  position:relative; overflow:hidden;
  display:flex; align-items:center; gap:22px;
  padding:22px 30px; margin-bottom:18px; border-radius:18px;
  background:linear-gradient(135deg,rgba(0,50,110,.62),rgba(0,25,60,.52) 40%,rgba(50,10,100,.46));
  border:1px solid rgba(0,200,255,.22);
  box-shadow:0 0 80px rgba(0,130,255,.14), inset 0 1px 0 rgba(255,255,255,.06);
}}
.tv-hdr-scan {{
  position:absolute; left:0; right:0; height:1.5px; top:0;
  background:linear-gradient(90deg,transparent,#00eeff 40%,#9b5cff 60%,transparent);
  animation:tv-scan 4s ease-in-out infinite; opacity:.75;
}}
@keyframes tv-scan {{ 0%{{top:-2px;opacity:0}} 8%{{opacity:.75}} 92%{{opacity:.75}} 100%{{top:100%;opacity:0}} }}
.tv-hdr::after {{
  content:''; position:absolute; bottom:0; left:7%; right:7%; height:1px;
  background:linear-gradient(90deg,transparent,#00c8ff,#9b5cff,transparent); opacity:.5;
}}
.tv-hdr-tb {{ flex:1; }}
.tv-hdr-brand {{
  font-family:'Space Grotesk',sans-serif; font-size:30px; font-weight:800;
  letter-spacing:7px; line-height:1;
  background:linear-gradient(90deg,#00eeff,#00c8ff 35%,#9b5cff 70%,#ffd840);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:tv-glow 3s ease-in-out infinite;
}}
@keyframes tv-glow {{ 0%,100%{{filter:brightness(1)}} 50%{{filter:brightness(1.3)}} }}
.tv-hdr-sub  {{ font-size:9.5px; letter-spacing:5px; text-transform:uppercase; color:#4a7a9b; margin-top:4px; font-weight:600; }}
.tv-hdr-tag  {{ font-size:12px; color:rgba(160,215,255,.68); margin-top:6px; }}
.tv-hdr-meta {{ text-align:right; flex-shrink:0; }}
.tv-hdr-clk  {{
  font-family:'JetBrains Mono',monospace; font-size:13px; color:#00eeff;
  text-shadow:0 0 14px rgba(0,238,255,.65); letter-spacing:1px;
}}
.tv-hdr-badge {{
  display:inline-flex; align-items:center; gap:7px;
  margin-top:8px; padding:4px 13px; border-radius:999px;
  background:rgba(0,255,176,.08); border:1px solid rgba(0,255,176,.32);
  font-size:9.5px; color:#00ffb0; letter-spacing:1.5px; font-weight:700;
}}
.tv-hdr-dot {{
  width:6px; height:6px; border-radius:50%; background:#00ffb0;
  box-shadow:0 0 10px #00ffb0; animation:tv-dot 1.8s ease-in-out infinite; display:inline-block;
}}
@keyframes tv-dot {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.35;transform:scale(.6)}} }}
</style>

<div class="tv-hdr">
  <div class="tv-hdr-scan"></div>
  {logo_tag}
  <div class="tv-hdr-tb">
    <div class="tv-hdr-brand">TAVRA</div>
    <div class="tv-hdr-sub">Mission Control Center</div>
    <div class="tv-hdr-tag">Space Debris &amp; Satellite Collision Risk Prediction Platform &mdash; LSTM + SHAP Explainability</div>
  </div>
  <div class="tv-hdr-meta">
    <div class="tv-hdr-clk">{now_utc}</div>
    <div class="tv-hdr-badge"><span class="tv-hdr-dot"></span>SYSTEM OPERATIONAL</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    if logo: st.markdown(f'<div class="tv-sidebar-logo"><img src="data:image/png;base64,{logo}"/></div>', unsafe_allow_html=True)
    total_hi = int((df["risk"] > -6).sum())
    st.markdown(f"""<div class="tv-sb-block">
      <div class="tv-sb-title">Mission Status</div>
      <div class="tv-sb-row"><span class="lbl">Status</span><span class="val"><span class="tv-status-dot tv-status-ok"></span>OPERATIONAL</span></div>
      <div class="tv-sb-row"><span class="lbl">Total Events</span><span class="val">{len(df):,}</span></div>
      <div class="tv-sb-row"><span class="lbl">High-Risk</span><span class="val" style="color:#ff2d5b">{total_hi:,}</span></div>
      <div class="tv-sb-row"><span class="lbl">Features</span><span class="val">{len(feat)}</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="tv-sb-block"><div class="tv-sb-title">Configuration</div>', unsafe_allow_html=True)
    sel_batch = st.selectbox("Event Batch", sorted(df["event_batch"].unique()))
    risk_thr  = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("</div>", unsafe_allow_html=True)
    gpus = tf.config.list_physical_devices("GPU")
    accel = f"{len(gpus)} GPU(s)" if gpus else "CPU only"
    cpu_txt = "N/A"
    try:
        import psutil; cpu_txt = f"{psutil.cpu_percent(interval=0.05):.0f}%  |  {psutil.virtual_memory().percent:.0f}% MEM"
    except ImportError: pass
    st.markdown(f"""<div class="tv-sb-block">
      <div class="tv-sb-title">System</div>
      <div class="tv-sb-row"><span class="lbl">Compute</span><span class="val">{accel}</span></div>
      <div class="tv-sb-row"><span class="lbl">CPU/RAM</span><span class="val">{cpu_txt}</span></div>
      <div class="tv-sb-row"><span class="lbl">LSTM</span><span class="val"><span class="tv-status-dot tv-status-ok"></span>LOADED</span></div>
    </div>""", unsafe_allow_html=True)

# ── Batch data ─────────────────────────────────────────────────────────────
batch  = df[df["event_batch"] == sel_batch]
hi     = int((batch["risk"] > -6).sum())
total  = len(batch); hi_pct = hi / total * 100 if total else 0
rc     = "risk-critical" if hi_pct > 15 else ("risk-high" if hi_pct > 5 else "risk-low")

# ── Metric cards ───────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(card("Total Events",     f"{total:,}",                    f"Batch #{sel_batch}", icon="📡"), unsafe_allow_html=True)
with c2: st.markdown(card("High-Risk Events", f"{hi:,}",                       f"Pc > 1e-6  ·  {hi_pct:.1f}%", rc, "⚠️"), unsafe_allow_html=True)
with c3: st.markdown(card("Avg log₁₀(Pc)",    f"{batch['risk'].mean():.2f}",   "Lower = safer", icon="📉"), unsafe_allow_html=True)
with c4: st.markdown(card("Avg Miss Distance", f"{batch['miss_distance'].mean():.3f}", "Scaled units", icon="📏"), unsafe_allow_html=True)

# ── Intelligence Feed ───────────────────────────────────────────────────────
st.write("")
top5 = batch.nsmallest(5, "risk")[["risk","miss_distance"]].reset_index()

BADGE_STYLE = {
    "CRITICAL": "background:rgba(255,45,91,.22);color:#ff6b8a;border:1px solid rgba(255,45,91,.55);",
    "HIGH":     "background:rgba(255,122,26,.18);color:#ffaa55;border:1px solid rgba(255,122,26,.45);",
    "LOW":      "background:rgba(0,255,176,.12);color:#00ffb0;border:1px solid rgba(0,255,176,.38);",
}
ICON = {"CRITICAL": "🔴", "HIGH": "🟡", "LOW": "🟢"}

# Build each row as a SINGLE LINE — no newlines or indentation inside
# (Streamlit markdown treats indented multi-line text as code blocks)
rows = []
for rank, (_, row) in enumerate(top5.iterrows(), 1):
    pc_val = float(row["risk"]); dist = float(row["miss_distance"])
    lvl   = "CRITICAL" if pc_val > -3 else ("HIGH" if pc_val > -6 else "LOW")
    bst   = BADGE_STYLE[lvl]
    bar_w = min(100, max(4, int((pc_val + 30) / 27 * 100)))
    bar_c = "#ff2d5b" if lvl=="CRITICAL" else ("#ff7a1a" if lvl=="HIGH" else "#00ffb0")
    evt   = f"CONJ-EVENT-{int(row['index']):05d}"
    pc_s  = f"log&#8321;&#8320;(Pc)={pc_val:.3f}"
    rows.append(
        f'<div style="display:flex;align-items:center;gap:14px;padding:11px 16px;border-radius:10px;margin-bottom:7px;background:rgba(0,18,42,.55);border:1px solid rgba(0,150,220,.16);">'
        f'<span style="font-family:monospace;font-size:13px;color:#4a7a9b;min-width:28px;font-weight:600;">#{rank:02d}</span>'
        f'<span style="font-size:9px;font-weight:800;letter-spacing:1.5px;padding:3px 10px;border-radius:999px;flex-shrink:0;text-transform:uppercase;{bst}">{ICON[lvl]} {lvl}</span>'
        f'<span style="font-family:monospace;font-size:12px;color:#c0e0ff;font-weight:600;flex:1;">{evt}</span>'
        f'<div style="flex:1;max-width:150px;">'
        f'<div style="font-size:10px;color:#4a7a9b;margin-bottom:3px;">Risk Level</div>'
        f'<div style="background:rgba(0,20,50,.6);border-radius:4px;height:5px;overflow:hidden;"><div style="width:{bar_w}%;height:100%;background:{bar_c};border-radius:4px;box-shadow:0 0 6px {bar_c};"></div></div>'
        f'<div style="font-family:monospace;font-size:10px;color:#7aadcc;margin-top:3px;">{pc_s}</div>'
        f'</div>'
        f'<div style="text-align:right;min-width:90px;">'
        f'<div style="font-size:10px;color:#4a7a9b;">Miss Distance</div>'
        f'<div style="font-family:monospace;font-size:12px;color:#00c8ff;font-weight:600;">{dist:.4f}</div>'
        f'</div>'
        f'</div>'
    )

feed_html = "".join(rows)
header_row = (
    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
    f'<div style="width:8px;height:8px;border-radius:50%;background:#00eeff;box-shadow:0 0 10px #00eeff;"></div>'
    f'<span style="font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#00eeff;font-weight:700;">&#9889; Mission Intelligence Feed</span>'
    f'<span style="font-size:10px;color:#4a7a9b;margin-left:auto;font-family:monospace;">TOP 5 DANGER EVENTS &middot; BATCH #{sel_batch}</span>'
    f'</div>'
)
st.markdown(
    f'<div style="border-radius:14px;padding:16px 18px;background:rgba(0,10,25,.60);border:1px solid rgba(0,180,255,.18);backdrop-filter:blur(14px);">{header_row}{feed_html}</div>',
    unsafe_allow_html=True,
)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Analysis", "🎯 Predictions", "🛰️ 3D Simulation", "🧠 Model Info"])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════
with tab1:
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="tv-section-title">Probability Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(batch, x="risk", nbins=60, color_discrete_sequence=["#00c8ff"],
                           labels={"risk":"log₁₀(Pc)"})
        fig.add_vline(x=-6, line_dash="dash", line_color="#ff2d5b",
                      annotation_text="Risk Threshold", annotation_font_color="#ff2d5b",
                      annotation_position="top right")
        fig.update_traces(marker_line_width=0, opacity=0.82)
        fig.update_layout(title="log₁₀(Collision Probability) Distribution", **DARK)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown('<div class="tv-section-title">Risk Category Breakdown</div>', unsafe_allow_html=True)
        low_  = int((batch["risk"] <= -10).sum())
        med_  = int(((batch["risk"] > -10) & (batch["risk"] <= -6)).sum())
        high_ = int((batch["risk"] > -6).sum())
        fig2 = go.Figure(go.Bar(
            x=["LOW\n(Pc ≤ 1e-10)", "MEDIUM\n(1e-10 – 1e-6)", "HIGH\n(Pc > 1e-6)"],
            y=[low_, med_, high_], text=[f"{low_:,}", f"{med_:,}", f"{high_:,}"],
            textposition="outside", textfont=dict(color="#c0e0ff"),
            marker=dict(color=["#00ffb0","#ffd840","#ff2d5b"],
                        line=dict(color="rgba(0,0,0,0)", width=0)),
        ))
        fig2.update_layout(title="Events by Risk Category", **DARK, showlegend=False,
                           yaxis=dict(gridcolor="rgba(0,150,220,0.10)"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="tv-section-title">Miss Distance vs Risk — 3-D Intelligence View</div>', unsafe_allow_html=True)
    # 3D scatter: miss_distance vs risk vs relative_speed (or third feature)
    third_col = "relative_speed" if "relative_speed" in batch.columns else feat[2]
    fig3d = px.scatter_3d(
        batch.sample(min(500, len(batch))), x="miss_distance", y="risk", z=third_col,
        color="risk", color_continuous_scale=["#00ffb0","#ffd840","#ff2d5b"],
        opacity=0.75, size_max=6,
        labels={"miss_distance":"Miss Distance","risk":"log₁₀(Pc)",third_col: third_col.replace("_"," ").title()},
    )
    fig3d.update_layout(
        title="3D Conjunction Risk Space", paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,8,20,0.0)",
            xaxis=dict(backgroundcolor="rgba(0,20,50,0.4)", gridcolor="rgba(0,150,220,0.2)", color="#5a88aa"),
            yaxis=dict(backgroundcolor="rgba(0,20,50,0.4)", gridcolor="rgba(0,150,220,0.2)", color="#5a88aa"),
            zaxis=dict(backgroundcolor="rgba(0,20,50,0.4)", gridcolor="rgba(0,150,220,0.2)", color="#5a88aa"),
        ),
        font=dict(color="#c0e0ff", family="Inter"),
        height=480, margin=dict(l=0,r=0,t=40,b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="tv-section-title">Real-Time Conjunction Risk Prediction</div>', unsafe_allow_html=True)
    p1, p2 = st.columns([1,1])
    with p1:
        idx    = st.slider("Sample Index", 0, max(len(batch)-1,0), 0)
        sample = batch.iloc[idx][feat].values.astype("float32")
        prob   = float(model.predict(sample.reshape(1,1,-1), verbose=0)[0][0])
        dec    = decide(prob); conf = abs(prob-0.5)*2

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob*100, 3),
            title=dict(text="Risk Probability (%)", font=dict(color="#c0e0ff", size=13)),
            number=dict(suffix="%", font=dict(color="#00eeff", size=40)),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#4a7a9b", tickfont=dict(color="#4a7a9b", size=10)),
                bar=dict(color="#00c8ff" if prob<0.5 else ("#ffd840" if prob<0.8 else "#ff2d5b"), thickness=0.3),
                bgcolor="rgba(0,20,45,0.5)", bordercolor="rgba(0,150,220,0.3)", borderwidth=1,
                steps=[
                    dict(range=[0,50],  color="rgba(0,255,176,0.07)"),
                    dict(range=[50,80], color="rgba(255,216,64,0.07)"),
                    dict(range=[80,100],color="rgba(255,45,91,0.09)"),
                ],
                threshold=dict(line=dict(color="#ff2d5b",width=2), thickness=0.75, value=80),
            ),
        ))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=270,
                            font=dict(color="#c0e0ff", family="Inter"))
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown(card("Confidence Score", f"{conf:.1%}", "Distance × 2 from 0.5", icon="🎯"), unsafe_allow_html=True)
        st.write("")
        if dec.risk_level == "CRITICAL":
            st.markdown(f'<div class="tv-alert">🔴 CRITICAL CONJUNCTION — {dec.action}</div>', unsafe_allow_html=True)
        elif dec.risk_level == "HIGH":
            st.markdown(f'<span class="tv-badge warn">🟡 HIGH RISK</span>&nbsp;&nbsp;{dec.action}', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="tv-badge ok">🟢 LOW RISK</span>&nbsp;&nbsp;{dec.action}', unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="tv-section-title">Decision Thresholds</div>', unsafe_allow_html=True)
        tbl = go.Figure(go.Table(
            header=dict(values=["Range","Level","Recommended Action"],
                        fill_color="rgba(0,25,55,0.85)", line_color="rgba(0,150,220,0.2)",
                        font=dict(color="#00eeff", size=12, family="Inter"), align="left", height=32),
            cells=dict(
                values=[["0.00 – 0.50","0.50 – 0.80","0.80 – 1.00"],
                        ["LOW","HIGH","CRITICAL"],
                        ["No Action Required","Increase Monitoring Frequency","Immediate Maneuver Required"]],
                fill_color=[["rgba(0,255,176,0.06)","rgba(255,216,64,0.06)","rgba(255,45,91,0.09)"]]*3,
                font=dict(color="#c0e0ff", size=11, family="Inter"), line_color="rgba(0,100,180,0.18)",
                align="left", height=30,
            ),
        ))
        tbl.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=170, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(tbl, use_container_width=True)

        st.markdown('<div class="tv-section-title">Top 10 Features — This Sample</div>', unsafe_allow_html=True)
        sf = pd.DataFrame({"Feature": feat[:10], "Value": sample[:10].round(5)})
        st.dataframe(sf, use_container_width=True, hide_index=True)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="tv-section-title">Live Orbital Conjunction Simulation</div>', unsafe_allow_html=True)
    st.caption("🌍 Three.js scene — Earth + atmosphere, 8 satellites on distinct orbital planes, 55 debris objects, real-time proximity alerts (⚡ orange banner when objects < 360 km), conjunction cone & danger line. Controls: on-screen panel or Space/R keys.")
    if SIM_PATH.exists():
        components.html(_sim(), height=790, scrolling=False)
    else:
        st.error(f"Simulation HTML not found: {SIM_PATH}")

# ══ TAB 4 ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="tv-section-title">Model Pipeline</div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.write("**Architecture:** LSTM (Long Short-Term Memory)")
        st.write(f"**Feature Count:** {len(feat)} engineered features")
        st.write("**Output:** P(Pc > 1e-6) — binary collision probability")
        st.write(f"**Dataset:** {len(df):,} conjunction events")
        st.write("**Input Shape:** `(batch, 1 timestep, n_features)`")
        st.write("**Training:** 80/20 stratified split, early stopping, ROC-AUC selection")
    with mc2:
        st.write("**Pipeline:** `preprocess.py` → `feature_engineering.py` → `train_models.py` → `decision_engine.py`")
        st.write("**Models Compared:** CNN · SimpleRNN · **LSTM** (winner by ROC-AUC)")
        st.write("**Explainability:** SHAP KernelExplainer (fallback: Gini importance)")
        st.write("**Thresholds:** LOW < 0.50 · HIGH 0.50–0.80 · CRITICAL > 0.80")
        st.write("**Backend:** FastAPI + uvicorn (port 8000) — `/predict`, `/statistics`, `/missions`")

    if MET_PATH.exists():
        st.markdown('<div class="tv-section-title" style="margin-top:18px">Validation Metrics</div>', unsafe_allow_html=True)
        met = json.loads(MET_PATH.read_text())
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(card("Accuracy",  f"{met.get('Accuracy',0):.3f}",  "Overall", icon="✅"), unsafe_allow_html=True)
        with m2: st.markdown(card("F1-Score",  f"{met.get('F1-Score',0):.3f}",  "Prec × Recall", icon="🎯"), unsafe_allow_html=True)
        with m3: st.markdown(card("ROC-AUC",   f"{met.get('ROC-AUC',0):.4f}",   "Discrimination", icon="📈"), unsafe_allow_html=True)
        with m4: st.markdown(card("Positive Rate", f"{met.get('positive_rate_test',0)*100:.1f}%",
                             f"n_test={met.get('n_test','?')}", icon="⚡"), unsafe_allow_html=True)
        st.caption("Source: `outputs/baseline_validation_metrics.json` — GradientBoostingClassifier baseline (sklearn), same train/test split as the LSTM.")

    if CMP_PATH.exists():
        st.markdown('<div class="tv-section-title" style="margin-top:18px">Model Comparison — CNN vs RNN vs LSTM</div>', unsafe_allow_html=True)
        cmp = pd.read_csv(CMP_PATH, index_col=0)
        st.dataframe(cmp, use_container_width=True)

    if FI_PATH.exists():
        st.markdown('<div class="tv-section-title" style="margin-top:18px">Feature Importance — Top 15</div>', unsafe_allow_html=True)
        fi = pd.read_csv(FI_PATH, index_col=0); fi.columns=["importance"]
        fi = fi.sort_values("importance",ascending=False).head(15).sort_values("importance")
        figf = go.Figure(go.Bar(
            x=fi["importance"], y=fi.index, orientation="h",
            text=fi["importance"].round(4), textposition="outside", textfont=dict(color="#c0e0ff"),
            marker=dict(color=fi["importance"], colorscale=[[0,"#004466"],[0.5,"#00c8ff"],[1,"#ffd840"]], showscale=False),
        ))
        figf.update_layout(title="Gini / SHAP Feature Importance (Top 15)", **DARK, height=460,
                           xaxis=dict(gridcolor="rgba(0,150,220,0.10)"),
                           yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(figf, use_container_width=True)
    else:
        st.info("Run `feature_importance_fallback.py` to generate feature importance.")

st.divider()
st.markdown('<div class="tv-footer">TAVRA MISSION CONTROL &nbsp;·&nbsp; Space Debris Collision Prediction &nbsp;·&nbsp; LSTM + SHAP &nbsp;·&nbsp; FastAPI + Streamlit</div>',
            unsafe_allow_html=True)
