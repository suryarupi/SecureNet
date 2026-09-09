"""
SecureNet - Real-Time NIDS Dashboard
======================================

Reads completed flow decisions written by live_capture/packet_capture.py
into data/securenet_live.db. Does NOT run its own packet sniffer and
does NOT perform ML prediction - it only visualizes what the live
pipeline has already decided.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from dashboard.live_data import (
    init_db,
    fetch_recent_flows,
    fetch_summary_counts,
    get_heartbeat,
    fetch_flow_features,
    PROJECT_ROOT,
)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================
# Configuration (mirrors live_capture/packet_capture.py)
# ============================================================

INTERFACE = "wlp0s20f3"
FLOW_TIMEOUT = 15
MIN_PACKETS = 2
HEARTBEAT_STALE_SECONDS = 10

CAPTURE_SCRIPT = os.path.join(PROJECT_ROOT, "live_capture", "packet_capture.py")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="SecureNet | NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=3000, key="securenet_autorefresh")

st.markdown("""
<style>
    .stApp { background-color: #0b0f14; }
    section[data-testid="stSidebar"] { background-color: #0e141b; }

    h1, h2, h3, h4, p, span, label, div { color: #d7e2ea; }

    .sn-title { font-size: 2.1rem; font-weight: 800; letter-spacing: 1px;
        color: #e8f6ff; margin-bottom: 0; }
    .sn-subtitle { color: #7fa8c9; font-size: 0.95rem; margin-top: -6px; }

    .status-pill { display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; }
    .status-live { background-color: rgba(0,255,157,0.12); color: #00ff9d;
        border: 1px solid #00ff9d; }
    .status-offline { background-color: rgba(255,59,59,0.12); color: #ff5c5c;
        border: 1px solid #ff5c5c; }

    div[data-testid="stMetric"] { background-color: #111823; border: 1px solid #1e2a38;
        border-radius: 10px; padding: 12px 8px; }

    .alert-box { border-radius: 10px; padding: 18px 22px; margin: 8px 0 20px 0;
        border-left: 6px solid; }
    .alert-attack { background-color: rgba(255,59,59,0.10); border-color: #ff3b3b; }
    .alert-suspicious { background-color: rgba(255,176,32,0.10); border-color: #ffb020; }
    .alert-clear { background-color: rgba(0,255,157,0.08); border-color: #00ff9d; }

    .badge { padding: 2px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 700; }
    .badge-normal { background: rgba(0,255,157,0.15); color: #00ff9d; }
    .badge-suspicious { background: rgba(255,176,32,0.18); color: #ffb020; }
    .badge-attack { background: rgba(255,59,59,0.18); color: #ff5c5c; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helpers
# ============================================================

def status_badge(status):
    cls = {
        "NORMAL": "badge-normal",
        "SUSPICIOUS": "badge-suspicious",
        "ATTACK": "badge-attack",
    }.get(status, "badge-normal")
    return f'<span class="badge {cls}">{status}</span>'


def severity_badge(severity):
    cls = {
        "NONE": "badge-normal",
        "LOW": "badge-suspicious",
        "MEDIUM": "badge-suspicious",
        "HIGH": "badge-attack",
    }.get(severity, "badge-normal")
    return f'<span class="badge {cls}">{severity}</span>'


def fmt_time(ts):
    try:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except Exception:
        return "-"


# ============================================================
# Header
# ============================================================

heartbeat = get_heartbeat()
is_live = (
    heartbeat is not None
    and (time.time() - heartbeat["last_seen"]) < HEARTBEAT_STALE_SECONDS
)

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown('<p class="sn-title">🛡️ SECURENET</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sn-subtitle">AI-Powered Real-Time Network Intrusion '
        'Detection System</p>',
        unsafe_allow_html=True,
    )

with header_right:
    pill_class = "status-live" if is_live else "status-offline"
    pill_text = "● LIVE" if is_live else "● OFFLINE"
    st.markdown(
        f'<div style="text-align:right;">'
        f'<span class="status-pill {pill_class}">{pill_text}</span><br>'
        f'<span style="color:#7fa8c9; font-size:0.85rem;">'
        f'Interface: {(heartbeat or {}).get("interface", INTERFACE)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ============================================================
# Data
# ============================================================

flows_df = fetch_recent_flows(limit=500)
summary = fetch_summary_counts()

if not flows_df.empty:
    flows_df["time_str"] = flows_df["timestamp"].apply(fmt_time)


# ============================================================
# Summary cards
# ============================================================

cards = st.columns(7)
card_data = [
    ("Total Flows", summary["total"]),
    ("Normal", summary["normal"]),
    ("Suspicious", summary["suspicious"]),
    ("Attacks", summary["attack"]),
    ("High Sev.", summary["high"]),
    ("Medium Sev.", summary["medium"]),
    ("Low Sev.", summary["low"]),
]
for col, (label, value) in zip(cards, card_data):
    col.metric(label, value)

st.markdown("")


# ============================================================
# Real-time alert panel
# ============================================================

st.markdown("### 🚨 Real-Time Alert")

if flows_df.empty:
    st.info("No live flows detected yet.")
else:
    latest = flows_df.iloc[0]

    if latest["status"] == "ATTACK":
        st.markdown(
            f"""
            <div class="alert-box alert-attack">
                <b>🚨 SECURITY ALERT</b><br><br>
                Status: <b>{latest['status']}</b> &nbsp;|&nbsp;
                Severity: <b>{latest['severity']}</b><br>
                Attack Type: <b>{latest['attack_type'] or 'Unknown'}</b><br>
                Attack Probability: <b>{latest['attack_probability']*100:.1f}%</b><br>
                Confidence: <b>{(latest['attack_confidence'] or 0)*100:.1f}%</b><br><br>
                <i>{latest['reason']}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif latest["status"] == "SUSPICIOUS":
        st.markdown(
            f"""
            <div class="alert-box alert-suspicious">
                <b>⚠️ SUSPICIOUS ACTIVITY</b><br><br>
                Status: <b>{latest['status']}</b> &nbsp;|&nbsp;
                Severity: <b>{latest['severity']}</b><br>
                Anomaly Score: <b>{latest['anomaly_score']:.4f}</b><br><br>
                <i>{latest['reason']}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-box alert-clear">
                ✅ No active threats. Latest flow classified as <b>NORMAL</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Live flow table
# ============================================================

st.markdown("### 📡 Live Flow Table")

TABLE_COLUMNS = [
    "id", "time_str", "src_ip", "src_port", "dst_ip", "dst_port",
    "protocol", "status", "severity", "attack_type",
    "attack_probability", "attack_confidence", "anomaly", "anomaly_score",
]

selected_id = None

if flows_df.empty:
    st.info("No live flows detected yet.")
else:
    table_df = flows_df[TABLE_COLUMNS].rename(columns={
        "time_str": "Timestamp", "src_ip": "Source IP", "src_port": "Src Port",
        "dst_ip": "Destination IP", "dst_port": "Dst Port", "protocol": "Protocol",
        "status": "Status", "severity": "Severity", "attack_type": "Attack Type",
        "attack_probability": "Attack Prob.", "attack_confidence": "Confidence",
        "anomaly": "Anomaly", "anomaly_score": "Anomaly Score",
    })

    event = st.dataframe(
        table_df.drop(columns=["id"]),
        use_container_width=True,
        height=380,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="flow_table",
        column_config={
            "Attack Prob.": st.column_config.ProgressColumn(
                "Attack Prob.", min_value=0, max_value=1, format="%.2f"
            ),
            "Anomaly Score": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    rows = event.selection.rows if event and event.selection else []
    if rows:
        selected_id = int(table_df.iloc[rows[0]]["id"])


# ============================================================
# Traffic statistics
# ============================================================

st.markdown("### 📊 Traffic Statistics")

if flows_df.empty:
    st.info("Charts will appear once flows start arriving.")
elif not PLOTLY_AVAILABLE:
    st.warning("Install `plotly` to see traffic charts (`pip install plotly`).")
else:
    chart_row1_c1, chart_row1_c2 = st.columns(2)

    with chart_row1_c1:
        status_counts = flows_df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(
            status_counts, names="Status", values="Count",
            title="Normal vs Suspicious vs Attack",
            color="Status",
            color_discrete_map={"NORMAL": "#00ff9d", "SUSPICIOUS": "#ffb020", "ATTACK": "#ff3b3b"},
        )
        fig.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
        st.plotly_chart(fig, use_container_width=True)

    with chart_row1_c2:
        sev_counts = flows_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig = px.bar(
            sev_counts, x="Severity", y="Count", title="Severity Distribution",
            color="Severity",
            color_discrete_map={"NONE": "#00ff9d", "LOW": "#ffd166", "MEDIUM": "#ffb020", "HIGH": "#ff3b3b"},
        )
        fig.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
        st.plotly_chart(fig, use_container_width=True)

    chart_row2_c1, chart_row2_c2 = st.columns(2)

    with chart_row2_c1:
        attack_df = flows_df[flows_df["status"] == "ATTACK"]
        if attack_df.empty:
            st.info("No attack flows yet - attack type breakdown will appear here.")
        else:
            type_counts = attack_df["attack_type"].fillna("Unknown").value_counts().reset_index()
            type_counts.columns = ["Attack Type", "Count"]
            fig = px.bar(
                type_counts, x="Attack Type", y="Count", title="Attack Types",
                color_discrete_sequence=["#ff3b3b"],
            )
            fig.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
            st.plotly_chart(fig, use_container_width=True)

    with chart_row2_c2:
        timeline_df = flows_df.sort_values("timestamp").tail(150).copy()
        timeline_df["time_str"] = timeline_df["timestamp"].apply(fmt_time)
        fig = px.line(
            timeline_df, x="time_str", y="attack_probability",
            title="Attack Probability Over Time", markers=True,
        )
        fig.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
        fig.update_traces(line_color="#ff5c5c")
        st.plotly_chart(fig, use_container_width=True)

    chart_row3_c1, chart_row3_c2 = st.columns(2)

    with chart_row3_c1:
        fig = px.line(
            timeline_df, x="time_str", y="anomaly_score",
            title="Anomaly Score Over Time", markers=True,
        )
        fig.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
        fig.update_traces(line_color="#ffb020")
        st.plotly_chart(fig, use_container_width=True)

    with chart_row3_c2:
        bucket_df = flows_df.copy()
        bucket_df["minute"] = pd.to_datetime(bucket_df["timestamp"], unit="s").dt.strftime("%H:%M")
        per_min = bucket_df.groupby("minute").size().reset_index(name="Flows")
        fig = px.bar(
            per_min, x="minute", y="Flows", title="Flows Over Time",
            color_discrete_sequence=["#4da3ff"],
        )
        fig.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Flow details + SHAP
# ============================================================

st.markdown("### 🔍 Flow Details")

if selected_id is None:
    st.caption("Select a row in the Live Flow Table above to see details.")
else:
    row = flows_df[flows_df["id"] == selected_id].iloc[0]

    net_col, ml_col, dec_col = st.columns(3)

    with net_col:
        st.markdown("**Network**")
        st.write(f"Source IP: `{row['src_ip']}`")
        st.write(f"Destination IP: `{row['dst_ip']}`")
        st.write(f"Source Port: `{row['src_port']}`")
        st.write(f"Destination Port: `{row['dst_port']}`")
        st.write(f"Protocol: `{row['protocol']}`")

    with ml_col:
        st.markdown("**ML**")
        st.write(f"Binary Prediction: `{'Attack' if row['status'] != 'NORMAL' or row['attack_probability'] >= 0.5 else 'Normal'}`")
        st.write(f"Attack Probability: `{row['attack_probability']:.4f}`")
        st.write(f"Attack Type: `{row['attack_type'] or 'None'}`")
        st.write(f"Attack Confidence: `{row['attack_confidence']}`")
        st.write(f"Anomaly: `{bool(row['anomaly'])}`")
        st.write(f"Anomaly Score: `{row['anomaly_score']:.4f}`")

    with dec_col:
        st.markdown("**Decision**")
        st.markdown(f"Status: {status_badge(row['status'])}", unsafe_allow_html=True)
        st.markdown(f"Severity: {severity_badge(row['severity'])}", unsafe_allow_html=True)
        st.write("Reason:")
        st.caption(row["reason"])

    # --------------------------------------------------------
    # Optional SHAP explanation - only offered for ATTACK flows
    # --------------------------------------------------------

    if row["status"] == "ATTACK":
        st.markdown("---")
        if st.button("🧠 Explain Prediction", key=f"explain_{selected_id}"):
            features = fetch_flow_features(int(selected_id))

            if features is None:
                st.warning(
                    "Raw feature data for this flow wasn't found, so it "
                    "can't be explained. This can happen for flows stored "
                    "before this feature was added."
                )
            else:
                try:
                    from prediction.explain import explain_binary, explain_multiclass

                    flow_series = pd.Series(features)

                    st.markdown("**Why was this flow classified as malicious?**")

                    binary_exp = explain_binary(flow_series).head(10)
                    fig = px.bar(
                        binary_exp, x="SHAP Value", y="Feature", orientation="h",
                        color="SHAP Value", color_continuous_scale=["#00ff9d", "#ff3b3b"],
                        title="Top Contributing Features (Binary Classification)",
                    )
                    fig.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
                    st.plotly_chart(fig, use_container_width=True)

                    if row["attack_type"]:
                        try:
                            predicted_type, multi_exp = explain_multiclass(flow_series)
                            multi_exp = multi_exp.head(10)
                            st.markdown(f"**Why was it classified as {predicted_type}?**")
                            fig2 = px.bar(
                                multi_exp, x="SHAP Value", y="Feature", orientation="h",
                                color="SHAP Value", color_continuous_scale=["#4da3ff", "#ffb020"],
                                title="Top Contributing Features (Attack Type)",
                            )
                            fig2.update_layout(paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", font_color="#d7e2ea")
                            st.plotly_chart(fig2, use_container_width=True)
                        except Exception as multi_err:
                            st.caption(f"Multiclass explanation unavailable: {multi_err}")

                except Exception as e:
                    st.warning(f"SHAP explanation unavailable right now: {e}")


# ============================================================
# Sidebar - monitoring controls
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Live Monitoring Controls")

    st.write(f"**Interface:** `{INTERFACE}`")
    st.write(f"**Flow timeout:** {FLOW_TIMEOUT} seconds")
    st.write(f"**Minimum packets:** {MIN_PACKETS}")

    st.markdown("---")

    if "capture_proc" not in st.session_state:
        st.session_state.capture_proc = None

    c1, c2 = st.columns(2)

    with c1:
        if st.button("▶ START CAPTURE", use_container_width=True):
            proc = st.session_state.capture_proc
            if proc is not None and proc.poll() is None:
                st.warning("A capture process started from this dashboard is already running.")
            else:
                try:
                    # No sudo is invoked here on purpose. Raw packet
                    # capture needs elevated privileges - the safe way
                    # to grant that without embedding sudo in a web
                    # app is to grant the venv's python the capability
                    # once via:
                    #   sudo setcap cap_net_raw,cap_net_admin=eip .venv/bin/python
                    # After that, packet_capture.py can run without sudo.
                    python_bin = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
                    st.session_state.capture_proc = subprocess.Popen(
                        [python_bin, CAPTURE_SCRIPT],
                        cwd=PROJECT_ROOT,
                    )
                    st.success("Capture process launched.")
                except Exception as e:
                    st.error(
                        f"Could not launch capture from the dashboard ({e}). "
                        "This is expected if raw-socket permissions aren't "
                        "granted to this process. Run it manually instead:\n\n"
                        f"sudo {VENV_PYTHON} {CAPTURE_SCRIPT}"
                    )

    with c2:
        if st.button("■ STOP CAPTURE", use_container_width=True):
            proc = st.session_state.capture_proc
            if proc is not None and proc.poll() is None:
                proc.terminate()
                st.info("Stop signal sent to capture process.")
            else:
                st.warning(
                    "No capture process was started from this dashboard. "
                    "If it's running in a separate terminal, stop it there "
                    "with CTRL+C."
                )

    if st.button("🔄 REFRESH", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption(
        "Recommended for the demo: run capture in its own terminal "
        "(`sudo .venv/bin/python live_capture/packet_capture.py`) and "
        "this dashboard in a second terminal. The buttons above are a "
        "convenience for unprivileged runs and won't fight over the "
        "network interface with a terminal-started process."
    )

    st.markdown("---")
    st.caption(f"Database: `{os.path.relpath(os.path.join(PROJECT_ROOT, 'data', 'securenet_live.db'), PROJECT_ROOT)}`")