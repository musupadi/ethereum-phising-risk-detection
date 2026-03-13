"""
app/streamlit_app.py
--------------------
Streamlit demo UI for the Ethereum Phishing Risk Registry.

Modes:
  1. Demo mode  – pick a row from test.csv via a slider; wallet auto-populated.
  2. Manual mode – paste a wallet + fill feature sliders.

Both modes call the FastAPI backend at http://localhost:8000.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
ROOT = Path(__file__).resolve().parent.parent

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ethereum Phishing Risk Registry",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS for risk colour cards ──────────────────────────────────────────
st.markdown(
    """
    <style>
    .risk-high   { background:#ff4b4b22; border-left:6px solid #ff4b4b; padding:12px; border-radius:6px; }
    .risk-medium { background:#ffa50022; border-left:6px solid #ffa500; padding:12px; border-radius:6px; }
    .risk-low    { background:#21c35422; border-left:6px solid #21c354; padding:12px; border-radius:6px; }
    .metric-box  { background:#f0f2f622; border:1px solid #e0e0e0; padding:14px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Ethereum Phishing Risk Registry")
st.caption("AI-powered phishing risk scoring anchored to a local Hardhat blockchain")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Input mode", ["🎲 Demo (test.csv row)", "✍️ Manual (enter wallet + features)"])
    st.divider()
    st.markdown(f"**API:** `{API_BASE}`")

    # health check
    try:
        hresp = requests.get(f"{API_BASE}/health", timeout=3)
        if hresp.ok:
            health = hresp.json()
            st.success("API ✅ online")
            chain_ok = health.get("chain_connected", False)
            if chain_ok:
                st.success("Chain ✅ connected")
            else:
                st.warning("Chain ⚠️ not connected (scores won't be stored on-chain)")
        else:
            st.error("API ❌ error")
    except Exception:
        st.error("API ❌ unreachable – start FastAPI first")

    st.divider()
    st.markdown(
        "**Risk levels:**\n"
        "- 🟢 LOW  0–39\n"
        "- 🟡 MEDIUM  40–69\n"
        "- 🔴 HIGH  70–100"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: render result
# ──────────────────────────────────────────────────────────────────────────────

def render_result(data: dict, wallet: str):
    score = data.get("risk_score", 0)
    level = data.get("risk_level", "UNKNOWN")
    prob  = data.get("probability", 0.0)
    tx    = data.get("tx_hash")

    level_map = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}
    emoji_map = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    css_class = level_map.get(level, "metric-box")
    emoji = emoji_map.get(level, "⚪")

    st.markdown(
        f"""
        <div class="{css_class}">
            <h2 style="margin:0">{emoji} Risk Level: <b>{level}</b></h2>
            <h3 style="margin:4px 0">Score: <b>{score} / 100</b></h3>
            <p style="margin:0">Phishing probability: <b>{prob:.2%}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(score / 100, text=f"Risk score: {score}")

    if tx:
        st.success(f"✅ Stored on-chain  |  tx hash: `{tx}`")
    else:
        st.info("ℹ️ Chain write skipped (CHAIN_ENABLED=false or node offline).")

    # ── Confusion-matrix image (if trained) ───────────────────────────────────
    cm_path = ROOT / "ml" / "artifacts" / "confusion_matrix.png"
    fi_path = ROOT / "ml" / "artifacts" / "feature_importance.png"

    if cm_path.exists() or fi_path.exists():
        st.divider()
        st.subheader("📊 Model Artifacts")
        col1, col2 = st.columns(2)
        if cm_path.exists():
            with col1:
                st.image(str(cm_path), caption="Confusion Matrix (test set)", width="stretch")
        if fi_path.exists():
            with col2:
                st.image(str(fi_path), caption="Feature Importance (Random Forest)", width="stretch")


def render_chain_record(wallet: str):
    """Fetch and show on-chain record."""
    try:
        resp = requests.get(f"{API_BASE}/risk/{wallet}", timeout=5)
        if resp.ok:
            rec = resp.json()
            if rec["timestamp"] == 0:
                st.info("No on-chain record found for this wallet.")
            else:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(rec["timestamp"]))
                st.markdown(
                    f"""
                    | Field | Value |
                    |---|---|
                    | Wallet | `{rec['wallet']}` |
                    | Risk Score | **{rec['risk_score']}** |
                    | Last Updated | {ts_str} |
                    | Model Version | `{rec['model_version']}` |
                    """
                )
        else:
            st.error(f"Chain lookup error: {resp.text}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Demo mode
# ──────────────────────────────────────────────────────────────────────────────
if mode.startswith("🎲"):
    st.subheader("🎲 Demo Mode — Test Set Row")
    st.info(
        "Select any row from `data/processed/test.csv`. "
        "The wallet address is the node ID from the dataset."
    )

    # Fetch row count
    try:
        n_rows_resp = requests.get(f"{API_BASE}/demo/rows", timeout=3)
        max_rows = n_rows_resp.json().get("total_rows", 100) if n_rows_resp.ok else 100
    except Exception:
        max_rows = 100

    col_a, col_b = st.columns([2, 1])
    with col_a:
        row_id = st.number_input(
            "Row ID (0-based)",
            min_value=0,
            max_value=max(0, max_rows - 1),
            value=0,
            step=1,
            help=f"Test set has {max_rows} rows",
        )
    with col_b:
        wallet = st.text_input("Wallet address", value="0xDemoWalletFromDataset000000000000000001")

    if st.button("🔍 Analyze", type="primary", key="demo_btn"):
        with st.spinner("Running ML inference…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"wallet": wallet, "row_id": int(row_id)},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    render_result(data, wallet)

                    # Optional: show true label
                    with st.expander("🔬 Raw ML response"):
                        st.json(data)
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    st.divider()
    st.subheader("🔗 On-chain Record")
    col1, col2 = st.columns([3, 1])
    with col1:
        lookup_wallet = st.text_input("Wallet to query (on-chain)", value=wallet, key="chain_lookup_demo")
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh from chain", key="refresh_demo"):
            render_chain_record(lookup_wallet)


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Manual mode
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.subheader("✍️ Manual Mode — Enter Wallet & Features")

    wallet = st.text_input("Wallet address (0x…)", value="0xAbCdef1234567890AbCdef1234567890AbCdef12")

    with st.expander("📐 Transaction Features", expanded=True):
        c1, c2, c3 = st.columns(3)
        in_degree      = c1.number_input("in_degree",      0, 10000, 5)
        out_degree     = c2.number_input("out_degree",     0, 10000, 2)
        unique_senders = c3.number_input("unique_senders", 0, 10000, 3)

        c4, c5, c6 = st.columns(3)
        unique_receivers = c4.number_input("unique_receivers", 0, 10000, 2)
        total_received   = c5.number_input("total_received",   0.0, 1e9, 1.5)
        total_sent       = c6.number_input("total_sent",       0.0, 1e9, 0.8)

        c7, c8 = st.columns(2)
        avg_in_amount  = c7.number_input("avg_in_amount",  0.0, 1e9, 0.3)
        avg_out_amount = c8.number_input("avg_out_amount", 0.0, 1e9, 0.4)

        c9, c10 = st.columns(2)
        max_in_amount  = c9.number_input("max_in_amount",  0.0, 1e9, 0.9)
        max_out_amount = c10.number_input("max_out_amount",0.0, 1e9, 0.5)

        c11, c12 = st.columns(2)
        min_in_amount  = c11.number_input("min_in_amount",  0.0, 1e9, 0.1)
        min_out_amount = c12.number_input("min_out_amount", 0.0, 1e9, 0.3)

        c13, c14 = st.columns(2)
        std_in_amount  = c13.number_input("std_in_amount",  0.0, 1e9, 0.25)
        std_out_amount = c14.number_input("std_out_amount", 0.0, 1e9, 0.1)

        c15, c16 = st.columns(2)
        in_out_ratio          = c15.number_input("in_out_ratio",          0.0, 1e6, 2.5)
        sent_received_ratio   = c16.number_input("sent_received_ratio",   0.0, 1e6, 0.53)

        c17, c18 = st.columns(2)
        timestamp_spread_in  = c17.number_input("timestamp_spread_in",  0.0, 1e9, 86400.0)
        timestamp_spread_out = c18.number_input("timestamp_spread_out", 0.0, 1e9, 3600.0)

    features = {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "unique_senders": unique_senders,
        "unique_receivers": unique_receivers,
        "total_received": total_received,
        "total_sent": total_sent,
        "avg_in_amount": avg_in_amount,
        "avg_out_amount": avg_out_amount,
        "max_in_amount": max_in_amount,
        "max_out_amount": max_out_amount,
        "min_in_amount": min_in_amount,
        "min_out_amount": min_out_amount,
        "std_in_amount": std_in_amount,
        "std_out_amount": std_out_amount,
        "in_out_ratio": in_out_ratio,
        "sent_received_ratio": sent_received_ratio,
        "timestamp_spread_in": timestamp_spread_in,
        "timestamp_spread_out": timestamp_spread_out,
    }

    if st.button("🔍 Analyze", type="primary", key="manual_btn"):
        with st.spinner("Running ML inference…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"wallet": wallet, "features": features},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    render_result(data, wallet)
                    with st.expander("🔬 Raw ML response"):
                        st.json(data)
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    st.divider()
    st.subheader("🔗 On-chain Record")
    col1, col2 = st.columns([3, 1])
    with col1:
        lookup_wallet = st.text_input("Wallet to query (on-chain)", value=wallet, key="chain_lookup_manual")
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh from chain", key="refresh_manual"):
            render_chain_record(lookup_wallet)
