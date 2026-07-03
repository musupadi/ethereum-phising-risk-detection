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
import re
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


def validate_eth_address(addr: str):
    """Return an error message if `addr` is not a valid Ethereum address, else None."""
    addr = (addr or "").strip()
    if not addr:
        return "Alamat masih kosong."
    if not addr.startswith("0x"):
        return "Alamat harus diawali `0x`."
    hexpart = addr[2:]
    bad = sorted({c for c in hexpart if c.lower() not in "0123456789abcdef"})
    if bad:
        return f"Ada karakter tidak valid: `{'`, `'.join(bad)}` (hanya boleh 0–9 dan a–f)."
    n = len(addr)
    if n < 42:
        return f"Alamat **kurang {42 - n} karakter** (sekarang {n}, harus 42)."
    if n > 42:
        return f"Alamat **kelebihan {n - 42} karakter** (sekarang {n}, harus 42)."
    return None

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
    mode = st.radio("Input mode", [
        "🎲 Demo (test.csv row)",
        "🔎 Scan by Address (dataset)",
        "✍️ Manual (enter wallet + features)",
    ])
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

    row_id = st.number_input(
        "Row ID (0-based)",
        min_value=0,
        max_value=max(0, max_rows - 1),
        value=0,
        step=1,
        help=f"Demo set has {max_rows} held-out wallets",
    )
    st.caption("ℹ️ Alamat wallet diambil **otomatis dari baris terpilih** — skor adalah "
               "prediksi untuk wallet asli dataset itu, bukan alamat yang diketik.")

    wallet = ""  # filled from the API response (real dataset wallet) after Analyze

    if st.button("🔍 Analyze", type="primary", key="demo_btn"):
        with st.spinner("Running ML inference…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"wallet": "0x0000000000000000000000000000000000000000",
                          "row_id": int(row_id)},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    wallet = data.get("wallet", "")
                    render_result(data, wallet)
                    if data.get("tx_hash"):
                        st.divider()
                        st.subheader("🔗 Tersimpan di smart contract")
                        render_chain_record(wallet)
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
# Mode: Scan by Address (paste wallet → lookup features in dataset → predict)
# ──────────────────────────────────────────────────────────────────────────────
elif mode.startswith("🔎"):
    st.subheader("🔎 Scan by Address — Tempel Wallet")
    st.info(
        "Tempel alamat wallet. Sistem mencari fitur wallet itu di dataset "
        "MulDiGraph (train + test, ~12.815 wallet), lalu memprediksi phishing/tidak. "
        "Alamat di luar dataset → gunakan mode Manual."
    )
    # Quick-pick examples (avoid copy/paste truncation). Choosing one overrides the text box.
    EXAMPLES = {
        "— ketik / tempel alamat sendiri —": "",
        "🔴 HIGH — Phishing (≈100)": "0x4dad23689bef624af5d1757a56135901c915fcfd",
        "🔴 HIGH — Phishing (≈81)": "0x8c155c3f4c79485dbb0eef5b4076fd426b3f8c65",
        "🟡 MEDIUM — Phishing ragu (≈62)": "0x35919ff0a88a0bdfd1b0ef3be574bcf6c1e90e01",
        "🟡 MEDIUM — Normal ragu (≈64)": "0x5141ffe7d645818996515e7336174a614def6380",
        "🟢 LOW — Normal (≈25)": "0x2a7cf914a7825bb067703042eff254d54963234c",
        "🟢 LOW — Normal (≈12)": "0x16bccf9ea461adbc202d5f0d7f740b43e2da514c",
    }
    pick = st.selectbox("Pilih contoh wallet (atau ketik manual di bawah)", list(EXAMPLES))
    typed = st.text_input(
        "Wallet address (0x…)",
        value="0x766316f65c88486bcb59fd57d5ce0087780c1424",
        key="scan_wallet",
        help="Harus 42 karakter (0x + 40 hex). Pastikan tidak terpotong saat menempel.",
    )
    # Selected example takes priority; otherwise use the typed address.
    scan_wallet = (EXAMPLES[pick] or typed).strip()
    if EXAMPLES[pick]:
        st.caption(f"Akan men-scan: `{scan_wallet}`")

    # Real-time address validation feedback.
    addr_err = validate_eth_address(scan_wallet)
    if addr_err:
        st.error(f"⚠️ {addr_err}")
    else:
        st.success("✅ Format alamat Ethereum valid (42 karakter).")

    if st.button("🔎 Scan", type="primary", key="scan_btn", disabled=bool(addr_err)):
        with st.spinner("Mencari fitur wallet & menjalankan inferensi…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"wallet": scan_wallet},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    render_result(data, scan_wallet)
                    tl = data.get("true_label")
                    if tl is not None and tl >= 0:
                        truth = "🔴 PHISHING" if tl == 1 else "🟢 NORMAL"
                        flagged = data.get("risk_score", 0) >= 40  # MEDIUM+ dianggap terdeteksi
                        match = (tl == 1) == flagged
                        st.caption(
                            f"Label asli dataset (`{data.get('dataset_split','')}`): **{truth}** — "
                            f"prediksi model {'cocok ✅' if match else 'berbeda ⚠️'}"
                        )
                    # Auto-tampilkan record yang baru tersimpan di blockchain
                    if data.get("tx_hash"):
                        st.divider()
                        st.subheader("🔗 Tersimpan di smart contract")
                        render_chain_record(data.get("wallet", scan_wallet))
                    with st.expander("🔬 Raw ML response"):
                        st.json(data)
                elif resp.status_code == 404:
                    st.warning(resp.json().get("detail", "Wallet tidak ditemukan di dataset."))
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    st.divider()
    st.subheader("🔗 On-chain Record")
    col1, col2 = st.columns([3, 1])
    with col1:
        lookup_wallet = st.text_input("Wallet to query (on-chain)", value=scan_wallet, key="chain_lookup_scan")
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh from chain", key="refresh_scan"):
            render_chain_record(lookup_wallet)


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Manual mode
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.subheader("✍️ Manual Mode — Enter Wallet & Features")

    wallet = st.text_input("Wallet address (0x…)", value="0xAbCdef1234567890AbCdef1234567890AbCdef12")
    _manual_err = validate_eth_address(wallet)
    if _manual_err:
        st.error(f"⚠️ {_manual_err}")

    st.caption("💡 Default di bawah dari profil **wallet normal nyata** (hasil ≈ LOW). "
               "Mode manual hanya mengisi 18 fitur graf (fitur Etherscan = 0); untuk "
               "skoring 26-fitur penuh pakai **Scan by Address**.")
    with st.expander("📐 Transaction Features", expanded=True):
        c1, c2, c3 = st.columns(3)
        in_degree      = c1.number_input("in_degree",      0, 10000, 3)
        out_degree     = c2.number_input("out_degree",     0, 10000, 4)
        unique_senders = c3.number_input("unique_senders", 0, 10000, 1)

        c4, c5, c6 = st.columns(3)
        unique_receivers = c4.number_input("unique_receivers", 0, 10000, 2)
        total_received   = c5.number_input("total_received",   0.0, 1e9, 108.7055)
        total_sent       = c6.number_input("total_sent",       0.0, 1e9, 0.1)

        c7, c8 = st.columns(2)
        avg_in_amount  = c7.number_input("avg_in_amount",  0.0, 1e9, 36.2352)
        avg_out_amount = c8.number_input("avg_out_amount", 0.0, 1e9, 0.025)

        c9, c10 = st.columns(2)
        max_in_amount  = c9.number_input("max_in_amount",  0.0, 1e9, 94.6856)
        max_out_amount = c10.number_input("max_out_amount",0.0, 1e9, 0.1)

        c11, c12 = st.columns(2)
        min_in_amount  = c11.number_input("min_in_amount",  0.0, 1e9, 7.0044)
        min_out_amount = c12.number_input("min_out_amount", 0.0, 1e9, 0.0)

        c13, c14 = st.columns(2)
        std_in_amount  = c13.number_input("std_in_amount",  0.0, 1e9, 41.3307)
        std_out_amount = c14.number_input("std_out_amount", 0.0, 1e9, 0.0433)

        c15, c16 = st.columns(2)
        in_out_ratio          = c15.number_input("in_out_ratio",          0.0, 1e6, 0.6)
        sent_received_ratio   = c16.number_input("sent_received_ratio",   0.0, 1e6, 0.0009)

        c17, c18 = st.columns(2)
        timestamp_spread_in  = c17.number_input("timestamp_spread_in",  0.0, 1e9, 283.0)
        timestamp_spread_out = c18.number_input("timestamp_spread_out", 0.0, 1e9, 12727162.0)

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
        # Wallet has graph features present; Etherscan features default to 0
        # (manual mode is graph-only — for full 26-feature scoring use Scan by Address).
        "graph_present": 1,
    }

    if st.button("🔍 Analyze", type="primary", key="manual_btn", disabled=bool(_manual_err)):
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


# ──────────────────────────────────────────────────────────────────────────────
# On-chain records list (immutable audit trail) — shown in every mode
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📜 Riwayat On-chain — semua wallet yang tercatat di smart contract"):
    if st.button("🔄 Muat / refresh daftar", key="load_records"):
        st.session_state["_show_records"] = True
    if st.session_state.get("_show_records"):
        try:
            resp = requests.get(f"{API_BASE}/records", timeout=10)
            if resp.ok:
                payload = resp.json()
                recs = payload.get("records", [])
                st.caption(f"Total wallet tercatat: **{payload.get('count', 0)}**")
                if recs:
                    rdf = pd.DataFrame(recs)
                    rdf["risk_level"] = rdf["risk_score"].apply(
                        lambda s: "🔴 HIGH" if s >= 70 else ("🟡 MEDIUM" if s >= 40 else "🟢 LOW")
                    )
                    rdf["waktu (UTC)"] = rdf["timestamp"].apply(
                        lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
                    )
                    st.dataframe(
                        rdf[["wallet", "risk_score", "risk_level", "model_version", "waktu (UTC)"]],
                        width="stretch", hide_index=True,
                    )
                else:
                    st.info("Belum ada record on-chain. Lakukan Scan/Analyze dulu.")
            else:
                st.error(f"Gagal memuat daftar: {resp.status_code} {resp.text}")
        except Exception as exc:
            st.error(f"Gagal memuat daftar: {exc}")
