"""
app/streamlit_app.py
--------------------
Streamlit demo UI for the Ethereum Phishing Risk Registry.

Single mode:
  Scan by Address – paste a dataset wallet; features are looked up server-side
  and scored by the production XGBoost model.

Calls the FastAPI backend at http://localhost:8000.
"""

import os
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
    verdict = data.get("verdict") or ("Phishing" if score >= 50 else "Bukan Phishing")
    is_phishing = data.get("is_phishing", score >= 50)

    # ── Keputusan biner tegas (Phishing / Bukan Phishing) ─────────────────────
    if is_phishing:
        st.error(f"### 🚨 {verdict}")
    else:
        st.success(f"### ✅ {verdict}")

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

    # ── Penjelasan asal skor: kontribusi fitur (SHAP) ─────────────────────────
    contribs = data.get("top_contributors")
    if contribs:
        st.markdown("**Mengapa skor ini?** Fitur paling berpengaruh (nilai SHAP):")
        cdf = pd.DataFrame(contribs)
        cdf["arah"] = cdf["contribution"].apply(lambda c: "⬆️ menaikkan risiko" if c > 0 else "⬇️ menurunkan risiko")
        cdf = cdf.rename(columns={"feature": "Fitur", "value": "Nilai", "contribution": "Kontribusi", "arah": "Arah"})
        st.dataframe(cdf[["Fitur", "Nilai", "Kontribusi", "Arah"]], hide_index=True, width="stretch")

    if tx:
        st.success(f"✅ Stored on-chain  |  tx hash: `{tx}`")
    elif data.get("out_of_distribution"):
        st.info(
            "ℹ️ Skor ini **sengaja tidak dicatat ke blockchain**. Wallet berada di luar "
            "dataset penelitian, sehingga hasilnya berupa estimasi — catatan permanen "
            "on-chain hanya diisi penilaian yang tervalidasi."
        )
    else:
        st.info("ℹ️ Chain write skipped (CHAIN_ENABLED=false or node offline).")

    # ── Model artifacts — SERVED model (XGBoost production) ────────────────────
    # Tampilkan gambar dari model yang benar-benar di-serve (XGBoost, 26 fitur),
    # bukan baseline Random Forest. Fallback ke RF hanya jika artefak prod hilang.
    cm_path = ROOT / "ml" / "artifacts" / "prod_cm_xgb.png"
    fi_path = ROOT / "ml" / "artifacts" / "prod_feature_importance_xgb.png"
    cm_caption = "Confusion Matrix — XGBoost (production, held-out set)"
    fi_caption = "Feature Importance — XGBoost (production, 26 fitur)"
    if not cm_path.exists():
        cm_path = ROOT / "ml" / "artifacts" / "confusion_matrix.png"
        cm_caption = "Confusion Matrix (RF baseline — fallback)"
    if not fi_path.exists():
        fi_path = ROOT / "ml" / "artifacts" / "feature_importance.png"
        fi_caption = "Feature Importance (RF baseline — fallback)"

    if cm_path.exists() or fi_path.exists():
        st.divider()
        st.subheader("📊 Model Artifacts")
        col1, col2 = st.columns(2)
        if cm_path.exists():
            with col1:
                st.image(str(cm_path), caption=cm_caption, width="stretch")
        if fi_path.exists():
            with col2:
                st.image(str(fi_path), caption=fi_caption, width="stretch")


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
# Scan by Address (paste wallet → lookup features in dataset → predict)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🔎 Scan by Address — Tempel Wallet")
st.caption(
    "**Dua jalur penilaian.** Wallet yang ada di dataset penelitian (8.595 alamat, "
    "snapshot EPTransNet 2019) dinilai oleh **model tesis** — XGBoost 26 fitur, F1 91,02%. "
    "Wallet di luar itu dinilai oleh **model perluasan**: fitur diambil langsung dari "
    "Etherscan saat itu juga, hasilnya berupa estimasi dan tidak dicatat ke blockchain."
)
# st.info(
#     "Tempel alamat wallet. Sistem mencari fitur wallet itu di dataset "
#     "MulDiGraph (train + test, ~12.815 wallet), lalu memprediksi phishing/tidak. "
#     "Hanya alamat yang ada di dataset (snapshot 2019) yang bisa di-scan."
# )
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
                # Mode live memanggil Etherscan berkali-kali, jadi butuh waktu lebih.
                timeout=180,
            )
            if resp.ok:
                data = resp.json()

                # Wallet di luar dataset -> dinilai model perluasan (mode live).
                # Peringatannya ditampilkan SEBELUM skor, supaya tidak terlewat.
                if data.get("insufficient_data"):
                    st.error(
                        "🚫 **Tidak dapat disimpulkan — riwayat terlalu tipis**\n\n"
                        f"Wallet ini hanya punya {data.get('n_transactions', 0)} transaksi. "
                        "Ke-18 fitur perilaku dihitung dari agregat transaksi, sehingga "
                        "wallet sebaru ini tidak membawa cukup sinyal untuk dinilai. "
                        "Skor di bawah ditampilkan apa adanya, tetapi **jangan** dibaca "
                        "sebagai kesimpulan."
                    )
                elif data.get("out_of_distribution"):
                    st.warning(
                        "⚠️ **Wallet di luar dataset penelitian — mode estimasi**\n\n"
                        f"{data.get('warning', '')}\n\n"
                        f"Fitur diambil langsung dari Etherscan"
                        + (f" ({data['n_transactions']:,} transaksi terbaca)."
                           if data.get("n_transactions") else ".")
                    )

                render_result(data, scan_wallet)

                if data.get("model_used") == "live":
                    st.caption(
                        "Model yang dipakai: **model perluasan (live)** — bukan model tesis "
                        "XGBoost F1 91,02%. Metrik tesis tidak berlaku untuk hasil ini, "
                        "dan skornya tidak dicatat ke blockchain."
                    )
                elif data.get("model_used") == "dataset":
                    st.caption(
                        "Model yang dipakai: **model tesis** — XGBoost, 26 fitur "
                        "(MulDiGraph + Etherscan), F1 91,02%."
                    )

                tl = data.get("true_label")
                if tl is not None and tl >= 0:
                    truth = "🔴 PHISHING" if tl == 1 else "🟢 NORMAL"
                    flagged = data.get("is_phishing", data.get("risk_score", 0) >= 50)
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
# On-chain records list (immutable audit trail)
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
