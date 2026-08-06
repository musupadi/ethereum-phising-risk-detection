"""
ml/predict_live.py
------------------
Inferensi untuk MODEL LIVE (model B) — wallet DI LUAR dataset EPTransNet.

Alurnya:
    alamat  →  ml/live_features.extract_live_features()  (panggil Etherscan)
            →  model_live_xgb.json                        (XGBoost model B)
            →  skor risiko 0–100

Bedakan dengan `ml/predict.py`:
    predict.py       model tesis (F1 0,9102) untuk wallet DI DALAM dataset.
    predict_live.py  model perluasan untuk wallet arbitrer.

Hasil model live SELALU ditandai `out_of_distribution=True`: ia menilai wallet di
luar populasi yang dievaluasi pada tesis, sehingga metriknya tidak berlaku dan
skornya tidak boleh dicatat ke blockchain.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import LIVE_MODEL_JSON, LIVE_SCHEMA_JSON, add_import_paths  # noqa: E402

add_import_paths()
LIVE_MODEL = LIVE_MODEL_JSON
LIVE_SCHEMA = LIVE_SCHEMA_JSON

log = logging.getLogger(__name__)

_model = None
_schema: dict | None = None
_extra_models: list = []  # anggota ensemble selain XGBoost (bila ada)

# Di bawah jumlah transaksi ini, riwayat wallet terlalu tipis untuk dinilai secara
# bermakna: 18 fitur perilaku dihitung dari agregat, dan agregat atas 1–4 transaksi
# nyaris tidak membawa sinyal. Hasilnya tetap ditampilkan, tetapi ditandai supaya
# tidak dibaca sebagai tuduhan yang meyakinkan.
MIN_TX_FOR_CONFIDENCE = 5

# Tingkat keyakinan berdasarkan banyaknya transaksi.
#
# Dasarnya bukan tebakan: pengujian pada wallet bursa nyata menunjukkan kedua model
# sama-sama salah pada wallet bertransaksi sedikit. Contohnya Binance: Hot Wallet
# dengan 17 transaksi masuk dari 16 pengirim dan NOL transaksi keluar — secara
# struktur on-chain identik dengan wallet pengumpul phishing yang belum menyapu
# dana. Perbedaannya tidak ada pada statistik node, melainkan pada konteks
# jaringan (siapa lawan transaksinya), yang tidak dilihat oleh 26 fitur ini.
#
# Karena itu sistem menyatakan "belum dapat disimpulkan" alih-alih menuduh.
# CATATAN PENTING soal pemilihan ambang "boleh divonis".
#
# Percobaan menahan vonis untuk seluruh wallet < 30 transaksi memang menihilkan
# salah tuduh pada alamat bursa, TETAPI ongkosnya diukur dan ternyata tidak dapat
# diterima: hanya 27,5% wallet phishing yang masih divonis — 72,5% sisanya menjadi
# "belum dapat disimpulkan". Sistem deteksi yang bungkam pada tiga dari empat
# penipu tidak ada gunanya.
#
# Sebabnya mendasar: jumlah transaksi TIDAK memisahkan kedua kelas. Median wallet
# phishing hanya 13 transaksi — rentang yang sama persis dengan alamat bursa yang
# masih baru. Karena itu ambang berbasis jumlah transaksi tidak akan pernah
# menyelesaikan masalah ini, seberapa pun disetel.
#
# Keputusan: vonis hanya ditahan bila riwayatnya benar-benar tidak bermakna
# (< 5 transaksi). Untuk selebihnya vonis tetap diberikan, tetapi disertai label
# keyakinan supaya pengguna tahu seberapa tipis dasarnya.
# Perbaikan sesungguhnya ada pada FITUR, bukan ambang — lihat README §11.
CONFIDENCE_TIERS = [
    (MIN_TX_FOR_CONFIDENCE, "sangat rendah", False),
    (30, "rendah", True),
    (200, "sedang", True),
    (float("inf"), "tinggi", True),
]


def confidence_of(n_tx: float) -> tuple[str, bool]:
    """Kembalikan (label keyakinan, apakah hasilnya boleh dianggap kesimpulan)."""
    for limit, label, conclusive in CONFIDENCE_TIERS:
        if n_tx < limit:
            return label, conclusive
    return "tinggi", True


class LiveModelUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    return LIVE_MODEL.exists() and LIVE_SCHEMA.exists()


def _load() -> None:
    global _model, _schema
    if _model is not None:
        return
    if not is_available():
        raise LiveModelUnavailable(
            "Model live belum tersedia. Jalankan: "
            "python new_system/ml/build_live_dataset.py lalu "
            "python new_system/ml/train_live_model.py"
        )
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(LIVE_MODEL))
    _model = booster
    _schema = json.loads(LIVE_SCHEMA.read_text(encoding="utf-8"))

    # Model live bisa berupa XGBoost tunggal ATAU ensemble rata-rata probabilitas
    # (XGBoost + RandomForest + ExtraTrees). Anggota tambahan dimuat hanya bila
    # skema menyatakannya, sehingga model versi lama tetap bisa dilayani.
    global _extra_models
    _extra_models = []
    if _schema.get("model_type") == "ensemble":
        import joblib

        for name in ("random_forest", "extra_trees"):
            fname = _schema.get("ensemble_files", {}).get(name)
            if not fname:
                continue
            path = LIVE_MODEL.parent / fname
            if path.exists():
                _extra_models.append((name, joblib.load(path)))
            else:
                log.warning("Anggota ensemble '%s' tidak ditemukan di %s — "
                            "prediksi memakai anggota yang tersedia saja.", name, path)
        log.info("Model live (ensemble) dimuat: xgboost + %s | %d fitur | ambang %.2f",
                 ", ".join(n for n, _ in _extra_models), len(_schema["features"]),
                 float(_schema.get("threshold", 0.5)))
    else:
        log.info("Model live dimuat (%d fitur)", len(_schema["features"]))


def risk_level(score: int) -> str:
    if score < 40:
        return "LOW"
    if score < 70:
        return "MEDIUM"
    return "HIGH"


def predict_from_feature_dict(features: dict[str, Any], explain_top: int = 5) -> dict:
    """Jalankan model live pada dict fitur yang sudah dihitung."""
    _load()
    assert _schema is not None
    cols: list[str] = _schema["features"]
    threshold: float = float(_schema.get("threshold", 0.5))

    row = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=float)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

    import xgboost as xgb

    dmat = xgb.DMatrix(row, feature_names=cols)
    probs = [float(_model.predict(dmat)[0])]
    for _name, model in _extra_models:
        probs.append(float(model.predict_proba(row)[0, 1]))
    prob = float(np.mean(probs))  # ensemble = rata-rata probabilitas anggota
    score = round(prob * 100)

    n_tx = float(features.get("transaction_count", 0.0))
    conf_label, conclusive = confidence_of(n_tx)
    flagged = bool(prob >= threshold)

    result: dict[str, Any] = {
        "probability": prob,
        "risk_score": score,
        "risk_level": risk_level(score),
        "is_phishing": flagged,
        # Vonis hanya dinyatakan bila datanya memang cukup. Menahan diri di sini
        # jauh lebih baik daripada menuduh alamat bursa sebagai phishing.
        "verdict": ("Phishing" if flagged else "Bukan Phishing") if conclusive
                   else "Belum dapat disimpulkan",
        "confidence": conf_label,
        "conclusive": conclusive,
        "model": "live",
        "out_of_distribution": True,
        "n_ensemble_members": len(probs),
    }
    if not conclusive:
        result["inconclusive_reason"] = (
            f"Hanya {int(n_tx)} transaksi terbaca. Ke-26 fitur adalah agregat "
            f"statistik, sehingga riwayat sependek ini tidak membawa cukup sinyal."
        )
    elif conf_label == "rendah" and flagged:
        # Kasus paling rawan salah tuduh: riwayat tipis + dituduh phishing.
        result["caution"] = (
            f"Keyakinan rendah — hanya {int(n_tx)} transaksi. Pada rentang ini "
            f"alamat deposit bursa yang masih baru secara struktur mirip wallet "
            f"pengumpul phishing: sama-sama menerima dari banyak pihak dan belum "
            f"mengirim keluar. Verifikasi lewat label Etherscan sebelum menyimpulkan."
        )

    if explain_top:
        # Catatan kejujuran: RandomForest dan ExtraTrees tidak menyediakan nilai
        # SHAP bawaan, sehingga penjelasan di bawah berasal dari komponen XGBoost
        # saja — arah dan urutan fiturnya indikatif, bukan dekomposisi persis dari
        # probabilitas ensemble.
        contribs = _model.predict(dmat, pred_contribs=True)[0]
        pairs = [
            {"feature": cols[i], "value": float(row[0, i]), "contribution": float(contribs[i])}
            for i in range(len(cols))
        ]
        pairs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        result["top_contributors"] = pairs[:explain_top]

    return result


def predict_address(address: str, api_key: str | None = None,
                    delay: float = 0.25, max_pages: int = 5,
                    explain_top: int = 5, short_circuit: bool = False) -> dict:
    """Ambil fitur dari Etherscan lalu skor wallet-nya. Butuh koneksi internet."""
    import label_registry  # noqa: E402

    # Alur keputusan (urutannya disengaja):
    #
    #   1. MODEL   hitung 26 fitur dari transaksi & lawan transaksi -> skor
    #   2. LABEL   cek LabelRegistry on-chain, lalu daftar lokal
    #   3. PUTUS   bila ada label terverifikasi, label menang atas model
    #
    # Label ditaruh di AKHIR, bukan di awal, supaya pendapat model tetap terekam
    # walau kemudian ditimpa. Ketidaksepakatan model vs label justru catatan paling
    # berharga: ia menunjukkan persis di mana model lemah, dan bisa dipakai sebagai
    # bahan pelatihan berikutnya.
    #
    # `short_circuit=True` melewati langkah 1 untuk alamat yang sudah berlabel —
    # menghemat panggilan Etherscan bila kecepatan lebih dipentingkan daripada
    # catatan ketidaksepakatan.
    entry = label_registry.lookup(address)
    if entry is not None and short_circuit:
        result = label_registry.as_result(address, entry)
        result["n_transactions"] = None
        result["insufficient_data"] = False
        result["model_consulted"] = False
        return result

    import live_features as _lf  # noqa: E402

    # Pengaman terhadap tabrakan nama: sistem tesis punya modul `live_features`
    # sendiri. Bila yang ter-import ternyata modul itu, hentikan dengan pesan
    # jelas alih-alih gagal dengan ImportError yang membingungkan.
    if getattr(_lf, "MODULE_VARIANT", None) != "new_system":
        raise LiveModelUnavailable(
            f"Modul live_features yang ter-import bukan milik new_system "
            f"(dimuat dari {getattr(_lf, '__file__', '?')}). "
            f"Panggil paths.add_import_paths() lebih dulu."
        )
    extract_live_features = _lf.extract_live_features

    feats = extract_live_features(address, api_key=api_key, delay=delay,
                                  max_pages=max_pages)
    result = predict_from_feature_dict(feats, explain_top=explain_top)
    result["decision_source"] = "ml_model"
    n_tx = int(feats.get("_n_total_tx", 0))
    result["features"] = {k: v for k, v in feats.items() if not k.startswith("_")}
    result["n_transactions"] = n_tx
    result["truncated"] = bool(feats.get("_truncated", 0))
    result["is_contract"] = bool(feats.get("is_contract", 0))
    result["insufficient_data"] = n_tx < MIN_TX_FOR_CONFIDENCE
    result["model_consulted"] = True

    # ── Lapisan terakhir: label menimpa model ────────────────────────────────
    if entry is None:
        result["label_override"] = False
        return result

    model_opinion = {
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "verdict": result["verdict"],
        "is_phishing": result["is_phishing"],
        "probability": result["probability"],
        "confidence": result.get("confidence"),
    }
    labeled = label_registry.as_result(address, entry)
    disagreement = bool(model_opinion["is_phishing"] != labeled["is_phishing"])

    # Label yang menang, tetapi pendapat model tetap dibawa serta — inilah yang
    # membuat ketidaksepakatan bisa dihitung, ditampilkan, dan ditindaklanjuti.
    result.update(labeled)
    result.update({
        "label_override": True,
        "model_opinion": model_opinion,
        "disagreement": disagreement,
        "n_transactions": n_tx,
        "insufficient_data": n_tx < MIN_TX_FOR_CONFIDENCE,
        "label_layer": "onchain" if entry.get("onchain") else "lokal",
    })
    if disagreement:
        result["disagreement_note"] = (
            f"Model menilai «{model_opinion['verdict']}» dengan skor "
            f"{model_opinion['risk_score']}, tetapi alamat ini terdaftar sebagai "
            f"«{labeled['label_name']}». Label yang dipakai. Ketidaksepakatan ini "
            f"dicatat sebagai bahan perbaikan model."
        )
    return result


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    targets = sys.argv[1:] or ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"]
    for t in targets:
        r = predict_address(t, explain_top=3)
        print(f"\n{t}")
        print(f"  skor {r['risk_score']} ({r['risk_level']}) -> {r['verdict']}  "
              f"| tx={r['n_transactions']} kontrak={r['is_contract']}")
        for c in r["top_contributors"]:
            print(f"    {c['feature']:24s} {c['value']:14.4f}  {c['contribution']:+.3f}")
