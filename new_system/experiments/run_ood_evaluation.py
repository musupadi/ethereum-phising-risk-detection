"""
experiments/run_ood_evaluation.py
---------------------------------
Evaluasi OUT-OF-DISTRIBUTION: membandingkan dua model pada wallet nyata yang
BERADA DI LUAR dataset EPTransNet.

  Model A  ml/artifacts/model_prod_xgb.json   model tesis (26 fitur, F1 0,9102)
  Model B  ml/artifacts/model_live_xgb.json   model perluasan (fitur live + hard negatives)

Alamat uji adalah alamat sah yang identitasnya diketahui publik (tokoh, router DEX,
kontrak resmi). Semuanya SEHARUSNYA dinilai "Bukan Phishing". Model A gagal pada
uji ini — itulah yang melatarbelakangi dibangunnya Model B.

Keluaran:
  outputs/ood_evaluation.csv
  outputs/ood_evaluation.json
  outputs/ood_evaluation_report.md

Jalankan:  python experiments/run_ood_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from paths import NEW_OUTPUTS, THESIS_ROOT, add_import_paths  # noqa: E402

add_import_paths()
ROOT = THESIS_ROOT

from live_features import extract_live_features  # noqa: E402
from ml.predict import predict_full as predict_thesis  # noqa: E402

OUTPUTS = NEW_OUTPUTS

# Alamat sah yang identitasnya publik — tidak satu pun boleh dinilai phishing.
LEGIT_ADDRESSES = [
    ("0xd8da6bf26964af9d7eed9e03e53415d37aa96045", "Vitalik Buterin"),
    ("0x7a250d5630b4cf539739df2c5dacb4c659f2488d", "Uniswap V2 Router"),
    ("0x00000000219ab540356cbb839cbe05303d7705fa", "ETH2 Deposit Contract"),
    ("0xab5801a7d398351b8be11c439e05c5b3259aec9b", "Vitalik (dompet lama)"),
    ("0x5041ed759dd4afc3a72b8192c143f72f4724081a", "OKX Exchange"),
    ("0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", "Token UNI"),
    ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "Token USDC"),
    ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "WETH"),
]

# Wallet phishing dari dataset — dipakai sebagai kontrol positif: kedua model
# tetap harus menandainya sebagai phishing.
PHISHING_CONTROL = [
    ("0x60afc5761212ed59608b0559b4ccbaad24fc74b1", "Phishing dataset #1"),
    ("0x903bb9cd3a276d8f18fa6efed49b9bc52ccf06e5", "Phishing dataset #2"),
]


def score_thesis_model(live: dict) -> dict:
    """
    Nilai memakai Model A. Fitur Etherscan milik Model A yang tidak tersedia
    di jalur live diisi nol — persis seperti perilaku aplikasi sebelum Model B ada.
    """
    feats = {k: v for k, v in live.items() if not k.startswith("_")}
    feats["graph_present"] = 0.0
    feats.setdefault("activity_score", min(100.0, live.get("transaction_count", 0.0)))
    feats.setdefault("avg_tx_value", (live.get("avg_in_amount", 0.0) + live.get("avg_out_amount", 0.0)) / 2.0)
    feats.setdefault("tx_frequency", min(100.0, live.get("transaction_count", 0.0)) / 30.0)
    feats.setdefault("created_at_ts", 0.0)
    return predict_thesis(feats)


def main() -> None:
    load_dotenv()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    try:
        import predict_live
        model_b_ready = predict_live.is_available()
    except Exception:  # noqa: BLE001
        model_b_ready = False
    if not model_b_ready:
        print("PERINGATAN: Model B belum ada — hanya Model A yang dievaluasi.\n"
              "Latih dulu: python new_system/ml/train_live_model.py")

    rows = []
    cases = [(a, n, 0) for a, n in LEGIT_ADDRESSES] + [(a, n, 1) for a, n in PHISHING_CONTROL]

    for addr, name, expected in cases:
        print(f"  mengambil {name} ...", flush=True)
        live = extract_live_features(addr, delay=0.32, max_pages=5)

        a_res = score_thesis_model(live)
        row = {
            "address": addr,
            "name": name,
            "expected_label": expected,
            "n_tx": int(live.get("_n_total_tx", 0)),
            "is_contract": int(live.get("is_contract", 0)),
            "modelA_score": a_res["risk_score"],
            "modelA_level": a_res["risk_level"],
            "modelA_verdict": a_res["verdict"],
            "modelA_correct": int(bool(a_res["is_phishing"]) == bool(expected)),
        }

        if model_b_ready:
            b_res = predict_live.predict_from_feature_dict(live, explain_top=0)
            row.update({
                "modelB_score": b_res["risk_score"],
                "modelB_level": b_res["risk_level"],
                "modelB_verdict": b_res["verdict"],
                "modelB_correct": int(bool(b_res["is_phishing"]) == bool(expected)),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "ood_evaluation.csv", index=False)

    summary = {
        "n_cases": int(len(df)),
        "n_legit": int((df["expected_label"] == 0).sum()),
        "n_phishing_control": int((df["expected_label"] == 1).sum()),
        "modelA_correct": int(df["modelA_correct"].sum()),
        "modelA_accuracy": float(df["modelA_correct"].mean()),
    }
    if model_b_ready:
        summary["modelB_correct"] = int(df["modelB_correct"].sum())
        summary["modelB_accuracy"] = float(df["modelB_correct"].mean())
        legit = df[df["expected_label"] == 0]
        summary["modelA_false_positives_on_legit"] = int((legit["modelA_correct"] == 0).sum())
        summary["modelB_false_positives_on_legit"] = int((legit["modelB_correct"] == 0).sum())
    (OUTPUTS / "ood_evaluation.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")

    # ── Laporan markdown ──────────────────────────────────────────────────────
    lines = [
        "# Evaluasi Out-of-Distribution: Model Tesis vs Model Perluasan", "",
        "Alamat di bawah ini **tidak ada** di dataset EPTransNet. Delapan pertama",
        "adalah alamat sah yang identitasnya publik dan seharusnya dinilai",
        "*Bukan Phishing*; dua terakhir adalah kontrol positif dari dataset.", "",
        "| Alamat | Identitas | Seharusnya | Model A (tesis) | Model B (perluasan) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        exp = "Phishing" if r["expected_label"] else "Bukan Phishing"
        a = f"{r['modelA_score']} {r['modelA_level']} {'✅' if r['modelA_correct'] else '❌'}"
        b = (f"{r['modelB_score']} {r['modelB_level']} {'✅' if r['modelB_correct'] else '❌'}"
             if model_b_ready else "—")
        lines.append(f"| `{r['address'][:10]}…` | {r['name']} | {exp} | {a} | {b} |")
    lines += ["", "## Ringkasan", "", "```json", json.dumps(summary, indent=2), "```"]
    (OUTPUTS / "ood_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + df.to_string(index=False))
    print("\nRingkasan:", json.dumps(summary, indent=2))
    print(f"\nTersimpan -> {OUTPUTS / 'ood_evaluation_report.md'}")


if __name__ == "__main__":
    main()
