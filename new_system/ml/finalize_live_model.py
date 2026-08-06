"""
ml/finalize_live_model.py
-------------------------
Menjalankan seluruh rangkaian finalisasi MODEL LIVE dalam satu perintah:

  1. Ringkasan komposisi dataset (memastikan sebaran ukuran wallet seimbang)
  2. Melatih model  ............................ ml/train_live_model.py
  3. Gerbang mutu out-of-distribution ......... 8 alamat sah held-out harus LOW
  4. Uji regresi .............................. wallet dataset tetap dinilai benar
  5. Ringkasan akhir + status LULUS/GAGAL

Uji tambahan yang memanggil Etherscan (butuh waktu) dijalankan terpisah:
  python new_system/experiments/run_false_positive_test.py
  python new_system/experiments/run_ood_evaluation.py

Jalankan:  python ml/finalize_live_model.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Konsol Windows memakai cp1252 dan akan melempar UnicodeEncodeError begitu
# bertemu karakter seperti "≈" atau em dash. Paksa keluaran ke UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:  # Python < 3.7
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    LIVE_DATASET_CSV, LIVE_REPORT_JSON, NEW_ML, THESIS_ROOT, add_import_paths,
)

add_import_paths()

PY = sys.executable
ROOT = THESIS_ROOT
LIVE_CSV = LIVE_DATASET_CSV
REPORT = LIVE_REPORT_JSON


def header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def main() -> None:
    # ── 1. Komposisi dataset ──────────────────────────────────────────────────
    header("1. KOMPOSISI DATASET")
    df = pd.read_csv(LIVE_CSV)
    print(f"Total baris     : {len(df)}")
    print(f"Label           : {df['label'].value_counts().to_dict()}")
    print(f"Sumber          : {df['source'].str.split(':').str[0].value_counts().to_dict()}")
    print("\nSebaran jumlah transaksi per kelas:")
    print(df.groupby("label")["n_tx"].describe()[["count", "25%", "50%", "75%"]].round(0).to_string())

    small = df[df["n_tx"] <= 60]
    pos_small = float((df[df.label == 1]["n_tx"] <= 60).mean())
    neg_small = float((df[df.label == 0]["n_tx"] <= 60).mean())
    print(f"\nProporsi wallet kecil (<=60 tx): positif {pos_small:.0%} | negatif {neg_small:.0%}")
    gap = abs(pos_small - neg_small)
    print(f"Selisih {gap:.0%} — {'seimbang' if gap < 0.25 else 'MASIH TIMPANG, model bisa memakai ukuran sebagai jalan pintas'}")

    # ── 2. Latih ──────────────────────────────────────────────────────────────
    header("2. PELATIHAN MODEL")
    res = subprocess.run([PY, "-u", str(NEW_ML / "train_live_model.py")],
                         cwd=str(THESIS_ROOT), capture_output=True, text=True)
    for line in (res.stderr or "").splitlines():
        if any(k in line for k in ("Dataset live", "Held-out", "Train ", "Test:",
                                   "Gerbang", "LULUS", "GAGAL", "Fitur paling")):
            print("  " + line.split("INFO")[-1].strip())
    if res.returncode != 0:
        print("PELATIHAN GAGAL:\n", res.stderr[-2000:])
        sys.exit(1)

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    m = report["test_metrics"]

    # ── 3. Uji regresi model tesis ────────────────────────────────────────────
    header("3. UJI REGRESI — MODEL TESIS TIDAK BOLEH BERUBAH")
    sys.path.insert(0, str(ROOT))
    # Sengaja lewat ml_service, bukan predict_full langsung: inilah jalur yang
    # benar-benar dipakai API, sehingga regresinya teruji ujung ke ujung
    # (termasuk penyaringan kolom meta dan pemilihan model).
    from api.services import ml_service  # noqa: E402

    checks = [("0x60afc5761212ed59608b0559b4ccbaad24fc74b1", 1, 100),
              ("0x903bb9cd3a276d8f18fa6efed49b9bc52ccf06e5", 1, 98)]
    regression_ok = True
    for addr, label, expected in checks:
        res = ml_service.predict_from_address(addr)
        got = res["risk_score"]
        ok = abs(got - expected) <= 1 and res.get("model_used") == "dataset"
        regression_ok &= ok
        print(f"  {addr[:12]}... label={label} diharapkan={expected} didapat={got} "
              f"model={res.get('model_used')}  {'OK' if ok else 'BERUBAH!'}")

    # ── 4. Ringkasan ──────────────────────────────────────────────────────────
    header("4. RINGKASAN")
    gate = report["ood_gate_passed"]
    print(f"Model live  F1={m['f1']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  Acc={m['accuracy']:.4f}")
    print(f"Confusion   TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    print(f"\nGerbang OOD (8 alamat sah held-out) : {'LULUS' if gate else 'GAGAL'}")
    for r in report["ood_gate"]:
        print(f"    {r['name'][:26]:26s} skor {r['score']:3d}  {r['level']:6s}  "
              f"{'OK' if r['pass'] else 'GAGAL'}")
    print(f"\nRegresi model tesis                 : {'AMAN' if regression_ok else 'BERUBAH'}")
    print(f"Keseimbangan ukuran wallet          : {'OK' if gap < 0.25 else 'TIMPANG'}")

    all_ok = gate and regression_ok
    print("\n" + ("SEMUA PEMERIKSAAN LULUS" if all_ok else "ADA PEMERIKSAAN YANG GAGAL"))
    print("\nLangkah berikutnya (memanggil Etherscan, beberapa menit):")
    print("  python new_system/experiments/run_false_positive_test.py 20")
    print("  python new_system/experiments/run_ood_evaluation.py")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
