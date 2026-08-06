"""
experiments/run_false_positive_test.py
--------------------------------------
Mengukur tingkat FALSE POSITIVE pada wallet Ethereum acak yang benar-benar segar —
alamat yang tidak pernah dipakai melatih model mana pun.

Kenapa perlu: metrik F1 dihitung pada test split yang berasal dari populasi yang
sama dengan data latih. Ia tidak menjawab pertanyaan yang sesungguhnya penting bagi
pengguna: *"kalau saya masukkan wallet sembarangan, seberapa sering sistem salah
menuduh?"* Skrip ini menjawab itu secara langsung.

Cara kerja: mengambil alamat dari blok Ethereum acak (seed berbeda dari yang dipakai
membangun data latih), menyaring yang sudah pernah dilatih, lalu menilainya dengan
kedua model.

Asumsi yang harus diakui: alamat acak dianggap BUKAN phishing. Ini asumsi, bukan
label terverifikasi — proporsi wallet phishing di antara alamat acak sangat kecil
(1.165 dari ~3 juta node pada EPTransNet), sehingga sebagian besar penilaian
"Phishing" pada sampel ini wajar dibaca sebagai false positive.

Keluaran:
  outputs/false_positive_test.csv
  outputs/false_positive_test.json

Jalankan:  python experiments/run_false_positive_test.py [jumlah_sampel]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from paths import NEW_OUTPUTS, THESIS_ROOT, add_import_paths  # noqa: E402

add_import_paths()
ROOT = THESIS_ROOT

from live_features import LIMITER, extract_live_features  # noqa: E402

OUTPUTS = NEW_OUTPUTS
LIVE_CSV = Path(__file__).resolve().parent.parent / "data" / "live_dataset.csv"
HYBRID_CSV = ROOT / "data" / "processed" / "hybrid_dataset.csv"
API = "https://api.etherscan.io/v2/api"
SEED = 999  # sengaja berbeda dari seed pembangun data latih


def sample_fresh(api_key: str, n_blocks: int, exclude: set[str]) -> list[str]:
    rng = random.Random(SEED)
    out: list[str] = []
    for _ in range(n_blocks):
        block = rng.randint(3_000_000, 25_000_000)
        try:
            r = requests.get(API, params=dict(
                chainid=1, module="proxy", action="eth_getBlockByNumber",
                tag=hex(block), boolean="true", apikey=api_key), timeout=25).json()
        except Exception:  # noqa: BLE001
            continue
        for tx in (r.get("result") or {}).get("transactions", [])[:40]:
            for key in ("from", "to"):
                a = str(tx.get(key) or "").lower()
                if len(a) == 42 and a not in exclude and a not in out:
                    out.append(a)
    rng.shuffle(out)
    return out


def main() -> None:
    load_dotenv()
    import os

    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    LIMITER.set_interval(0.32)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    import predict_live
    if not predict_live.is_available():
        print("Model live belum ada. Jalankan new_system/ml/train_live_model.py dulu.")
        sys.exit(1)

    exclude: set[str] = set()
    for csv, col in ((LIVE_CSV, "address"), (HYBRID_CSV, "node_id")):
        if csv.exists():
            exclude |= set(pd.read_csv(csv)[col].astype(str).str.lower())
    print(f"Alamat yang dikecualikan (pernah dilatih/dievaluasi): {len(exclude):,}")

    candidates = sample_fresh(api_key, n_blocks=5, exclude=exclude)
    print(f"Kandidat segar: {len(candidates)}\n")

    rows = []
    print(f"{'alamat':44s} {'tx':>6s} {'skor':>5s}  level")
    print("-" * 72)
    for addr in candidates:
        if len(rows) >= target:
            break
        try:
            feats = extract_live_features(addr, api_key=api_key, delay=0.32, max_pages=3)
        except Exception:  # noqa: BLE001
            continue
        if feats["_n_total_tx"] == 0:
            continue
        res = predict_live.predict_from_feature_dict(feats, explain_top=0)
        n_tx = int(feats["_n_total_tx"])
        rows.append({
            "address": addr,
            "n_tx": n_tx,
            "is_contract": int(feats["is_contract"]),
            "score": res["risk_score"],
            "level": res["risk_level"],
            "flagged_phishing": int(res["is_phishing"]),
            "insufficient_data": int(n_tx < predict_live.MIN_TX_FOR_CONFIDENCE),
        })
        flag = "  <-- ditandai phishing" if res["is_phishing"] else ""
        print(f"{addr:44s} {n_tx:6d} {res['risk_score']:5d}  {res['risk_level']}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "false_positive_test.csv", index=False)

    n = len(df)
    flagged = int(df["flagged_phishing"].sum())
    solid = df[df["insufficient_data"] == 0]
    summary = {
        "n_sampled": n,
        "n_flagged_phishing": flagged,
        "false_positive_rate": round(flagged / max(n, 1), 4),
        "n_with_enough_history": int(len(solid)),
        "false_positive_rate_excluding_thin_wallets":
            round(int(solid["flagged_phishing"].sum()) / max(len(solid), 1), 4),
        "baseline_model_A": "7/8 (0.88) pada pengujian awal — lihat laporan belajar/13",
        "seed": SEED,
    }
    (OUTPUTS / "false_positive_test.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")

    print("-" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nTersimpan -> {OUTPUTS / 'false_positive_test.csv'}")


if __name__ == "__main__":
    main()
