"""
experiments/revision_ops.py
---------------------------
Operational measurements requested by IJECE Reviewer B (Paper ID 42848):

  B-2.5  production model file size, deserialisation time, memory footprint,
         single-wallet and batch inference latency.
  B-2.2  end-to-end latency of the live (Etherscan) graph-to-tabular path for
         wallets outside the dataset, broken into network vs. feature computation,
         plus the number of Etherscan calls per wallet (drives the rate-limit bound).
  B-2.5  RiskRegistry.setRisk() cost on public networks from the measured gas.

Outputs: outputs/revision/ops_metrics.json
Run    : .venv/Scripts/python.exe experiments/revision_ops.py [--no-live]
"""

from __future__ import annotations

import gc
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "revision"
MODEL = ROOT / "ml" / "artifacts" / "model_prod_xgb.json"
SCHEMA = ROOT / "ml" / "artifacts" / "prod_feature_schema.json"
GAS = ROOT / "ml" / "artifacts" / "blockchain_metrics.json"

# Gas prices (gwei) bracket for the public-network cost table. ETH/USD is passed on the
# command line (--eth-usd) so the date of the price quote is explicit in the paper.
GAS_PRICES_GWEI = [0.1, 0.5, 1, 5, 10, 30]


def rss_mb() -> float:
    gc.collect()
    return psutil.Process().memory_info().rss / 2**20


def pct(xs, q):
    return float(np.percentile(xs, q))


def model_metrics() -> dict:
    import xgboost as xgb

    base = rss_mb()
    t = time.perf_counter()
    booster = xgb.Booster()
    booster.load_model(str(MODEL))
    load_ms = (time.perf_counter() - t) * 1000
    after = rss_mb()

    feats = json.loads(SCHEMA.read_text())["features"]
    test = pd.read_csv(ROOT / "data" / "processed" / "hybrid_test.csv")
    X = test.reindex(columns=feats).fillna(0).values.astype(np.float32)

    single = []
    for i in range(1000):
        row = X[i % len(X)].reshape(1, -1)
        t = time.perf_counter()
        booster.predict(xgb.DMatrix(row, feature_names=feats))
        single.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    booster.predict(xgb.DMatrix(X, feature_names=feats))
    batch_ms = (time.perf_counter() - t) * 1000

    # Full service path (feature dict -> schema check -> probability -> score -> SHAP-free explain)
    from ml.predict import predict_full
    rec = dict(zip(feats, X[0].tolist()))
    predict_full(rec, explain_top=5)  # warm-up / lazy load
    full = []
    for i in range(300):
        rec = dict(zip(feats, X[i % len(X)].tolist()))
        t = time.perf_counter()
        predict_full(rec, explain_top=5)
        full.append((time.perf_counter() - t) * 1000)

    cfg = json.loads(booster.save_config())
    return {
        "model_file": MODEL.name,
        "model_file_size_kb": round(MODEL.stat().st_size / 1024, 1),
        "n_trees": booster.num_boosted_rounds(),
        "n_features": len(feats),
        "max_depth": int(cfg["learner"]["gradient_booster"]["updater"]["grow_quantile_histmaker"]["train_param"]["max_depth"])
        if "grow_quantile_histmaker" in cfg["learner"]["gradient_booster"].get("updater", {}) else None,
        "deserialise_ms": round(load_ms, 1),
        "process_rss_before_load_mb": round(base, 1),
        "process_rss_after_load_mb": round(after, 1),
        "model_resident_increment_mb": round(after - base, 1),
        "process_rss_after_inference_mb": round(rss_mb(), 1),
        "single_wallet_booster_predict_ms": {"median": round(st.median(single), 3), "p95": round(pct(single, 95), 3)},
        "single_wallet_predict_full_ms": {"median": round(st.median(full), 3), "p95": round(pct(full, 95), 3)},
        "batch": {"n": len(X), "total_ms": round(batch_ms, 1), "per_wallet_us": round(batch_ms * 1000 / len(X), 2)},
    }


def live_metrics(addresses: list[tuple[str, str]]) -> dict:
    import requests
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not os.getenv("ETHERSCAN_API_KEY", "").strip():
        return {"skipped": "ETHERSCAN_API_KEY not set"}

    from ml import live_features

    calls = {"n": 0, "net_s": 0.0}
    real_get = requests.get

    def counting_get(*a, **k):
        t = time.perf_counter()
        try:
            return real_get(*a, **k)
        finally:
            calls["n"] += 1
            calls["net_s"] += time.perf_counter() - t

    requests.get = counting_get
    delay = float(os.getenv("LIVE_REQUEST_DELAY", "0.25"))
    runs = []
    try:
        for addr, kind in addresses:
            calls.update(n=0, net_s=0.0)
            t = time.perf_counter()
            status, n_tx, score = "ok", None, None
            try:
                r = live_features.predict_address(addr, max_pages=5, request_delay=delay)
                n_tx, score = r["n_transactions"], r.get("risk_score")
            except live_features.WalletNotFoundError:
                status = "wallet_not_found (no txlist)"
            total = time.perf_counter() - t
            runs.append({
                "address": addr, "kind": kind, "status": status, "n_transactions": n_tx, "risk_score": score,
                "etherscan_calls": calls["n"], "total_s": round(total, 3),
                "network_s": round(calls["net_s"], 3),
                "throttle_sleep_s": round(calls["n"] * delay, 3),
                "compute_s": round(max(total - calls["net_s"] - calls["n"] * delay, 0), 3),
            })
            print(runs[-1])
    finally:
        requests.get = real_get

    ok = [r for r in runs if r["status"] == "ok"]
    tot = [r["total_s"] for r in ok]
    return {
        "request_delay_s": delay,
        "runs": runs,
        "summary_found_wallets": {
            "n": len(ok),
            "total_s_median": round(st.median(tot), 3) if tot else None,
            "total_s_max": round(max(tot), 3) if tot else None,
            "calls_median": st.median([r["etherscan_calls"] for r in ok]) if ok else None,
            "compute_s_median": round(st.median([r["compute_s"] for r in ok]), 4) if ok else None,
        },
    }


def gas_table(eth_usd: float | None) -> dict:
    g = json.loads(GAS.read_text())
    gas = g["gas_used"]["mean"]
    rows = []
    for gwei in GAS_PRICES_GWEI:
        eth = gas * gwei * 1e-9
        rows.append({"gas_price_gwei": gwei, "cost_eth": eth, "cost_usd": eth * eth_usd if eth_usd else None})
    return {"gas_used_mean": gas, "gas_used_range": [g["gas_used"]["min"], g["gas_used"]["max"]],
            "local_commit_latency_ms": g["latency_ms"], "eth_usd_assumed": eth_usd, "table": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    eth_usd = float(args[args.index("--eth-usd") + 1]) if "--eth-usd" in args else None

    res = {"host": {"cpu": os.cpu_count(), "python": sys.version.split()[0]}}
    res["model"] = model_metrics()
    print(json.dumps(res["model"], indent=2))
    res["gas"] = gas_table(eth_usd)

    if "--no-live" not in args:
        df = pd.read_csv(ROOT / "data" / "processed" / "hybrid_dataset.csv")
        rng = np.random.default_rng(42)
        pick = []
        for lab, kind in ((1, "phishing (dataset)"), (0, "normal (dataset)")):
            ids = df.loc[df.label == lab, "node_id"].astype(str).tolist()
            pick += [(a, kind) for a in rng.choice(ids, 5, replace=False)]
        # A synthetic vanity address with almost no history (it had 1 tx when measured).
        pick.append(("0x" + "00" * 19 + "de", "synthetic low-activity address"))
        res["live"] = live_metrics(pick)
    else:
        # Keep the earlier live measurement instead of silently discarding it.
        prev = OUT / "ops_metrics.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            if "live" in old:
                res["live"] = old["live"]

    (OUT / "ops_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print("saved", OUT / "ops_metrics.json")


if __name__ == "__main__":
    main()
