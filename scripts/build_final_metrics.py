"""
scripts/build_final_metrics.py
------------------------------
Collects every reported revision-v2 number into one long table,
outputs/revision/FINAL_VERIFIED_METRICS.csv, with its source file and protocol.
Nothing is computed from scratch here except aggregation of the per-run files.

Run (after revision_ijece.py, revision_extra.py, revision_ops.py):
    .venv/Scripts/python.exe scripts/build_final_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "outputs" / "revision"
rows: list[dict] = []


def add(section, scenario, model, metric, mean, std, n, source, protocol, valid=True):
    rows.append({"section": section, "scenario": scenario, "model": model, "metric": metric,
                 "mean": None if mean is None else round(float(mean), 6),
                 "std": None if std is None or pd.isna(std) else round(float(std), 6),
                 "n_runs": n, "valid_performance_estimate": valid, "protocol": protocol,
                 "source_file": source, "experiment_version": "v2"})


def from_runs(section, df, keys, metrics, source, protocol):
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        scen = k[0] if "scenario" in keys else (k[0] if len(k) else "")
        model = k[keys.index("model")] if "model" in keys else ""
        label = " / ".join(str(x) for x in k)
        for m in metrics:
            add(section, scen if "scenario" in keys else label, model, m, g[m].mean(), g[m].std(), len(g), source, protocol)


def main() -> None:
    std_proto = "stratified 80:20, seeds 42,0-8; XGBoost/GraphSAGE early stopping on a train-internal 20% slice"
    mets = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "TN", "FP", "FN", "TP"]
    runs = pd.read_csv(REV / "multiseed_all_runs.csv")
    from_runs("main_4x4_and_fusion", runs, ["scenario", "model"], mets, "outputs/revision/multiseed_all_runs.csv", std_proto)

    for _, r in pd.read_csv(REV / "graphsage_protocol_decomposition.csv").iterrows():
        for m in ["f1", "roc_auc", "pr_auc", "precision", "recall"]:
            add("graphsage_protocol_decomposition", "muldigraph_plus_etherscan (seed 42)", r.protocol, m, r[m], None, 1,
                "outputs/revision/graphsage_protocol_decomposition.csv", "single split seed 42",
                bool(r.valid_performance_estimate))

    cal = pd.read_csv(REV / "calibration_all_runs.csv")
    from_runs("calibration", cal, ["method"], ["brier", "ece_10bin", "log_loss", "roc_auc", "pr_auc", "f1_at_0.5"],
              "outputs/revision/calibration_all_runs.csv",
              "XGBoost S2 fit 60% / early stopping 20% / calibration 20% of train; test once")

    cost = pd.read_csv(REV / "cost_threshold_all_runs.csv")
    from_runs("cost_sensitive_threshold", cost, ["cost_ratio_FN_to_FP", "policy"],
              ["threshold", "FP", "FN", "total_cost", "precision", "recall", "f1"],
              "outputs/revision/cost_threshold_all_runs.csv", "threshold chosen on the calibration slice; test once")

    rat = pd.read_csv(REV / "majority_ratio_all_runs.csv")
    from_runs("majority_ratio", rat, ["ratio"], ["precision", "recall", "f1", "roc_auc", "pr_auc", "scale_pos_weight"],
              "outputs/revision/majority_ratio_all_runs.csv", "XGBoost S2, normals subsampled, 10 seeds")

    tmp = pd.read_csv(REV / "temporal_split_all_runs.csv")
    from_runs("temporal_split", tmp, ["scenario", "model"], ["precision", "recall", "f1", "roc_auc", "pr_auc", "FP", "FN"],
              "outputs/revision/temporal_split_all_runs.csv",
              "earliest 80% by first MulDiGraph tx -> train; latest 20% -> test (294 phishing); 10 model seeds")

    prev = pd.read_csv(REV / "prevalence_sensitivity_all_runs.csv")
    from_runs("prevalence_sensitivity_S1", prev, ["ratio", "model"], ["precision", "recall", "f1", "roc_auc", "pr_auc"],
              "outputs/revision/prevalence_sensitivity_all_runs.csv",
              "S1 only; extra normals featurised from MulDiGraph; 5 seeds")
    for _, r in pd.read_csv(REV / "prevalence_prior_shift_estimate.csv").iterrows():
        add("prevalence_prior_shift_ESTIMATE", r.scenario, r.model, f"estimated_precision @ {r.prevalence}",
            r.estimated_precision, None, None, "outputs/revision/prevalence_prior_shift_estimate.csv",
            "analytic: TPR*pi/(TPR*pi+FPR*(1-pi)); not measured", False)

    fs = pd.read_csv(REV / "feature_selection_cv_all_runs.csv")
    from_runs("feature_selection_train_cv", fs, ["subset"], ["cv_f1_train", "f1", "pr_auc"],
              "outputs/revision/feature_selection_cv_all_runs.csv", "subset chosen by 5-fold CV inside train")
    sel = fs[fs.selected_by_cv]
    add("feature_selection_train_cv", "muldigraph_plus_etherscan", "CV-selected subset", "f1", sel.f1.mean(),
        sel.f1.std(), len(sel), "outputs/revision/feature_selection_cv_all_runs.csv", "subset chosen by CV, test once")

    ops = json.loads((REV / "ops_metrics.json").read_text(encoding="utf-8"))
    mo = ops["model"]
    for k, v in [("model_file_size_kb", mo["model_file_size_kb"]), ("deserialise_ms", mo["deserialise_ms"]),
                 ("model_resident_increment_mb", mo["model_resident_increment_mb"]),
                 ("process_rss_after_inference_mb", mo["process_rss_after_inference_mb"]),
                 ("single_wallet_booster_predict_ms_median", mo["single_wallet_booster_predict_ms"]["median"]),
                 ("single_wallet_predict_full_ms_median", mo["single_wallet_predict_full_ms"]["median"]),
                 ("single_wallet_predict_full_ms_p95", mo["single_wallet_predict_full_ms"]["p95"]),
                 ("batch_per_wallet_us", mo["batch"]["per_wallet_us"])]:
        add("operational_measured", "production", "XGBoost S2 (model_prod_xgb.json)", k, v, None, None,
            "outputs/revision/ops_metrics.json", "measured, revision_ops.py")
    live = ops.get("live", {}).get("summary_found_wallets", {})
    for k in ["total_s_median", "total_s_max", "calls_median", "compute_s_median"]:
        if k in live:
            add("operational_measured", "live Etherscan path", "", k, live[k], None, live.get("n"),
                "outputs/revision/ops_metrics.json", "measured on 11 wallets, 27 Oct 2026")
    g = ops["gas"]
    add("operational_measured", "RiskRegistry Hardhat local", "", "gas_used_mean", g["gas_used_mean"], None, 20,
        "ml/artifacts/blockchain_metrics.json", "measured, Hardhat local")
    add("operational_measured", "RiskRegistry Hardhat local", "", "commit_latency_ms_mean",
        g["local_commit_latency_ms"]["mean"], g["local_commit_latency_ms"]["stdev"], 20,
        "ml/artifacts/blockchain_metrics.json", "measured, Hardhat local")
    for t in g["table"]:
        add("operational_ESTIMATE", "public network", "", f"cost_usd @ {t['gas_price_gwei']} gwei", t["cost_usd"], None,
            None, "outputs/revision/ops_metrics.json", f"derived: gas x price x ETH {g.get('eth_usd_assumed')} USD", False)

    out = pd.DataFrame(rows)
    out.to_csv(REV / "FINAL_VERIFIED_METRICS.csv", index=False)
    print(f"{len(out)} rows -> {REV / 'FINAL_VERIFIED_METRICS.csv'}")


if __name__ == "__main__":
    main()
