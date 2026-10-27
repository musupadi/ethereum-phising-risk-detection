"""
experiments/ijece_final_audit.py
--------------------------------
Builds the reviewer evidence package in outputs/ijece_final_audit/ for IJECE Paper ID 42848.

It does NOT retrain any of the reported models. It
  * re-reads the experiment outputs in outputs/revision/ (produced by revision_ijece.py and
    revision_ops.py) and re-shapes them into the requested deliverables, and
  * runs cheap integrity checks that were never run before: dataset counts, address
    normalisation, label agreement with MulDiGraph `isp`, reproduction of the 1:10 normal
    sample, feature joins, BigQuery coverage, correlation pairs behind the feature-ranking
    pruning, split sizes per seed, and production model / schema / API compatibility.

Run: .venv/Scripts/python.exe experiments/ijece_final_audit.py
"""

from __future__ import annotations

import json
import pickle
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REV = ROOT / "outputs" / "revision"
OUT = ROOT / "outputs" / "ijece_final_audit"
DATA = ROOT / "data"
ART = ROOT / "ml" / "artifacts"
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
ADDR_RE = re.compile(r"^0x[0-9a-f]{40}$")


def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                            encoding="utf-8")


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "processed" / "hybrid_dataset.csv")
    schema = json.loads((ART / "hybrid_feature_schema.json").read_text(encoding="utf-8"))
    prod_schema = json.loads((ART / "prod_feature_schema.json").read_text(encoding="utf-8"))
    sc = schema["scenarios"]
    groups = schema["feature_groups"]

    # ── 1. dataset audit ──────────────────────────────────────────────────────
    addr = df["node_id"].astype(str)
    base = pd.read_csv(DATA / "raw" / "addresses.csv")
    base["a"] = base["address"].astype(str).str.lower().str.strip()
    norm = pd.read_csv(DATA / "raw" / "addresses_normal.csv")
    phish = pd.read_csv(DATA / "raw" / "addresses_phishing.csv")
    eth = pd.read_csv(DATA / "raw" / "addresses_etherscan_enriched.csv")
    eth["a"] = eth["address"].astype(str).str.lower().str.strip()

    lab = dict(zip(addr.str.lower(), df["label"]))
    base_lab = dict(zip(base["a"], base["label"]))
    label_mismatch_vs_addresses_csv = sum(int(lab[a] != int(base_lab[a])) for a in lab if a in base_lab)

    with open(DATA / "raw" / "MulDiGraph.pkl", "rb") as f:
        G = pickle.load(f)
    # MulDiGraph node ids are mixed-case-insensitive hex strings; map lowercase -> node
    in_graph = 0
    isp_mismatch = 0
    for a, y in lab.items():
        if a in G:
            in_graph += 1
            isp_mismatch += int(int(G.nodes[a].get("isp", 0) or 0) != int(y))
    n_isp1_in_graph = sum(1 for _, d in G.nodes(data=True) if int(d.get("isp", 0) or 0) == 1)

    # Reproduce the 1:10 normal sample exactly as ml/extract_addresses_from_graph.py does.
    addr_re = re.compile(r"^0x[a-fA-F0-9]{40}$")
    normals = sorted({str(n) for n, d in G.nodes(data=True)
                      if addr_re.match(str(n)) and int(d.get("isp", 0) or 0) == 0})
    phish_all = sorted({str(n) for n, d in G.nodes(data=True)
                        if addr_re.match(str(n)) and int(d.get("isp", 0) or 0) == 1})
    random.seed(42)
    idx = random.sample(range(len(normals)), min(len(phish_all) * 10, len(normals)))
    repro_normals = {normals[i] for i in idx}
    committed_normals = set(norm["address"].astype(str))
    sample_reproduced = repro_normals == committed_normals

    feature_cols = sorted(set(sum(sc.values(), [])))
    nan_raw = {c: int(df[c].isna().sum()) for c in feature_cols + ["graph_present"] if df[c].isna().any()}
    bq_cols = groups["bigquery"]
    eth_cols = groups["etherscan"]
    bq_nonzero = (df[bq_cols].fillna(0).abs().sum(axis=1) > 0)

    dataset_audit = {
        "source_file": "data/processed/hybrid_dataset.csv",
        "rows": len(df),
        "unique_addresses_lowercase": int(addr.str.lower().nunique()),
        "duplicate_addresses": int(len(df) - addr.str.lower().nunique()),
        "all_addresses_lowercase_0x40hex": bool(addr.map(lambda s: bool(ADDR_RE.match(s))).all()),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "class_ratio_normal_per_phishing": round(int((df.label == 0).sum()) / int((df.label == 1).sum()), 4),
        "matches_expected_12815_1165_11650": bool(len(df) == 12815 and (df.label == 1).sum() == 1165
                                                   and (df.label == 0).sum() == 11650),
        "label_integrity": {
            "label_mismatch_vs_data_raw_addresses_csv": label_mismatch_vs_addresses_csv,
            "addresses_found_in_muldigraph": in_graph,
            "label_mismatch_vs_muldigraph_isp": isp_mismatch,
            "isp1_nodes_in_full_muldigraph": n_isp1_in_graph,
            "all_isp1_nodes_included": bool((df.label == 1).sum() == n_isp1_in_graph),
        },
        "sampling_reproducibility": {
            "procedure": "ml/extract_addresses_from_graph.py: all isp=1 nodes; normals = random.sample over the "
                         "address-sorted list of valid isp=0 hex nodes, random.seed(42), n = 10 x phishing",
            "valid_normal_nodes_in_graph": len(normals),
            "reproduced_sample_size": len(repro_normals),
            "identical_to_committed_addresses_normal_csv": bool(sample_reproduced),
            "committed_phishing_csv_rows": len(phish),
        },
        "graph_present": {"unique_values": sorted(df["graph_present"].unique().tolist()),
                          "role": "audit flag only; excluded from every scenario and from prod schema",
                          "in_any_scenario": any("graph_present" in v for v in sc.values()),
                          "in_prod_schema_features": "graph_present" in prod_schema["features"]},
        "feature_joins": {
            "etherscan_rows": len(eth), "etherscan_addresses_matched": int(addr.str.lower().isin(eth["a"]).sum()),
            "etherscan_rows_with_error": int(eth["error"].notna().sum()) if "error" in eth else None,
            "etherscan_transaction_count_zero": int((df["transaction_count"].fillna(0) == 0).sum()),
            "etherscan_transaction_count_at_10000_cap": int((df["transaction_count"] >= 10000).sum()),
            "bigquery_rows_with_any_nonzero_value": int(bq_nonzero.sum()),
        },
        "missing_values_in_hybrid_dataset_before_fillna": nan_raw,
        "missing_value_policy": "all model inputs use .fillna(0) (scenario scripts, revision_ijece.py, "
                                "train_production_model.py, predict.py defaults absent keys to 0.0)",
        "constant_predictors": [c for c in feature_cols if df[c].fillna(0).nunique() <= 1],
        "scenario_feature_counts": {k: len(v) for k, v in sc.items()},
        "feature_group_counts": {k: len(v) for k, v in groups.items()},
        "graph_feature_count": len(sc["muldigraph_only"]),
        "scenarios_match_expected_18_25_28_35": [len(sc[k]) for k in
            ["muldigraph_only", "muldigraph_plus_etherscan", "muldigraph_plus_bigquery",
             "muldigraph_plus_etherscan_plus_bigquery"]] == [18, 25, 28, 35],
        "graph_time_range_utc": None,
    }
    ts = [d.get("timestamp") for _, _, d in G.edges(data=True) if d.get("timestamp") is not None]
    dataset_audit["graph_time_range_utc"] = [
        datetime.fromtimestamp(min(ts), timezone.utc).isoformat(), datetime.fromtimestamp(max(ts), timezone.utc).isoformat()]
    del ts
    dump("dataset_audit.json", dataset_audit)
    print("dataset audit done")

    # ── 2. feature dictionary ─────────────────────────────────────────────────
    desc = {
        "in_degree": "Number of incoming transaction edges", "out_degree": "Number of outgoing transaction edges",
        "unique_senders": "Distinct predecessor addresses", "unique_receivers": "Distinct successor addresses",
        "total_received": "Sum of incoming ETH", "total_sent": "Sum of outgoing ETH",
        "avg_in_amount": "Mean incoming amount", "avg_out_amount": "Mean outgoing amount",
        "max_in_amount": "Maximum incoming amount", "max_out_amount": "Maximum outgoing amount",
        "min_in_amount": "Minimum incoming amount", "min_out_amount": "Minimum outgoing amount",
        "std_in_amount": "Population std. dev. of incoming amounts (0 if <2 edges)",
        "std_out_amount": "Population std. dev. of outgoing amounts (0 if <2 edges)",
        "in_out_ratio": "in_degree / (out_degree + 1)", "sent_received_ratio": "total_sent / (total_received + 1e-9)",
        "timestamp_spread_in": "max - min timestamp of incoming edges, seconds (0 if <2 edges)",
        "timestamp_spread_out": "max - min timestamp of outgoing edges, seconds (0 if <2 edges)",
        "graph_present": "1 if the address is a MulDiGraph node (join-quality audit flag)",
        "created_at_ts": "First-transaction timestamp from Etherscan (never populated; constant 0)",
        "balance": "Current ETH balance (Etherscan account/balance)",
        "transaction_count": "Normal transactions (Etherscan txlist, paged, capped at 10,000)",
        "activity_score": "Transactions in the latest 100-transaction txlist window",
        "avg_tx_value": "Mean ETH value in that window", "tx_frequency": "activity_score / 30",
        "unique_counterparties": "Distinct counterparties in that window (self excluded)",
        "bq_tx_count_out": "Outgoing tx count, BigQuery 90-day window", "bq_tx_count_in": "Incoming tx count, 90-day window",
        "bq_total_out_eth": "ETH sent, 90-day window", "bq_total_in_eth": "ETH received, 90-day window",
        "bq_unique_to": "Distinct recipients, 90-day window", "bq_unique_from": "Distinct senders, 90-day window",
        "bq_active_days_out": "Days with outgoing activity, 90-day window",
        "bq_active_days_in": "Days with incoming activity, 90-day window",
        "bq_first_seen_ts": "Earliest tx timestamp in the 90-day window",
        "bq_last_seen_ts": "Latest tx timestamp in the 90-day window",
    }
    code_src = {"graph": "ml/preprocess.py:extract_node_features", "etherscan": "ml/data_fetchers.py:EtherscanFetcher",
                "bigquery": "ml/enrich_real_addresses_bigquery.py (SQL)"}
    rows = []
    for grp, cols in [("graph", groups["graph"]), ("audit", ["graph_present"]),
                      ("etherscan", eth_cols), ("bigquery", bq_cols)]:
        for c in cols:
            rows.append({
                "attribute": c, "source": {"graph": "MulDiGraph (1-hop)", "audit": "MulDiGraph",
                                           "etherscan": "Etherscan API", "bigquery": "BigQuery crypto_ethereum"}[grp],
                "role": "audit-only (not a predictor)" if grp == "audit" else "predictor",
                "definition": desc[c], "code": code_src.get(grp, "ml/build_hybrid_dataset.py"),
                **{s: c in sc[k] for s, k in [("S1", "muldigraph_only"), ("S2", "muldigraph_plus_etherscan"),
                                               ("S3", "muldigraph_plus_bigquery"),
                                               ("S4", "muldigraph_plus_etherscan_plus_bigquery")]},
                "in_production_model": c in prod_schema["features"],
                "nonzero_rows": int((df[c].fillna(0) != 0).sum()),
                "constant": bool(df[c].fillna(0).nunique() <= 1),
            })
    fd = pd.DataFrame(rows)
    assert len(fd) == 36, len(fd)
    fd.insert(0, "no", range(1, 37))
    fd.to_csv(OUT / "feature_dictionary.csv", index=False)

    # ── 3/4. per-seed and summary tables ──────────────────────────────────────
    runs = pd.read_csv(REV / "multiseed_all_runs.csv")
    y = df["label"].values
    counts = {}
    for s in SEEDS:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=s, stratify=y)
        fit, val = train_test_split(tr, test_size=0.2, random_state=s, stratify=y[tr])
        counts[s] = dict(n_train=len(tr), n_fit=len(fit), n_val=len(val), n_test=len(te),
                         train_phishing=int(y[tr].sum()), fit_phishing=int(y[fit].sum()), test_phishing=int(y[te].sum()))

    def split_cols(r):
        c = counts[r.seed]
        uses_val = r.model.startswith(("XGBoost", "GraphSAGE"))
        return pd.Series({"n_train_used_for_fitting": c["n_fit"] if uses_val else c["n_train"],
                          "n_validation": c["n_val"] if uses_val else 0, "n_test": c["n_test"],
                          "test_phishing": c["test_phishing"]})
    per = pd.concat([runs, runs.apply(split_cols, axis=1)], axis=1)
    per.insert(3, "feature_count", per["scenario"].map(lambda k: len(sc[k]) if k in sc else
                                                       (135 if "eth7" in str(k) else 128)))
    per.loc[per.model == "XGBoost[emb128 only]", "feature_count"] = 128
    per.loc[per.model == "XGBoost[emb128||eth7]", "feature_count"] = 135
    per.to_csv(OUT / "model_results_per_seed.csv", index=False)

    mets = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "TN", "FP", "FN", "TP"]
    summ = per.groupby(["scenario", "model"], sort=False)[mets].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["n_seeds"] = per.groupby(["scenario", "model"], sort=False).size().values
    summ.to_csv(OUT / "model_results_summary.csv", index=False)

    # Figure 2 data (F1 mean ± std per scenario x model, tabular scenarios only)
    fig2 = summ[summ.scenario.isin(sc.keys())][["scenario", "model", "f1_mean", "f1_std"]]
    fig2.to_csv(OUT / "figure2_f1_mean_std.csv", index=False)

    # best-by-metric + paired comparison XGB vs RF on S2
    best = {}
    tab = summ[summ.scenario.isin(sc.keys())]
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        r = tab.loc[tab[f"{m}_mean"].idxmax()]
        best[m] = {"scenario": r.scenario, "model": r.model, "mean": round(r[f"{m}_mean"], 4), "std": round(r[f"{m}_std"], 4)}
    r = tab.loc[tab["FN_mean"].idxmin()]
    best["fewest_FN"] = {"scenario": r.scenario, "model": r.model, "mean": r.FN_mean}
    r = tab.loc[tab["FP_mean"].idxmin()]
    best["fewest_FP"] = {"scenario": r.scenario, "model": r.model, "mean": r.FP_mean}
    s2 = per[per.scenario == "muldigraph_plus_etherscan"].pivot(index="seed", columns="model", values="f1")
    paired = {
        "rf_minus_xgb_f1_per_seed": (s2["RandomForest"] - s2["XGBoost"]).round(4).to_dict(),
        "seeds_rf_higher_f1": int((s2["RandomForest"] > s2["XGBoost"]).sum()),
        "mean_diff_rf_minus_xgb": round(float((s2["RandomForest"] - s2["XGBoost"]).mean()), 4),
        "std_diff": round(float((s2["RandomForest"] - s2["XGBoost"]).std()), 4),
    }
    try:
        from scipy.stats import wilcoxon
        paired["wilcoxon_p_two_sided"] = round(float(wilcoxon(s2["RandomForest"], s2["XGBoost"]).pvalue), 4)
    except Exception as e:  # scipy is a sklearn dependency; keep going if absent
        paired["wilcoxon_p_two_sided"] = f"unavailable: {e}"
    for m in ["recall", "pr_auc", "roc_auc"]:
        p = per[per.scenario == "muldigraph_plus_etherscan"].pivot(index="seed", columns="model", values=m)
        paired[f"seeds_xgb_higher_{m}"] = int((p["XGBoost"] > p["RandomForest"]).sum())
    pFN = per[per.scenario == "muldigraph_plus_etherscan"].pivot(index="seed", columns="model", values="FN")
    paired["seeds_xgb_fewer_FN"] = int((pFN["XGBoost"] < pFN["RandomForest"]).sum())
    dump("best_model_by_metric.json", {"best": best, "S2_xgboost_vs_random_forest": paired,
                                       "split_sizes_per_seed": counts})

    # ── 5. sampling ratio ─────────────────────────────────────────────────────
    rat = pd.read_csv(REV / "majority_ratio_all_runs.csv")
    rs = rat.groupby("ratio", sort=False)[["n_total", "scale_pos_weight", "prevalence_test", "precision", "recall",
                                            "f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])
    rs.columns = [f"{a}_{b}" for a, b in rs.columns]
    rs = rs.reset_index()
    rs.insert(1, "n_seeds", rat.groupby("ratio", sort=False).size().values)
    rs.insert(2, "row_type", "summary")
    rat2 = rat.assign(row_type="per_seed")
    pd.concat([rs, rat2], ignore_index=True).to_csv(OUT / "sampling_ratio_sensitivity.csv", index=False)

    # ── 6. GraphSAGE protocol audit ───────────────────────────────────────────
    dec = pd.read_csv(REV / "graphsage_protocol_decomposition.csv").assign(seed=42, row_type="decomposition")
    g = per[per.model.str.startswith("GraphSAGE")][["seed", "scenario", "model", "epochs_run", "precision", "recall",
                                                    "f1", "roc_auc", "pr_auc", "TN", "FP", "FN", "TP"]]
    g = g.assign(row_type="multiseed", early_stopping_on="train-internal validation F1 (patience 25)",
                 edges="MulDiGraph only",
                 training_topology=np.where(g.model == "GraphSAGE-inductive",
                                            "edges touching test wallets removed during training",
                                            "full induced graph (test nodes unlabeled, present)"))
    pd.concat([dec, g], ignore_index=True).to_csv(OUT / "graphsage_protocol_audit.csv", index=False)

    # ── 7. GraphSAGE + fusion results ─────────────────────────────────────────
    fus = summ[(summ.scenario == "late_fusion") | (summ.model.str.startswith("GraphSAGE")) |
               ((summ.scenario == "muldigraph_plus_etherscan") & (summ.model == "XGBoost"))].copy()
    fus.insert(0, "source", "outputs/revision/multiseed_all_runs.csv (10 seeds)")
    single = pd.read_csv(ROOT / "outputs" / "late_fusion_comparison_table_v2.csv")
    single = single.rename(columns={"Model": "model", "F1": "f1_mean", "Precision": "precision_mean",
                                    "Recall": "recall_mean", "ROC-AUC": "roc_auc_mean", "PR-AUC": "pr_auc_mean",
                                    "Accuracy": "accuracy_mean"})
    single.insert(0, "source", "outputs/late_fusion_comparison_table_v2.csv (single split, seed 42)")
    pd.concat([fus, single], ignore_index=True).to_csv(OUT / "graphsage_fusion_results.csv", index=False)

    # ── 8. BigQuery coverage ──────────────────────────────────────────────────
    sql = (ROOT / "ml" / "enrich_real_addresses_bigquery.py").read_text(encoding="utf-8")
    bq = pd.read_csv(DATA / "raw" / "addresses_bigquery_enriched.csv")
    bqn = bq[bq[bq_cols].fillna(0).abs().sum(axis=1) > 0]
    first = pd.to_datetime(pd.to_numeric(bqn["bq_first_seen_ts"], errors="coerce"), unit="s", utc=True)
    last = pd.to_datetime(pd.to_numeric(bqn["bq_last_seen_ts"], errors="coerce"), unit="s", utc=True)
    span_days = (last.max() - first.min()).total_seconds() / 86400
    dump("bigquery_coverage_audit.json", {
        "query_source": "ml/enrich_real_addresses_bigquery.py",
        "table_default": "bigquery-public-data.crypto_ethereum.transactions",
        "window_clause_in_sql": "WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_window DAY)"
                                if "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_window DAY)" in sql else "NOT FOUND",
        "days_window_default": 90 if 'os.getenv("BIGQUERY_DAYS_WINDOW", "90")' in sql else "check script",
        "rows": len(bq), "rows_with_any_nonzero_bq_value": len(bqn),
        "coverage_pct": round(100 * len(bqn) / len(bq), 2),
        "nonzero_by_label": {str(k): int(v) for k, v in bqn["label"].value_counts().items()},
        "observed_bq_first_seen_min": str(first.min()), "observed_bq_last_seen_max": str(last.max()),
        "observed_span_days": round(span_days, 1),
        "any_covered_activity_inside_muldigraph_period": bool(first.min() <= pd.Timestamp("2019-01-19", tz="UTC")),
        "muldigraph_edge_time_range_utc": dataset_audit["graph_time_range_utc"],
        "conclusion": ("Coverage is limited by the query window, not by missing history: the SQL restricts "
                       "block_timestamp to the 90 days before the query date, every observed timestamp of the "
                       f"{len(bqn)} covered wallets lies in {first.min().date()}..{last.max().date()} "
                       f"(span {span_days:.0f} days), and none overlaps the 2015-2019 MulDiGraph period. "
                       "The full-history public table was never queried for 2015-2019, so this is a design "
                       "limitation of the enrichment step."),
    })

    # ── 9. feature ranking audit ──────────────────────────────────────────────
    fs = json.loads((ART / "feature_selection_report.json").read_text(encoding="utf-8"))
    imp = pd.DataFrame(fs["importance_ranking"])
    imp.insert(0, "rank", range(1, len(imp) + 1))
    clean = pd.read_csv(DATA / "processed" / "hybrid_dataset_clean.csv")
    feats = list(imp["feature"])
    Xtr, _ = train_test_split(clean[feats], test_size=0.2, random_state=42, stratify=clean["label"])
    from ml.feature_selection import apply_caps, fit_caps   # identical p1/p99 train capping
    Xc = apply_caps(Xtr, fit_caps(Xtr))
    corr = Xc.corr().abs()
    rankd = dict(zip(imp.feature, imp["rank"]))
    partners = {f: [] for f in feats}
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if corr.loc[a, b] >= fs["correlation_threshold"]:
                partners[a].append(f"{b} ({corr.loc[a, b]:.3f})")
                partners[b].append(f"{a} ({corr.loc[a, b]:.3f})")
    imp["corr_partners_ge_threshold"] = imp.feature.map(lambda f: "; ".join(partners[f]))
    imp["dropped_by_correlation_filter"] = imp.feature.isin(fs["dropped_by_correlation"])
    imp["in_top_20_subset"] = imp.feature.isin(fs["experiments"]["top_20"]["features"])
    imp["in_corr_pruned_subset"] = imp.feature.isin(fs["experiments"]["corr_pruned"]["features"])
    imp["in_production_model"] = imp.feature.isin(prod_schema["features"])
    imp.to_csv(OUT / "feature_ranking_audit.csv", index=False)
    rank_meta = {
        "formula": "combined(f) = 0.5 * [ gain_XGB(f)/sum gain_XGB + MDI_RF(f)/sum MDI_RF ]",
        "xgb_importance": "XGBClassifier.feature_importances_ (importance_type=gain, sklearn API default)",
        "rf_importance": "RandomForestClassifier(n_estimators=400, class_weight=balanced).feature_importances_ (MDI)",
        "fitted_on": "train split (seed 42) of hybrid_dataset_clean.csv after p1/p99 capping fitted on train",
        "n_features_ranked": len(imp), "created_at_ts_removed_before_ranking": "created_at_ts" not in feats,
        "order": "ranking first, correlation filter afterwards (filter uses the ranking to decide which of a pair to drop)",
        "threshold_operator": ">= " + str(fs["correlation_threshold"]),
        "dropped": fs["dropped_by_correlation"],
        "subset_results_f1": {k: v["f1"] for k, v in fs["experiments"].items()},
        "used_by_production_model": False,
        "why_in_degree_is_top_ranked": "The ranking is computed on all 24 features before pruning, so in_degree keeps "
                                       "rank 2. The correlation filter then removes it from the 'corr_pruned' subset "
                                       "only. 'top_k' subsets are cut from the unpruned ranking and still contain it, "
                                       "and the production model uses all 25 S2 features (no selection applied).",
    }
    dump("feature_ranking_method.json", rank_meta)

    # ── 10. calibration ───────────────────────────────────────────────────────
    cal = pd.read_csv(REV / "calibration_all_runs.csv").assign(row_type="per_seed")
    cs = cal.groupby("method", sort=False)[["brier", "ece_10bin", "log_loss", "roc_auc", "f1_at_0.5"]].agg(["mean", "std"])
    cs.columns = [f"{a}_{b}" for a, b in cs.columns]
    cs = cs.reset_index().assign(row_type="summary")
    pd.concat([cs, cal], ignore_index=True).to_csv(OUT / "calibration_results.csv", index=False)

    # ── 11. cost-sensitive thresholds ─────────────────────────────────────────
    cost = pd.read_csv(REV / "cost_threshold_all_runs.csv")
    w = cost.pivot_table(index=["seed", "cost_ratio_FN_to_FP"], columns="policy",
                         values=["threshold", "FP", "FN", "total_cost", "f1"]).reset_index()
    w.columns = ["_".join([c for c in col if c]) for col in w.columns]
    w["test_cost_change_vs_default"] = w["total_cost_cost_optimal"] - w["total_cost_default_0.5"]
    w["validation_selected_worse_on_test"] = w["test_cost_change_vs_default"] > 0
    w.insert(0, "row_type", "per_seed")
    ag = w.groupby("cost_ratio_FN_to_FP").agg(
        threshold_mean=("threshold_cost_optimal", "mean"), threshold_std=("threshold_cost_optimal", "std"),
        cost_default_mean=("total_cost_default_0.5", "mean"), cost_optimal_mean=("total_cost_cost_optimal", "mean"),
        FN_default_mean=("FN_default_0.5", "mean"), FP_default_mean=("FP_default_0.5", "mean"),
        FN_optimal_mean=("FN_cost_optimal", "mean"), FP_optimal_mean=("FP_cost_optimal", "mean"),
        seeds_optimal_worse_than_default=("validation_selected_worse_on_test", "sum"),
    ).reset_index()
    ag["mean_cost_change_pct"] = (100 * (ag.cost_optimal_mean - ag.cost_default_mean) / ag.cost_default_mean).round(2)
    ag.insert(0, "row_type", "summary")
    pd.concat([ag, w], ignore_index=True).to_csv(OUT / "cost_sensitive_thresholds.csv", index=False)

    # ── 12. operational benchmarks ────────────────────────────────────────────
    ops = json.loads((REV / "ops_metrics.json").read_text(encoding="utf-8"))
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(ART / "model_prod_xgb.json"))
    prod_check = {
        "booster_feature_names_equal_schema_order": list(booster.feature_names or []) == prod_schema["features"],
        "schema_equals_hybrid_S2_order": prod_schema["features"] == sc["muldigraph_plus_etherscan"],
        "n_features": len(prod_schema["features"]),
        "schema_metrics": prod_schema.get("metrics"),
        "matches_seed42_xgboost_S2_in_revision_runs": None,
        "training_rows": "80% inner fit slice of the seed-42 training split (8,201 wallets); not refitted on all data",
    }
    r42 = per[(per.seed == 42) & (per.scenario == "muldigraph_plus_etherscan") & (per.model == "XGBoost")].iloc[0]
    prod_check["matches_seed42_xgboost_S2_in_revision_runs"] = bool(abs(r42.f1 - prod_schema["metrics"]["f1"]) < 1e-4)
    live = ops.get("live", {})
    dump("operational_benchmarks.json", {
        "measured": {
            "host": ops["host"],
            "model": ops["model"],
            "live_etherscan_path": {"request_delay_s": live.get("request_delay_s"),
                                     "summary": live.get("summary_found_wallets"),
                                     "n_wallets": len(live.get("runs", [])),
                                     "calls_per_wallet": sorted({r["etherscan_calls"] for r in live.get("runs", [])}),
                                     "total_s_range": [min(r["total_s"] for r in live["runs"]),
                                                       max(r["total_s"] for r in live["runs"])] if live.get("runs") else None,
                                     "note": "measured 27 Oct 2026 from the author workstation while another experiment was using "
                                             "the CPU; 11 wallets only"},
            "riskregistry_hardhat_local": {"source": "ml/artifacts/blockchain_metrics.json (20 setRisk calls, "
                                                     "chain/scripts/measure_gas.py)",
                                            "gas_used": ops["gas"]["gas_used_mean"], "gas_range": ops["gas"]["gas_used_range"],
                                            "commit_latency_ms": ops["gas"]["local_commit_latency_ms"]},
        },
        "derived_estimates_not_measured": {
            "public_network_cost_formula": "cost_ETH = gas_used x gas_price_gwei x 1e-9",
            "eth_usd_assumption": ops["gas"].get("eth_usd_assumed"), "eth_usd_source": ops["gas"].get("eth_usd_source"),
            "table": ops["gas"]["table"],
            "mainnet_commit_latency": "about one block interval (~12 s); NOT measured",
            "throughput_bound_free_tier": "3 calls/s / 5 calls per wallet = 0.6 wallets/s; 100,000 calls/day / 5 = "
                                          "20,000 wallets/day per account. Limits taken from Etherscan documentation "
                                          "(docs.etherscan.io rate-limits page, Oct 2026); not load-tested.",
        },
        "production_model_compatibility": prod_check,
        "api_behaviour_notes": [
            "Dataset wallets are scored from precomputed rows of hybrid_dataset.csv, which includes the ~80% training "
            "wallets: their scores are in-sample and must not be presented as held-out performance.",
            "predict.py fills any missing schema feature with 0.0 without warning.",
            "Live mode: empty Etherscan txlist -> WalletNotFoundError -> HTTP 404, no score; fewer than "
            "MIN_MEANINGFUL_TX = 5 transactions -> scored with insufficient_data flag; all live scores flagged "
            "out_of_distribution and never written on-chain.",
        ],
    })

    # ── reconciliation vs. originally submitted numbers ───────────────────────
    old_sc = git("show", "6f98c6c:ml/artifacts/dataset_scenarios_comparison.csv")
    old_lf = git("show", "b632480:outputs/late_fusion_metrics_v2.json")
    old_ab = git("show", "b632480:outputs/embedding_only_ablation_metrics.json")
    rec = []
    if old_sc:
        from io import StringIO
        o = pd.read_csv(StringIO(old_sc))
        for _, r in o.iterrows():
            m = "GraphSAGE" if r.model.startswith("GNN") else r.model
            n = summ[(summ.scenario == r.scenario) & (summ.model == m)]
            n42 = per[(per.seed == 42) & (per.scenario == r.scenario) & (per.model == m)]
            rec.append({"item": f"{r.scenario} / {m} F1", "submitted_8595": round(r.f1, 4),
                        "revised_seed42_12815": round(float(n42.f1.iloc[0]), 4) if len(n42) else None,
                        "revised_mean_10seeds": round(float(n.f1_mean.iloc[0]), 4) if len(n) else None,
                        "revised_std": round(float(n.f1_std.iloc[0]), 4) if len(n) else None,
                        "old_source": "git 6f98c6c ml/artifacts/dataset_scenarios_comparison.csv",
                        "note": "old GraphSAGE used test-F1 early stopping + Etherscan edges" if m == "GraphSAGE" else ""})
    if old_lf:
        j = json.loads(old_lf)
        for key, label in [("late_fusion_v2", "Late fusion F1"), ("graphsage_baseline", "GraphSAGE graph-only (fusion encoder) F1")]:
            if key in j:
                rec.append({"item": label, "submitted_8595": round(j[key]["F1"], 4), "old_source": "git b632480 outputs/late_fusion_metrics_v2.json"})
    if old_ab:
        j = json.loads(old_ab)
        rec.append({"item": "embedding-only ablation JSON (old)", "submitted_8595": json.dumps(
            {k: round(v.get("F1", float("nan")), 4) for k, v in j.get("models", {}).items()} if isinstance(j.get("models"), dict) else "see file"),
            "old_source": "git b632480 outputs/embedding_only_ablation_metrics.json"})
    rec += [
        {"item": "wallet population", "submitted_8595": 8595, "revised_seed42_12815": 12815, "note": "1,165 phishing in both; normals 7,430 -> 11,650"},
        {"item": "S2 predictor count", "submitted_8595": 26, "revised_seed42_12815": 25, "note": "graph_present removed (constant audit flag)"},
        {"item": "BigQuery coverage %", "submitted_8595": 0.93, "revised_seed42_12815": round(100 * len(bqn) / len(bq), 2),
         "note": "0.93% is quoted by Reviewer A from the manuscript; the 8,595-wallet BigQuery file is not in the "
                 "repository, so the old count (~80 wallets if 0.93%) cannot be verified. Revised: 126 of 12,815."},
        {"item": "identity of 0.8184 / 0.8189 in the submission", "submitted_8595": "0.8184 = embedding-only XGBoost; 0.8189 = GraphSAGE graph-only",
         "note": "both: validation early stopping, 19 graph inputs (incl. graph_present), MulDiGraph edges, 8,595 wallets. "
                 "Table 3's 0.8960: early stopping on test F1, 26 inputs, MulDiGraph + Etherscan edges."},
        {"item": "LogisticRegression S2 F1", "submitted_8595": 0.854, "revised_seed42_12815": 0.7285,
         "note": "largest change of any model (-0.13); same code path (class_weight=balanced). Attributed to the larger "
                 "1:10 negative class; not separately investigated."},
        {"item": "production model F1", "submitted_8595": 0.9102, "revised_seed42_12815": prod_schema["metrics"]["f1"]},
    ]
    pd.DataFrame(rec).to_csv(OUT / "reconciliation_submitted_vs_revised.csv", index=False)

    # ── environment ───────────────────────────────────────────────────────────
    import sklearn, torch, torch_geometric, xgboost  # noqa: E401
    dump("environment.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit_of_inputs": git("rev-parse", "HEAD"), "git_dirty_files": git("status", "--short").splitlines(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "device": "cpu" if not torch.cuda.is_available() else "cuda",
        "seeds": SEEDS,
    })
    print("audit package written to", OUT)


if __name__ == "__main__":
    main()
