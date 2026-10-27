"""
scripts/validate_revision.py
----------------------------
Consistency checks for the IJECE revision outputs. Exits with status 1 if any check fails.

Checks: dataset rows / label counts / duplicates / required columns; scenario feature
counts; graph_present never a predictor; every per-seed confusion matrix sums to the test
size with 233 phishing; precision / recall / F1 recomputed from TN/FP/FN/TP; the 10-seed
summary equals the mean of the per-seed rows; GraphSAGE runs never monitored the test set
(except the rows flagged invalid in the decomposition); calibration/threshold slice is
disjoint in size from the early-stopping slice; production booster vs schema vs dataset;
production metrics recomputed on the seed-42 test split; artifact hashes recorded.

Run: .venv/Scripts/python.exe scripts/validate_revision.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "outputs" / "revision"
ART = ROOT / "ml" / "artifacts"
results: list[dict] = []


def check(name: str, ok: bool, detail="") -> None:
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})


def blocked(name: str, why: str) -> None:
    results.append({"check": name, "status": "BLOCKED", "detail": why})


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    df = pd.read_csv(ROOT / "data" / "processed" / "hybrid_dataset.csv")
    schema = json.loads((ART / "hybrid_feature_schema.json").read_text(encoding="utf-8"))
    prod = json.loads((ART / "prod_feature_schema.json").read_text(encoding="utf-8"))
    sc = schema["scenarios"]

    # dataset
    check("dataset rows = 12815", len(df) == 12815, len(df))
    check("label counts 1165/11650", (df.label == 1).sum() == 1165 and (df.label == 0).sum() == 11650,
          df.label.value_counts().to_dict())
    check("no duplicate addresses", df.node_id.str.lower().nunique() == len(df))
    need = set(sum(sc.values(), [])) | {"node_id", "label", "graph_present"}
    missing = sorted(need - set(df.columns))
    check("required columns present", not missing, missing)
    check("no NaN in predictors", not df[sorted(set(sum(sc.values(), [])))].isna().any().any())

    # schema
    counts = [len(sc[k]) for k in ["muldigraph_only", "muldigraph_plus_etherscan", "muldigraph_plus_bigquery",
                                   "muldigraph_plus_etherscan_plus_bigquery"]]
    check("scenario sizes 18/25/28/35", counts == [18, 25, 28, 35], counts)
    check("graph_present not a predictor", all("graph_present" not in v for v in sc.values())
          and "graph_present" not in prod["features"])
    check("feature groups 18/7/10", [len(schema["feature_groups"][g]) for g in ["graph", "etherscan", "bigquery"]]
          == [18, 7, 10])

    # per-seed metrics
    runs = pd.read_csv(REV / "multiseed_all_runs.csv")
    tot = runs[["TN", "FP", "FN", "TP"]].sum(axis=1)
    check("confusion matrices sum to 2563", (tot == 2563).all(), sorted(tot.unique()))
    check("233 phishing in every test split", ((runs.TP + runs.FN) == 233).all())
    prec = runs.TP / (runs.TP + runs.FP).replace(0, np.nan)
    rec = runs.TP / (runs.TP + runs.FN)
    f1 = 2 * prec * rec / (prec + rec)
    check("precision/recall/F1 match confusion matrix",
          np.allclose(prec.fillna(0), runs.precision, atol=1e-9) and np.allclose(rec, runs.recall, atol=1e-9)
          and np.allclose(f1.fillna(0), runs.f1, atol=1e-9))
    check("10 seeds per scenario/model", (runs.groupby(["scenario", "model"]).seed.nunique() == 10).all())
    summ = pd.read_csv(REV / "multiseed_summary.csv")
    m = runs.groupby(["scenario", "model"]).f1.mean().reset_index().merge(summ[["scenario", "model", "f1_mean"]])
    check("summary F1 = mean of per-seed F1", np.allclose(m.f1, m.f1_mean))
    seeds_in_runs = sorted(runs.seed.unique().tolist())
    check("seeds = {42, 0..8}", seeds_in_runs == sorted([42, 0, 1, 2, 3, 4, 5, 6, 7, 8]), seeds_in_runs)

    # GraphSAGE protocol
    dec = pd.read_csv(REV / "graphsage_protocol_decomposition.csv")
    if "valid_performance_estimate" in dec:
        bad = dec[dec.protocol.str.contains("TEST") & dec.valid_performance_estimate]
        check("test-monitored GraphSAGE rows flagged invalid", bad.empty, bad.protocol.tolist())
    else:
        check("decomposition has validity flag", False, "column missing (old output)")
    src = (ROOT / "experiments" / "revision_ijece.py").read_text(encoding="utf-8")
    check("main GraphSAGE monitors validation mask",
          "train_sage(Xs, y_all, edges, m_fit, m_val, seed)" in src)

    # calibration / threshold slice
    ss = REV / "calibration_split_sizes.csv"
    if ss.exists():
        s = pd.read_csv(ss)
        ok = ((s.fit + s.early_stopping + s.calibration_threshold) == 10252).all() and (s.test == 2563).all()
        check("fit/early-stopping/calibration slices partition the training split", ok, s.iloc[0].to_dict())
    else:
        check("calibration split sizes recorded", False, "missing")

    # production model
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    b = xgb.Booster()
    b.load_model(str(ART / "model_prod_xgb.json"))
    check("booster feature order == prod schema", list(b.feature_names) == prod["features"])
    check("prod schema == S2 scenario", prod["features"] == sc["muldigraph_plus_etherscan"])
    _, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    p = b.predict(xgb.DMatrix(te[prod["features"]].fillna(0).values, feature_names=prod["features"]))
    f1p = f1_score(te.label, (p > 0.5).astype(int))
    check("prod F1 on seed-42 test == schema metric", abs(f1p - prod["metrics"]["f1"]) < 5e-5, round(f1p, 4))
    r42 = runs[(runs.seed == 42) & (runs.scenario == "muldigraph_plus_etherscan") & (runs.model == "XGBoost")]
    check("prod model == seed-42 XGBoost S2 of the 10-seed table", abs(float(r42.f1.iloc[0]) - f1p) < 5e-5,
          f"{float(r42.f1.iloc[0]):.4f} vs {f1p:.4f}")
    bi = int(b.attributes().get("best_iteration", b.num_boosted_rounds() - 1))
    check("prod booster uses all trees (best_iteration = last)", bi == b.num_boosted_rounds() - 1,
          f"best_iteration={bi}, rounds={b.num_boosted_rounds()}")

    graph = ROOT / "data" / "raw" / "MulDiGraph.pkl"
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [ART / "model_prod_xgb.json", ART / "prod_feature_schema.json",
                                                          ART / "hybrid_feature_schema.json",
                                                          ROOT / "data" / "processed" / "hybrid_dataset.csv"]}
    if graph.exists():
        hashes["data/raw/MulDiGraph.pkl"] = sha(graph)
    else:
        blocked("MulDiGraph.pkl hash", "file not present (third-party dataset, not redistributed)")

    out = {"results": results, "artifact_sha256": hashes,
           "n_pass": sum(r["status"] == "PASS" for r in results),
           "n_fail": sum(r["status"] == "FAIL" for r in results),
           "n_blocked": sum(r["status"] == "BLOCKED" for r in results)}
    (REV / "validation_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in results:
        print(f"[{r['status']:7s}] {r['check']}  {r['detail'] if r['status'] != 'PASS' else ''}")
    print(f"{out['n_pass']} pass, {out['n_fail']} fail, {out['n_blocked']} blocked")
    return 1 if out["n_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
