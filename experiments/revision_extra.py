"""
experiments/revision_extra.py
-----------------------------
Additional robustness analyses for the IJECE revision (Paper ID 42848). Nothing here
changes the main protocol of experiments/revision_ijece.py.

  T  Temporal split: wallets ordered by their first MulDiGraph transaction; the earliest
     80% train, the latest 20% test. Graph features still summarise each wallet's whole
     2015-2019 history (they are not truncated at the cut-off), so this is a
     distribution-shift test, not a strict prospective simulation.
  P  Lower-prevalence sensitivity (S1 only, because Etherscan features exist only for the
     12,815 enriched wallets and re-enrichment would need API calls): extra normal wallets
     are sampled from MulDiGraph and featurised with the same ml/preprocess.py code, giving
     phishing:normal = 1:10, 1:20, 1:50, 1:100. Plus an analytic prior-shift estimate of
     precision for S2 at lower prevalence from the measured TPR/FPR.
  F  Feature selection without test labels: the subset (full / corr_pruned / top_20 /
     top_18 / top_15) is chosen by 5-fold CV F1 inside the training split, then scored once
     on test. The original ml/feature_selection.py chose the subset by TEST F1.

Outputs: outputs/revision/temporal_split*.csv, prevalence_sensitivity*.csv,
         prevalence_prior_shift_estimate.csv, feature_selection_cv*.csv
Run    : .venv/Scripts/python.exe experiments/revision_extra.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT))
from preprocess import extract_node_features  # noqa: E402
from xgboost_model import XGBoostModel  # noqa: E402
from ml.feature_selection import apply_caps, fit_caps  # noqa: E402

SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
OUT = ROOT / "outputs" / "revision"
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("extra")
logging.getLogger("xgboost_model").setLevel(logging.WARNING)


def met(y, p, thr=0.5):
    pr = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, pr, zero_division=0), "recall": recall_score(y, pr, zero_division=0),
            "f1": f1_score(y, pr, zero_division=0), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def xgb_proba(X_tr, y_tr, X_te, names, seed):
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    m = XGBoostModel(n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, random_state=seed)
    m.fit(X_fit, y_fit, X_val, y_val, feature_names=names, early_stopping_rounds=40)
    return m.predict_proba(X_te)[:, 1]


def three_models(X_tr, y_tr, X_te, names, seed):
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(sc.transform(X_tr), y_tr)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                                random_state=seed).fit(X_tr, y_tr)
    return {"LogisticRegression": lr.predict_proba(sc.transform(X_te))[:, 1],
            "RandomForest": rf.predict_proba(X_te)[:, 1],
            "XGBoost": xgb_proba(X_tr, y_tr, X_te, names, seed)}


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(ROOT / "data" / "processed" / "hybrid_dataset.csv")
    df["addr"] = df["node_id"].astype(str).str.lower()
    schema = json.loads((ROOT / "ml" / "artifacts" / "hybrid_feature_schema.json").read_text(encoding="utf-8"))
    sc = schema["scenarios"]
    s1, s2 = sc["muldigraph_only"], sc["muldigraph_plus_etherscan"]
    y = df["label"].values.astype(int)
    with open(ROOT / "data" / "raw" / "MulDiGraph.pkl", "rb") as f:
        G = pickle.load(f)
    log.info("graph loaded (%.0fs)", time.time() - t0)

    # ── T. temporal split ─────────────────────────────────────────────────────
    first = []
    for a in df["addr"]:
        ts = [d.get("timestamp") for _, _, d in G.in_edges(a, data=True)] + \
             [d.get("timestamp") for _, _, d in G.out_edges(a, data=True)]
        ts = [t for t in ts if t is not None]
        first.append(min(ts) if ts else np.nan)
    df["first_seen_ts"] = first
    order = df["first_seen_ts"].rank(method="first").values
    cut = np.quantile(order, 0.8)
    tr, te = np.where(order <= cut)[0], np.where(order > cut)[0]
    cut_time = pd.to_datetime(df["first_seen_ts"].iloc[tr].max(), unit="s", utc=True)
    trows = []
    for scen, cols in [("muldigraph_only", s1), ("muldigraph_plus_etherscan", s2)]:
        X = df[cols].fillna(0).values.astype(float)
        for seed in SEEDS:
            for name, p in three_models(X[tr], y[tr], X[te], cols, seed).items():
                trows.append({"split": "temporal", "scenario": scen, "model": name, "seed": seed, **met(y[te], p)})
    tdf = pd.DataFrame(trows)
    tdf.to_csv(OUT / "temporal_split_all_runs.csv", index=False)
    tsum = tdf.groupby(["scenario", "model"], sort=False)[["precision", "recall", "f1", "roc_auc", "pr_auc", "FP", "FN"]] \
        .agg(["mean", "std"])
    tsum.columns = [f"{a}_{b}" for a, b in tsum.columns]
    tsum = tsum.reset_index()
    tsum["train_n"], tsum["train_phishing"] = len(tr), int(y[tr].sum())
    tsum["test_n"], tsum["test_phishing"] = len(te), int(y[te].sum())
    tsum["cutoff_first_seen_utc"] = str(cut_time)
    tsum.to_csv(OUT / "temporal_split_summary.csv", index=False)
    log.info("temporal done: train %d (%d phish) test %d (%d phish) cut %s", len(tr), y[tr].sum(), len(te),
             y[te].sum(), cut_time)

    # ── P. lower prevalence (S1) ──────────────────────────────────────────────
    import re
    are = re.compile(r"^0x[a-fA-F0-9]{40}$")
    used = set(df["addr"])
    pool = [str(n) for n, d in G.nodes(data=True)
            if are.match(str(n)) and int(d.get("isp", 0) or 0) == 0 and str(n).lower() not in used]
    pool.sort()
    rng = np.random.default_rng(2026)
    n_extra = 100 * 1165 - 11650
    extra_nodes = list(rng.choice(pool, n_extra, replace=False))
    t1 = time.time()
    extra = pd.DataFrame([extract_node_features(G, n) for n in extra_nodes])[s1]
    extra["label"] = 0
    log.info("featurised %d extra normals in %.0fs", n_extra, time.time() - t1)
    base = df[s1 + ["label"]]
    phish, norm0 = base[base.label == 1], base[base.label == 0]
    allnorm = pd.concat([norm0, extra], ignore_index=True)  # first 11,650 = original sample (nested design)
    prow = []
    for r in [10, 20, 50, 100]:
        sub = pd.concat([phish, allnorm.iloc[: r * 1165]], ignore_index=True)
        X, yy = sub[s1].fillna(0).values.astype(float), sub["label"].values
        for seed in SEEDS[:5]:
            Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.2, random_state=seed, stratify=yy)
            ps = three_models(Xtr, ytr, Xte, s1, seed)
            for name in ["RandomForest", "XGBoost"]:
                prow.append({"scenario": "muldigraph_only", "ratio": f"1:{r}", "n_total": len(sub),
                             "prevalence_test": float(yte.mean()), "model": name, "seed": seed, **met(yte, ps[name])})
        log.info("prevalence 1:%d done", r)
    pdf = pd.DataFrame(prow)
    pdf.to_csv(OUT / "prevalence_sensitivity_all_runs.csv", index=False)
    psum = pdf.groupby(["ratio", "model"], sort=False)[["n_total", "prevalence_test", "precision", "recall", "f1",
                                                         "roc_auc", "pr_auc", "FP", "FN"]].agg(["mean", "std"])
    psum.columns = [f"{a}_{b}" for a, b in psum.columns]
    psum.reset_index().to_csv(OUT / "prevalence_sensitivity_summary.csv", index=False)

    # Analytic prior-shift estimate: precision(pi) = TPR*pi / (TPR*pi + FPR*(1-pi)), threshold 0.5.
    runs = pd.read_csv(OUT / "multiseed_all_runs.csv")
    est = []
    for scen in ["muldigraph_only", "muldigraph_plus_etherscan"]:
        for model in ["RandomForest", "XGBoost"]:
            r_ = runs[(runs.scenario == scen) & (runs.model == model)]
            tpr = (r_.TP / (r_.TP + r_.FN)).mean()
            fpr = (r_.FP / (r_.FP + r_.TN)).mean()
            for label, pi in [("1:10 (study)", 1 / 11), ("1:20", 1 / 21), ("1:50", 1 / 51), ("1:100", 1 / 101),
                              ("1:1000", 1 / 1001), ("MulDiGraph prevalence 0.039%", 1165 / 2973489)]:
                prec = tpr * pi / (tpr * pi + fpr * (1 - pi))
                est.append({"scenario": scen, "model": model, "prevalence": label, "pi": pi, "TPR": tpr, "FPR": fpr,
                            "estimated_precision": prec, "estimated_f1": 2 * prec * tpr / (prec + tpr),
                            "type": "analytic estimate (assumes TPR/FPR transfer unchanged)"})
    pd.DataFrame(est).to_csv(OUT / "prevalence_prior_shift_estimate.csv", index=False)

    # ── F. feature selection by train-only CV ─────────────────────────────────
    clean = pd.read_csv(ROOT / "data" / "processed" / "hybrid_dataset_clean.csv")
    feats = [c for c in s2 if c in clean.columns and c != "created_at_ts"]
    frows, chosen = [], []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(clean[feats], clean["label"].values, test_size=0.2,
                                              random_state=seed, stratify=clean["label"])
        caps = fit_caps(Xtr)
        Xtr_c, Xte_c = apply_caps(Xtr, caps), apply_caps(Xte, caps)
        spw = float((ytr == 0).sum() / (ytr == 1).sum())
        mk = lambda: XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85,  # noqa: E731
                                   colsample_bytree=0.85, scale_pos_weight=spw, eval_metric="logloss",
                                   n_jobs=-1, random_state=seed)
        xg = mk().fit(Xtr_c, ytr)
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1,
                                    random_state=seed).fit(Xtr_c, ytr)
        imp = pd.DataFrame({"f": feats, "x": xg.feature_importances_, "r": rf.feature_importances_})
        imp["c"] = (imp.x / imp.x.sum() + imp.r / imp.r.sum()) / 2
        ranked = list(imp.sort_values("c", ascending=False).f)
        rank = {f: i for i, f in enumerate(ranked)}
        corr = Xtr_c.corr().abs()
        drop = set()
        for i, a in enumerate(feats):
            for b in feats[i + 1:]:
                if corr.loc[a, b] >= 0.95:
                    drop.add(a if rank[a] > rank[b] else b)
        subsets = {"full": ranked, "corr_pruned": [f for f in ranked if f not in drop],
                   "top_20": ranked[:20], "top_18": ranked[:18], "top_15": ranked[:15]}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        cvf1 = {}
        for name, fl in subsets.items():
            s = []
            for a, b in cv.split(Xtr_c, ytr):
                m = mk().fit(Xtr_c.iloc[a][fl], ytr[a])
                s.append(f1_score(ytr[b], m.predict(Xtr_c.iloc[b][fl])))
            cvf1[name] = float(np.mean(s))
        best = max(cvf1, key=cvf1.get)
        for name, fl in subsets.items():
            p = mk().fit(Xtr_c[fl], ytr).predict_proba(Xte_c[fl])[:, 1]
            frows.append({"seed": seed, "subset": name, "n_features": len(fl), "cv_f1_train": cvf1[name],
                          "selected_by_cv": name == best, "dropped_by_corr": ";".join(sorted(drop)), **met(yte, p)})
        chosen.append(best)
        log.info("feature selection seed %d: chose %s", seed, best)
    fdf = pd.DataFrame(frows)
    fdf.to_csv(OUT / "feature_selection_cv_all_runs.csv", index=False)
    fs = fdf.groupby("subset", sort=False)[["n_features", "cv_f1_train", "f1", "precision", "recall", "pr_auc"]] \
        .agg(["mean", "std"])
    fs.columns = [f"{a}_{b}" for a, b in fs.columns]
    fs = fs.reset_index()
    fs["times_selected_by_cv"] = fs.subset.map(pd.Series(chosen).value_counts()).fillna(0).astype(int)
    sel = fdf[fdf.selected_by_cv]
    fs.to_csv(OUT / "feature_selection_cv_summary.csv", index=False)
    (OUT / "feature_selection_cv_selected.json").write_text(json.dumps({
        "selected_per_seed": dict(zip(SEEDS, chosen)),
        "test_f1_of_cv_selected_subset_mean": float(sel.f1.mean()), "std": float(sel.f1.std())}, indent=2))
    log.info("all done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
