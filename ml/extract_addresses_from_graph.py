import os
import pickle
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
OUTPUT_CSV = DATA_RAW / "addresses.csv"
PHISHING_CSV = DATA_RAW / "addresses_phishing.csv"
NORMAL_CSV = DATA_RAW / "addresses_normal.csv"

GRAPH_CANDIDATES = [
    DATA_RAW / "MulDiGraph.pkl",
    ROOT / "Ethereum Phishing Transaction Network" / "MulDiGraph.pkl",
]

ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
RANDOM_SEED = 42


def find_graph_path() -> Path | None:
    for path in GRAPH_CANDIDATES:
        if path.exists():
            return path
    return None


def main() -> None:
    graph_path = find_graph_path()
    if graph_path is None:
        print("MulDiGraph.pkl not found.")
        print("Put the file in one of these locations:")
        for path in GRAPH_CANDIDATES:
            print(f"- {path}")
        return

    print(f"Loading graph: {graph_path}")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    print(f"Graph loaded. Total nodes: {graph.number_of_nodes():,}")

    phishing_rows = []
    normal_rows = []

    for node, attrs in graph.nodes(data=True):
        node_str = str(node)
        if not ADDRESS_RE.match(node_str):
            continue

        label = int(attrs.get("isp", 0) or 0)
        row = {
            "address": node_str,
            "label": label,
            "class_name": "phishing" if label == 1 else "normal",
            "source_dataset": "EPTransNet",
        }
        if label == 1:
            phishing_rows.append(row)
        else:
            normal_rows.append(row)

    phishing_df = pd.DataFrame(phishing_rows).drop_duplicates(subset=["address"]).sort_values("address")
    normal_df = pd.DataFrame(normal_rows).drop_duplicates(subset=["address"]).sort_values("address")

    # Keep all phishing addresses, sample normal addresses with configurable ratio.
    ratio = int(os.getenv("NORMAL_SAMPLE_RATIO", "10"))
    max_normals = len(phishing_df) * ratio if len(phishing_df) > 0 else len(normal_df)
    max_normals = min(max_normals, len(normal_df))
    random.seed(RANDOM_SEED)
    if max_normals < len(normal_df):
        sampled_indices = random.sample(range(len(normal_df)), max_normals)
        normal_sample_df = normal_df.iloc[sampled_indices].sort_values("address").reset_index(drop=True)
    else:
        normal_sample_df = normal_df.reset_index(drop=True)

    combined_df = pd.concat([phishing_df, normal_sample_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["address"]).reset_index(drop=True)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    phishing_df.to_csv(PHISHING_CSV, index=False)
    normal_sample_df.to_csv(NORMAL_CSV, index=False)
    combined_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Valid phishing addresses: {len(phishing_df):,}")
    print(f"Valid normal addresses : {len(normal_df):,}")
    print(f"Sampled normal saved   : {len(normal_sample_df):,}")
    print(f"Combined enrich input  : {len(combined_df):,}")
    print(f"Saved: {PHISHING_CSV}")
    print(f"Saved: {NORMAL_CSV}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
