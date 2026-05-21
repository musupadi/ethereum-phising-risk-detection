"""
Generate sample dataset untuk testing ML pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # Go up to project root
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print(f"ROOT: {ROOT}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")

# ── Generate synthetic data ────────────────────────────────────────────────────
np.random.seed(42)

n_train = 500
n_test = 125

# Feature names (18 features)
features = [
    "in_degree", "out_degree",
    "unique_senders", "unique_receivers",
    "total_received", "total_sent",
    "avg_in_amount", "avg_out_amount",
    "max_in_amount", "max_out_amount",
    "min_in_amount", "min_out_amount",
    "std_in_amount", "std_out_amount",
    "in_out_ratio", "sent_received_ratio",
    "timestamp_spread_in", "timestamp_spread_out"
]

# ── Create training set ────────────────────────────────────────────────────────
# Normal transactions (label=0)
normal_data = []
for i in range(int(n_train * 0.85)):
    row = {
        "node_id": f"0x_normal_{i:05d}",
        "label": 0,
        "in_degree": np.random.poisson(3),
        "out_degree": np.random.poisson(3),
        "unique_senders": np.random.poisson(2),
        "unique_receivers": np.random.poisson(2),
        "total_received": np.random.exponential(1.0),
        "total_sent": np.random.exponential(1.0),
        "avg_in_amount": np.random.exponential(0.5),
        "avg_out_amount": np.random.exponential(0.5),
        "max_in_amount": np.random.exponential(2.0),
        "max_out_amount": np.random.exponential(2.0),
        "min_in_amount": np.random.exponential(0.1),
        "min_out_amount": np.random.exponential(0.1),
        "std_in_amount": np.random.exponential(0.3),
        "std_out_amount": np.random.exponential(0.3),
        "in_out_ratio": np.random.uniform(0.5, 2.0),
        "sent_received_ratio": np.random.uniform(0.5, 2.0),
        "timestamp_spread_in": np.random.exponential(100) * 3600,  # seconds
        "timestamp_spread_out": np.random.exponential(100) * 3600,
    }
    normal_data.append(row)

# Phishing transactions (label=1) - different distribution
phishing_data = []
for i in range(int(n_train * 0.15)):
    row = {
        "node_id": f"0x_phish_{i:05d}",
        "label": 1,
        "in_degree": np.random.poisson(8),  # Higher activity
        "out_degree": np.random.poisson(8),
        "unique_senders": np.random.poisson(6),
        "unique_receivers": np.random.poisson(6),
        "total_received": np.random.exponential(3.0),
        "total_sent": np.random.exponential(3.0),
        "avg_in_amount": np.random.exponential(1.5),
        "avg_out_amount": np.random.exponential(1.5),
        "max_in_amount": np.random.exponential(5.0),
        "max_out_amount": np.random.exponential(5.0),
        "min_in_amount": np.random.exponential(0.05),
        "min_out_amount": np.random.exponential(0.05),
        "std_in_amount": np.random.exponential(0.8),
        "std_out_amount": np.random.exponential(0.8),
        "in_out_ratio": np.random.uniform(0.2, 5.0),  # More extreme ratios
        "sent_received_ratio": np.random.uniform(0.2, 5.0),
        "timestamp_spread_in": np.random.exponential(300) * 3600,
        "timestamp_spread_out": np.random.exponential(300) * 3600,
    }
    phishing_data.append(row)

train_data = pd.DataFrame(normal_data + phishing_data)
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
train_data.to_csv(DATA_PROCESSED / "train.csv", index=False)

print(f"✓ Generated train.csv: {len(train_data)} rows")

# ── Create test set (same distribution) ────────────────────────────────────────
normal_test = []
for i in range(int(n_test * 0.85)):
    row = {
        "node_id": f"0x_ntest_{i:05d}",
        "label": 0,
        "in_degree": np.random.poisson(3),
        "out_degree": np.random.poisson(3),
        "unique_senders": np.random.poisson(2),
        "unique_receivers": np.random.poisson(2),
        "total_received": np.random.exponential(1.0),
        "total_sent": np.random.exponential(1.0),
        "avg_in_amount": np.random.exponential(0.5),
        "avg_out_amount": np.random.exponential(0.5),
        "max_in_amount": np.random.exponential(2.0),
        "max_out_amount": np.random.exponential(2.0),
        "min_in_amount": np.random.exponential(0.1),
        "min_out_amount": np.random.exponential(0.1),
        "std_in_amount": np.random.exponential(0.3),
        "std_out_amount": np.random.exponential(0.3),
        "in_out_ratio": np.random.uniform(0.5, 2.0),
        "sent_received_ratio": np.random.uniform(0.5, 2.0),
        "timestamp_spread_in": np.random.exponential(100) * 3600,
        "timestamp_spread_out": np.random.exponential(100) * 3600,
    }
    normal_test.append(row)

phishing_test = []
for i in range(int(n_test * 0.15)):
    row = {
        "node_id": f"0x_ptest_{i:05d}",
        "label": 1,
        "in_degree": np.random.poisson(8),
        "out_degree": np.random.poisson(8),
        "unique_senders": np.random.poisson(6),
        "unique_receivers": np.random.poisson(6),
        "total_received": np.random.exponential(3.0),
        "total_sent": np.random.exponential(3.0),
        "avg_in_amount": np.random.exponential(1.5),
        "avg_out_amount": np.random.exponential(1.5),
        "max_in_amount": np.random.exponential(5.0),
        "max_out_amount": np.random.exponential(5.0),
        "min_in_amount": np.random.exponential(0.05),
        "min_out_amount": np.random.exponential(0.05),
        "std_in_amount": np.random.exponential(0.8),
        "std_out_amount": np.random.exponential(0.8),
        "in_out_ratio": np.random.uniform(0.2, 5.0),
        "sent_received_ratio": np.random.uniform(0.2, 5.0),
        "timestamp_spread_in": np.random.exponential(300) * 3600,
        "timestamp_spread_out": np.random.exponential(300) * 3600,
    }
    phishing_test.append(row)

test_data = pd.DataFrame(normal_test + phishing_test)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data.to_csv(DATA_PROCESSED / "test.csv", index=False)

print(f"✓ Generated test.csv: {len(test_data)} rows")
print(f"\nDataset Summary:")
print(f"  Train: {len(train_data[train_data['label']==0])} normal, {len(train_data[train_data['label']==1])} phishing")
print(f"  Test:  {len(test_data[test_data['label']==0])} normal, {len(test_data[test_data['label']==1])} phishing")
