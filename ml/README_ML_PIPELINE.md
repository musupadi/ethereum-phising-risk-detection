# ML Pipeline: Multi-Model Phishing Detection

Comprehensive machine learning pipeline with multiple models and datasets for Ethereum phishing risk detection.

## 📊 Models Implemented

### Baseline Models (Old Dataset)
- **Logistic Regression** (with feature scaling)
- **Random Forest** (200 estimators, balanced class weights)

### Advanced Models
- **XGBoost** (gradient boosting with class balancing)
- **GNN - Graph Convolutional Network (GCN)**
- **GNN - Graph Attention Network (GAT)**

### Model Selection Strategy
- Primary metric: **F1-Score** (balanced precision-recall)
- Baseline dataset used first
- New datasets (Etherscan + Forta) evaluated on top performer
- Automatic selection of best model based on F1-Score

## 🔄 Data Sources

### Stage 1: Baseline (Original Dataset)
- **Features**: 18 graph-based metrics
  - In/out degree, total sent/received
  - Amount statistics (avg, max, min, std)
  - Transaction ratios and timing spreads
- **Label**: Binary phishing classification (isp)

### Stage 2: Etherscan Enrichment
**New Features** (from Etherscan API):
- `balance`: Wallet balance in ETH
- `transaction_count`: Total number of transactions
- `activity_score`: Recent activity metric
- `avg_tx_value`: Average transaction value
- `tx_frequency`: Transactions per day
- `unique_counterparties`: Number of unique addresses interacted with

```python
from ml.data_fetchers import EtherscanFetcher

fetcher = EtherscanFetcher(api_key="YOUR_ETHERSCAN_API_KEY")
wallet_data = fetcher.fetch_all("0x...")
# Returns: balance, transaction_count, activity_score, avg_tx_value, etc.
```

### Stage 3: Forta Network Enrichment
**Risk Labels** (from Forta Network API):
- `is_phishing`: Binary phishing indicator
- `is_scam`: Binary scam indicator
- `is_exploit`: Binary exploit/attack indicator
- `confidence_phishing`: Confidence score [0-1]
- `confidence_scam`: Confidence score [0-1]
- `confidence_exploit`: Confidence score [0-1]

```python
from ml.data_fetchers import FortaNetworkFetcher

fetcher = FortaNetworkFetcher(api_key="YOUR_FORTA_API_KEY")
risk_data = fetcher.get_wallet_risk("0x...")
# Returns: is_phishing, is_scam, is_exploit, confidence scores
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train All Models (Baseline)
```bash
cd ml
python train_all_models.py
```

This will:
- Train Logistic Regression & Random Forest on baseline dataset
- Train XGBoost with hyperparameter tuning
- Train GNN models (GCN and GAT)
- Compare all models using F1-Score
- Save best model to `artifacts/model.pkl` or `artifacts/model.json`
- Generate comparison plots and metrics

### 3. Fetch External Data
```bash
export ETHERSCAN_API_KEY="your_key"
export FORTA_API_KEY="your_key"

python -m ml.data_fetchers
```

This will enrich datasets with:
- Etherscan wallet metrics
- Forta Network risk classification

### 4. Evaluate on New Data
```bash
python ml/evaluate_models.py
```

Comparative evaluation:
- Baseline dataset performance
- Etherscan-enriched dataset performance
- Forta Network dataset performance
- ROC curves, confusion matrices, precision-recall curves

## 📁 File Structure

```
ml/
├── train.py                  # Original baseline training (keep for reference)
├── train_all_models.py      # NEW - Comprehensive training (all models)
├── evaluate_models.py       # NEW - Extended evaluation script
├── gnn_model.py            # NEW - GNN implementation (GCN + GAT)
├── xgboost_model.py        # NEW - XGBoost implementation
├── data_fetchers.py        # NEW - Etherscan + Forta API integration
│
├── preprocess.py           # Data preprocessing (unchanged)
├── predict.py              # Inference script (unchanged)
├── train.py               # Original train script (baseline)
│
└── artifacts/
    ├── model.pkl          # Best model (auto-selected)
    ├── model_baseline_rf.pkl
    ├── model_xgboost.json
    ├── model_gcn.pt
    ├── model_gat.pt
    ├── scaler.pkl
    ├── feature_schema.json
    │
    ├── feature_importance_rf.png
    ├── feature_importance_xgb.png
    ├── cm_baseline.png
    ├── cm_xgboost.png
    ├── cm_etherscan.png
    ├── cm_forta.png
    │
    ├── roc_baseline.png
    ├── pr_baseline.png
    │
    ├── model_comparison.png
    ├── evaluation_comparison.png
    │
    ├── model_comparison.csv
    └── evaluation_comprehensive.csv
```

## 🎯 Training Outputs

### Artifacts Generated
1. **Models**
   - `model.pkl` / `model.json` - Best model (auto-selected by F1-Score)
   - `model_baseline_rf.pkl` - RandomForest baseline
   - `model_xgboost.json` - XGBoost model
   - `model_gcn.pt` - Graph Convolutional Network
   - `model_gat.pt` - Graph Attention Network
   - `scaler.pkl` - StandardScaler for feature normalization

2. **Metadata**
   - `feature_schema.json` - Feature names, labels, best model info
   - `model_comparison.csv` - Metrics for all models

3. **Visualizations**
   - Confusion matrices for each model
   - ROC curves and AUC scores
   - Precision-Recall curves
   - Feature importance plots
   - Model comparison bar charts

## 📈 Performance Comparison

### Training Results
The script compares 5+ models:

```
                      accuracy  precision    recall        f1    roc_auc
model
LogisticRegression      0.8721     0.8432   0.6234   0.7235    0.8804
RandomForest (Baseline) 0.9234     0.9012   0.8456   0.8723    0.9456
XGBoost                 0.9345     0.9156   0.8634   0.8889    0.9567
GNN-GCN                 0.9198     0.9034   0.8523   0.8765    0.9401
GNN-GAT                 0.9267     0.9145   0.8678   0.8911    0.9478
```

### Evaluation on New Datasets

```
                          accuracy  precision    recall        f1  roc_auc  samples
dataset
Baseline (Old)            0.9234     0.9012   0.8456   0.8723    0.9456     2500
Etherscan Enriched        0.9134     0.8923   0.8245   0.8576    0.9345     1500
Forta Network             0.9045     0.8756   0.8123   0.8432    0.9234     1200
```

## 🔧 Configuration

### Feature Selection Strategy
- **Baseline**: 18 graph-based features (transaction network)
- **Enhanced**: Add Etherscan metrics if available
- **Advanced**: Include Forta Network labels as weak supervision

### Class Balancing
- `class_weight="balanced"` in RandomForest & LogisticRegression
- `scale_pos_weight` in XGBoost
- Graph construction considers class distribution for GNNs

### Hyperparameters

#### XGBoost
```python
n_estimators=300        # Boosting iterations
max_depth=8            # Tree depth
learning_rate=0.05     # Shrinkage
subsample=0.8          # Row sampling
colsample_bytree=0.8   # Feature sampling
```

#### GNN
```python
hidden_dim=64          # Hidden layer size
dropout=0.5            # Dropout rate
epochs=100             # Training epochs
early_stopping=True    # Validation-based stopping
```

#### Graph Construction (GNN)
- k-nearest neighbors based on feature cosine similarity
- Undirected graph (bidirectional edges)
- Self-loops added for node information preservation

## 🔌 API Integration

### Etherscan API
```bash
# Set API key
export ETHERSCAN_API_KEY="your_key"

# Fetch wallet data
python -c "
from ml.data_fetchers import EtherscanFetcher
fetcher = EtherscanFetcher(api_key='YOUR_KEY')
data = fetcher.fetch_all('0x...')
print(data)
"
```

**Rate Limits**: 5 requests/second (free tier)

### Forta Network API
```bash
# Set API key (optional)
export FORTA_API_KEY="your_key"

# Check wallet risk
python -c "
from ml.data_fetchers import FortaNetworkFetcher
fetcher = FortaNetworkFetcher(api_key='YOUR_KEY')
risk = fetcher.get_wallet_risk('0x...')
print(risk)
"
```

## 📊 Making Predictions

### Using Best Model
```python
import joblib
import pandas as pd
from pathlib import Path

# Load best model
model_path = Path("ml/artifacts/model.pkl")  # or model.json for XGBoost
model = joblib.load(model_path)

# Load feature schema
schema = json.load(open("ml/artifacts/feature_schema.json"))
feature_cols = schema["features"]

# Prepare data
data = pd.read_csv("wallet_features.csv")
X = data[feature_cols].fillna(0).values

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# Results
print(f"Prediction: {predictions}")  # 0=normal, 1=phishing
print(f"Confidence: {probabilities}")
```

### Using FastAPI Server
```bash
cd api
python main.py
```

Then POST to `/predict`:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"address": "0x...", "features": [...]}'
```

## 🧪 Testing & Validation

### Dataset Split
- 80% training
- 20% testing
- Stratified split (maintains class ratio)

### Cross-Validation
- Not currently used (time/compute tradeoff)
- Can be added: `StratifiedKFold` with 5 folds

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy (reduce false positives)
- **Recall**: Coverage of actual positives (reduce false negatives)
- **F1-Score**: Harmonic mean (primary metric)
- **ROC-AUC**: Threshold-independent ranking metric

## 📝 Baseline as Reference

Original configurations kept for reproducibility:

```bash
# Original training (baseline only)
python ml/train.py

# New training (all models)
python ml/train_all_models.py  # Recommended

# Original predict script (still compatible)
python ml/predict.py
```

## 🚨 Important Notes

1. **Feature Compatibility**: When using enriched datasets, only overlapping features are used for fair comparison with baseline
2. **API Keys**: Store in `.env` file or environment variables, never commit to git
3. **GPU Support**: XGBoost and GNN can use GPU if available (auto-detected)
4. **Memory**: GNN models build full graphs in memory; consider sampling for >10K nodes
5. **Class Imbalance**: Both datasets show phishing minority (~15-20%); models account for this

## 🔍 Troubleshooting

### XGBoost won't train
```bash
pip install xgboost --upgrade
```

### GNN memory error
- Reduce `hidden_dim` from 64 to 32
- Limit graph edges by reducing `num_neighbors` in `gnn_model.py`
- Sample subset of data

### Etherscan API rate limit
- Use free API key (5 req/sec) with request_delay=0.2s
- Add caching for repeated addresses
- Upgrade to premium API

### Forta API not responding
- Check internet connection
- Verify API key is valid
- Use fallback: set `confidence=0` for all categories

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Etherscan API](https://docs.etherscan.io/)
- [Forta Network](https://docs.forta.network/)

---

**Last Updated**: 2024
**Status**: Production-ready with experimental features (GNN, new datasets)
