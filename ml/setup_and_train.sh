#!/bin/bash
# ml/setup_and_train.sh
# Quick setup script for ML pipeline

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Ethereum Phishing Detection - ML Pipeline Setup           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Install dependencies ─────────────────────────────────────────────
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# ── Step 2: Create directories ───────────────────────────────────────────────
echo "[2/5] Creating output directories..."
mkdir -p ml/artifacts
mkdir -p data/processed
mkdir -p data/raw
echo "✓ Directories created"
echo ""

# ── Step 3: Setup environment variables ──────────────────────────────────────
echo "[3/5] Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file (update with your API keys)"
    echo ""
    echo "➜ Please add your API keys to .env:"
    echo "  - ETHERSCAN_API_KEY"
    echo "  - FORTA_API_KEY"
    echo ""
else
    echo "✓ .env file already exists"
fi
echo ""

# ── Step 4: Train baseline models ────────────────────────────────────────────
echo "[4/5] Training all models (baseline + advanced)..."
cd ml
python train_all_models.py
cd ..
echo "✓ Training complete"
echo ""

# ── Step 5: Generate evaluation report ────────────────────────────────────────
echo "[5/5] Generating evaluation report..."
cd ml
python evaluate_models.py
cd ..
echo "✓ Evaluation complete"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup complete!                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Artifacts saved to: ml/artifacts/"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Fetch external data (optional):"
echo ""
echo "   # Set API keys in .env first"
echo "   export ETHERSCAN_API_KEY=\"your_key\""
echo "   export FORTA_API_KEY=\"your_key\""
echo "   python -m ml.data_fetchers"
echo ""
echo "2️⃣  Start prediction server:"
echo ""
echo "   cd api"
echo "   python main.py"
echo ""
echo "3️⃣  View model comparison:"
echo ""
echo "   cat ml/artifacts/model_comparison.csv"
echo "   open ml/artifacts/model_comparison.png"
echo ""
echo "4️⃣  Check documentation:"
echo ""
echo "   cat ml/README_ML_PIPELINE.md"
echo ""
