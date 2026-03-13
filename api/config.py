"""
api/config.py
-------------
Central configuration loaded from environment variables (with sane defaults
for the local Hardhat demo).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Hardhat / Web3 ────────────────────────────────────────────────────────────
RPC_URL: str = os.getenv("RPC_URL", "http://127.0.0.1:8545")

# Path to the deployment.json produced by deploy.ts
DEPLOYMENT_JSON: Path = Path(
    os.getenv(
        "DEPLOYMENT_JSON",
        str(ROOT / "chain" / "artifacts" / "deployment.json"),
    )
)

# Path to the ABI JSON produced by deploy.ts
ABI_JSON: Path = Path(
    os.getenv(
        "ABI_JSON",
        str(ROOT / "chain" / "artifacts" / "RiskRegistry.abi.json"),
    )
)

# Which Hardhat account index to use as the tx signer (default: account[0])
SIGNER_PRIVATE_KEY: str = os.getenv(
    "SIGNER_PRIVATE_KEY",
    # Hardhat default account #0 private key – NEVER use in production
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
)

MODEL_VERSION: str = os.getenv("MODEL_VERSION", "rf-v1")

# ── ML ────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR: Path = ROOT / "ml" / "artifacts"
TEST_CSV: Path = ROOT / "data" / "processed" / "test.csv"

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Blockchain write toggle ───────────────────────────────────────────────────
# Set CHAIN_ENABLED=false to skip writing to chain during demo without a node
CHAIN_ENABLED: bool = os.getenv("CHAIN_ENABLED", "true").lower() in ("1", "true", "yes")
