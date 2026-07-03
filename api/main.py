"""
api/main.py
-----------
FastAPI application exposing:
  POST /analyze    – ML inference + optional chain write
  GET  /risk/{wallet} – On-chain risk record lookup
  GET  /health     – liveness probe
  GET  /demo/rows  – number of test rows available (demo mode)
"""

import logging
import re
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.schemas import AnalyzeRequest, AnalyzeResponse, RiskRecord
from api.services import ml_service, chain_service
from api.config import CHAIN_ENABLED, MODEL_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ethereum Phishing Risk Registry API",
    description=(
        "ML-powered phishing risk scoring for Ethereum wallets "
        "with optional on-chain result anchoring via Hardhat/Solidity."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "chain_enabled": CHAIN_ENABLED,
        "chain_connected": chain_service.is_connected(),
    }


@app.get("/demo/rows", tags=["meta"])
def demo_rows():
    """Return the number of test rows available for demo mode."""
    try:
        return {"total_rows": ml_service.get_test_row_count()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/records", tags=["risk"])
def records():
    """List every wallet scored on-chain (latest per wallet), from event logs."""
    try:
        recs = chain_service.get_all_records()
        return {"count": len(recs), "records": recs}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Chain read error: {exc}")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["risk"])
def analyze(req: AnalyzeRequest):
    """
    Run ML phishing risk inference for a wallet address.
    - Provide `row_id` for fast demo mode (features loaded from test.csv).
    - Provide `features` dict for manual input.
    - Provide only `wallet` to scan-by-address: features are looked up from the
      dataset (train+test) automatically.
    On success, stores the result on-chain (if CHAIN_ENABLED=true).
    """
    # ── ML inference ──────────────────────────────────────────────────────────
    try:
        if req.row_id is not None:
            ml_result = ml_service.predict_from_row(req.row_id)
        elif req.features:
            ml_result = ml_service.predict_from_features(req.features)
        else:
            # Scan-by-address: look up precomputed features for this wallet.
            ml_result = ml_service.predict_from_address(req.wallet)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0] if exc.args else "Wallet not found.")
    except (FileNotFoundError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception("ML inference error")
        raise HTTPException(status_code=500, detail=f"ML error: {exc}")

    probability: float = ml_result["probability"]
    risk_score: int = ml_result["risk_score"]
    risk_level: str = ml_result["risk_level"]

    # Canonical wallet: for demo/scan the score belongs to the dataset wallet
    # (node_id), NOT a free-typed label — so we report & store that real address.
    node_id = str(ml_result.get("node_id", "") or "")
    wallet = node_id if re.fullmatch(r"0x[0-9a-fA-F]{40}", node_id) else req.wallet

    # ── Chain write ───────────────────────────────────────────────────────────
    tx_hash = None
    if CHAIN_ENABLED:
        try:
            tx_hash = chain_service.set_risk(wallet, risk_score, MODEL_VERSION)
        except Exception as exc:
            log.warning("Chain write failed (non-fatal): %s", exc)
            tx_hash = None

    return AnalyzeResponse(
        wallet=wallet,
        probability=round(probability, 6),
        risk_score=risk_score,
        risk_level=risk_level,
        tx_hash=tx_hash,
        true_label=ml_result.get("true_label"),
        dataset_split=ml_result.get("split"),
    )


@app.get("/risk/{wallet}", response_model=RiskRecord, tags=["risk"])
def get_risk(wallet: str):
    """
    Retrieve the on-chain risk record for a wallet address.
    Returns zeros if the wallet has never been assessed or chain is disabled.
    """
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise HTTPException(status_code=422, detail="Invalid Ethereum address format.")
    try:
        record = chain_service.get_risk(wallet)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Chain read error: {exc}")

    return RiskRecord(
        wallet=wallet,
        risk_score=record["risk_score"],
        timestamp=record["timestamp"],
        model_version=record.get("model_version", ""),
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from api.config import API_HOST, API_PORT
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
