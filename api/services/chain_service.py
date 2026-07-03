"""
api/services/chain_service.py
------------------------------
web3.py bridge to the local Hardhat RiskRegistry contract.

If CHAIN_ENABLED=false (env) every write returns a dummy tx hash and
every read returns zeroed values – useful for demos without a running node.
"""

import json
import logging
from typing import Optional

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from api.config import (
    RPC_URL, DEPLOYMENT_JSON, ABI_JSON,
    SIGNER_PRIVATE_KEY, MODEL_VERSION, CHAIN_ENABLED,
)

log = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_w3: Optional[Web3] = None
_contract = None
_account = None


def _init():
    global _w3, _contract, _account

    if not CHAIN_ENABLED:
        log.warning("CHAIN_ENABLED=false – chain calls are mocked.")
        return

    if _w3 is not None:
        return

    _w3 = Web3(Web3.HTTPProvider(RPC_URL))
    # Inject POA middleware (needed for Hardhat / Geth clique)
    _w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    if not _w3.is_connected():
        raise ConnectionError(f"Cannot connect to Ethereum node at {RPC_URL}")

    # Load ABI
    with open(ABI_JSON) as f:
        abi = json.load(f)

    # Load deployed address
    with open(DEPLOYMENT_JSON) as f:
        deployment = json.load(f)
    contract_address = Web3.to_checksum_address(deployment["address"])

    _contract = _w3.eth.contract(address=contract_address, abi=abi)
    _account = _w3.eth.account.from_key(SIGNER_PRIVATE_KEY)
    log.info("Connected to chain. Contract: %s  Signer: %s", contract_address, _account.address)


# ── Public API ─────────────────────────────────────────────────────────────────

def set_risk(wallet: str, score: int, model_version: str = MODEL_VERSION) -> str:
    """
    Call setRisk on the contract.

    Returns
    -------
    str – transaction hash (hex)
    """
    if not CHAIN_ENABLED:
        log.info("Mock set_risk for %s score=%d", wallet, score)
        return "0x" + "0" * 64

    _init()
    wallet_addr = Web3.to_checksum_address(wallet)

    nonce = _w3.eth.get_transaction_count(_account.address)
    tx = _contract.functions.setRisk(wallet_addr, score, model_version).build_transaction({
        "from": _account.address,
        "nonce": nonce,
        "gas": 200_000,
        "gasPrice": _w3.eth.gas_price,
    })
    signed = _w3.eth.account.sign_transaction(tx, private_key=SIGNER_PRIVATE_KEY)
    tx_hash = _w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = _w3.eth.wait_for_transaction_receipt(tx_hash)
    hex_hash = receipt.transactionHash.hex()
    log.info("setRisk tx: %s  (block %d)", hex_hash, receipt.blockNumber)
    return hex_hash


def get_risk(wallet: str) -> dict:
    """
    Call getRisk on the contract.

    Returns
    -------
    dict with keys: risk_score, timestamp, model_version
    """
    if not CHAIN_ENABLED:
        return {"risk_score": 0, "timestamp": 0, "model_version": ""}

    _init()
    wallet_addr = Web3.to_checksum_address(wallet)
    score, ts, mv = _contract.functions.getRisk(wallet_addr).call()
    return {"risk_score": score, "timestamp": ts, "model_version": mv}


def get_all_records() -> list:
    """
    Return every wallet ever scored on-chain (latest record per wallet),
    reconstructed from the contract's `RiskUpdated` event log.

    Returns a list of dicts {wallet, risk_score, timestamp, model_version},
    sorted by timestamp (newest first). Empty list if chain disabled / no events.
    """
    if not CHAIN_ENABLED:
        return []

    _init()
    try:
        events = _contract.events.RiskUpdated().get_logs(from_block=0)
    except Exception as exc:
        log.warning("Could not read RiskUpdated events: %s", exc)
        return []

    latest: dict = {}
    for ev in events:
        a = ev["args"]
        wallet = a["wallet"]
        rec = {
            "wallet": wallet,
            "risk_score": int(a["riskScore"]),
            "timestamp": int(a["timestamp"]),
            "model_version": a.get("modelVersion", ""),
            "block": ev["blockNumber"],
        }
        # keep the most recent update per wallet (highest block)
        if wallet not in latest or rec["block"] >= latest[wallet]["block"]:
            latest[wallet] = rec

    records = sorted(latest.values(), key=lambda r: r["timestamp"], reverse=True)
    for r in records:
        r.pop("block", None)
    return records


def is_connected() -> bool:
    """Health-check: is the web3 provider reachable?"""
    if not CHAIN_ENABLED:
        return False
    try:
        _init()
        return _w3.is_connected()
    except Exception:
        return False
