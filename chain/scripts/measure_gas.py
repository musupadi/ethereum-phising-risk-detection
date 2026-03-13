"""
chain/scripts/measure_gas.py
-----------------------------
Measures gas used and commit latency for setRisk() transactions.
Runs N_CALLS calls to the deployed RiskRegistry contract and reports stats.
Run AFTER Hardhat node + contract are deployed.
"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

# ── Config ────────────────────────────────────────────────────────────────────
RPC_URL      = "http://127.0.0.1:8545"
DEPLOYMENT   = ROOT / "chain" / "artifacts" / "deployment.json"
ABI_FILE     = ROOT / "chain" / "artifacts" / "RiskRegistry.abi.json"
PRIVATE_KEY  = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
N_CALLS      = 20
MODEL_VERSION = "rf-v1"

# ── Connect ───────────────────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(RPC_URL))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
assert w3.is_connected(), "Cannot connect to Hardhat node"

with open(DEPLOYMENT)  as f: deployment = json.load(f)
with open(ABI_FILE)    as f: abi = json.load(f)
contract_address = Web3.to_checksum_address(deployment["address"])
contract  = w3.eth.contract(address=contract_address, abi=abi)
account   = w3.eth.account.from_key(PRIVATE_KEY)

# ── Measure ───────────────────────────────────────────────────────────────────
latencies = []
gas_useds = []

# Use deterministic but varied wallets (Hardhat pre-funded accounts recycled)
WALLETS = [
    "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
    "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
    "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
    "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
    "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
    "0x976EA74026E726554dB657fA54763abd0C3a0aa9",
    "0x14dC79964da2C08b23698B3D3cc7Ca32193d9955",
    "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f",
    "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720",
    "0xBcd4042DE499D14e55001CcbB24a551F3b954096",
]

print(f"Measuring {N_CALLS} setRisk() transactions against RiskRegistry at {contract_address}…\n")

for i in range(N_CALLS):
    wallet  = Web3.to_checksum_address(WALLETS[i % len(WALLETS)])
    score   = random.randint(0, 100)
    nonce   = w3.eth.get_transaction_count(account.address)

    # Estimate gas before sending
    gas_estimate = contract.functions.setRisk(wallet, score, MODEL_VERSION).estimate_gas(
        {"from": account.address}
    )

    tx = contract.functions.setRisk(wallet, score, MODEL_VERSION).build_transaction({
        "from":     account.address,
        "nonce":    nonce,
        "gas":      gas_estimate + 10_000,
        "gasPrice": w3.eth.gas_price,
    })

    t0 = time.perf_counter()
    signed  = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    latencies.append(elapsed_ms)
    gas_useds.append(receipt.gasUsed)
    print(f"  [{i+1:2d}/{N_CALLS}]  wallet={wallet[:10]}…  score={score:3d}  "
          f"gas={receipt.gasUsed:,}  latency={elapsed_ms:.1f}ms  "
          f"tx={receipt.transactionHash.hex()[:14]}…")

# ── Report ────────────────────────────────────────────────────────────────────
import statistics

print(f"""
{'='*65}
Blockchain Performance Report ({N_CALLS} transactions)
{'='*65}
Latency (ms):
  Min    : {min(latencies):.1f}
  Max    : {max(latencies):.1f}
  Mean   : {statistics.mean(latencies):.1f}
  Median : {statistics.median(latencies):.1f}
  Stdev  : {statistics.stdev(latencies):.1f}

Gas Used (setRisk):
  Min    : {min(gas_useds):,}
  Max    : {max(gas_useds):,}
  Mean   : {statistics.mean(gas_useds):,.0f}
  Median : {statistics.median(gas_useds):,.0f}
  Stdev  : {statistics.stdev(gas_useds):,.0f}
{'='*65}
""")

# Save results
import json
results = {
    "n_calls": N_CALLS,
    "latency_ms": {
        "min":    round(min(latencies), 2),
        "max":    round(max(latencies), 2),
        "mean":   round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "stdev":  round(statistics.stdev(latencies), 2),
    },
    "gas_used": {
        "min":    min(gas_useds),
        "max":    max(gas_useds),
        "mean":   round(statistics.mean(gas_useds)),
        "median": int(statistics.median(gas_useds)),
    },
    "raw_latencies": [round(x, 2) for x in latencies],
    "raw_gas":       gas_useds,
}

out = ROOT / "ml" / "artifacts" / "blockchain_metrics.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {out}")
