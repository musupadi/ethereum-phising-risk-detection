"""
new_system/chain/deploy_label_registry.py
-----------------------------------------
Deploy `LabelRegistry.sol` ke node Hardhat lokal, lalu isi dengan daftar label
dari `new_system/data/address_labels.json`.

Setelah ini, pengecekan label berjalan ON-CHAIN: aplikasi membaca label langsung
dari smart contract, bukan dari berkas lokal. Berkas lokal tetap ada sebagai
cadangan bila node mati.

Kenapa memakai solcjs lewat Hardhat milik sistem tesis? Karena kompilernya sudah
terpasang di sana. Kontrak disalin sementara, dikompilasi, lalu salinannya dihapus
— `chain/contracts/` kembali hanya berisi `RiskRegistry.sol`.

Jalankan (node Hardhat harus sudah hidup):
    cd chain ; npx hardhat node          # terminal lain
    python new_system\\chain\\deploy_label_registry.py

Keluaran: new_system/chain/deployment.json  (berisi alamat kontrak)
Lalu setel:  $env:LABEL_REGISTRY_ADDRESS = "<alamat>"
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

NEW_CHAIN = Path(__file__).resolve().parent
NEW_ROOT = NEW_CHAIN.parent
THESIS_ROOT = NEW_ROOT.parent
THESIS_CHAIN = THESIS_ROOT / "chain"

SOL_SRC = NEW_CHAIN / "LabelRegistry.sol"
TMP_SOL = THESIS_CHAIN / "contracts" / "_LabelRegistry_build.sol"
ARTIFACT = (THESIS_CHAIN / "artifacts" / "contracts" /
            "_LabelRegistry_build.sol" / "LabelRegistry.json")
LABELS_JSON = NEW_ROOT / "data" / "address_labels.json"
DEPLOYMENT_OUT = NEW_CHAIN / "deployment.json"

# Akun #0 Hardhat — kunci publik yang diketahui semua orang, khusus demo lokal.
DEFAULT_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

LABEL_SAFE, LABEL_PHISHING = 1, 2  # sesuai enum Label di kontrak


def compile_contract() -> tuple[list, str]:
    """Kompilasi kontrak memakai Hardhat, lalu bersihkan jejaknya."""
    print("Mengompilasi LabelRegistry.sol ...")
    shutil.copy(SOL_SRC, TMP_SOL)
    try:
        res = subprocess.run(["npx", "hardhat", "compile"], cwd=str(THESIS_CHAIN),
                             capture_output=True, text=True, shell=True)
        if res.returncode != 0:
            print(res.stdout[-1500:], res.stderr[-1500:])
            raise RuntimeError("Kompilasi gagal")
        data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        print("  berhasil")
        return data["abi"], data["bytecode"]
    finally:
        TMP_SOL.unlink(missing_ok=True)
        # Buang artefak sementara supaya folder sistem tesis tetap bersih.
        for leftover in (
            THESIS_CHAIN / "artifacts" / "contracts" / "_LabelRegistry_build.sol",
            THESIS_CHAIN / "typechain-types" / "_LabelRegistry_build.sol",
            THESIS_CHAIN / "typechain-types" / "factories" / "_LabelRegistry_build.sol",
        ):
            shutil.rmtree(leftover, ignore_errors=True)
        # Hardhat ikut menulis ulang berkas indeks typechain agar memuat kontrak
        # sementara tadi. Kompilasi sekali lagi setelah kontraknya dihapus supaya
        # indeks kembali persis seperti milik sistem tesis.
        # `--force` wajib: tanpa itu Hardhat memakai cache dan berkas indeks
        # typechain tetap memuat kontrak sementara yang sudah dihapus.
        print("  merapikan artefak sistem tesis ...")
        subprocess.run(["npx", "hardhat", "compile", "--force"], cwd=str(THESIS_CHAIN),
                       capture_output=True, text=True, shell=True)


def main() -> None:
    load_dotenv(THESIS_ROOT / ".env")
    rpc = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        print(f"Tidak bisa terhubung ke {rpc}. Jalankan dulu: cd chain ; npx hardhat node")
        sys.exit(1)

    abi, bytecode = compile_contract()
    acct = w3.eth.account.from_key(os.getenv("SIGNER_PRIVATE_KEY", DEFAULT_KEY))
    print(f"Deployer: {acct.address}")

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor().build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 3_000_000,
        "gasPrice": w3.eth.gas_price,
    })
    signed = w3.eth.account.sign_transaction(tx, private_key=acct.key)
    receipt = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(signed.raw_transaction))
    address = receipt.contractAddress
    print(f"LabelRegistry di-deploy ke: {address}")

    # ── Isi daftar label ─────────────────────────────────────────────────────
    registry = w3.eth.contract(address=address, abi=abi)
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))["labels"]

    wallets, enums, names, sources = [], [], [], []
    for addr, meta in labels.items():
        wallets.append(Web3.to_checksum_address(addr))
        enums.append(LABEL_PHISHING if int(meta["label"]) == 1 else LABEL_SAFE)
        names.append(meta.get("name", ""))
        sources.append(meta.get("source", ""))

    print(f"Menulis {len(wallets)} label ke rantai ...")
    tx = registry.functions.setVerifiedLabelBatch(
        wallets, enums, names, sources).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 8_000_000,
            "gasPrice": w3.eth.gas_price,
        })
    signed = w3.eth.account.sign_transaction(tx, private_key=acct.key)
    r2 = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(signed.raw_transaction))
    print(f"  selesai di block {r2.blockNumber}, gas terpakai {r2.gasUsed:,}")

    # ── Verifikasi bacaan ────────────────────────────────────────────────────
    probe = Web3.to_checksum_address("0x0068eb681ec52dbd9944517d785727310b494575")
    lbl, ts, name, source = registry.functions.getVerifiedLabel(probe).call()
    print(f"Uji baca: {probe[:12]}... -> label={lbl} nama='{name}' sumber='{source}'")

    DEPLOYMENT_OUT.write_text(json.dumps({
        "address": address, "deployer": acct.address,
        "network": "localhost", "chainId": w3.eth.chain_id,
        "labels_written": len(wallets), "abi": abi,
    }, indent=2), encoding="utf-8")
    print(f"\nTersimpan -> {DEPLOYMENT_OUT}")
    print("\nAktifkan pengecekan on-chain dengan:")
    print(f'  $env:LABEL_REGISTRY_ADDRESS = "{address}"')


if __name__ == "__main__":
    main()
