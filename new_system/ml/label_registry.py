"""
new_system/ml/label_registry.py
-------------------------------
LAPISAN KEPUTUSAN PERTAMA: daftar label alamat yang identitasnya sudah diketahui.

Kenapa lapisan ini ada
======================
Pengujian pada wallet bursa nyata menunjukkan batas yang tidak bisa ditembus oleh
model manapun: alamat deposit bursa yang masih baru **secara struktur on-chain
identik** dengan wallet pengumpul phishing. Contohnya `Binance: Hot Wallet` —
17 transaksi masuk dari 16 pengirim, nol transaksi keluar. Pola itu sama persis
dengan phishing yang belum menyapu dana.

Perbedaannya bukan pada statistik transaksi, melainkan pada **identitas** — dan
identitas itu pengetahuan dari luar rantai, bukan sesuatu yang bisa disimpulkan
dari angka.

Karena itu urutan keputusan sistem menjadi:

    1. Alamat ada di daftar label?  -> pakai labelnya, SELESAI (tanpa menebak)
    2. Tidak ada di daftar          -> baru jalankan model ML

Ini bukan jalan pintas untuk memperbagus angka: ini praktik standar industri.
Etherscan, Chainalysis, dan hampir semua produk keamanan blockchain memakai
daftar label untuk alamat yang dikenal, dan menyisakan model untuk sisanya.

Batasnya juga harus jujur: daftar label hanya menutup alamat yang SUDAH dikenal.
Ia tidak menggeneralisasi. Justru karena itu model ML tetap dibutuhkan.

Perluasan on-chain
==================
Lihat `new_system/chain/LabelRegistry.sol`: daftar ini dapat dipindahkan ke smart
contract sehingga pelabelan menjadi terbuka dan dapat diaudit, serta bisa menerima
usulan label dari pengguna.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LABELS_JSON = Path(__file__).resolve().parent.parent / "data" / "address_labels.json"
ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")

_labels: dict[str, dict] | None = None
_meta: dict = {}


def _load() -> dict[str, dict]:
    global _labels, _meta
    if _labels is not None:
        return _labels
    if not LABELS_JSON.exists():
        log.warning("Daftar label tidak ditemukan di %s — lapisan label dilewati.", LABELS_JSON)
        _labels, _meta = {}, {}
        return _labels
    raw = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    _meta = raw.get("_meta", {})
    _labels = {k.lower(): v for k, v in raw.get("labels", {}).items()}
    n_safe = sum(1 for v in _labels.values() if v.get("label") == 0)
    log.info("Daftar label dimuat: %d alamat (%d aman, %d phishing)",
             len(_labels), n_safe, len(_labels) - n_safe)
    return _labels


def reload() -> None:
    """Paksa muat ulang — dipakai setelah daftar diubah saat aplikasi berjalan."""
    global _labels
    _labels = None
    _load()


def lookup_local(address: str) -> dict | None:
    """Cari di daftar label lokal (JSON)."""
    return _load().get(str(address).strip().lower())


def lookup_onchain(address: str) -> dict | None:
    """
    Cari label di smart contract `LabelRegistry` — inilah lapisan terdesentralisasi.

    Aktif hanya bila `LABEL_REGISTRY_ADDRESS` diisi di lingkungan. Bila kontrak
    belum di-deploy atau node tidak dapat dihubungi, kembalikan None secara diam
    supaya sistem otomatis jatuh ke daftar lokal — kegagalan lapisan opsional
    tidak boleh menjatuhkan penilaian.
    """
    import os

    contract_addr = os.getenv("LABEL_REGISTRY_ADDRESS", "").strip()
    if not contract_addr:
        return None

    try:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(os.getenv("RPC_URL", "http://127.0.0.1:8545")))
        if not w3.is_connected():
            return None
        contract = w3.eth.contract(address=Web3.to_checksum_address(contract_addr),
                                   abi=ONCHAIN_ABI)
        label, ts, name, source = contract.functions.getVerifiedLabel(
            Web3.to_checksum_address(address)).call()
        if int(label) == 0:  # enum Label.Unknown -> belum pernah dilabeli
            return None
        return {
            "label": 1 if int(label) == 2 else 0,  # 1=Safe, 2=Phishing pada enum
            "name": name or "(tanpa nama)",
            "category": "onchain",
            "source": source or "label_registry_onchain",
            "onchain": True,
            "onchain_timestamp": int(ts),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Pembacaan label on-chain gagal (%s) — memakai daftar lokal.", exc)
        return None


def lookup(address: str, prefer_onchain: bool = True) -> dict | None:
    """
    Cari label alamat.

    Urutannya: kontrak on-chain lebih dulu (sumber kebenaran yang dapat diaudit
    siapa pun), lalu daftar lokal sebagai cadangan.
    """
    if prefer_onchain:
        entry = lookup_onchain(address)
        if entry is not None:
            return entry
    return lookup_local(address)


# ABI minimal untuk membaca LabelRegistry — cukup fungsi yang dipakai di sini.
ONCHAIN_ABI = [{
    "inputs": [{"internalType": "address", "name": "wallet", "type": "address"}],
    "name": "getVerifiedLabel",
    "outputs": [
        {"internalType": "uint8", "name": "label", "type": "uint8"},
        {"internalType": "uint64", "name": "timestamp", "type": "uint64"},
        {"internalType": "string", "name": "name", "type": "string"},
        {"internalType": "string", "name": "source", "type": "string"},
    ],
    "stateMutability": "view", "type": "function",
}]


def is_labeled(address: str) -> bool:
    return lookup(address) is not None


def count() -> int:
    return len(_load())


def as_result(address: str, entry: dict) -> dict[str, Any]:
    """
    Susun hasil penilaian dari entri label — bentuknya sama dengan keluaran model
    supaya pemanggil (API/UI) tidak perlu membedakan asalnya.
    """
    label = int(entry.get("label", 0))
    is_phishing = label == 1
    return {
        "probability": 1.0 if is_phishing else 0.0,
        "risk_score": 100 if is_phishing else 0,
        "risk_level": "HIGH" if is_phishing else "LOW",
        "is_phishing": is_phishing,
        "verdict": "Phishing" if is_phishing else "Bukan Phishing",
        "confidence": "pasti",
        "conclusive": True,
        # Penanda asal keputusan — WAJIB dibedakan dari hasil model, supaya tidak
        # ada yang mengira angka ini prediksi.
        "decision_source": "label_registry",
        "model": "label",
        "out_of_distribution": False,
        "label_name": entry.get("name", ""),
        "label_category": entry.get("category", ""),
        "label_source": entry.get("source", ""),
        "explanation": (
            f"Alamat ini terdaftar sebagai «{entry.get('name', 'tidak bernama')}» "
            f"({entry.get('category', 'tanpa kategori')}), bersumber dari "
            f"{entry.get('source', 'tidak diketahui')}. Identitasnya sudah diketahui, "
            f"sehingga tidak perlu ditebak oleh model."
        ),
    }


def add_label(address: str, label: int, name: str, category: str = "user_submitted",
              source: str = "manual", persist: bool = True) -> dict:
    """
    Tambahkan atau perbarui satu entri label.

    Dipakai oleh alur pelaporan pengguna: ketika seseorang menandai sebuah alamat
    sebagai phishing (atau aman) dan penandaan itu disetujui, entrinya masuk ke sini.
    """
    addr = str(address).strip().lower()
    if not ADDRESS_RE.fullmatch(addr):
        raise ValueError(f"Alamat tidak valid: {address}")
    if label not in (0, 1):
        raise ValueError("label harus 0 (aman) atau 1 (phishing)")

    raw = (json.loads(LABELS_JSON.read_text(encoding="utf-8"))
           if LABELS_JSON.exists() else {"_meta": {}, "labels": {}})
    entry = {"label": int(label), "name": name, "category": category, "source": source}
    raw.setdefault("labels", {})[addr] = entry

    if persist:
        LABELS_JSON.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        reload()
    return entry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Total alamat berlabel: {count()}")
    for a in ("0x0068eb681ec52dbd9944517d785727310b494575",
              "0x631fc1ea2270e98fbd9d92658ece0f5a269aa161",
              "0x60afc5761212ed59608b0559b4ccbaad24fc74b1"):
        e = lookup(a)
        print(f"  {a} -> {e['name'] if e else 'TIDAK BERLABEL (lanjut ke model ML)'}")
