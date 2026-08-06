"""
new_system/ml/paths.py
----------------------
Satu tempat untuk semua lokasi berkas, supaya batas antara SISTEM TESIS dan
SISTEM BARU tidak pernah kabur.

  THESIS_ROOT   akar repo — sistem tesis. Dibaca saja, TIDAK PERNAH ditulis.
  NEW_ROOT      folder new_system/ — semua keluaran sistem baru menetap di sini.

Aturan yang dijaga file ini: sistem baru boleh MEMBACA data sistem tesis
(daftar alamat phishing, dataset hybrid, model tesis untuk pembanding), tetapi
tidak boleh MENIMPA apa pun di luar new_system/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# new_system/ml/paths.py -> new_system/ -> akar repo
NEW_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = NEW_ROOT.parent

# ── Keluaran sistem baru (boleh ditulis) ──────────────────────────────────────
NEW_ML = NEW_ROOT / "ml"
NEW_ARTIFACTS = NEW_ROOT / "artifacts"
NEW_DATA = NEW_ROOT / "data"
NEW_OUTPUTS = NEW_ROOT / "outputs"
NEW_EXPERIMENTS = NEW_ROOT / "experiments"

LIVE_DATASET_CSV = NEW_DATA / "live_dataset.csv"
LIVE_MODEL_JSON = NEW_ARTIFACTS / "model_live_xgb.json"
LIVE_SCHEMA_JSON = NEW_ARTIFACTS / "live_feature_schema.json"
LIVE_REPORT_JSON = NEW_ARTIFACTS / "live_model_report.json"

# ── Masukan dari sistem tesis (hanya dibaca) ──────────────────────────────────
THESIS_RAW = THESIS_ROOT / "data" / "raw"
THESIS_PROCESSED = THESIS_ROOT / "data" / "processed"
PHISHING_CSV = THESIS_RAW / "addresses_phishing.csv"
HYBRID_CSV = THESIS_PROCESSED / "hybrid_dataset.csv"


def ensure_dirs() -> None:
    for d in (NEW_ARTIFACTS, NEW_DATA, NEW_OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)


def add_import_paths() -> None:
    """
    Siapkan sys.path agar `import live_features` dan `from ml.predict import ...`
    sama-sama jalan.

    URUTANNYA PENTING. Sistem tesis punya `ml/live_features.py` sendiri dengan nama
    modul yang persis sama, tetapi antarmuka berbeda (`build_live_features`, bukan
    `extract_live_features`). Bila folder `ml/` milik tesis lebih dulu di sys.path,
    modul yang salah yang ter-import dan sistem baru gagal dengan ImportError.

    Karena itu daftar path dibersihkan lebih dulu lalu dipasang ulang dengan
    `new_system/ml` selalu di posisi pertama.
    """
    wanted = [str(NEW_ML), str(THESIS_ROOT), str(THESIS_ROOT / "ml")]
    for p in wanted:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(wanted):
        sys.path.insert(0, p)
