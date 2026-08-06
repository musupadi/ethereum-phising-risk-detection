"""
scripts/profile_muldigraph.py
-----------------------------
Membuka data/raw/MulDiGraph.pkl lalu menghasilkan bahan deskripsi data untuk tesis:

  1. Statistik graph (node, edge, label, degree, amount, rentang waktu)  -> JSON + Markdown
  2. Potongan edge list mentah (contoh 10 transaksi)                     -> CSV + Markdown
  3. Gambar ego-network 1-hop: 1 wallet phishing vs 1 wallet normal      -> PNG
  4. Distribusi degree (log-log) phishing vs normal                      -> PNG

Semua keluaran ditulis ke ml/artifacts/graph_profile/.

Jalankan:  python scripts/profile_muldigraph.py
Perlu ~2-5 menit dan RAM cukup besar (pickle 1,2 GB).
"""

import json
import pickle
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "raw" / "MulDiGraph.pkl"
OUT_DIR = ROOT / "ml" / "artifacts" / "graph_profile"

RANDOM_SEED = 42
# rentang derajat contoh wallet supaya gambar masih terbaca mata
EGO_MIN_DEG, EGO_MAX_DEG = 8, 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _fmt_ts(ts):
    """Unix timestamp -> string tanggal, aman terhadap nilai aneh."""
    try:
        return pd.to_datetime(float(ts), unit="s").strftime("%Y-%m-%d")
    except Exception:
        return "n/a"


# ── 1. Statistik graph ────────────────────────────────────────────────────────

def collect_stats(G):
    log.info("Menghitung statistik node & label…")
    phishing_nodes, normal_nodes = [], []
    for node, data in G.nodes(data=True):
        (phishing_nodes if data.get("isp", 0) == 1 else normal_nodes).append(node)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    log.info("Menghitung derajat…")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_all = np.array([in_deg[n] + out_deg[n] for n in G.nodes()], dtype=float)
    deg_phish = np.array([in_deg[n] + out_deg[n] for n in phishing_nodes], dtype=float)
    deg_norm = np.array([in_deg[n] + out_deg[n] for n in normal_nodes], dtype=float)

    log.info("Memindai atribut edge (amount & timestamp)…")
    amounts, timestamps = [], []
    zero_amount = 0
    for _, _, d in G.edges(data=True):
        a = d.get("amount", 0) or 0
        amounts.append(float(a))
        if a == 0:
            zero_amount += 1
        ts = d.get("timestamp")
        if ts is not None:
            timestamps.append(float(ts))
    amounts = np.asarray(amounts, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    # berapa banyak pasangan wallet yang bertransaksi >1 kali (inti dari "Multi")
    n_unique_pairs = len({(u, v) for u, v in G.edges()})

    stats = {
        "tipe_objek": type(G).__name__,
        "jumlah_node": n_nodes,
        "jumlah_edge": n_edges,
        "jumlah_pasangan_unik": n_unique_pairs,
        "edge_paralel_rata2_per_pasangan": round(n_edges / n_unique_pairs, 3),
        "node_phishing": len(phishing_nodes),
        "node_normal": len(normal_nodes),
        "persen_phishing": round(100 * len(phishing_nodes) / n_nodes, 4),
        "kepadatan_graph": n_edges / (n_nodes * (n_nodes - 1)),
        "degree_rata2_semua": round(float(deg_all.mean()), 3),
        "degree_median_semua": float(np.median(deg_all)),
        "degree_maks_semua": float(deg_all.max()),
        "degree_rata2_phishing": round(float(deg_phish.mean()), 3),
        "degree_median_phishing": float(np.median(deg_phish)),
        "degree_rata2_normal": round(float(deg_norm.mean()), 3),
        "degree_median_normal": float(np.median(deg_norm)),
        "node_tanpa_transaksi_masuk": int((np.array(list(in_deg.values())) == 0).sum()),
        "node_tanpa_transaksi_keluar": int((np.array(list(out_deg.values())) == 0).sum()),
        "amount_total_eth": float(amounts.sum()),
        "amount_rata2_eth": float(amounts.mean()),
        "amount_median_eth": float(np.median(amounts)),
        "amount_maks_eth": float(amounts.max()),
        "edge_amount_nol": zero_amount,
        "timestamp_min": _fmt_ts(timestamps.min()) if len(timestamps) else "n/a",
        "timestamp_maks": _fmt_ts(timestamps.max()) if len(timestamps) else "n/a",
        "edge_tanpa_timestamp": n_edges - len(timestamps),
    }
    return stats, phishing_nodes, normal_nodes, in_deg, out_deg, deg_phish, deg_norm


# ── 2. Potongan edge list mentah ──────────────────────────────────────────────

def sample_edge_list(G, phishing_nodes, n=10):
    """Ambil beberapa edge yang menyentuh wallet phishing, supaya contohnya relevan."""
    rng = random.Random(RANDOM_SEED)
    phish_set = set(phishing_nodes)
    rows = []
    for node in rng.sample(phishing_nodes, min(200, len(phishing_nodes))):
        for succ in G.successors(node):
            for d in G[node][succ].values():
                rows.append({
                    "from_address": node,
                    "to_address": succ,
                    "amount_eth": d.get("amount", 0),
                    "timestamp": d.get("timestamp"),
                    "tanggal": _fmt_ts(d.get("timestamp")),
                    "from_isp": 1,
                    "to_isp": int(succ in phish_set),
                })
                if len(rows) >= n * 5:
                    break
            if len(rows) >= n * 5:
                break
        if len(rows) >= n * 5:
            break
    df = pd.DataFrame(rows)
    if len(df) > n:
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ── 3. Ego-network 1-hop ──────────────────────────────────────────────────────

def pick_example(nodes, in_deg, out_deg):
    """Pilih wallet dengan derajat sedang supaya gambarnya terbaca."""
    rng = random.Random(RANDOM_SEED)
    pool = [n for n in nodes if EGO_MIN_DEG <= in_deg[n] + out_deg[n] <= EGO_MAX_DEG]
    if not pool:
        pool = list(nodes)
    return rng.choice(sorted(pool))


def draw_ego(ax, G, center, title, center_color):
    neighbors = set(G.predecessors(center)) | set(G.successors(center))
    sub = nx.DiGraph()
    sub.add_node(center)
    for nb in neighbors:
        sub.add_node(nb)
        if G.has_edge(nb, center):
            sub.add_edge(nb, center, k=len(G[nb][center]))
        if G.has_edge(center, nb):
            sub.add_edge(center, nb, k=len(G[center][nb]))

    pos = nx.spring_layout(sub, seed=RANDOM_SEED, k=0.9)
    incoming = {n for n in neighbors if G.has_edge(n, center)}

    nx.draw_networkx_edges(
        sub, pos, ax=ax, edge_color="#9aa5b1", width=1.1,
        arrowsize=11, arrowstyle="-|>", connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        sub, pos, ax=ax, nodelist=[n for n in neighbors if n in incoming],
        node_color="#7fb3d5", node_size=210, edgecolors="#33556e", linewidths=0.6,
    )
    nx.draw_networkx_nodes(
        sub, pos, ax=ax, nodelist=[n for n in neighbors if n not in incoming],
        node_color="#f0c987", node_size=210, edgecolors="#8a6d23", linewidths=0.6,
    )
    nx.draw_networkx_nodes(
        sub, pos, ax=ax, nodelist=[center], node_color=center_color,
        node_size=800, edgecolors="#1b1b1b", linewidths=1.4,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.axis("off")
    return sub


def figure_ego(G, phish_node, norm_node, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    sub_p = draw_ego(
        axes[0], G, phish_node,
        f"Wallet PHISHING (isp=1)\n{phish_node[:10]}…{phish_node[-6:]}",
        "#e05c5c",
    )
    sub_n = draw_ego(
        axes[1], G, norm_node,
        f"Wallet NORMAL (isp=0)\n{norm_node[:10]}…{norm_node[-6:]}",
        "#4c9a5a",
    )
    fig.suptitle(
        "Ego-network 1-hop pada MulDiGraph — lingkup yang dibaca extract_node_features()",
        fontsize=13, y=0.98,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=13, mfc="#e05c5c", mec="k", label="wallet phishing (pusat)"),
        plt.Line2D([], [], marker="o", ls="", ms=13, mfc="#4c9a5a", mec="k", label="wallet normal (pusat)"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#7fb3d5", mec="k", label="tetangga pengirim (in)"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#f0c987", mec="k", label="tetangga penerima (out)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return sub_p, sub_n


# ── 4. Distribusi degree ──────────────────────────────────────────────────────

def figure_degree(deg_phish, deg_norm, out_path):
    """
    Dinormalisasi per kelas (proporsi, bukan jumlah). Kalau memakai jumlah absolut,
    batang normal (2,97 juta) menelan batang phishing (1.165) dan bentuk sebarannya
    tidak terbaca.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    p = deg_phish[deg_phish > 0]
    n = deg_norm[deg_norm > 0]
    bins = np.logspace(0, np.log10(max(p.max(), n.max()) + 1), 40)

    wn = np.ones_like(n) / len(n)
    wp = np.ones_like(p) / len(p)
    ax.hist(n, bins=bins, weights=wn, alpha=0.65, label=f"normal (n={len(deg_norm):,})", color="#4c9a5a")
    ax.hist(p, bins=bins, weights=wp, alpha=0.7, label=f"phishing (n={len(deg_phish):,})", color="#e05c5c")

    ax.axvline(np.median(n), color="#2f6b3c", ls="--", lw=1.2)
    ax.axvline(np.median(p), color="#a33", ls="--", lw=1.2)
    ax.annotate(f"median normal = {np.median(deg_norm):.0f}", xy=(np.median(n), 0.30),
                xytext=(1.3, 0.34), fontsize=8.5, color="#2f6b3c")
    ax.annotate(f"median phishing = {np.median(deg_phish):.0f}", xy=(np.median(p), 0.22),
                xytext=(16, 0.25), fontsize=8.5, color="#a33")

    ax.set_xscale("log")
    ax.set_xlabel("total degree (in + out)")
    ax.set_ylabel("proporsi wallet dalam kelasnya")
    ax.set_title("Sebaran degree MulDiGraph, dinormalisasi per kelas")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GRAPH_PATH.exists():
        raise SystemExit(f"Tidak menemukan {GRAPH_PATH}")

    log.info("Memuat graph dari %s (1,2 GB, mohon tunggu)…", GRAPH_PATH)
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    log.info("Graph dimuat: %s node, %s edge", f"{G.number_of_nodes():,}", f"{G.number_of_edges():,}")

    stats, phishing_nodes, normal_nodes, in_deg, out_deg, deg_phish, deg_norm = collect_stats(G)

    # contoh wallet
    phish_node = pick_example(phishing_nodes, in_deg, out_deg)
    norm_node = pick_example(normal_nodes, in_deg, out_deg)
    stats["contoh_wallet_phishing"] = phish_node
    stats["contoh_wallet_normal"] = norm_node

    # simpan array degree supaya gambar bisa dibuat ulang tanpa memuat pickle 1,2 GB lagi
    np.save(OUT_DIR / "degree_phishing.npy", deg_phish)
    np.save(OUT_DIR / "degree_normal.npy", deg_norm)

    log.info("Menulis statistik…")
    (OUT_DIR / "graph_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Mengambil contoh edge list…")
    edges_df = sample_edge_list(G, phishing_nodes, n=10)
    edges_df.to_csv(OUT_DIR / "sample_edge_list.csv", index=False)

    log.info("Menggambar ego-network…")
    sub_p, sub_n = figure_ego(G, phish_node, norm_node, OUT_DIR / "ego_network_1hop.png")
    stats["ego_phishing_tetangga"] = sub_p.number_of_nodes() - 1
    stats["ego_normal_tetangga"] = sub_n.number_of_nodes() - 1

    log.info("Menggambar distribusi degree…")
    figure_degree(deg_phish, deg_norm, OUT_DIR / "degree_distribution.png")

    (OUT_DIR / "graph_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Selesai. Keluaran di %s", OUT_DIR)
    for k, v in stats.items():
        log.info("  %-32s %s", k, v)


if __name__ == "__main__":
    main()
