from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, x, y, w, h, text, fc="#f6f8fa", ec="#2f3b52", fs=9):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return box


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#2f3b52"),
    )


def main():
    out = Path("gambar_4_3_implementasi_smart_contract.png")

    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Gambar 4.3 Implementasi Smart Contract RiskRegistry",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Header
    add_box(ax, 0.06, 0.84, 0.2, 0.08, "Deployer")
    add_box(ax, 0.38, 0.84, 0.24, 0.08, "RiskRegistry Contract", fc="#e9f2ff")
    add_box(ax, 0.74, 0.84, 0.2, 0.08, "Owner = msg.sender")
    arrow(ax, 0.26, 0.88, 0.38, 0.88)
    arrow(ax, 0.62, 0.88, 0.74, 0.88)

    # setRisk flow
    add_box(ax, 0.05, 0.63, 0.23, 0.08, "setRisk(wallet, score,\nmodelVersion)", fc="#fff2e8")
    add_box(ax, 0.34, 0.63, 0.2, 0.08, "onlyOwner?", fc="#fff7d6")
    add_box(ax, 0.6, 0.63, 0.16, 0.08, "score <= 100?", fc="#fff7d6")
    add_box(ax, 0.8, 0.63, 0.16, 0.08, "ts = block.timestamp", fc="#e8fff1")

    arrow(ax, 0.28, 0.67, 0.34, 0.67)
    arrow(ax, 0.54, 0.67, 0.6, 0.67)
    arrow(ax, 0.76, 0.67, 0.8, 0.67)

    add_box(ax, 0.18, 0.5, 0.32, 0.08, "revert if not owner", fc="#ffeaea", ec="#a12d2d")
    add_box(ax, 0.54, 0.5, 0.32, 0.08, "revert if score > 100", fc="#ffeaea", ec="#a12d2d")
    arrow(ax, 0.44, 0.63, 0.34, 0.58)
    arrow(ax, 0.68, 0.63, 0.7, 0.58)

    # Storage + event
    add_box(ax, 0.18, 0.34, 0.32, 0.1, "_records[wallet] =\nRecord(score, ts, modelVersion)", fc="#e8fff1")
    add_box(ax, 0.54, 0.34, 0.32, 0.1, "emit RiskUpdated(wallet,\nscore, ts, modelVersion)", fc="#e8fff1")
    arrow(ax, 0.88, 0.63, 0.66, 0.44)
    arrow(ax, 0.5, 0.39, 0.54, 0.39)

    # Read and ownership
    add_box(ax, 0.06, 0.14, 0.28, 0.1, "getRisk(wallet)\nreturn riskScore, timestamp, modelVersion", fc="#eef5ff")
    add_box(ax, 0.38, 0.14, 0.28, 0.1, "transferOwnership(newOwner)\nonlyOwner and newOwner != 0x0", fc="#eef5ff")
    add_box(ax, 0.7, 0.14, 0.24, 0.1, "owner = newOwner", fc="#e8fff1")
    arrow(ax, 0.66, 0.19, 0.7, 0.19)

    ax.text(
        0.5,
        0.03,
        "Smart contract menyimpan skor risiko phishing wallet secara on-chain\n"
        "dengan kontrol akses owner-only dan jejak event untuk audit.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )

    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    fig.savefig(out, dpi=300)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
