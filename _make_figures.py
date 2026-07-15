import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS = "data/analysis_250.json"
IMG_DIR = "../makale_latex/images"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "figure.dpi": 100,
})

C = {
    "asr": "#d62728",
    "f1": "#1f77b4",
    "benign": "#2ca02c",
    "block": "#ff7f0e",
    "neutral": "#7f7f7f",
}

def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    eps = os.path.join(IMG_DIR, name + ".eps")
    png = os.path.join(IMG_DIR, name + ".png")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", eps, "+ png")

def fig_main(a):
    order = ["baseline", "react", "spotlighting_only", "detector_only",
             "dual_signed", "dual_signed_full"]
    labels = ["Baseline", "ReAct", "Spotlighting\nOnly", "Detector\nOnly",
              "Spotlight-Guard", "Spotlight-Guard\n(Full)"]
    h = a["headline"]
    asr = [h[k]["ASR_percent"] for k in order]
    f1 = [h[k]["f1"] * 100 for k in order]

    x = range(len(order))
    w = 0.38
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    b1 = ax1.bar([i - w / 2 for i in x], asr, w, label="ASR (%)",
                 color=C["asr"], edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Attack Success Rate (%)", color=C["asr"])
    ax1.tick_params(axis="y", labelcolor=C["asr"])
    ax1.set_ylim(0, max(asr) * 1.25)

    ax2 = ax1.twinx()
    b2 = ax2.bar([i + w / 2 for i in x], f1, w, label="F1 (x100)",
                 color=C["f1"], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("F1 Score (x100)", color=C["f1"])
    ax2.tick_params(axis="y", labelcolor=C["f1"])
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=9)
    for rect in b1:
        ax1.annotate(f"{rect.get_height():.1f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     ha="center", va="bottom", fontsize=8)
    for rect in b2:
        ax2.annotate(f"{rect.get_height():.1f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     ha="center", va="bottom", fontsize=8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=9,
               framealpha=0.95)
    save(fig, "fig_main_comparison")

def fig_ablation(a):
    order = ["dual_signed_full", "abl_no_signing", "abl_no_heuristic",
             "abl_no_quarantine", "abl_no_fallback", "abl_no_spotlighting"]
    labels = ["Full\nSystem", "w/o\nSigning", "w/o\nHeuristic", "w/o\nQuarantine",
              "w/o\nFallback", "w/o\nSpotlighting"]
    h = a["headline"]
    asr = [h[k]["ASR_percent"] for k in order]
    benign = [h[k]["BENIGN_success_percent"] for k in order]

    x = range(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], asr, w, label="ASR (%)",
                color=C["asr"], edgecolor="black", linewidth=0.5)
    b2 = ax.bar([i + w / 2 for i in x], benign, w, label="Benign Success (%)",
                color=C["benign"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 118)
    for rect in list(b1) + list(b2):
        ax.annotate(f"{rect.get_height():.1f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=7.5)
    ax.legend(loc="upper center", fontsize=9, ncol=2, framealpha=0.95)
    save(fig, "fig_ablation")

def fig_category(a):
    cats = ["Financial Data", "Physical Data", "Others"]
    defs = ["baseline", "react", "dual_signed_full"]
    dlabels = ["Baseline", "ReAct", "Spotlight-Guard (Full)"]
    colors = [C["neutral"], C["block"], C["f1"]]

    x = range(len(cats))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for j, dv in enumerate(defs):
        vals = [a["by_category"][dv][c]["asr"] for c in cats]
        bars = ax.bar([i + (j - 1) * w for i in x], vals, w, label=dlabels[j],
                      color=colors[j], edgecolor="black", linewidth=0.5)
        for rect in bars:
            ax.annotate(f"{rect.get_height():.1f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylim(0, max(a["by_category"]["baseline"][c]["asr"] for c in cats) * 1.25)
    ax.legend(fontsize=9, loc="upper right", ncol=3, framealpha=0.95)
    save(fig, "fig_category_asr")

def fig_adaptive(a):
    h = a["headline"]
    names = ["Standard\nAttacks", "Adaptive\nAttacks"]
    asr = [h["dual_signed_full"]["ASR_percent"], h["adaptive_dual_signed"]["ASR_percent"]]
    f1 = [h["dual_signed_full"]["f1"] * 100, h["adaptive_dual_signed"]["f1"] * 100]

    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    b1 = ax.bar([i - w / 2 for i in x], asr, w, label="ASR (%)",
                color=C["asr"], edgecolor="black", linewidth=0.5)
    b2 = ax.bar([i + w / 2 for i in x], f1, w, label="F1 (x100)",
                color=C["f1"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Value")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    for rect in list(b1) + list(b2):
        ax.annotate(f"{rect.get_height():.1f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    save(fig, "fig_adaptive")

def main():
    a = json.load(open(ANALYSIS, encoding="utf-8"))
    fig_main(a)
    fig_ablation(a)
    fig_category(a)
    fig_adaptive(a)
    print("\nAll figures written to", IMG_DIR)

if __name__ == "__main__":
    main()
