import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

IMG_DIR = "../makale_latex/images"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": False,
})

PANEL = {
    "blue":   ("#d6e8f5", "#2c7fb8"),
    "yellow": ("#fdeecb", "#e0a800"),
    "green":  ("#d8efd3", "#3c9a3c"),
    "red":    ("#f8d7da", "#d6455a"),
    "gray":   ("#e9ecef", "#868e96"),
    "purple": ("#e3d7f0", "#7e57c2"),
}
BOX = {
    "blue":   ("#eaf4fb", "#2c7fb8"),
    "yellow": ("#fff6e0", "#e0a800"),
    "green":  ("#eaf7e6", "#3c9a3c"),
    "red":    ("#fdecee", "#d6455a"),
    "gray":   ("#f3f5f7", "#868e96"),
    "purple": ("#f0eaf8", "#7e57c2"),
    "white":  ("#ffffff", "#495057"),
}

def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    eps = os.path.join(IMG_DIR, name + ".eps")
    png = os.path.join(IMG_DIR, name + ".png")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", eps, "+ png")

def panel(ax, x, y, w, h, title, color):
    fc, ec = PANEL[color]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=ec, zorder=3)

def box(ax, x, y, w, h, text, color, fontsize=10, bold=False, zorder=4):
    fc, ec = BOX[color]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color="#212529", zorder=zorder + 1)
    return (x, y, w, h)

def arrow(ax, p_from, p_to, color="#495057", lw=2.0, style="-|>", ls="-",
          rad=0.0, zorder=5):
    a = FancyArrowPatch(p_from, p_to, arrowstyle=style, mutation_scale=16,
                        linewidth=lw, color=color, zorder=zorder,
                        linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

def label(ax, x, y, text, color="#495057", fontsize=8.5, style="italic"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontstyle=style, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

def diagram_system_overview():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 10)
    ax.axis("off")

    pw, ph, py = 5.0, 9.2, 0.4
    gaps = [0.3, 6.0, 12.0, 21.5, 27.0]

    px = 0.3
    panel(ax, px, py, pw, ph, "Data Preparation", "blue")
    box(ax, px + 0.6, 7.2, 3.8, 1.0, "InjecAgent\nDataset", "blue", 9.5, True)
    box(ax, px + 0.6, 5.4, 3.8, 1.0, "Stratified\nSampling (seed)", "blue", 9.5)
    box(ax, px + 0.6, 3.6, 3.8, 1.0, "Benign\nMatching", "blue", 9.5)
    box(ax, px + 0.6, 1.8, 3.8, 1.0, "test_cases.json\n(250 adv + 250 benign)", "blue", 8.5)
    for y0, y1 in [(7.2, 6.4), (5.4, 4.6), (3.6, 2.8)]:
        arrow(ax, (px + 2.5, y0), (px + 2.5, y1 + 0.0))

    px = 6.0
    panel(ax, px, py, pw, ph, "Evaluation Engine", "yellow")
    box(ax, px + 0.6, 7.0, 3.8, 1.1, "Experiment\nController", "yellow", 9.5, True)
    box(ax, px + 0.6, 4.9, 3.8, 1.4, "Ollama Runtime\nqwen2.5:7b\nmistral:7b\ndeepseek-coder", "yellow", 8.3)
    box(ax, px + 0.6, 2.9, 3.8, 1.1, "Automated\nLLM Judge", "yellow", 9.5, True)
    arrow(ax, (px + 2.5, 7.0), (px + 2.5, 6.4))
    arrow(ax, (px + 2.5, 4.9), (px + 2.5, 4.1))

    px = 12.0
    panel(ax, px, py, 8.5, ph, "Defense Configurations", "green")
    cfgs = [
        ("Baseline", "Static System Prompt"),
        ("ReAct", "Reasoning + Self-check"),
        ("Spotlighting", "Encoding / Delimiting"),
        ("Detector", "Guard Model"),
        ("Spotlight-Guard", "Spotlighting + HMAC"),
        ("Spotlight-Guard (Full)", "Proposed System"),
    ]
    cy = 7.7
    for i, (name, desc) in enumerate(cfgs):
        col = "green" if name != "Spotlight-Guard (Full)" else "purple"
        bold = name == "Spotlight-Guard (Full)"
        box(ax, px + 0.5, cy, 7.5, 0.78, f"{name}  \u2014  {desc}", col, 8.5, bold)
        cy -= 1.0

    px = 21.5
    panel(ax, px, py, pw, ph, "Metrics", "red")
    box(ax, px + 0.6, 7.0, 3.8, 1.1, "ASR + 95% CI\n(bootstrap)", "red", 9.0, True)
    box(ax, px + 0.6, 5.2, 3.8, 1.1, "F1 / Precision\n/ Recall", "red", 9.0)
    box(ax, px + 0.6, 3.4, 3.8, 1.1, "Benign Success\nRate", "red", 9.0)
    box(ax, px + 0.6, 1.6, 3.8, 1.1, "Confusion\nMatrix", "red", 9.0)

    px = 27.0
    panel(ax, px, py, 2.7, ph, "Output", "gray")
    box(ax, px + 0.35, 6.8, 2.0, 1.2, "Per-run\nJSON", "gray", 9.0, True)
    box(ax, px + 0.35, 4.7, 2.0, 1.2, "Aggregated\nResults", "gray", 9.0)
    box(ax, px + 0.35, 2.6, 2.0, 1.2, "Figures &\nTables", "gray", 9.0)

    big = dict(lw=3.5, color="#7f7f7f")
    arrow(ax, (5.4, 5.0), (5.9, 5.0), **big)
    arrow(ax, (11.1, 5.0), (11.9, 5.0), **big)
    arrow(ax, (20.6, 5.0), (21.4, 5.0), **big)
    arrow(ax, (26.6, 5.0), (26.9, 5.0), **big)

    save(fig, "fig_system_overview")

def diagram_defense_architecture():
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_xlim(0, 34)
    ax.set_ylim(0, 15)
    ax.axis("off")

    phases = [(3.3, "Input Phase"), (10.0, "Signing & Spotlighting"),
              (21.8, "Detection & Recovery"), (31.1, "Output Phase")]
    for x, name in phases:
        ax.text(x, 14.3, name, ha="center", va="center", fontsize=13,
                fontweight="bold", color="#343a40")
    for xd in [6.8, 13.5, 28.5]:
        ax.plot([xd, xd], [0.5, 13.6], linestyle=(0, (4, 4)),
                color="#adb5bd", linewidth=1.3, zorder=0)

    box(ax, 0.8, 10.4, 5.0, 1.5, "Trusted\nInstructions", "blue", 11, True)
    box(ax, 0.8, 3.2, 5.0, 1.5, "Untrusted\nTool Content", "red", 11, True)

    box(ax, 7.4, 10.4, 5.2, 1.5, "HMAC Signer\n(sign instructions)", "yellow", 10, True)
    box(ax, 8.3, 7.4, 3.4, 1.3, "Verify?", "white", 11, True)
    box(ax, 8.3, 4.8, 3.4, 1.2, "Stop", "red", 11, True)
    box(ax, 7.4, 2.4, 5.2, 1.5, "Spotlighting /\nEncoding", "purple", 10, True)

    arrow(ax, (5.8, 11.15), (7.3, 11.15))
    arrow(ax, (5.8, 3.95), (7.3, 3.15))
    arrow(ax, (10.0, 10.4), (10.0, 8.75))
    arrow(ax, (10.0, 7.4), (10.0, 6.05), color="#d6455a")
    label(ax, 10.5, 6.75, "N", color="#d6455a", fontsize=10, style="normal")

    box(ax, 14.3, 7.2, 4.6, 2.0, "Guard Model\n(Detector)", "blue", 11, True)
    arrow(ax, (11.7, 8.05), (14.2, 8.2))
    label(ax, 13.0, 8.55, "Y", color="#3c9a3c", fontsize=10, style="normal")

    arrow(ax, (10.0, 2.4), (10.0, 1.6), color="#7e57c2")
    arrow(ax, (10.0, 1.6), (16.6, 1.6), color="#7e57c2")
    arrow(ax, (16.6, 1.6), (16.6, 7.1), color="#7e57c2")

    box(ax, 20.0, 10.6, 3.4, 1.1, "Allow\n(high conf.)", "green", 9, True)
    box(ax, 20.0, 7.7, 3.4, 1.1, "Escalate\n(low conf.)", "yellow", 9, True)
    box(ax, 20.0, 4.6, 3.4, 1.1, "Block\n(attack)", "red", 9, True)
    arrow(ax, (18.9, 8.8), (19.9, 11.0), color="#3c9a3c")
    arrow(ax, (18.9, 8.2), (19.9, 8.25), color="#e0a800")
    arrow(ax, (18.9, 7.6), (19.9, 5.15), color="#d6455a")

    box(ax, 24.2, 10.6, 3.6, 1.1, "Heuristic\nCheck", "gray", 9.5, True)
    box(ax, 24.2, 7.7, 3.6, 1.2, "Quarantine\nModel", "yellow", 9.5, True)
    box(ax, 24.2, 5.5, 3.6, 1.2, "Fallback\nModel", "yellow", 9.5, True)
    box(ax, 24.2, 3.3, 3.6, 1.2, "Sanitized\nResponse", "green", 9.5, True)
    arrow(ax, (23.4, 11.15), (24.1, 11.15), color="#3c9a3c")
    arrow(ax, (26.0, 10.6), (26.0, 8.95), color="#e0a800", ls="--")
    label(ax, 27.2, 9.8, "Override", color="#e0a800", fontsize=8.5)
    arrow(ax, (23.4, 8.25), (24.1, 8.3), color="#e0a800")
    arrow(ax, (26.0, 7.7), (26.0, 6.75), color="#e0a800")
    arrow(ax, (26.0, 5.5), (26.0, 4.55), color="#3c9a3c")

    box(ax, 29.2, 8.5, 3.8, 1.6, "Secure\nExecution", "green", 10, True)
    box(ax, 29.2, 3.0, 3.8, 1.4, "Security\nNotification", "red", 9.5, True)
    arrow(ax, (27.8, 11.0), (29.1, 9.7), color="#3c9a3c")

    arrow(ax, (27.8, 3.9), (28.4, 3.9), color="#3c9a3c")
    arrow(ax, (28.4, 3.9), (28.4, 9.0), color="#3c9a3c")
    arrow(ax, (28.4, 9.0), (29.1, 9.0), color="#3c9a3c")

    arrow(ax, (21.7, 4.6), (21.7, 2.4), color="#d6455a")
    arrow(ax, (21.7, 2.4), (31.1, 2.4), color="#d6455a")
    arrow(ax, (31.1, 2.4), (31.1, 2.95), color="#d6455a")

    save(fig, "fig_defense_architecture")

def diagram_data_preparation():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(ax, 0.6, 3.6, 4.4, 2.8, "InjecAgent\nDataset\n(1,054 cases)", "yellow", 11, True)

    panel(ax, 6.5, 1.6, 4.6, 7.0, "Attack Categories", "green")
    box(ax, 7.0, 6.4, 3.6, 1.3, "Financial Data", "white", 10, True)
    box(ax, 7.0, 4.3, 3.6, 1.3, "Physical Data", "white", 10, True)
    box(ax, 7.0, 2.2, 3.6, 1.3, "Others", "white", 10, True)

    box(ax, 13.0, 3.6, 4.4, 2.8, "Stratified\nSampling", "blue", 11, True)
    box(ax, 13.4, 7.2, 3.6, 1.3, "Random Seed\n(20250917)", "purple", 9.5, True)
    arrow(ax, (15.2, 7.2), (15.2, 6.5), color="#7e57c2")

    box(ax, 19.4, 3.6, 4.4, 2.8, "Benign\nMatching", "red", 11, True)

    box(ax, 25.6, 3.6, 3.9, 2.8, "Processed\nDataset (JSON)\n250 adv + 250 benign", "gray", 9.5, True)

    arrow(ax, (5.0, 5.0), (6.4, 5.0), lw=2.6)
    arrow(ax, (11.1, 5.0), (12.9, 5.0), lw=2.6)
    arrow(ax, (17.4, 5.0), (19.3, 5.0), lw=2.6)
    arrow(ax, (23.8, 5.0), (25.5, 5.0), lw=2.6)

    save(fig, "fig_data_preparation")

if __name__ == "__main__":
    diagram_system_overview()
    diagram_defense_architecture()
    diagram_data_preparation()
    print("\nAll diagrams written to", IMG_DIR)
