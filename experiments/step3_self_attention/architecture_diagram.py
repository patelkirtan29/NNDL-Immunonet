"""
GeneAttention Architecture Evolution — single vertical flow diagram (dark theme)
Output: results/geneattention_architecture_evolution.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#1A1A2E"   # page background
C_INPUT     = "#3A3A3A"   # input / residual gate boxes
C_ATTN_BG   = "#2D2060"   # attention section background
C_ATTN_V3   = "#3D3D3D"   # v3 single-layer sub-box
C_ATTN_V4   = "#1B5E20"   # v4+ deeper attention (green)
C_ATTN_SOFT = "#3A2E7A"   # softmax row inside attention block
C_ENC_BG    = "#0D2137"   # encoder section background
C_ENC_ITEM  = "#1A3550"   # encoder sub-box
C_LOSS_BG   = "#4A1200"   # loss section background
C_LOSS_V3   = "#3A3A3A"   # v3 base loss
C_LOSS_V4   = "#1B5E20"   # v4 new losses (green)
C_LOSS_V6   = "#4A1572"   # v6 new losses (purple)
C_LOSS_V5   = "#2A2A2A"   # v5 note
C_RES_V3    = "#3A3A3A"
C_RES_V4    = "#2A3A2A"
C_RES_V5    = "#1B5E20"   # best
C_RES_V6    = "#3A2A2A"
WHITE       = "#FFFFFF"
LGRAY       = "#CCCCCC"
GREEN       = "#4CAF50"
PURPLE      = "#CE93D8"

# ── Figure ─────────────────────────────────────────────────────────────────────
FW, FH = 9, 17
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 9)
ax.set_ylim(0, 17)
ax.axis("off")

CX = 4.5   # centre x
BW = 7.0   # default box width

# ── Helpers ────────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, text="", fc=C_INPUT, ec=WHITE, lw=1.2,
         fs=10, tc=WHITE, bold=False, alpha=1.0, pad=0.2, va="center"):
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle=f"round,pad={pad}",
                           facecolor=fc, edgecolor=ec,
                           linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(patch)
    if text:
        ax.text(cx, cy, text, ha="center", va=va, fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=4, multialignment="center",
                linespacing=1.4)

def arr(ax, cx, y0, y1, color=LGRAY):
    ax.annotate("", xy=(cx, y1 + 0.04), xytext=(cx, y0 - 0.04),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14), zorder=5)

def section_bg(ax, cx, cy, w, h, fc, ec="#555555", lw=1.0, alpha=0.95):
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.3",
                           facecolor=fc, edgecolor=ec,
                           linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(patch)

# ══════════════════════════════════════════════════════════════════════════════
# 1 · INPUT
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, CX, 16.3, BW, 0.7, "Input x  (2,701 genes)",
     fc=C_INPUT, ec=LGRAY, lw=1.5, fs=12, bold=True)
arr(ax, CX, 15.95, 15.45)

# ══════════════════════════════════════════════════════════════════════════════
# 2 · ATTENTION BRANCH SECTION
# ══════════════════════════════════════════════════════════════════════════════
ATT_CY, ATT_H = 13.55, 3.6
section_bg(ax, CX, ATT_CY, BW + 0.4, ATT_H, C_ATTN_BG, ec="#7B68EE", lw=1.5)

# Section label
ax.text(CX, ATT_CY + ATT_H/2 - 0.3, "Attention branch",
        ha="center", va="top", fontsize=11, fontweight="bold",
        color="#9B8FEF", zorder=4)

# v3 sub-box (left)
rbox(ax, CX - 1.65, 14.7, 2.9, 0.85,
     "Dense(128, relu)\nv3:  single layer",
     fc=C_ATTN_V3, ec=LGRAY, lw=1.0, fs=9, tc=LGRAY)

# v4+ sub-box (right, green)
rbox(ax, CX + 1.65, 14.7, 2.9, 0.85,
     "Dense(256) + Drop(0.3)\nDense(128) + Drop(0.2)",
     fc=C_ATTN_V4, ec=GREEN, lw=1.5, fs=9, tc=WHITE)
ax.text(CX + 1.65, 14.15, "v4+:  deeper attention",
        ha="center", va="top", fontsize=8, color=GREEN, zorder=4,
        fontstyle="italic")

# Softmax row
rbox(ax, CX, 13.25, BW - 0.4, 0.65,
     "Dense(2701)  +  Softmax(T=2.0)  =  w",
     fc=C_ATTN_SOFT, ec="#9B8FEF", lw=1.0, fs=9.5)

arr(ax, CX, 15.45, 14.5)      # input → attention sub-boxes implied by section
arr(ax, CX, 13.55, 12.9)      # section bottom → residual gate (approx)

# small note bridging v3→softmax and v4→softmax
for dx, label in [(-1.65, ""), (1.65, "")]:
    ax.plot([CX + dx, CX], [14.28, 13.57],
            color="#888888", lw=1.0, ls="--", zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# 3 · RESIDUAL GATE
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, CX, 12.55, BW, 0.65,
     "x_residual  =  x · w · 2701  +  x",
     fc=C_INPUT, ec=LGRAY, lw=1.2, fs=10)
arr(ax, CX, 12.22, 11.75)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · SHARED ENCODER TRUNK
# ══════════════════════════════════════════════════════════════════════════════
ENC_CY, ENC_H = 10.25, 3.5
section_bg(ax, CX, ENC_CY, BW + 0.4, ENC_H, C_ENC_BG, ec="#4A90D9", lw=1.5)

ax.text(CX, ENC_CY + ENC_H/2 - 0.3, "Shared encoder trunk  (all versions)",
        ha="center", va="top", fontsize=11, fontweight="bold",
        color="#7EB8E8", zorder=4)

enc_layers = [
    "Dense(512, relu)  +  LayerNorm  +  Drop(0.4)",
    "Dense(256, relu)  +  LayerNorm  +  Drop(0.3)",
    "Dense(128, relu)  [latent  z]   +  Drop(0.2)",
    "Dense(6, softmax)   [output]",
]
enc_ys = [11.3, 10.45, 9.6, 8.75]
for txt, ey in zip(enc_layers, enc_ys):
    rbox(ax, CX, ey, BW - 0.5, 0.6, txt,
         fc=C_ENC_ITEM, ec="#4A90D9", lw=0.8, fs=9.5)

for i in range(len(enc_ys) - 1):
    arr(ax, CX, enc_ys[i] - 0.3, enc_ys[i+1] + 0.3, color="#4A90D9")

arr(ax, CX, 8.42, 7.9)

# ══════════════════════════════════════════════════════════════════════════════
# 5 · LOSS FUNCTIONS SECTION
# ══════════════════════════════════════════════════════════════════════════════
LOSS_CY, LOSS_H = 5.5, 4.6
section_bg(ax, CX, LOSS_CY, BW + 0.4, LOSS_H, C_LOSS_BG, ec="#CC4400", lw=1.5)

ax.text(CX, LOSS_CY + LOSS_H/2 - 0.3, "Loss functions  (version evolution)",
        ha="center", va="top", fontsize=11, fontweight="bold",
        color="#FF8A65", zorder=4)

# Row 1 — three loss version boxes
lx = [2.4, 4.5, 6.6]
lw_box = 2.4
loss_row1 = [
    (C_LOSS_V3, LGRAY,  "v3  base",           "CE + MMD + Centroid"),
    (C_LOSS_V4, GREEN,  "+ v4  adds",         "Mixup + Entropy min"),
    (C_LOSS_V6, PURPLE, "+ v6  adds",         "Deep CORAL + label smooth"),
]
for x, (fc, ec, title, sub) in zip(lx, loss_row1):
    rbox(ax, x, 6.85, lw_box, 1.0, f"{title}\n{sub}",
         fc=fc, ec=ec, lw=1.5, fs=9, tc=WHITE)

# Row 2 — v5 note
rbox(ax, CX, 5.65, BW - 0.4, 0.7,
     "v5  =  v4 architecture  +  preprocessing fixes  (not an architecture change)\n"
     "Independent target standardisation · dead gene removal · Fano HVG",
     fc=C_LOSS_V5, ec="#888888", lw=1.0, fs=8.5, tc=LGRAY)

# Row 3 — result badges
res = [
    (2.05, C_RES_V3, LGRAY,  "v3:  0.407",   "Target F1"),
    (3.75, C_RES_V4, LGRAY,  "v4:  0.420",   "Target F1"),
    (5.45, C_RES_V5, GREEN,  "v5:  0.438",   "Best target F1"),
    (7.15, C_RES_V6, "#CC4444", "v6:  0.392", "Regressed"),
]
for x, fc, ec, line1, line2 in res:
    rbox(ax, x, 4.55, 1.55, 0.8, f"{line1}\n{line2}",
         fc=fc, ec=ec, lw=1.5, fs=8.5, tc=WHITE,
         bold=("0.438" in line1))

plt.tight_layout(pad=0.3)
out = Path("results/geneattention_architecture_evolution.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.show()
