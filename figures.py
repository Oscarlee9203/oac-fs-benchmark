"""Regenerate Figures 1-4 from results/*.json produced by run_all.py.

Figures are sized for a 170 mm (full-page-width) journal column and written as
600-dpi PNG plus vector PDF. Method colours are fixed across figures (Okabe-Ito
palette) and every series also carries a distinct marker, so identity never
depends on colour alone. Greys encode the evaluation setting (in-sample ->
within-cohort held-out -> cross-cohort), never a method.
"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

W = 6.69          # 170 mm in inches
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e4e4e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 600,
})

# method -> (display name, colour, marker)
M = {
    "SupervisedDE": ("Supervised DE", "#0072B2", "o"),
    "CADA-AE":      ("CADA-AE",       "#D55E00", "s"),
    "PERSIST":      ("PERSIST",       "#CC79A7", "D"),
    "DFS-AE":       ("DFS-AE",        "#009E73", "^"),
    "HighVar":      ("High variance", "#E69F00", "v"),
    "PCA":          ("PCA",           "#56B4E9", "P"),
    "Random":       ("Random",        "#4d4d4d", "X"),
}
M["SupDE"] = M["SupervisedDE"]
# evaluation setting -> grey (ordered by stringency)
G_IN, G_WITHIN, G_CROSS = "#cfcfca", "#8f8f8a", "#2f2f2f"


def _save(fig, o, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{o}/{name}.{ext}", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    from PIL import Image                      # flatten to RGB (no alpha channel) for journal upload
    Image.open(f"{o}/{name}.png").convert("RGB").save(f"{o}/{name}.png", dpi=(600, 600))


def _panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", pad=6)


def _line(ax, xs, tr, key, lo_clip=None):
    name, c, mk = M[key]
    m = np.array([t[0] for t in tr]); lo = np.array([t[1] for t in tr]); hi = np.array([t[2] for t in tr])
    if lo_clip is not None:
        lo = np.maximum(lo, lo_clip)
    ax.fill_between(xs, lo, hi, color=c, alpha=.15, lw=0)
    ax.plot(xs, m, color=c, lw=1.3, marker=mk, ms=4.2, mec="white", mew=0.6, label=name,
            clip_on=False, zorder=3)


def _method_handles(keys):
    return [Line2D([0], [0], color=M[k][1], lw=1.3, marker=M[k][2], ms=4.2, mec="white",
                   mew=0.6, label=M[k][0]) for k in keys]


def f1(o):
    """Ground-truth benchmark (semi-synthetic data) and sensitivity to training length."""
    A = json.load(open(f"{o}/f2ci_A.json")); B = json.load(open(f"{o}/f2ci_B.json"))
    S = json.load(open(f"{o}/f2_sensitivity.json"))
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.9)); ax = axs.ravel()
    fcs = A["fcs"]; fk = [str(f) for f in fcs]
    for k in ["Random", "PCA", "HighVar", "DFS-AE", "SupervisedDE", "CADA-AE"]:
        _line(ax[0], fcs, [A["methods"][k][f] for f in fk], k, lo_clip=0)
    null = float(np.mean([A["perm_null"][f][0] for f in fk]))
    dots = (0, (1, 1.5))
    ax[0].axhline(null, color=MUTED, lw=0.8, ls=dots)
    _panel(ax[0], "A", "Recovery of true genes")
    ax[0].set_xlabel("Disease effect size (log2 fold-change)"); ax[0].set_ylabel("AUPRC against known truth")
    ax[0].set_ylim(0, 1); ax[0].set_xticks(fcs)
    cs = B["corrs"]; ck = [str(c) for c in cs]
    for k in ["DFS-AE", "PERSIST", "SupervisedDE", "CADA-AE"]:
        _line(ax[1], cs, [B["auprc"][k][c] for c in ck], k, lo_clip=0)
        _line(ax[2], cs, [B["contam"][k][c] for c in ck], k, lo_clip=0)
    ax[1].axhline(null, color=MUTED, lw=0.8, ls=dots)
    _panel(ax[1], "B", "Recovery under confounding")
    ax[1].set_xlabel("Disease–confounder correlation"); ax[1].set_ylabel("AUPRC against known truth")
    ax[1].set_ylim(0, 1); ax[1].set_xticks(cs)
    _panel(ax[2], "C", "Confounder contamination")
    ax[2].set_xlabel("Disease–confounder correlation"); ax[2].set_ylabel("Confounder genes in top 100 (%)")
    ax[2].set_ylim(0, 50); ax[2].set_xticks(cs)
    eps = sorted(S, key=int); xe = np.arange(len(eps))
    for k in ["DFS-AE", "PERSIST", "SupervisedDE", "CADA-AE"]:
        _line(ax[3], xe, [S[e]["B"]["contam"][k]["0.4"] for e in eps], k, lo_clip=0)
    _panel(ax[3], "D", "Contamination vs training length")
    ax[3].set_xlabel("Training epochs of the deep selectors (correlation 0.4)")
    ax[3].set_ylabel("Confounder genes in top 100 (%)")
    ax[3].set_ylim(0, 50); ax[3].set_xticks(xe); ax[3].set_xticklabels(eps); ax[3].set_xlim(-0.25, len(eps) - 0.75)
    h = _method_handles(["SupervisedDE", "CADA-AE", "PERSIST", "DFS-AE", "HighVar", "PCA", "Random"])
    h.append(Line2D([0], [0], color=MUTED, lw=0.8, ls=dots, label="Random-ranking null"))
    fig.legend(handles=h, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.035),
               columnspacing=1.1, handletextpad=0.4, handlelength=1.6)
    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    _save(fig, o, "Figure1")


def f2(o):
    """Accuracy saturates; DAV does not (eight real contrasts)."""
    rows = json.load(open(f"{o}/f5_panel.json")); xi = np.arange(len(rows))
    names = [r["name"].replace("-", "\n", 1) for r in rows]
    keys = ["SupDE", "HighVar", "Random"]
    fig, ax = plt.subplots(2, 1, figsize=(W, 4.1), sharex=True)
    off = {"SupDE": -0.18, "HighVar": 0.0, "Random": 0.18}
    for k in keys:
        name, c, mk = M[k]
        ax[0].plot(xi + off[k], [r[k + "_AUC"] for r in rows], ls="none", marker=mk, ms=5,
                   color=c, mec="white", mew=0.6, label=name)
    ax[0].set_ylim(0.5, 1.02); ax[0].set_ylabel("Cross-validated AUC")
    _panel(ax[0], "A", "Classification accuracy is high for every panel, including random genes")
    w = 0.24
    for j, k in enumerate(keys):
        ax[1].bar(xi + (j - 1) * (w + 0.02), [r[k + "_DAV"] for r in rows], w, color=M[k][1], label=M[k][0])
    ax[1].set_ylabel("Disease-attributable variance (DAV)"); ax[1].set_ylim(0, 1)
    _panel(ax[1], "B", "In-sample DAV of the same panels")
    ax[1].set_xticks(xi); ax[1].set_xticklabels(names)
    ax[1].tick_params(axis="x", length=0)
    fig.legend(handles=[Line2D([0], [0], ls="none", marker=M[k][2], ms=5, color=M[k][1], mec="white",
                               mew=0.6, label=M[k][0]) for k in keys],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.035), handletextpad=0.3)
    fig.tight_layout(h_pad=1.4)
    _save(fig, o, "Figure2")


def f3(o):
    """Cross-cohort transfer and the within-cohort split-half control."""
    cart = json.load(open(f"{o}/f3_cartilage.json")); syn = json.load(open(f"{o}/f3_synovium.json"))
    sh = json.load(open(f"{o}/splithalf.json")); shr = json.load(open(f"{o}/splithalf_random.json"))
    rnd = json.load(open(f"{o}/f3_random.json"))
    order = ["SupervisedDE", "CADA-AE", "HighVar", "DFS-AE", "PCA", "Random"]
    short = {"SupervisedDE": "Sup. DE", "HighVar": "High var."}
    labs = [short.get(k, M[k][0]) for k in order]; x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.7))
    for a, dat, letter, ttl in [(ax[0, 0], cart, "A", "Cartilage OA (three platforms)"),
                                (ax[0, 1], syn, "B", "Synovium OA (same platform)")]:
        a.bar(x - w / 2 - 0.01, [dat[k]["in"] for k in order], w, color=G_IN)
        a.bar(x + w / 2 + 0.01, [dat[k]["cross"] for k in order], w, color=G_CROSS)
        a.set_xticks(x); a.set_xticklabels(labs); a.tick_params(axis="x", length=0)
        a.set_ylim(0, 0.8); a.set_ylabel("DAV of selected panel"); _panel(a, letter, ttl)
    ax[0, 0].legend(handles=[Patch(color=G_IN, label="In-sample (selection cohort)"),
                             Patch(color=G_CROSS, label="Cross-cohort (independent cohort)")],
                    loc="upper right", handlelength=1.2)
    a = ax[1, 0]
    a.plot(x - 0.09, [cart[k]["cross_auc"] for k in order], ls="none", marker="o", ms=5, color=G_CROSS,
           mec="white", mew=0.6, label="Cartilage OA")
    a.plot(x + 0.09, [syn[k]["cross_auc"] for k in order], ls="none", marker="s", ms=4.6, mfc="white",
           mec=G_CROSS, mew=1.0, label="Synovium OA")
    a.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    a.text(len(order) - 0.55, 0.515, "chance", ha="right", va="bottom", color=MUTED, fontsize=6.5)
    a.set_xticks(x); a.set_xticklabels(labs); a.tick_params(axis="x", length=0)
    a.set_ylim(0.4, 1.02); a.set_ylabel("Cross-cohort AUC"); a.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), handletextpad=0.3)
    _panel(a, "C", "Cross-cohort accuracy stays high")
    a = ax[1, 1]
    ks = list(sh.keys()); cl = ["GSE" + k.split("_")[-1] for k in ks]; xs = np.arange(len(ks))
    m = np.array([sh[k][0] for k in ks]); lo = np.array([sh[k][1] for k in ks]); hi = np.array([sh[k][2] for k in ks])
    a.bar(xs - w / 2 - 0.01, m, w, color=G_WITHIN, yerr=[m - lo, hi - m],
          error_kw=dict(ecolor=INK, elinewidth=0.7, capsize=2, capthick=0.7))
    a.bar(xs + w / 2 + 0.01, [cart["SupervisedDE"]["cross"]] * len(ks), w, color=G_CROSS)
    # floor: DAV of random panels in the same evaluation samples
    for i, k in enumerate(ks):
        a.hlines(shr[k][0], i - w - 0.01, i - 0.01, color="white", lw=2.2)
        a.hlines(shr[k][0], i - w - 0.01, i - 0.01, color="#D55E00", lw=1.2)
        a.hlines(rnd["cartilage"]["mean"], i + 0.01, i + w + 0.01, color="white", lw=2.2)
        a.hlines(rnd["cartilage"]["mean"], i + 0.01, i + w + 0.01, color="#D55E00", lw=1.2)
    a.set_xticks(xs); a.set_xticklabels(cl); a.tick_params(axis="x", length=0)
    a.set_ylim(0, 0.8); a.set_ylabel("Held-out DAV (supervised DE)")
    a.legend(handles=[Patch(color=G_WITHIN, label="Within-cohort (split-half)"),
                      Patch(color=G_CROSS, label="Cross-cohort (mean)"),
                      Line2D([0], [0], color="#D55E00", lw=1.2, label="Random panels, same samples")],
             loc="upper right", handlelength=1.2)
    _panel(a, "D", "Split-half control")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    _save(fig, o, "Figure3")


F4_NAMES = {"ConsensusDE": "Consensus DE (two cohorts)", "XC-CADA-AE": "XC-CADA-AE",
            "HighVar": "High variance (one cohort)", "DE_single": "Supervised DE (one cohort)",
            "XC-CADA-AE-v2": "XC-CADA-AE v2", "DFS-AE_single": "DFS-AE (one cohort)"}


def f4(o):
    """Leave-one-cohort-out held-out DAV against matched random panels."""
    d = json.load(open(f"{o}/f4_xc.json")); mean = d["mean"]
    order = sorted(F4_NAMES, key=lambda k: mean[k]); y = np.arange(len(order))
    rm, (rl, rh) = d["random_mean"], d["random_range"]
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.3))
    ax.grid(False); ax.xaxis.grid(True)
    ax.axvspan(rl, rh, color="#bdbdb8", alpha=.45, lw=0)
    ax.axvline(rm, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    ax.barh(y, [mean[k] for k in order], 0.58, color=G_CROSS)
    for i, k in enumerate(order):
        ax.text(mean[k] + 0.0012, i, f"{mean[k]:.3f}", va="center", ha="left", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    ax.set_yticks(y); ax.set_yticklabels([F4_NAMES[k] for k in order]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.08); ax.set_xlabel("Held-out cohort DAV (leave-one-cohort-out mean)")
    ax.text(rm, len(order) - 0.3, f"{d['n_random']} random panels: mean {rm:.3f}\n(range {rl:.3f}–{rh:.3f})",
            ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_ylim(-0.6, len(order) + 0.75)
    fig.tight_layout()
    _save(fig, o, "Figure4")


def make_all(o="results"):
    for nm, fn in [("Figure1", f1), ("Figure2", f2), ("Figure3", f3), ("Figure4", f4)]:
        try:
            fn(o); print(f"  {nm} saved")
        except FileNotFoundError as e:
            print(f"  {nm} skipped ({e})")


if __name__ == "__main__":
    import sys
    make_all(sys.argv[1] if len(sys.argv) > 1 else "results")
