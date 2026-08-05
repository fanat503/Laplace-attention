# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Paper figures: the three plots the paper's argument stands on.

Each figure obeys one rule (Anthropic style): ONE takeaway per figure,
statable in the caption's first sentence.

  fig1_twin_divergence : val loss, base vs HLA twins (same init/data/steps)
                         + inset of the gap in std units of seed noise.
  fig2_litm_curves     : positional recall (pos_10..pos_90) at N checkpoints
                         - does HLA flatten the U while base keeps sagging?
  fig3_gap_closure     : H5 causal patching - base / franken / hla bars per
                         retrieval probe with the pre-registered 50%/20%
                         thresholds drawn as decision lines.

Inputs are the artifacts training already produces (train_log_*.csv,
causal JSON from scripts/causal_patch.py). Works on CPU.

    python scripts/make_paper_figures.py --base-log runs/b/train_log_b.csv \
        --hla-log runs/h/train_log_h.csv --causal-json runs/causal.json \
        --out-dir figures/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List


def require_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })
    return plt


BASE_COLOR, HLA_COLOR, FRANKEN_COLOR = "#606060", "#B0413E", "#3E6FB0"


def read_log(path: str) -> Dict[str, List[float]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] and not row[0].startswith("#"):
                rows.append(row)
    header, data = rows[0], rows[1:]
    out: Dict[str, List[float]] = {k: [] for k in header}
    for r in data:
        for k, v in zip(header, r):
            try:
                out[k].append(float(v))
            except ValueError:
                out[k].append(float("nan"))
    return out


def fig1_twin_divergence(base_log, hla_log, out, seed_std: float = 0.0):
    plt = require_matplotlib()
    b, h = read_log(base_log), read_log(hla_log)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for log, name, color in ((b, "base", BASE_COLOR), (h, "HLA", HLA_COLOR)):
        xs = [t for t, v in zip(log["tokens_seen"], log["val_loss"])
              if not math.isnan(v)]
        ys = [v for v in log["val_loss"] if not math.isnan(v)]
        ax.plot(xs, ys, label=name, color=color, lw=1.4)
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("val loss")
    ax.legend()
    # Inset: the actual decision quantity - gap in units of seed noise.
    if seed_std > 0:
        bx = {t: v for t, v in zip(b["tokens_seen"], b["val_loss"]) if not math.isnan(v)}
        hx = {t: v for t, v in zip(h["tokens_seen"], h["val_loss"]) if not math.isnan(v)}
        common = sorted(set(bx) & set(hx))
        if common:
            ia = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
            ia.plot(common, [(bx[t] - hx[t]) / seed_std for t in common],
                    color=HLA_COLOR, lw=1.2)
            ia.axhline(0, color="k", lw=0.6)
            ia.set_title("gap / seed σ", fontsize=7)
            ia.tick_params(labelsize=6)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig2_litm_curves(base_log, hla_log, out, n_checkpoints: int = 3):
    plt = require_matplotlib()
    cols = ["pos_10", "pos_30", "pos_50", "pos_70", "pos_90"]
    depths = [10, 30, 50, 70, 90]
    b, h = read_log(base_log), read_log(hla_log)
    if "pos_10" not in b or all(math.isnan(v) for v in b["pos_10"]):
        raise SystemExit("no LITM columns in logs (need svd_every cadence rows)")
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(3.0 * n_checkpoints, 2.6),
                             sharey=True)
    for log, name, color in ((b, "base", BASE_COLOR), (h, "HLA", HLA_COLOR)):
        idx = [i for i, v in enumerate(log["pos_10"]) if not math.isnan(v)]
        picks = [idx[round(j * (len(idx) - 1) / (n_checkpoints - 1))]
                 for j in range(n_checkpoints)] if len(idx) >= n_checkpoints else idx
        for ax, i in zip(axes, picks):
            ax.plot(depths, [log[c][i] for c in cols], "o-", ms=3, lw=1.2,
                    label=name, color=color)
            ax.set_title(f"{log['tokens_seen'][i]:,.0f} tokens", fontsize=8)
            ax.set_xlabel("needle depth (%)")
    axes[0].set_ylabel("P(needle)")
    axes[0].legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig3_gap_closure(causal_json, out):
    plt = require_matplotlib()
    res = json.load(open(causal_json, encoding="utf-8"))
    gc = res.get("gap_closure")
    if not gc:
        raise SystemExit("causal JSON has no gap_closure block (rerun causal_patch.py)")
    metrics = [m for m, r in gc.items() if not r.get("closure_note")]
    if not metrics:
        raise SystemExit("all gaps flagged too-small - nothing to plot honestly")
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.9 * len(metrics), 3.2),
                             sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        r = gc[m]
        names = ["base", "franken", "HLA"]
        vals = [r["base"], r["franken"], r["hla"]]
        colors = [BASE_COLOR, FRANKEN_COLOR, HLA_COLOR]
        ax.bar(names, vals, color=colors, width=0.62)
        # Pre-registered decision lines on the base->hla span
        for frac, style in ((0.2, ":"), (0.5, "--")):
            ax.axhline(r["base"] + frac * r["gap"], color="k", ls=style, lw=0.8)
        ax.set_title(f"{m}\nclosure={r['closure']:.2f}"
                     + (f" ±{r['closure_std_bound']:.2f}" if "closure_std_bound" in r else ""),
                     fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("H5 causal patching: does retrieval geometry carry the gain?",
                 fontsize=9, y=1.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig4_knockout_context(analysis_json, out):
    """Mechanism knockout Dloss vs context length (H2: the long-range
    mechanisms must MATTER MORE as context grows - the quantitative
    long-context evidence, one line per mechanism)."""
    plt = require_matplotlib()
    res = json.load(open(analysis_json, encoding="utf-8"))
    ko = res.get("knockout_by_context_length")
    if not ko:
        raise SystemExit("analysis JSON lacks knockout_by_context_length")
    lengths = sorted(int(k) for k in ko)
    mechs = sorted({m for L in ko.values() for m in L if m.startswith("ko_")})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for m in mechs:
        ys = [ko[str(L)].get(m, float("nan")) for L in lengths]
        ax.plot(lengths, ys, "o-", ms=3.5, lw=1.3,
                label=m.replace("ko_", "").replace("_delta", ""))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("knockout \u0394loss")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig5_mechanism_trajectories(base_log, hla_log, out):
    """Internal dynamics over training (GDN/Diff reviewer lesson: show WHEN
    mechanisms wake up, not just that they exist): gate/salience activity,
    saturation, and retrieval probes on one page, base vs HLA."""
    plt = require_matplotlib()
    panels = [
        ("Gate activity", ["gate_k_mean", "gate_v_mean"]),
        ("Mix envelope", ["mix_k_mean", "mix_v_mean"]),
        ("Saturation", ["gate_k_sat_frac", "angle_q_sat_frac"]),
        ("Retrieval probes", ["induction", "distractor_margin"]),
        ("LITM scalars", ["litm_middle_drop", "litm_worst_frac"]),
        ("Head interference", ["qk_interference", "qk_ov_separation"]),
    ]
    logs = {"base": read_log(base_log), "HLA": read_log(hla_log)}
    colors = {"base": BASE_COLOR, "HLA": HLA_COLOR}
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (title, cols) in zip(axes.flat, panels):
        drew = False
        for name, log in logs.items():
            for i, col in enumerate(cols):
                if col not in log:
                    continue
                pairs = [(x, y) for x, y in zip(log["tokens_seen"], log[col])
                         if not math.isnan(y)]
                if not pairs:
                    continue
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                        ["-", "--"][i % 2], color=colors[name], lw=1.2,
                        label=f"{name}:{col}", alpha=0.9)
                drew = True
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("tokens", fontsize=8)
        if drew:
            ax.legend(fontsize=5.5)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-grade figures from run artifacts")
    ap.add_argument("--base-log")
    ap.add_argument("--hla-log")
    ap.add_argument("--causal-json")
    ap.add_argument("--analysis-json")
    ap.add_argument("--seed-std", type=float, default=0.0,
                    help="val-loss std across seeds (for the gap inset)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.base_log and args.hla_log:
        fig1_twin_divergence(args.base_log, args.hla_log,
                             os.path.join(args.out_dir, "fig1_twin_divergence.png"),
                             seed_std=args.seed_std)
        try:
            fig2_litm_curves(args.base_log, args.hla_log,
                             os.path.join(args.out_dir, "fig2_litm_curves.png"))
        except SystemExit as e:
            print(f"[skip fig2] {e}")
    if args.causal_json:
        try:
            fig3_gap_closure(args.causal_json,
                             os.path.join(args.out_dir, "fig3_gap_closure.png"))
        except SystemExit as e:
            print(f"[skip fig3] {e}")
    if args.base_log and args.hla_log:
        try:
            fig5_mechanism_trajectories(
                args.base_log, args.hla_log,
                os.path.join(args.out_dir, "fig5_mechanism_trajectories.png"))
        except SystemExit as e:
            print(f"[skip fig5] {e}")
    if args.analysis_json:
        try:
            fig4_knockout_context(args.analysis_json,
                                  os.path.join(args.out_dir, "fig4_knockout_context.png"))
        except (SystemExit, OSError) as e:
            print(f"[skip fig4] {e}")


if __name__ == "__main__":
    main()
