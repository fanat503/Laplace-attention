"""Generate paper-ready plots from HLA-v4 logs and analysis JSON files.

This script is optional and depends on matplotlib/pandas. It is intentionally not
part of TPU training requirements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise RuntimeError("Install matplotlib to use make_plots.py") from e


def read_log(path: str) -> Dict[str, List[float]]:
    header = None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
            else:
                rows.append(dict(zip(header, row)))
    if header is None:
        raise RuntimeError(f"No header in {path}")
    out: Dict[str, List[float]] = {k: [] for k in header}
    for r in rows:
        for k in header:
            v = r.get(k, "nan")
            try:
                out[k].append(float(v))
            except Exception:
                out[k].append(float("nan"))
    return out


def plot_loss(base_log: str, hla_log: str, out: str) -> None:
    plt = require_matplotlib()
    base, hla = read_log(base_log), read_log(hla_log)
    plt.figure(figsize=(7, 4))
    plt.plot(base["tokens_seen"], base["val_loss"], label="base val")
    plt.plot(hla["tokens_seen"], hla["val_loss"], label="HLA val")
    plt.xlabel("tokens seen")
    plt.ylabel("validation loss")
    plt.legend()
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def heatmap(matrix, out: str, title: str, xlabel: str = "head", ylabel: str = "layer") -> None:
    plt = require_matplotlib()
    plt.figure(figsize=(8, 5))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_analysis(analysis_json: str, out_dir: str, prefix: str) -> None:
    with open(analysis_json, "r", encoding="utf-8") as f:
        a = json.load(f)
    if "attention_entropy_normalized" in a:
        heatmap(a["attention_entropy_normalized"], os.path.join(out_dir, f"{prefix}_entropy_norm.png"), f"{prefix} normalized attention entropy")
    if "attention_entropy" in a:
        heatmap(a["attention_entropy"], os.path.join(out_dir, f"{prefix}_entropy.png"), f"{prefix} attention entropy")

    # Gate/scale summaries by layer.
    if "layer_metrics" in a:
        plt = require_matplotlib()
        layers = [x["layer"] for x in a["layer_metrics"]]
        for key in ["gate_k", "gate_v", "mix_k", "mix_v"]:
            vals = [x[key]["mean"] for x in a["layer_metrics"]]
            plt.plot(layers, vals, label=key)
        plt.xlabel("layer")
        plt.ylabel("mean activation / scale")
        plt.legend()
        plt.tight_layout()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_gate_scale_means.png"))
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-log")
    ap.add_argument("--hla-log")
    ap.add_argument("--base-analysis")
    ap.add_argument("--hla-analysis")
    ap.add_argument("--kl-json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.base_log and args.hla_log:
        plot_loss(args.base_log, args.hla_log, os.path.join(args.out_dir, "loss_curves.png"))
    if args.base_analysis:
        plot_analysis(args.base_analysis, args.out_dir, "base")
    if args.hla_analysis:
        plot_analysis(args.hla_analysis, args.out_dir, "hla")
    if args.kl_json:
        with open(args.kl_json, "r", encoding="utf-8") as f:
            kl = json.load(f)
        heatmap(kl["kl_base_to_hla_per_layer_head"], os.path.join(args.out_dir, "attention_kl.png"), "KL(base || HLA) attention")
    print(f"wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
