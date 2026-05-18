#!/usr/bin/env python3
"""Analyze GOODRL step2 training logs.

Usage:
    python tools/analyze.py runs/<run_id>                # plot single run
    python tools/analyze.py runs/*                       # compare multiple runs
    python tools/analyze.py runs/<run_id> --csv          # dump CSVs
    python tools/analyze.py runs/<run_id> --out fig.png  # save figure
"""
import re, sys, os, json, argparse
from pathlib import Path

EP_RE = re.compile(
    r"Episode-(\d+):\s+ep_meanFlowTime:\s+([0-9.eE+-]+)\s+all_loss:\s+(-?[0-9.eE+-]+)"
    r"\s+p_loss:\s+(-?[0-9.eE+-]+)\s+e_loss:\s+(-?[0-9.eE+-]+)"
    r"\s+v_loss:\s+([0-9.eE+-]+)\s+v_mre:\s+([0-9.eE+-]+)"
    r"\s+grad_changes:\s+([0-9.eE+-]+)\s+time_elapsed:\s+([0-9.eE+-]+)")
VAL_RE = re.compile(
    r"Vlidation at update-([a-zA-Z0-9]+):\s+mean_flowtime_deterministic:\s+([0-9.eE+-]+)"
    r"\+/-([0-9.eE+-]+)\s+mean_Entropy:\s+([0-9.eE+-]+)\+/-([0-9.eE+-]+)"
    r"\s+record:\s+([0-9.eE+-]+)")
HEFT_RE = re.compile(
    r"Vlidation at HEFT:\s+mean_flowtime_deterministic:\s+([0-9.eE+-]+)\+/-([0-9.eE+-]+)")


def parse(log_path):
    eps, vals, heft = [], [], None
    with open(log_path) as f:
        for L in f:
            m = EP_RE.search(L)
            if m:
                eps.append({
                    "episode": int(m.group(1)),
                    "ep_meanFlowTime": float(m.group(2)),
                    "all_loss": float(m.group(3)),
                    "p_loss": float(m.group(4)),
                    "e_loss": float(m.group(5)),
                    "v_loss": float(m.group(6)),
                    "v_mre": float(m.group(7)),
                    "grad_changes": float(m.group(8)),
                    "time_elapsed_h": float(m.group(9)),
                })
                continue
            m = VAL_RE.search(L)
            if m:
                upd_str = m.group(1)
                try:
                    upd = int(upd_str)
                except ValueError:
                    upd = -1
                vals.append({
                    "update": upd,
                    "val_flowtime": float(m.group(2)),
                    "val_std": float(m.group(3)),
                    "val_entropy": float(m.group(4)),
                    "val_entropy_std": float(m.group(5)),
                    "record": float(m.group(6)),
                })
                continue
            m = HEFT_RE.search(L)
            if m:
                heft = {"flowtime": float(m.group(1)), "std": float(m.group(2))}
    return {"episodes": eps, "validations": vals, "heft": heft}


def dump_csv(parsed, prefix):
    import csv
    ep_path = prefix + "_episodes.csv"
    if parsed["episodes"]:
        keys = list(parsed["episodes"][0].keys())
        with open(ep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(parsed["episodes"])
    val_path = prefix + "_validations.csv"
    if parsed["validations"]:
        keys = list(parsed["validations"][0].keys())
        with open(val_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(parsed["validations"])
    return ep_path, val_path


def plot(runs, out_path=None, smooth=10):
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_val, ax_loss), (ax_vmre, ax_grad) = axes

    def moving_avg(x, w):
        if len(x) < w: return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    for label, parsed in runs:
        eps = parsed["episodes"]
        vals = parsed["validations"]
        if not eps:
            continue
        ep_x = [e["episode"] for e in eps]
        v_loss = [e["v_loss"] for e in eps]
        v_mre = [e["v_mre"] for e in eps]
        p_loss = [e["p_loss"] for e in eps]
        grad = [e["grad_changes"] for e in eps]

        val_x = [v["update"] for v in vals if v["update"] >= 0]
        val_y = [v["val_flowtime"] for v in vals if v["update"] >= 0]
        val_rec = [v["record"] for v in vals if v["update"] >= 0]

        ax_val.plot(val_x, val_y, "-o", label=f"{label} val", alpha=0.7, markersize=3)
        ax_val.plot(val_x, val_rec, "--", label=f"{label} record", alpha=0.5)
        if parsed["heft"]:
            ax_val.axhline(parsed["heft"]["flowtime"], linestyle=":", alpha=0.3)

        w = min(smooth, max(1, len(v_loss)//5))
        if w > 1:
            ax_loss.plot(ep_x[w-1:], moving_avg(v_loss, w), label=f"{label} v_loss(MA{w})")
            ax_loss.plot(ep_x[w-1:], moving_avg(p_loss, w), label=f"{label} p_loss(MA{w})", linestyle="--")
            ax_vmre.plot(ep_x[w-1:], moving_avg(v_mre, w), label=f"{label} v_mre(MA{w})")
            ax_grad.plot(ep_x[w-1:], moving_avg(grad, w), label=f"{label} grad_changes(MA{w})")
        else:
            ax_loss.plot(ep_x, v_loss, label=f"{label} v_loss")
            ax_loss.plot(ep_x, p_loss, label=f"{label} p_loss", linestyle="--")
            ax_vmre.plot(ep_x, v_mre, label=f"{label} v_mre")
            ax_grad.plot(ep_x, grad, label=f"{label} grad_changes")

    ax_val.set_title("Validation mean flowtime (deterministic)")
    ax_val.set_xlabel("Update"); ax_val.set_ylabel("Mean Flowtime"); ax_val.legend(fontsize=7); ax_val.grid(alpha=0.3)
    ax_loss.set_title("PPO losses (moving avg)")
    ax_loss.set_xlabel("Episode"); ax_loss.set_ylabel("Loss"); ax_loss.legend(fontsize=7); ax_loss.grid(alpha=0.3)
    ax_loss.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax_vmre.set_title("Critic mean relative error (v_mre)")
    ax_vmre.set_xlabel("Episode"); ax_vmre.set_ylabel("v_mre (%)"); ax_vmre.legend(fontsize=7); ax_vmre.grid(alpha=0.3)
    ax_grad.set_title("Actor gradient L2 norm")
    ax_grad.set_xlabel("Episode"); ax_grad.set_ylabel("grad_changes"); ax_grad.legend(fontsize=7); ax_grad.grid(alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run directories OR log files")
    ap.add_argument("--csv", action="store_true", help="Dump CSVs alongside log")
    ap.add_argument("--out", default=None, help="Save figure to this path")
    ap.add_argument("--smooth", type=int, default=10, help="Moving avg window")
    args = ap.parse_args()

    runs = []
    for r in args.runs:
        p = Path(r)
        if p.is_dir():
            log = p / "step2.log"
            label = p.name
        elif p.is_file():
            log = p
            label = p.stem
        else:
            print(f"Skipping {r}: not found", file=sys.stderr); continue
        if not log.exists():
            print(f"Skipping {label}: no log at {log}", file=sys.stderr); continue
        parsed = parse(log)
        print(f"[{label}] episodes={len(parsed['episodes'])}, validations={len(parsed['validations'])}")
        runs.append((label, parsed))
        if args.csv:
            prefix = str(log).rsplit(".log",1)[0]
            ep_csv, val_csv = dump_csv(parsed, prefix)
            print(f"  -> {ep_csv}, {val_csv}")

    if runs and not args.csv:
        plot(runs, out_path=args.out, smooth=args.smooth)


if __name__ == "__main__":
    main()
