#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def apply_paper_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def main():
    ap = argparse.ArgumentParser(description="Plot diversity from compare_baseline_vs_ours.csv")
    ap.add_argument("--csv", type=str, default="results/features/comparison/compare_baseline_vs_ours.csv",
                    help="Path to compare_baseline_vs_ours.csv")
    ap.add_argument("--outdir", type=str, default="results/features/comparison")
    ap.add_argument("--style", type=str, default="paper", choices=["paper","default"])
    args = ap.parse_args()

    if args.style == "paper":
        apply_paper_style()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    # clean and sort
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step")

    out_base = Path(args.outdir) / (csv_path.stem + "__plots")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # 1) Diversity traces
    fig1, ax1 = plt.subplots(figsize=(6,3.2), dpi=300)
    ax1.plot(df["step"], df["baseline_diversity_trace"], "o-", label="Baseline", color="#1f77b4")
    ax1.plot(df["step"], df["ours_diversity_trace"], "s-", label="Ours", color="#d62728")
    ax1.set_xlabel("step"); ax1.set_ylabel("diversity (trace)")
    ax1.legend(frameon=False); ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_base.with_name(out_base.stem + "__diversity_traces.png"), transparent=True)
    fig1.savefig(out_base.with_name(out_base.stem + "__diversity_traces.pdf"), transparent=True)
    fig1.savefig(out_base.with_name(out_base.stem + "__diversity_traces.svg"), transparent=True)
    plt.close(fig1)

    # 2) Improvement (%) + Delta
    fig2, ax2 = plt.subplots(figsize=(6,3.2), dpi=300)
    ax2.plot(df["step"], df["improvement_pct"], "o-", color="#2ca02c", label="Improvement %")
    ax2.set_xlabel("step"); ax2.set_ylabel("improvement (%)")
    ax2.grid(True, alpha=0.3)
    ax2_t = ax2.twinx()
    ax2_t.plot(df["step"], df["delta"], "s-", color="#9467bd", label="Delta (Ours-Baseline)")
    ax2_t.set_ylabel("delta")
    # legend above
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_t.get_legend_handles_labels()
    fig2.legend(lines+lines2, labels+labels2, frameon=False,
                loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig2.tight_layout(rect=[0,0,1,0.95])
    fig2.savefig(out_base.with_name(out_base.stem + "__improvement.png"), transparent=True)
    fig2.savefig(out_base.with_name(out_base.stem + "__improvement.pdf"), transparent=True)
    fig2.savefig(out_base.with_name(out_base.stem + "__improvement.svg"), transparent=True)
    plt.close(fig2)

    print(f"[OK] wrote plots → {out_base.parent}")


if __name__ == "__main__":
    main()

