#!/usr/bin/env python3
"""
Generate benchmark comparison chart for autoresearch-swift README.

Hardcoded data from actual benchmark runs on M4 Max 128GB.
Matches the metrics highlighted in the README table.

Usage:
    python3 scripts/benchmark_chart.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_chart(output_path="benchmark_chart.png"):
    labels = ["Swift (MLX)", "Python (MLX)"]
    colors = ["#FF6B35", "#4ECDC4"]

    metrics = [
        {
            "title": "Training Quality (val_bpb)",
            "subtitle": "Lower is better",
            "values": [1.418, 1.863],
            "fmt": ".3f",
            "invert": True,  # lower is better → highlight the lower bar
        },
        {
            "title": "Startup Time",
            "subtitle": "Lower is better",
            "values": [0.1, 1.4],
            "fmt": ".1f",
            "suffix": "s",
            "invert": True,
        },
        {
            "title": "Throughput (tok/sec)",
            "subtitle": "Higher is better",
            "values": [50035, 50000],
            "fmt": ",.0f",
            "invert": False,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, m in zip(axes, metrics):
        values = m["values"]
        winner_idx = values.index(min(values)) if m.get("invert") else values.index(max(values))

        # Muted colors for loser, full color for winner
        bar_colors = []
        for i, c in enumerate(colors):
            if i == winner_idx:
                bar_colors.append(c)
            else:
                bar_colors.append(c + "66")  # 40% opacity via hex alpha

        bars = ax.barh(
            range(len(values)), values,
            color=bar_colors, height=0.55,
            edgecolor="white", linewidth=2,
            zorder=3,
        )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=12, fontweight="600")
        ax.set_title(m["title"], fontsize=13, fontweight="bold", pad=14, loc="left")
        ax.set_xlabel(m["subtitle"], fontsize=9, color="#888", style="italic")
        ax.invert_yaxis()

        # Value labels
        max_val = max(values)
        for i, (bar, val) in enumerate(zip(bars, values)):
            suffix = m.get("suffix", "")
            text = f"{val:{m['fmt']}}{suffix}"
            if i == winner_idx:
                text = f"  {text}"  # slight padding
            ax.text(
                bar.get_width() + max_val * 0.03,
                bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=11,
                fontweight="bold" if i == winner_idx else "normal",
                color="#222" if i == winner_idx else "#888",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#DDD")
        ax.tick_params(left=False, bottom=True, colors="#AAA")
        ax.set_xlim(0, max_val * 1.35)
        ax.grid(axis="x", color="#EEE", linewidth=0.5, zorder=0)

    fig.suptitle(
        "autoresearch-swift vs Python MLX  ·  M4 Max 128GB  ·  5-min training budget",
        fontsize=10, color="#666", y=0.02,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1], pad=2)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    make_chart()
