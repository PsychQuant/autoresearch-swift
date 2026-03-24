#!/usr/bin/env python3
"""
Generate benchmark comparison chart for autoresearch-swift README.
4-panel: val_bpb, tok/sec, startup, memory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_chart(output_path="benchmark_chart.png"):
    labels = ["Swift (MLX)", "Python (MLX)"]
    colors = ["#FF6B35", "#4ECDC4"]

    metrics = [
        {
            "title": "Training Quality (val_bpb)",
            "subtitle": "Lower is better",
            "values": [1.406, 1.863],
            "fmt": ".3f",
            "invert": True,
        },
        {
            "title": "Throughput (tok/sec)",
            "subtitle": "Higher is better",
            "values": [54051, 50000],
            "fmt": ",.0f",
            "invert": False,
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
            "title": "Peak Memory (GB)",
            "subtitle": "Lower is better",
            "values": [23.2, 21.7],
            "fmt": ".1f",
            "suffix": " GB",
            "invert": True,
        },
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, m in zip(axes, metrics):
        values = m["values"]
        winner_idx = values.index(min(values)) if m.get("invert") else values.index(max(values))

        bar_colors = []
        for i, c in enumerate(colors):
            bar_colors.append(c if i == winner_idx else c + "66")

        bars = ax.barh(
            range(len(values)), values,
            color=bar_colors, height=0.55,
            edgecolor="white", linewidth=2, zorder=3,
        )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11, fontweight="600")
        ax.set_title(m["title"], fontsize=12, fontweight="bold", pad=12, loc="left")
        ax.set_xlabel(m["subtitle"], fontsize=8, color="#888", style="italic")
        ax.invert_yaxis()

        max_val = max(values)
        for i, (bar, val) in enumerate(zip(bars, values)):
            suffix = m.get("suffix", "")
            text = f"{val:{m['fmt']}}{suffix}"
            ax.text(
                bar.get_width() + max_val * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"  {text}" if i == winner_idx else text,
                va="center", fontsize=10,
                fontweight="bold" if i == winner_idx else "normal",
                color="#222" if i == winner_idx else "#888",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#DDD")
        ax.tick_params(left=False, bottom=True, colors="#AAA")
        ax.set_xlim(0, max_val * 1.4)
        ax.grid(axis="x", color="#EEE", linewidth=0.5, zorder=0)

    fig.suptitle(
        "autoresearch-swift vs Python MLX  ·  M4 Max 128GB  ·  5-min training budget",
        fontsize=9, color="#666", y=0.02,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1], pad=1.5)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    make_chart()
