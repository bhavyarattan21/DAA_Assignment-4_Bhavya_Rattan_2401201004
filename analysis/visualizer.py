"""
Visualization Module for Dynamic Programming Tables and Runtime Curves
Generates publication-quality annotated heatmaps and performance plots.
All generated plots are saved directly to the 'graphes/' directory.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

GRAPHES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graphes")
os.makedirs(GRAPHES_DIR, exist_ok=True)


def plot_dp_table(
    dp_table: List[List[float]],
    title: str = "DP Table",
    xlabel: str = "j",
    ylabel: str = "i",
    x_ticks: Optional[List[str]] = None,
    y_ticks: Optional[List[str]] = None,
    highlight_cells: Optional[List[Tuple[int, int]]] = None,
    cmap: str = "Blues",
    save_filename: Optional[str] = None
) -> str:
    """
    Renders a 2-D DP table as an annotated heatmap with optional optimal-path cell highlights.
    Returns the path of the saved figure.
    """
    arr = np.array(dp_table, dtype=float)
    # Replace inf with NaN for visualization if any (like MCM upper triangle unused)
    masked_arr = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(max(6, arr.shape[1] * 0.7), max(5, arr.shape[0] * 0.6)))
    im = ax.imshow(masked_arr, cmap=cmap, aspect="auto")

    # Annotate cell values
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val) or np.isinf(val):
                text_str = "∞" if np.isinf(val) else "-"
            else:
                text_str = str(int(val)) if val == int(val) else f"{val:.1f}"
            
            # Determine text color and font weight based on highlighting
            is_highlighted = highlight_cells and (i, j) in highlight_cells
            color = "red" if is_highlighted else ("white" if masked_arr[i, j] > (masked_arr.max() * 0.65) else "black")
            weight = "bold" if is_highlighted else "normal"
            fontsize = 9 if is_highlighted else 8

            ax.text(j, i, text_str, ha="center", va="center", color=color, fontweight=weight, fontsize=fontsize)

            # Draw highlight borders if cell is in optimal traceback path
            if is_highlighted:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="crimson", linewidth=2.5)
                ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="DP Value")
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)

    if x_ticks and len(x_ticks) == arr.shape[1]:
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, fontsize=9)
    if y_ticks and len(y_ticks) == arr.shape[0]:
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels(y_ticks, fontsize=9)

    plt.tight_layout()

    filename = save_filename or (title.replace(" ", "_").lower() + ".png")
    out_path = os.path.join(GRAPHES_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dp_runtime_growth(
    input_sizes: List[int],
    runtimes: List[float],
    algo_name: str,
    theoretical_complexity: Optional[str] = None,
    save_filename: Optional[str] = None
) -> str:
    """
    Plots the empirical runtime growth curve against input sizes with aesthetic styling.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(input_sizes, runtimes, marker="o", linewidth=2.2, color="#0066CC", label="Measured Runtime (seconds)")

    ax.set_xlabel("Input Size (n)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
    title_text = f"{algo_name} — Empirical Runtime Growth"
    if theoretical_complexity:
        title_text += f"\n[Theoretical: {theoretical_complexity}]"
    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="#f8f9fa")

    plt.tight_layout()
    filename = save_filename or f"{algo_name.replace(' ', '_').lower()}_runtime.png"
    out_path = os.path.join(GRAPHES_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_recursion_vs_dp(
    sizes: List[int],
    rec_times: List[float],
    dp_times: List[float],
    algo_name: str,
    save_filename: Optional[str] = None
) -> str:
    """
    Plots a dual-axis or logarithmic comparative study between naive recursive (exponential)
    and dynamic programming (polynomial) execution times.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(sizes, rec_times, marker="s", color="#D9381E", linewidth=2.2, label="Naive Recursive (Exponential)")
    ax.plot(sizes, dp_times, marker="o", color="#107C41", linewidth=2.2, label="DP Tabulation (Polynomial)")

    ax.set_xlabel("Problem Size (n)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_title(f"{algo_name}: Naive Recursion vs Dynamic Programming", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    filename = save_filename or f"recursion_vs_dp_{algo_name.replace(' ', '_').lower()}.png"
    out_path = os.path.join(GRAPHES_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
