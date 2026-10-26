"""
Benchmarking and Analysis Engine
Automates empirical runtime experiments, recursion vs DP comparisons,
generates datasets in 'data_csv/', and renders visualization artifacts in 'graphes/'.
"""

import os
import sys
import time
import random
import string
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from algorithms.lcs import lcs, lcs_recursive, lcs_traceback_path
from algorithms.knapsack import knapsack_01, knapsack_recursive, knapsack_space_optimized, knapsack_traceback_path
from algorithms.matrix_chain import matrix_chain_order, print_optimal_parens, matrix_chain_recursive
from analysis.visualizer import plot_dp_table, plot_dp_runtime_growth, plot_recursion_vs_dp

DATA_CSV_DIR = os.path.join(BASE_DIR, "data_csv")
GRAPHES_DIR = os.path.join(BASE_DIR, "graphes")
os.makedirs(DATA_CSV_DIR, exist_ok=True)
os.makedirs(GRAPHES_DIR, exist_ok=True)


def generate_random_string(length: int, alphabet: str = "ACGT") -> str:
    """Generates random string mimicking DNA / sequence characters."""
    return "".join(random.choices(alphabet, k=length))


def benchmark_lcs():
    print("-> Benchmarking Longest Common Subsequence (LCS)...")
    sizes = [50, 100, 150, 200, 250, 300, 400, 500]
    results = []

    for n in sizes:
        s1 = generate_random_string(n)
        s2 = generate_random_string(n)
        
        start_t = time.perf_counter()
        length, _, _ = lcs(s1, s2)
        elapsed = time.perf_counter() - start_t
        
        theoretical_ops = n * n
        results.append({
            "String_Length_m": n,
            "String_Length_n": n,
            "LCS_Length": length,
            "Theoretical_Ops_mn": theoretical_ops,
            "Runtime_Seconds": elapsed
        })
        print(f"   LCS size {n}x{n}: {elapsed:.5f}s (LCS length={length})")

    df = pd.DataFrame(results)
    csv_path = os.path.join(DATA_CSV_DIR, "lcs_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Plot runtime
    plot_dp_runtime_growth(
        sizes,
        df["Runtime_Seconds"].tolist(),
        algo_name="Longest Common Subsequence",
        theoretical_complexity="O(m * n)",
        save_filename="lcs_runtime.png"
    )
    return df


def benchmark_knapsack():
    print("-> Benchmarking 0/1 Knapsack...")
    item_counts = [50, 100, 150, 200, 250, 300, 400, 500]
    capacity = 200
    results = []

    for n in item_counts:
        weights = [random.randint(1, 40) for _ in range(n)]
        values = [random.randint(10, 100) for _ in range(n)]

        start_t = time.perf_counter()
        max_val, selected, _ = knapsack_01(weights, values, capacity)
        elapsed = time.perf_counter() - start_t

        start_t_opt = time.perf_counter()
        max_val_opt = knapsack_space_optimized(weights, values, capacity)
        elapsed_opt = time.perf_counter() - start_t_opt

        results.append({
            "Num_Items_n": n,
            "Capacity_W": capacity,
            "Theoretical_Ops_nW": n * capacity,
            "Max_Profit": max_val,
            "Selected_Count": len(selected),
            "Runtime_2D_Seconds": elapsed,
            "Runtime_1D_Seconds": elapsed_opt
        })
        print(f"   Knapsack n={n}, W={capacity}: 2D={elapsed:.5f}s, 1D={elapsed_opt:.5f}s")

    df = pd.DataFrame(results)
    csv_path = os.path.join(DATA_CSV_DIR, "knapsack_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Plot runtime
    plot_dp_runtime_growth(
        item_counts,
        df["Runtime_2D_Seconds"].tolist(),
        algo_name="0-1 Knapsack",
        theoretical_complexity="O(n * W)",
        save_filename="knapsack_runtime.png"
    )
    return df


def benchmark_matrix_chain():
    print("-> Benchmarking Matrix Chain Multiplication (MCM)...")
    matrix_counts = [10, 20, 30, 40, 50, 60, 75, 90, 110]
    results = []

    for n in matrix_counts:
        dims = [random.randint(5, 50) for _ in range(n + 1)]
        
        start_t = time.perf_counter()
        min_cost, split_tab, _ = matrix_chain_order(dims)
        elapsed = time.perf_counter() - start_t

        results.append({
            "Num_Matrices_n": n,
            "Theoretical_Ops_n3": n ** 3,
            "Min_Scalar_Mults": min_cost,
            "Runtime_Seconds": elapsed
        })
        print(f"   MCM n={n}: {elapsed:.5f}s (Min cost: {min_cost})")

    df = pd.DataFrame(results)
    csv_path = os.path.join(DATA_CSV_DIR, "matrix_chain_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Plot runtime
    plot_dp_runtime_growth(
        matrix_counts,
        df["Runtime_Seconds"].tolist(),
        algo_name="Matrix Chain Multiplication",
        theoretical_complexity="O(n^3)",
        save_filename="matrix_chain_runtime.png"
    )
    return df


def benchmark_recursion_vs_dp():
    print("-> Benchmarking Naive Recursion vs Dynamic Programming (LCS)...")
    test_lengths = [4, 6, 8, 10, 12, 14, 15, 16]
    rec_times = []
    dp_times = []

    for n in test_lengths:
        s1 = generate_random_string(n, alphabet="ABCD")
        s2 = generate_random_string(n, alphabet="ABCD")

        # DP benchmark
        t0 = time.perf_counter()
        for _ in range(5):
            _ = lcs(s1, s2)
        dp_t = (time.perf_counter() - t0) / 5.0

        # Naive recursive benchmark
        t0 = time.perf_counter()
        _ = lcs_recursive(s1, s2)
        rec_t = time.perf_counter() - t0

        rec_times.append(rec_t)
        dp_times.append(dp_t)
        print(f"   Size {n}: Recursive = {rec_t:.6f}s vs DP = {dp_t:.6f}s (Ratio: {rec_t/max(dp_t, 1e-9):.1f}x)")

    df = pd.DataFrame({
        "Sequence_Length": test_lengths,
        "Naive_Recursive_Seconds": rec_times,
        "DP_Tabulation_Seconds": dp_times,
        "Speedup_Ratio": [r / max(d, 1e-9) for r, d in zip(rec_times, dp_times)]
    })
    csv_path = os.path.join(DATA_CSV_DIR, "recursion_vs_dp.csv")
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    plot_recursion_vs_dp(
        test_lengths,
        rec_times,
        dp_times,
        algo_name="LCS (Sequence Analysis)",
        save_filename="recursion_vs_dp_comparison.png"
    )
    return df


def generate_complexity_comparison_table():
    print("-> Generating Complexity Comparison Table...")
    data = [
        {
            "Algorithm": "Longest Common Subsequence",
            "Domain": "Sequence Analysis",
            "Time Complexity": "O(m * n)",
            "Space Complexity": "O(m * n)",
            "Optimized Space": "O(min(m, n))",
            "Subproblems Count": "m * n",
            "Dependency Pattern": "dp[i-1][j-1], dp[i-1][j], dp[i][j-1]"
        },
        {
            "Algorithm": "0/1 Knapsack",
            "Domain": "Resource Allocation",
            "Time Complexity": "O(n * W)",
            "Space Complexity": "O(n * W)",
            "Optimized Space": "O(W) 1-D array",
            "Subproblems Count": "n * W",
            "Dependency Pattern": "dp[i-1][w], dp[i-1][w - weights[i-1]]"
        },
        {
            "Algorithm": "Matrix Chain Multiplication",
            "Domain": "Computation Optimization",
            "Time Complexity": "O(n^3)",
            "Space Complexity": "O(n^2)",
            "Optimized Space": "O(n^2)",
            "Subproblems Count": "O(n^2) table entries",
            "Dependency Pattern": "cost[i][k] + cost[k+1][j] for k in i..j-1"
        }
    ]
    df = pd.DataFrame(data)
    csv_path = os.path.join(DATA_CSV_DIR, "complexity_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")
    return df


def render_all_sample_heatmaps():
    print("-> Generating DP Heatmaps for Lab Questions 2, 3, 4...")
    # 1. LCS sample from Question 2:
    x_sample, y_sample = "ABCBDAB", "BDCABA"
    lcs_len, lcs_subseq, lcs_dp = lcs(x_sample, y_sample)
    lcs_path = lcs_traceback_path(x_sample, y_sample)
    x_ticks = ["Ø"] + list(y_sample)
    y_ticks = ["Ø"] + list(x_sample)
    plot_dp_table(
        lcs_dp,
        title=f"LCS DP Table: '{x_sample}' vs '{y_sample}' (Length={lcs_len})",
        xlabel=f"String Y ({y_sample})",
        ylabel=f"String X ({x_sample})",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        highlight_cells=lcs_path,
        cmap="Blues",
        save_filename="lcs_dp_table.png"
    )

    # 2. Knapsack sample from Question 3:
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    max_val, chosen, kp_dp = knapsack_01(weights, values, capacity)
    kp_path = knapsack_traceback_path(weights, values, capacity)
    y_ticks_kp = ["0 items"] + [f"Item {i+1} (w={weights[i]},v={values[i]})" for i in range(len(weights))]
    x_ticks_kp = [f"w={w}" for w in range(capacity + 1)]
    plot_dp_table(
        kp_dp,
        title=f"0-1 Knapsack DP Table (Capacity={capacity}, Max Value={max_val})",
        xlabel="Knapsack Capacity (w)",
        ylabel="Items Considered",
        x_ticks=x_ticks_kp,
        y_ticks=y_ticks_kp,
        highlight_cells=kp_path,
        cmap="Greens",
        save_filename="knapsack_dp_table.png"
    )

    # 3. Matrix Chain Multiplication sample from Question 4:
    dims = [10, 30, 5, 60]  # A1(10x30), A2(30x5), A3(5x60)
    min_cost, split_tab, cost_tab = matrix_chain_order(dims)
    n = len(dims) - 1
    # Clean up matrix for 1-indexed to 1..n
    clean_cost = [[cost_tab[i][j] if j >= i else 0 for j in range(1, n + 1)] for i in range(1, n + 1)]
    ticks_mcm = [f"A{i}" for i in range(1, n + 1)]
    plot_dp_table(
        clean_cost,
        title=f"Matrix Chain Cost Table (Min Multiplications={min_cost})",
        xlabel="Right Matrix Index (j)",
        ylabel="Left Matrix Index (i)",
        x_ticks=ticks_mcm,
        y_ticks=ticks_mcm,
        cmap="Purples",
        save_filename="mcm_cost_table.png"
    )
    print("   All heatmaps generated successfully in graphes/!")


def run_full_analysis():
    random.seed(42)
    np.random.seed(42)
    print("=" * 60)
    print("RUNNING LAB 4 DYNAMIC PROGRAMMING BENCHMARKS AND ANALYSIS")
    print("=" * 60)
    generate_complexity_comparison_table()
    render_all_sample_heatmaps()
    benchmark_lcs()
    benchmark_knapsack()
    benchmark_matrix_chain()
    benchmark_recursion_vs_dp()
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
