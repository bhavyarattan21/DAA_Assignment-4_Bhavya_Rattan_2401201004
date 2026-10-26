"""
Smart Optimization Assistant using Dynamic Programming
Lab 4 - Design and Analysis of Algorithms (DAA)

Interactive Web Application built with Streamlit.
Implements:
1. Longest Common Subsequence (Sequence Analysis)
2. 0/1 Knapsack (Resource Allocation)
3. Matrix Chain Multiplication (Computation Optimization)
4. Empirical Benchmarks, DP Heatmaps, and Complexity Analysis
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Smart Optimization Assistant | DP Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set path to import modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from algorithms.lcs import lcs, lcs_traceback_path, lcs_recursive
from algorithms.knapsack import knapsack_01, knapsack_space_optimized, knapsack_traceback_path, knapsack_recursive
from algorithms.matrix_chain import matrix_chain_order, print_optimal_parens, compute_standard_left_to_right_cost
from analysis.visualizer import plot_dp_table

DATA_CSV_DIR = os.path.join(BASE_DIR, "data_csv")
GRAPHES_DIR = os.path.join(BASE_DIR, "graphes")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)


def render_heatmap_ui(dp_arr, title, x_labels, y_labels, highlight_cells=None, cmap="Blues"):
    """Renders a matplotlib heatmap directly into streamlit."""
    arr = np.array(dp_arr, dtype=float)
    masked_arr = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(max(5.5, arr.shape[1] * 0.65), max(4.0, arr.shape[0] * 0.55)))
    im = ax.imshow(masked_arr, cmap=cmap, aspect="auto")

    # Annotate cell numbers
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val) or np.isinf(val):
                text_val = "—"
            else:
                text_val = str(int(val)) if val == int(val) else f"{val:.1f}"

            is_hl = highlight_cells and (i, j) in highlight_cells
            txt_color = "red" if is_hl else ("white" if masked_arr[i, j] > (masked_arr.max() * 0.65) else "black")
            txt_weight = "bold" if is_hl else "normal"
            ax.text(j, i, text_val, ha="center", va="center", color=txt_color, fontweight=txt_weight, fontsize=8)

            if is_hl:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="crimson", linewidth=2.5)
                ax.add_patch(rect)

    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=0)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚡ DP Assistant")
st.sidebar.caption("DAA Lab 4 | Dynamic Programming")

module = st.sidebar.radio(
    "Select Optimization Module:",
    [
        "🧬 Sequence Analysis (LCS)",
        "🎒 Resource Allocation (0/1 Knapsack)",
        "📐 Computation Optimization (MCM)",
        "📊 Performance & Empirical Benchmarks",
        "📑 Complexity & Theoretical Analysis",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Smart Optimization Assistant**\n\n"
    "• Optimal Substructure Verification\n"
    "• Overlapping Subproblem Tabulation\n"
    "• Traceback Path Reconstruction\n"
    "• Empirical Big-O Analysis"
)


# ==========================================
# MODULE 1: SEQUENCE ANALYSIS (LCS)
# ==========================================
if module == "🧬 Sequence Analysis (LCS)":
    st.markdown('<div class="main-header">🧬 Sequence Analysis — Longest Common Subsequence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discovers the longest sequence of characters appearing in both strings in relative order (Bioinformatics & Diff tools).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Configuration")
        preset = st.selectbox(
            "Choose a Preset or Custom:",
            [
                "Lab 4 Question 2 Sample ('ABCBDAB' vs 'BDCABA')",
                "DNA Sequence Alignment ('AGCCCTA' vs 'AATGCC')",
                "Git Diff Token Comparison ('INT_MAX_VAL' vs 'INT_MIN_VAL')",
                "Custom Input"
            ]
        )

        if preset == "Lab 4 Question 2 Sample ('ABCBDAB' vs 'BDCABA')":
            default_x, default_y = "ABCBDAB", "BDCABA"
        elif preset == "DNA Sequence Alignment ('AGCCCTA' vs 'AATGCC')":
            default_x, default_y = "AGCCCTA", "AATGCC"
        elif preset == "Git Diff Token Comparison ('INT_MAX_VAL' vs 'INT_MIN_VAL')":
            default_x, default_y = "INT_MAX_VAL", "INT_MIN_VAL"
        else:
            default_x, default_y = "BANANA", "ATANA"

        seq_x = st.text_input("First Sequence (X):", value=default_x).strip()
        seq_y = st.text_input("Second Sequence (Y):", value=default_y).strip()

        run_btn = st.button("🚀 Analyze Sequences", type="primary", use_container_width=True)

    with col2:
        if seq_x and seq_y:
            t0 = time.perf_counter()
            length, subsequence, dp_table = lcs(seq_x, seq_y)
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            path = lcs_traceback_path(seq_x, seq_y)

            # Display metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("LCS Length", length)
            with m2:
                st.metric("Subsequence", subsequence if subsequence else "Ø")
            with m3:
                st.metric("Table Entries", f"{len(seq_x)+1} x {len(seq_y)+1}")
            with m4:
                st.metric("Execution Time", f"{runtime_ms:.3f} ms")

            st.markdown("---")
            st.subheader("Dynamic Programming Table & Traceback Path")
            st.caption("Red outlined cells indicate the optimal decision path traversed during traceback.")

            x_ticks = ["Ø"] + list(seq_y)
            y_ticks = ["Ø"] + list(seq_x)
            render_heatmap_ui(
                dp_table,
                f"LCS DP Table: '{seq_x}' vs '{seq_y}' (LCS = '{subsequence}')",
                x_ticks,
                y_ticks,
                highlight_cells=path,
                cmap="Blues"
            )

            with st.expander("📝 View Recurrence Relation & Traceback Explanation"):
                st.markdown(r"""
                **Recurrence Relation:**
                $$dp[i][j] = \begin{cases} 
                0 & \text{if } i=0 \text{ or } j=0 \\
                dp[i-1][j-1] + 1 & \text{if } X[i-1] == Y[j-1] \\
                \max(dp[i-1][j], dp[i][j-1]) & \text{if } X[i-1] \neq Y[j-1]
                \end{cases}$$

                **Traceback Logic:**
                Starting at cell `(m, n)`, if `X[i-1] == Y[j-1]`, character is included in LCS, and we move diagonally to `(i-1, j-1)`. Otherwise, step towards the adjacent predecessor with the maximum value (`dp[i-1][j]` or `dp[i][j-1]`).
                """)


# ==========================================
# MODULE 2: RESOURCE ALLOCATION (KNAPSACK)
# ==========================================
elif module == "🎒 Resource Allocation (0/1 Knapsack)":
    st.markdown('<div class="main-header">🎒 Resource Allocation — 0/1 Knapsack Problem</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Maximizes total utility under a rigid capacity budget constraint without item fractionalization.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        preset_kp = st.selectbox(
            "Select Scenario Preset:",
            [
                "Lab 4 Question 3 Sample (W=5, 4 Items)",
                "Cloud Compute Resource Allocation (RAM vs Profit)",
                "Emergency Aid Cargo Flight (Payload vs Lives Saved)",
                "Custom Allocation"
            ]
        )

        if preset_kp == "Lab 4 Question 3 Sample (W=5, 4 Items)":
            def_wts, def_vals, def_cap = "2, 3, 4, 5", "3, 4, 5, 6", 5
        elif preset_kp == "Cloud Compute Resource Allocation (RAM vs Profit)":
            def_wts, def_vals, def_cap = "4, 8, 12, 16, 24", "100, 240, 380, 520, 800", 32
        elif preset_kp == "Emergency Aid Cargo Flight (Payload vs Lives Saved)":
            def_wts, def_vals, def_cap = "5, 10, 15, 22, 25", "40, 75, 110, 160, 200", 40
        else:
            def_wts, def_vals, def_cap = "2, 3, 4, 5", "3, 4, 5, 6", 5

        wts_input = st.text_input("Item Weights (comma-separated):", value=def_wts)
        vals_input = st.text_input("Item Values (comma-separated):", value=def_vals)
        cap_input = st.number_input("Knapsack Capacity (W):", min_value=1, max_value=200, value=def_cap, step=1)

        run_kp_btn = st.button("🚀 Allocate Resources", type="primary", use_container_width=True)

    with col2:
        try:
            weights = [int(x.strip()) for x in wts_input.split(",") if x.strip()]
            values = [int(x.strip()) for x in vals_input.split(",") if x.strip()]

            if len(weights) != len(values):
                st.error("Error: Number of weights must match number of values.")
            else:
                t0 = time.perf_counter()
                max_profit, selected, dp_table = knapsack_01(weights, values, cap_input)
                time_2d = (time.perf_counter() - t0) * 1000.0

                # 1D space optimized calculation
                opt_val = knapsack_space_optimized(weights, values, cap_input)
                path = knapsack_traceback_path(weights, values, cap_input)

                tot_weight_used = sum(weights[i] for i in selected)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Max Profit ($/Utility)", max_profit)
                with m2:
                    st.metric("Selected Items", f"{len(selected)} / {len(weights)}")
                with m3:
                    st.metric("Capacity Used", f"{tot_weight_used} / {cap_input}")
                with m4:
                    st.metric("2D DP Runtime", f"{time_2d:.3f} ms")

                st.markdown("#### Selected Items Breakdown:")
                if selected:
                    item_data = [{
                        "Item Index": f"Item {idx + 1}",
                        "Weight (kg/units)": weights[idx],
                        "Value ($/utility)": values[idx],
                        "Ratio (Val/Wt)": f"{values[idx]/weights[idx]:.2f}"
                    } for idx in selected]
                    st.dataframe(pd.DataFrame(item_data), use_container_width=True)
                else:
                    st.info("No items could be selected within the given capacity.")

                st.markdown("---")
                st.subheader("2D Dynamic Programming Tabulation Table")
                st.caption("Red outlined cells indicate the dynamic decisions made during backward traceback.")

                y_ticks = ["0 items"] + [f"Item {i+1} (w={weights[i]},v={values[i]})" for i in range(len(weights))]
                x_ticks = [f"{w}" for w in range(cap_input + 1)]
                render_heatmap_ui(
                    dp_table,
                    f"Knapsack DP: Capacity={cap_input}, Max Profit={max_profit}",
                    x_ticks,
                    y_ticks,
                    highlight_cells=path,
                    cmap="Greens"
                )

                st.success(f"💡 **1-D Space Optimization Verification**: 1-D Rolling array computed max value = **{opt_val}**, using only **O(W)** memory instead of O(n*W)!")

        except ValueError as e:
            st.error(f"Input Parsing Error: Please ensure numbers are integers separated by commas. ({e})")


# ==========================================
# MODULE 3: MATRIX CHAIN MULTIPLICATION
# ==========================================
elif module == "📐 Computation Optimization (MCM)":
    st.markdown('<div class="main-header">📐 Computation Optimization — Matrix Chain Multiplication</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Determines the optimal parenthesization order that minimizes total scalar multiplications in linear algebra pipelines.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Matrix Dimension Configuration")
        preset_mcm = st.selectbox(
            "Select Dimension Preset:",
            [
                "Lab 4 Question 4 Sample [10, 30, 5, 60] (3 Matrices)",
                "Computer Graphics 3D Pipeline [40, 20, 30, 10, 30] (4 Matrices)",
                "Neural Network Dense Projections [30, 35, 15, 5, 10, 20, 25] (6 Matrices)",
                "Custom Matrix Dimensions"
            ]
        )

        if preset_mcm == "Lab 4 Question 4 Sample [10, 30, 5, 60] (3 Matrices)":
            def_dims = "10, 30, 5, 60"
        elif preset_mcm == "Computer Graphics 3D Pipeline [40, 20, 30, 10, 30] (4 Matrices)":
            def_dims = "40, 20, 30, 10, 30"
        elif preset_mcm == "Neural Network Dense Projections [30, 35, 15, 5, 10, 20, 25] (6 Matrices)":
            def_dims = "30, 35, 15, 5, 10, 20, 25"
        else:
            def_dims = "10, 20, 30, 40, 30"

        dims_input = st.text_input("Matrix Dimensions p0, p1, ..., pn:", value=def_dims)
        run_mcm_btn = st.button("🚀 Optimize Multiplication Chain", type="primary", use_container_width=True)

    with col2:
        try:
            dims = [int(x.strip()) for x in dims_input.split(",") if x.strip()]
            if len(dims) < 3:
                st.error("Error: At least 3 dimension values are required to form 2 or more matrices.")
            else:
                n = len(dims) - 1
                t0 = time.perf_counter()
                min_cost, split_table, cost_table = matrix_chain_order(dims)
                mcm_time = (time.perf_counter() - t0) * 1000.0

                optimal_parens = print_optimal_parens(split_table, 1, n)
                naive_cost = compute_standard_left_to_right_cost(dims)
                saved_ops = naive_cost - min_cost
                reduction_pct = (saved_ops / max(naive_cost, 1)) * 100.0

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Optimal Multiplications", f"{min_cost:,}")
                with m2:
                    st.metric("Naive Mult. Cost", f"{naive_cost:,}")
                with m3:
                    st.metric("Operations Saved", f"{saved_ops:,} ({reduction_pct:.1f}%)")
                with m4:
                    st.metric("DP Runtime", f"{mcm_time:.3f} ms")

                st.markdown(f"### Optimal Parenthesization Formula:")
                st.success(f"### `{optimal_parens}`")

                st.markdown("---")
                st.subheader("Matrix Chain Cost Table (Upper Triangular)")

                clean_cost = [[cost_table[i][j] if j >= i else 0 for j in range(1, n + 1)] for i in range(1, n + 1)]
                ticks = [f"A{i}" for i in range(1, n + 1)]
                render_heatmap_ui(
                    clean_cost,
                    f"MCM Cost Table (Min Cost = {min_cost:,})",
                    ticks,
                    ticks,
                    cmap="Purples"
                )

                st.markdown("#### Split Positions (k):")
                split_display = [[split_table[i][j] if j > i else "-" for j in range(1, n + 1)] for i in range(1, n + 1)]
                st.dataframe(pd.DataFrame(split_display, index=ticks, columns=ticks), use_container_width=True)

        except ValueError as e:
            st.error(f"Dimension Parsing Error: Please enter positive integers separated by commas. ({e})")


# ==========================================
# MODULE 4: PERFORMANCE & BENCHMARKS
# ==========================================
elif module == "📊 Performance & Empirical Benchmarks":
    st.markdown('<div class="main-header">📊 Performance Analysis & Empirical Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical runtime verification against theoretical Big-O complexity bounds and comparison with naive recursion.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Runtime Growth Curves", "⚡ Naive Recursion vs DP Study", "📁 Raw Benchmark Datasets"])

    with tab1:
        st.subheader("Empirical Growth Curves vs Theoretical Bounds")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Longest Common Subsequence")
            p = os.path.join(GRAPHES_DIR, "lcs_runtime.png")
            if os.path.exists(p):
                st.image(p, caption="LCS Runtime Growth: O(m * n)")
            else:
                st.warning("Graph not yet generated. Run benchmark.py.")

        with c2:
            st.markdown("#### 0/1 Knapsack Problem")
            p = os.path.join(GRAPHES_DIR, "knapsack_runtime.png")
            if os.path.exists(p):
                st.image(p, caption="0/1 Knapsack Runtime Growth: O(n * W)")
            else:
                st.warning("Graph not yet generated. Run benchmark.py.")

        with c3:
            st.markdown("#### Matrix Chain Multiplication")
            p = os.path.join(GRAPHES_DIR, "matrix_chain_runtime.png")
            if os.path.exists(p):
                st.image(p, caption="MCM Runtime Growth: O(n³)")
            else:
                st.warning("Graph not yet generated. Run benchmark.py.")

    with tab2:
        st.subheader("Exponential Explosion (Recursion) vs Polynomial Efficiency (DP)")
        st.markdown(
            "Naive recursion recomputes overlapping subproblems exponentially many times ($O(2^n)$). "
            "Tabulation caches intermediate answers, bringing the runtime down to polynomial order ($O(mn)$)."
        )
        rec_graph = os.path.join(GRAPHES_DIR, "recursion_vs_dp_comparison.png")
        if os.path.exists(rec_graph):
            st.image(rec_graph, caption="Log-Scale Runtime Comparison: Exponential Recursive vs DP Tabulation")
        
        rec_csv = os.path.join(DATA_CSV_DIR, "recursion_vs_dp.csv")
        if os.path.exists(rec_csv):
            st.markdown("##### Empirical Comparison Data:")
            st.dataframe(pd.read_csv(rec_csv), use_container_width=True)

    with tab3:
        st.subheader("Exported Benchmark Datasets (CSV)")
        csv_files = {
            "LCS Benchmark": "lcs_benchmark.csv",
            "0/1 Knapsack Benchmark": "knapsack_benchmark.csv",
            "Matrix Chain Benchmark": "matrix_chain_benchmark.csv",
            "Recursion vs DP Study": "recursion_vs_dp.csv",
            "Complexity Matrix": "complexity_comparison.csv",
        }
        selected_csv = st.selectbox("Select CSV Dataset:", list(csv_files.keys()))
        csv_file_path = os.path.join(DATA_CSV_DIR, csv_files[selected_csv])
        if os.path.exists(csv_file_path):
            df_view = pd.read_csv(csv_file_path)
            st.dataframe(df_view, use_container_width=True)
            st.download_button(
                f"📥 Download {csv_files[selected_csv]}",
                data=df_view.to_csv(index=False),
                file_name=csv_files[selected_csv],
                mime="text/csv"
            )


# ==========================================
# MODULE 5: COMPLEXITY & THEORY REFERENCE
# ==========================================
elif module == "📑 Complexity & Theoretical Analysis":
    st.markdown('<div class="main-header">📑 Complexity Comparison & Theoretical Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive theoretical reference addressing Lab 4 Questions 5 and 7.</div>', unsafe_allow_html=True)

    st.subheader("Question 5: Complexity Comparison Matrix")
    comp_csv = os.path.join(DATA_CSV_DIR, "complexity_comparison.csv")
    if os.path.exists(comp_csv):
        st.dataframe(pd.read_csv(comp_csv), use_container_width=True)

    st.markdown("---")
    st.subheader("Core Dynamic Programming Properties")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 1. Optimal Substructure
        An optimization problem exhibits **optimal substructure** if an optimal solution to the problem contains within it optimal solutions to subproblems:
        - **LCS**: An optimal subsequence of $X[1..m]$ and $Y[1..n]$ incorporates the optimal subsequence of their prefixes.
        - **0/1 Knapsack**: Optimal selection of $n$ items with capacity $W$ includes the optimal subset for $(n-1)$ items with capacity $(W - w_n)$.
        - **MCM**: Optimal parenthesization of $A_i \dots A_j$ split at $k$ consists of optimal parenthesizations of $A_i \dots A_k$ and $A_{k+1} \dots A_j$.
        """)

    with col2:
        st.markdown("""
        ### 2. Overlapping Subproblems
        A problem has **overlapping subproblems** if recursive algorithms revisit the same smaller subproblems repeatedly rather than generating new subproblems:
        - In naive recursion, the call tree spawns identical state calls with identical parameters thousands of times.
        - **Dynamic Programming** solves each unique subproblem exactly once and stores results in a table (tabulation) for $O(1)$ constant-time lookup.
        """)

    st.markdown("---")
    st.subheader("Real-World Industrial Applications")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.info("""
        **🧬 Sequence Analysis (LCS)**
        - **Diff Tools (`git diff`)**: Generates minimal line insertion/deletion diff patches between codebase commits.
        - **Bioinformatics**: Aligning DNA, RNA, and protein sequences to detect genomic mutations and evolutionary lineage.
        """)
    with a2:
        st.info("""
        **🎒 Resource Allocation (Knapsack)**
        - **Cloud Resource Scheduling**: Maximizing server revenue by packing containerized microservices under CPU/RAM caps.
        - **Financial Portfolio Management**: Selecting discrete investment projects under finite budget constraints.
        """)
    with a3:
        st.info("""
        **📐 Computation Optimization (MCM)**
        - **3D Graphics & Game Engines**: Optimizing matrix multiplication transforms across camera, perspective, and world coordinates.
        - **Machine Learning Tensor Compilers**: Optimizing associative matrix-matrix multiplication in deep neural network inference.
        """)
