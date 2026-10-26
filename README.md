# ⚡ Smart Optimization Assistant using Dynamic Programming
### Design and Analysis of Algorithms (DAA) — Lab 4

An interactive optimization assistant powered by **Dynamic Programming (DP)** that solves, tabulates, visualizes, and benchmarks three classic algorithmic problems:
1. **Sequence Analysis:** Longest Common Subsequence (LCS)
2. **Resource Allocation:** 0/1 Knapsack Problem
3. **Computation Optimization:** Matrix Chain Multiplication (MCM)

---

## 📁 Repository Structure
The workspace is organized into modular directories matching the laboratory architecture:

```
DAA Assignment 4/
│
├── algorithms/                 # Core Dynamic Programming implementations
│   ├── __init__.py
│   ├── lcs.py                  # Bottom-up DP, traceback, and naive recursion
│   ├── knapsack.py             # 2D DP, 1D space-optimized, and traceback
│   └── matrix_chain.py         # Cost & split tabulation, parenthesization string
│
├── analysis/                   # Benchmark and visualization engine
│   ├── benchmark.py            # Automated runtime and recursion comparative engine
│   └── visualizer.py           # Matplotlib annotated heatmap and plot renderers
│
├── data_csv/                   # Experimental benchmark datasets
│   ├── complexity_comparison.csv
│   ├── knapsack_benchmark.csv
│   ├── lcs_benchmark.csv
│   ├── matrix_chain_benchmark.csv
│   └── recursion_vs_dp.csv
│
├── graphes/                    # Exported heatmaps and runtime growth curves
│   ├── knapsack_dp_table.png
│   ├── knapsack_runtime.png
│   ├── lcs_dp_table.png
│   ├── lcs_runtime.png
│   ├── matrix_chain_runtime.png
│   ├── mcm_cost_table.png
│   └── recursion_vs_dp_comparison.png
│
├── notebooks/                  # Interactive Jupyter Notebook for Lab 4
│   └── lab4_smart_optimization_assistant.ipynb
│
├── reports/                    # Comprehensive academic & technical report
│   └── Lab4_Report.md
│
├── .gitignore                  # Standard Python / Jupyter git ignore file
├── README.md                   # Project overview and execution guide
├── app.py                      # Interactive Streamlit Web Application
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Clone the repository and install the dependencies:
```bash
# Optional: create virtual environment
python -m venv venv
venv\Scripts\activate       # On Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard (`app.py`)
Run the interactive Streamlit assistant:
```bash
streamlit run app.py
```
This opens the web browser interface where you can:
- Test custom sequences, items, and matrix chains or select industry presets.
- View real-time 2D annotated heatmaps with the optimal decision paths highlighted in red.
- Compare standard 2D DP against 1D space-optimized memory models.
- Inspect empirical runtime benchmarks and download raw CSV datasets.

### 3. Run Automated Benchmarks & Generate Graphs
To re-run all experimental runs, regenerate CSVs in `data_csv/`, and update plots in `graphes/`:
```bash
python analysis/benchmark.py
```

### 4. Open Jupyter Notebook
To interact with step-by-step code cells and visualizations:
```bash
jupyter notebook notebooks/lab4_smart_optimization_assistant.ipynb
```

---

## 📊 Summary of Implemented Algorithms

| Algorithm | Domain | Recurrence Relation | Time | Space | Practical Application |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LCS** | Sequence Analysis | $dp[i][j] = dp[i-1][j-1]+1$ if match else $\max(dp[i-1][j], dp[i][j-1])$ | $O(mn)$ | $O(mn)$ | `git diff`, DNA alignment |
| **0/1 Knapsack** | Resource Allocation | $dp[i][w] = \max(dp[i-1][w], v_i + dp[i-1][w-w_i])$ | $O(nW)$ | $O(nW) \to O(W)$ | Cloud VM scheduling, Cargo flight |
| **Matrix Chain** | Computation Opt. | $cost[i][j] = \min_k \{ cost[i][k] + cost[k+1][j] + p_{i-1}p_kp_j \}$ | $O(n^3)$ | $O(n^2)$ | 3D graphics pipeline, Tensor compilers |

---

## 📈 Key Visualizations
- **Heatmaps (`graphes/`):** Display DP tables with exact cell values and red-boxed optimal traceback paths.
- **Runtime Curves (`graphes/`):** Empirical measurements verifying $O(mn)$, $O(nW)$, and $O(n^3)$ bounds.
- **Recursion vs DP (`graphes/recursion_vs_dp_comparison.png`):** Demonstrates exponential blowup ($O(2^n)$) vs polynomial DP ($O(mn)$).

---

## 📜 Full Lab Report
A detailed academic report covering optimal substructure, overlapping subproblems, complexity analysis, and real-world case studies is available at [`reports/Lab4_Report.md`](reports/Lab4_Report.md).
