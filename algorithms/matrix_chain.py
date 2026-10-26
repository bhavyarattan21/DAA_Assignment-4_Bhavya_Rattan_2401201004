"""
Computation Optimization Module: Matrix Chain Multiplication (MCM)
Determines the optimal parenthesization order that minimizes the scalar multiplications
required to multiply a chain of matrices.
"""

from typing import Tuple, List


def matrix_chain_order(dims: List[int]) -> Tuple[int, List[List[int]], List[List[int]]]:
    """
    dims: list of matrix dimensions, e.g. [10, 30, 5, 60] means
          matrices A1(10x30), A2(30x5), A3(5x60).
    Returns (min_cost, split_table, cost_table) where split_table is used to
    reconstruct the optimal parenthesization.
    
    Time Complexity: O(n^3) where n = len(dims) - 1
    Space Complexity: O(n^2) for the cost and split tables
    """
    n = len(dims) - 1  # number of matrices
    cost = [[0] * (n + 1) for _ in range(n + 1)]
    split = [[0] * (n + 1) for _ in range(n + 1)]

    for chain_len in range(2, n + 1):
        for i in range(1, n - chain_len + 2):
            j = i + chain_len - 1
            cost[i][j] = float("inf")
            for k in range(i, j):
                q = cost[i][k] + cost[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if q < cost[i][j]:
                    cost[i][j] = q
                    split[i][j] = k

    return int(cost[1][n]), split, cost


def print_optimal_parens(split: List[List[int]], i: int, j: int) -> str:
    """
    Reconstructs the optimal parenthesization as a formatted mathematical string, e.g. '((A1 A2) A3)'.
    """
    if i == j:
        return f"A{i}"
    k = split[i][j]
    left = print_optimal_parens(split, i, k)
    right = print_optimal_parens(split, k + 1, j)
    return f"({left} x {right})"


def compute_standard_left_to_right_cost(dims: List[int]) -> int:
    """
    Computes scalar multiplications required if matrices are multiplied
    naively in fixed left-to-right order: (...((A1 x A2) x A3)... x An).
    Used to showcase the speedup achieved by DP optimization.
    """
    n = len(dims) - 1
    if n <= 1:
        return 0
    total_cost = 0
    # Initial result has dimension dims[0] x dims[1]
    curr_rows = dims[0]
    curr_cols = dims[1]
    for i in range(2, n + 1):
        next_cols = dims[i]
        total_cost += curr_rows * curr_cols * next_cols
        curr_cols = next_cols
    return total_cost


def matrix_chain_recursive(dims: List[int], i: int, j: int) -> int:
    """
    Naive recursive calculation of Matrix Chain Multiplication cost without memoization.
    Time Complexity: Exponential (Catalan number growth, ~ O(4^n / n^(3/2))).
    Used for empirical comparison showing why DP tabulation is necessary.
    """
    if i == j:
        return 0

    min_ops = float("inf")
    for k in range(i, j):
        count = (
            matrix_chain_recursive(dims, i, k)
            + matrix_chain_recursive(dims, k + 1, j)
            + dims[i - 1] * dims[k] * dims[j]
        )
        if count < min_ops:
            min_ops = count

    return min_ops


if __name__ == "__main__":
    dims_test = [10, 30, 5, 60]     # A1: 10x30, A2: 30x5, A3: 5x60
    min_c, split_tab, cost_tab = matrix_chain_order(dims_test)
    print(f"Minimum scalar multiplications: {min_c}")
    print("Optimal order:", print_optimal_parens(split_tab, 1, len(dims_test) - 1))
    naive_c = compute_standard_left_to_right_cost(dims_test)
    print(f"Naive Left-to-Right cost: {naive_c}")
    print(f"Saved multiplications: {naive_c - min_c} ({((naive_c - min_c)/naive_c)*100:.1f}% reduction)")
