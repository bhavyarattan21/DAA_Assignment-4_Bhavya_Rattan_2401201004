"""
Resource Allocation Module: 0/1 Knapsack Problem
Provides 2D DP tabulation, 1D space-optimized DP, item traceback reconstruction,
and naive recursive implementations for complexity and performance analysis.
"""

from typing import Tuple, List


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int], List[List[int]]]:
    """
    Solves 0/1 Knapsack using a standard 2D DP table.
    Returns (max_value, selected_item_indices, dp_table).
    
    Time Complexity: O(n * W)
    Space Complexity: O(n * W)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    # Traceback to find which items were selected
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.reverse()

    return dp[n][capacity], selected, dp


def knapsack_traceback_path(weights: List[int], values: List[int], capacity: int) -> List[Tuple[int, int]]:
    """
    Returns the coordinates (i, w) visited during decision traceback.
    """
    n = len(weights)
    _, _, dp = knapsack_01(weights, values, capacity)
    
    path = []
    w = capacity
    for i in range(n, 0, -1):
        path.append((i, w))
        if dp[i][w] != dp[i - 1][w]:
            w -= weights[i - 1]
    path.append((0, w))
    return path


def knapsack_space_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Solves 0/1 Knapsack with a 1-D rolling array (backward iteration to avoid overwriting).
    
    Time Complexity: O(n * W)
    Space Complexity: O(W)
    """
    dp = [0] * (capacity + 1)
    for wt, val in zip(weights, values):
        for w in range(capacity, wt - 1, -1):
            dp[w] = max(dp[w], dp[w - wt] + val)
    return dp[capacity]


def knapsack_recursive(weights: List[int], values: List[int], capacity: int, n: int = None) -> int:
    """
    Naive recursive calculation of 0/1 Knapsack without memoization.
    Time Complexity: O(2^n)
    Space Complexity: O(n) call stack depth.
    Used for comparative empirical analysis against the DP tabulation approach.
    """
    if n is None:
        n = len(weights)

    # Base Case
    if n == 0 or capacity == 0:
        return 0

    # If weight of the nth item is more than Knapsack capacity, it cannot be included
    if weights[n - 1] > capacity:
        return knapsack_recursive(weights, values, capacity, n - 1)

    # Return the maximum of two cases:
    # (1) nth item included
    # (2) not included
    return max(
        values[n - 1] + knapsack_recursive(weights, values, capacity - weights[n - 1], n - 1),
        knapsack_recursive(weights, values, capacity, n - 1)
    )


if __name__ == "__main__":
    weights_test = [2, 3, 4, 5]
    values_test = [3, 4, 5, 6]
    capacity_test = 5
    max_val, chosen, dp_tab = knapsack_01(weights_test, values_test, capacity_test)
    print(f"Maximum achievable profit: {max_val}")
    print(f"Selected item indices: {chosen}")
    print(f"Space-optimized value: {knapsack_space_optimized(weights_test, values_test, capacity_test)}")
