"""
Sequence Analysis Module: Longest Common Subsequence (LCS)
Provides bottom-up dynamic programming tabulation, optimal traceback path reconstruction,
and naive recursive implementations for complexity and performance analysis.
"""

from typing import Tuple, List, Set


def lcs(x: str, y: str) -> Tuple[int, str, List[List[int]]]:
    """
    Returns (length, subsequence_string, dp_table) for the LCS of x and y.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Traceback to reconstruct the actual subsequence
    i, j = m, n
    seq = []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            seq.append(x[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    seq.reverse()

    return dp[m][n], "".join(seq), dp


def lcs_traceback_path(x: str, y: str) -> List[Tuple[int, int]]:
    """
    Computes the cells (i, j) traversed during the traceback phase
    to highlight the optimal path on a visual heatmap.
    """
    m, n = len(x), len(y)
    _, _, dp = lcs(x, y)
    
    i, j = m, n
    path = [(i, j)]
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    return path


def lcs_recursive(x: str, y: str, i: int = None, j: int = None) -> int:
    """
    Naive recursive calculation of LCS length without memoization.
    Time Complexity: O(2^(m+n)) in the worst case (branching factor 2).
    Space Complexity: O(m + n) call stack depth.
    Used for comparative empirical analysis against the DP tabulation approach.
    """
    if i is None:
        i = len(x)
    if j is None:
        j = len(y)

    if i == 0 or j == 0:
        return 0

    if x[i - 1] == y[j - 1]:
        return 1 + lcs_recursive(x, y, i - 1, j - 1)
    else:
        return max(lcs_recursive(x, y, i - 1, j), lcs_recursive(x, y, i, j - 1))


if __name__ == "__main__":
    x_test, y_test = "ABCBDAB", "BDCABA"
    length, subsequence, dp_table = lcs(x_test, y_test)
    print(f"LCS length: {length}")
    print(f"LCS string: {subsequence}")
    print("DP Table:")
    for row in dp_table:
        print(row)
