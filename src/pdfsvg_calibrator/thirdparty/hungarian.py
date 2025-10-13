from __future__ import annotations

from typing import List, Sequence, Tuple


def solve(cost_matrix: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Solve the rectangular assignment problem using the Hungarian algorithm."""
    if not cost_matrix:
        return []
    num_rows = len(cost_matrix)
    num_cols = len(cost_matrix[0]) if cost_matrix[0] else 0
    if num_cols == 0:
        return []
    # Ensure all rows have the same number of columns
    matrix = [list(row) + [0.0] * (num_cols - len(row)) for row in cost_matrix]
    n = len(matrix)
    m = len(matrix[0])
    if n > m:
        raise ValueError("Hungarian algorithm requires columns >= rows")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment: List[Tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment.append((p[j] - 1, j - 1))
    return assignment
