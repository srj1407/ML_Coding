'''

# Problem 6: Row/Column Means, Normalization and Standardization

**Difficulty:** Easy | **Topic:** NumPy basics / Preprocessing

## Task
Implement four functions on a matrix in **plain Python**.

## Function signatures
```python
def column_means(A: list[list[float]]) -> list[float]:
    ...

def row_means(A: list[list[float]]) -> list[float]:
    ...

def normalize_rows(A: list[list[float]]) -> list[list[float]]:
    ...

def standardize_columns(A: list[list[float]]) -> list[list[float]]:
    ...
```

## Definitions
- `column_means`: the mean of each column. `row_means`: the mean of each row.
- `normalize_rows`: divide each row by its L2 norm. A row with norm `0` stays unchanged.
- `standardize_columns`: for each column, `(x - mean) / std`, using the **population** standard deviation (divide by `m`, not `m - 1`). If a column's `std` is `0`, set every value in that column to `0.0`.

## Input
- `A`: an `m x n` matrix (list of lists)

## Output
- `column_means`: a list of length `n`. `row_means`: a list of length `m`.
- `normalize_rows` and `standardize_columns`: an `m x n` matrix.
- Round every output value to 4 decimals.

## Examples

**Example 1 (means)**
```
A = [[1, 2],
     [3, 4],
     [5, 6]]

column_means(A) -> [3.0, 4.0]
row_means(A)    -> [1.5, 3.5, 5.5]
```

**Example 2 (normalize_rows)**
```
A = [[3, 4],
     [0, 0],
     [1, 0]]

Output: [[0.6, 0.8],
         [0, 0],
         [1.0, 0.0]]
```

**Example 3 (standardize_columns)**
```
A = [[1, 2],
     [3, 4],
     [5, 6]]

Output: [[-1.2247, -1.2247],
         [0.0, 0.0],
         [1.2247, 1.2247]]
```

**Example 4 (constant column)**
```
A = [[1, 5],
     [3, 5]]

Output: [[-1.0, 0.0],
         [1.0, 0.0]]
```

## Constraints
- `1 <= m, n <= 100`
- Inputs are non-empty and rectangular

## Rules
- Plain Python only. No NumPy.
- Reuse `lp_norm` from Problem 4 in `normalize_rows`.
- Don't modify the input.

## Bonus
1. In NumPy, what do `A.mean(axis=0)` and `A.mean(axis=1)` compute? Which one matches `column_means`?
2. Why is standardizing columns usually done before k-NN or gradient descent? Give one concrete reason for each.

Send your code when ready.

'''
from vector_norms import lp_norm

def mean(A: list[float]) ->  float:
    return sum(A) / len(A)

def std(A: list[float]) ->  float:
    mean_val = mean(A)
    return sum([(x - mean_val) ** 2 for x in A]) ** 1/2

def column_means(A: list[list[float]]) -> list[float]:
    n = len(A[0])
    return [mean(A[:][i]) for i in range(n)]

def row_means(A: list[list[float]]) -> list[float]:
    m = len(A)
    return [mean(A[i][:]) for i in range(m)]

def normalize_rows(A: list[list[float]]) -> list[list[float]]:
    m = len(A)
    n = len(A[0])
    return [[A[i][j] / lp_norm(A[i][:], 2) for j in range(n)] for i in range(m)]

def standardize_columns(A: list[list[float]]) -> list[list[float]]: