'''

# Problem 2: Transpose and Reshape

**Difficulty:** Easy | **Topic:** Linear Algebra / NumPy basics

## Task
Implement two functions using **plain Python only**:
1. `transpose(A)` returns the transpose of a matrix.
2. `reshape(A, new_shape)` reshapes a matrix into `new_shape` in **row-major order**. If the total number of elements doesn't match, return `[]`.

## Function signatures
```python
def transpose(A: list[list[float]]) -> list[list[float]]:
    ...

def reshape(A: list[list[float]], new_shape: tuple[int, int]) -> list[list[float]]:
    ...
```

## Input
- `A`: an `m x n` matrix (list of lists)
- `new_shape`: a tuple `(r, c)`

## Output
- `transpose`: an `n x m` matrix
- `reshape`: an `r x c` matrix filled in row-major order, or `[]` if `m*n != r*c`

## Examples

**Example 1 (transpose)**
```
A = [[1, 2, 3],
     [4, 5, 6]]

Output: [[1, 4],
         [2, 5],
         [3, 6]]
```

**Example 2 (reshape)**
```
A = [[1, 2, 3],
     [4, 5, 6]]
new_shape = (3, 2)

Output: [[1, 2],
        [3, 4],
        [5, 6]]
```

**Example 3 (reshape, impossible)**
```
A = [[1, 2, 3],
     [4, 5, 6]]
new_shape = (4, 2)

Output: []
```

## Constraints
- `1 <= m, n, r, c <= 100`
- Inputs are non-empty and rectangular

## Rules
- Plain Python loops or comprehensions only. No NumPy and no `zip(*A)`.
- Don't modify the input.

## Bonus
1. What is the time complexity of each function?
2. In NumPy, what is the difference between `A.reshape(...)` and `A.T`, and do they copy data?

Send your code when ready.

'''

def transpose(A: list[list[float]]) -> list[list[float]]:

    # Time Complexity -> O(m * n)
    # Space Complexity -> O(m * n)

    m = len(A)
    n = len(A[0])

    rows = []

    for i in range(n):
        cols = []
        for j in range(m):
            cols.append(A[j][i])
        rows.append(cols)

    return rows

def reshape(A: list[list[float]], new_shape: tuple[int, int]) -> list[list[float]]:

    # Time Complexity -> O(m * n)
    # Space Complexity -> O(m * n)

    m = len(A)
    n = len(A[0])

    r, c = new_shape

    if m * n != r * c:
        return []

    # Approach 1 ->

    # ele = []

    # for i in range(m):
    #     for j in range(n):
    #         ele.append(A[i][j])

    # rows = []

    # for i in range(r):
    #     cols = []
    #     for j in range(c):
    #         cols.append(ele[i * c + j])
    #     rows.append(cols)

    # return rows

    # Approach 2 ->

    return [[A[(i * c + j) // n][(i * c + j) % n] for j in range(c)] for i in range(r)]

A = [[1, 2, 3],
     [4, 5, 6]]

new_shape1 = (3, 2)

new_shape2 = (4, 2)

res1 = transpose(A)

res2 = reshape(A, new_shape1)

res3 = reshape(A, new_shape2)

print(res1)

print(res2)

print(res3)

# In numpy A.reshape reshapes the matrix while A.T transposes the matrix.
# A.T never copies. It returns a view with the strides swapped.
# A.reshape(...) returns a view when the memory layout allows it, and silently makes a copy when it doesn't (for example, reshaping a transposed array). np.shares_memory(A, B) tells you which happened.
# This matters because modifying a view changes the original array.