'''

# Problem 3: Dot Product and Matrix-Vector Multiplication

**Difficulty:** Easy | **Topic:** Linear Algebra

## Task
Implement two functions in **plain Python**:
1. `dot(a, b)` returns the dot product of two vectors.
2. `matvec(A, v)` returns the product of a matrix and a vector.

Return `-1` if the dimensions are incompatible.

## Function signatures
```python
def dot(a: list[float], b: list[float]):
    ...

def matvec(A: list[list[float]], v: list[float]):
    ...
```

## Input
- `a`, `b`: vectors of length `n` and `k`
- `A`: an `m x n` matrix, `v`: a vector of length `p`

## Output
- `dot`: a single number, or `-1` if `len(a) != len(b)`
- `matvec`: a list of length `m`, or `-1` if `n != p`

## Examples

**Example 1 (dot)**
```
a = [1, 2, 3]
b = [4, 5, 6]

Output: 32
```
Explanation: `1*4 + 2*5 + 3*6 = 32`

**Example 2 (dot, mismatch)**
```
a = [1, 2, 3]
b = [4, 5]

Output: -1
```

**Example 3 (matvec)**
```
A = [[1, 2, 3],
     [4, 5, 6]]
v = [1, 0, -1]

Output: [-2, -2]
```

**Example 4 (matvec, mismatch)**
```
A = [[1, 2],
     [3, 4]]
v = [1, 2, 3]

Output: -1
```

## Constraints
- `1 <= m, n <= 1000`
- Values are integers or floats

## Rules
- Plain Python only. No NumPy.
- **Reuse your `dot` function inside `matvec`.** This is the point of pairing them.

## Bonus
1. If `A` is `m x n`, what is the time complexity of `matvec`?
2. A matrix-vector product can be seen two ways: as `m` dot products (one per row), or as a linear combination of the columns of `A`. Which one does your code implement, and which is more cache-friendly in row-major storage?

Send your code when ready.

'''

def dot(a: list[float], b: list[float]):
    n = len(a)
    k = len(b)

    if n != k:
        return -1

    res = 0

    for i in range(n):
        res += a[i] * b[i]

    return res

def matvec(A: list[list[float]], v: list[float]):

    # Time Complexity -> O(m * n)

    m = len(A)
    n = len(A[0])
    p = len(v)

    if n != p:
        return -1

    res = []

    # Approach 1 ->

    # for i in range(m):
    #     temp = 0
    #     for j in range(n):
    #         temp += A[i][j] * v[j]
    #     res.append(temp)

    # Approach 2 ->

    for i in range(m):
        res.append(dot(A[i], v))

    return res

# a = [1, 2, 3]
# b = [4, 5, 6]

# a = [1, 2, 3]
# b = [4, 5]

# res = dot(a, b)

# A = [[1, 2, 3],
#      [4, 5, 6]]
# v = [1, 0, -1]

A = [[1, 2],
     [3, 4]]
v = [1, 2, 3]

res = matvec(A, v)

print(res)

'''

Bonus 2 explained

There are two ways to read A @ v. Take A = [[1,2,3],[4,5,6]] and v = [1,0,-1].

View 1: rows (dot products). Each output entry is one row of A dotted with v:

result[0] = dot([1,2,3], v) = -2
result[1] = dot([4,5,6], v) = -2

This is what your code does.

View 2: columns (linear combination). The result is the columns of A, each scaled by the matching entry of v, then added:

1 * [1,4]  +  0 * [2,5]  +  (-1) * [3,6]  =  [-2, -2]

Same answer, different way of seeing it. This view matters later: it explains why A @ v always lands in the column space of A, which is the key idea behind rank, PCA, and linear regression.

Which is more cache-friendly? In row-major storage (NumPy's default, and C), the elements of one row sit next to each other in memory. Your row-wise approach reads A in order, so the CPU can prefetch it efficiently. A naive column-wise loop jumps by a whole row length at each step, so it's slower on large arrays. The answer is the row view. The effect is muted for Python lists of lists, but it matters in NumPy and C, and interviewers like this question.

'''