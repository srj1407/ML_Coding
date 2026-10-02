'''

# Problem 1: Matrix Multiplication

**Difficulty:** Easy | **Topic:** Linear Algebra

## Task
Implement a function that multiplies two matrices. If the matrices can't be multiplied, return `-1`.

## Function signature
```python
def matrix_multiply(A: list[list[float]], B: list[list[float]]):
    ...
```

## Input
- `A`: an `m x n` matrix as a list of lists
- `B`: a `p x q` matrix as a list of lists

## Output
- The `m x q` product as a list of lists, if `n == p`
- `-1` if `n != p`

## Examples

**Example 1**
```
A = [[1, 2],
     [3, 4]]
B = [[5, 6],
     [7, 8]]

Output: [[19, 22],
         [43, 50]]
```
Explanation: `19 = 1*5 + 2*7`, `22 = 1*6 + 2*8`, and so on.

**Example 2**
```
A = [[1, 2, 3]]          # 1x3
B = [[4], [5], [6]]      # 3x1

Output: [[32]]
```

**Example 3**
```
A = [[1, 2],
     [3, 4]]            # 2x2
B = [[1, 2, 3]]          # 1x3

Output: -1
```

## Constraints
- `1 <= m, n, p, q <= 100`
- Values are integers or floats
- Inputs are non-empty and rectangular (every row has the same length)

## Rules
- Use **plain Python loops only**. Don't use `np.dot`, `np.matmul`, or the `@` operator.
- Don't modify the input matrices.

## Bonus (after it works)
1. What is the time complexity of your solution?
2. How would you write it in one line using NumPy?

Send me your code and I'll check correctness, edge cases, and style. Then we'll move to Problem 2.

'''

import numpy as np

def matrix_multiply(A: list[list[float]], B:list[list[float]]):
# Python -->

# Time Complexity -> m * n * q

    m = len(A)
    n = len(A[0])
    p = len(B)
    q = len(B[0])
    
    if n != p:
       return -1

# Approach 1 ->

#     rows = []

#     for i in range(m):
#         cols = []
#         for j in range(q):
#             s = 0
#             for k in range(n):
#                 s += A[i][k] * B[k][j]
#             cols.append(s)
#         rows.append(cols)

#     return rows

# Approach 2 ->

    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(q)]
            for i in range(m)]

# Numpy -->

     # A_arr = np.array(A)
     # B_Arr = np.array(B)

     # m, n = np.shape(A_arr)
     # p, q = np.shape(B_Arr)

     # if n != p:
     #      return -1

# Approach 1 ->

     # return np.dot(A_arr, B_Arr).tolist()

# Approach 2 ->

     # return np.matmul(A_arr, B_Arr)

# Approach 3 ->

     # return (A_arr @ B_Arr).tolist()

A = [[1, 2],
     [3, 4]]
B = [[5, 6],
     [7, 8]]

# A = [[1, 2, 3]]          # 1x3
# B = [[4], [5], [6]] 

# A = [[1, 2],
#      [3, 4]]            # 2x2
# B = [[1, 2, 3]]          # 1x3


res = matrix_multiply(A, B)
print(res)

