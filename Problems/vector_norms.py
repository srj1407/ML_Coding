'''

# Review: Problem 3

**Verdict: correct.** All four examples pass.

**What's good**
- `dot` checks the length mismatch before computing, and `matvec` reuses `dot` as asked.
- `O(m·n)` is the right time complexity for `matvec`, and `dot` alone is `O(n)`.
- `dot` returning `-1` can never leak into `matvec`'s output, because you check `n != p` first.

**Minor points**
- `matvec` could also be written as `[dot(row, v) for row in A]`. Yours is fine.
- In a real codebase, returning `-1` as an error is poor style (a vector can legitimately have a dot product of `-1`). Raising `ValueError` is better, but the problem specified `-1`.

## Bonus 2 explained

There are two ways to read `A @ v`. Take `A = [[1,2,3],[4,5,6]]` and `v = [1,0,-1]`.

**View 1: rows (dot products).** Each output entry is one row of `A` dotted with `v`:
```
result[0] = dot([1,2,3], v) = -2
result[1] = dot([4,5,6], v) = -2
```
This is what your code does.

**View 2: columns (linear combination).** The result is the columns of `A`, each scaled by the matching entry of `v`, then added:
```
1 * [1,4]  +  0 * [2,5]  +  (-1) * [3,6]  =  [-2, -2]
```
Same answer, different way of seeing it. This view matters later: it explains why `A @ v` always lands in the column space of `A`, which is the key idea behind rank, PCA, and linear regression.

**Which is more cache-friendly?** In row-major storage (NumPy's default, and C), the elements of one row sit next to each other in memory. Your row-wise approach reads `A` in order, so the CPU can prefetch it efficiently. A naive column-wise loop jumps by a whole row length at each step, so it's slower on large arrays. The answer is the row view. The effect is muted for Python lists of lists, but it matters in NumPy and C, and interviewers like this question.

---

# Problem 4: Vector Norms

**Difficulty:** Easy | **Topic:** Linear Algebra

## Task
Implement a function that computes the **Lp norm** of a vector, plus a normalizer that scales a vector to unit length.

## Function signatures
```python
def lp_norm(v: list[float], p) -> float:
    ...

def normalize(v: list[float], p=2) -> list[float]:
    ...
```

## Definitions
- For `p >= 1`: `||v||_p = (sum(|v_i|^p))^(1/p)`
- For `p = float('inf')`: `||v||_inf = max(|v_i|)`

## Input
- `v`: a non-empty list of numbers
- `p`: a number `>= 1`, or `float('inf')`

## Output
- `lp_norm`: the norm as a float, rounded to 4 decimals
- `normalize`: a list where each entry is `v_i / ||v||_p`, each rounded to 4 decimals. If the norm is `0`, return the original vector unchanged.

## Examples

**Example 1**
```
v = [3, -4]
p = 1
Output: 7.0
```

**Example 2**
```
v = [3, -4]
p = 2
Output: 5.0
```

**Example 3**
```
v = [1, -5, 3]
p = float('inf')
Output: 5.0
```

**Example 4 (normalize)**
```
v = [3, 4]
p = 2
Output: [0.6, 0.8]
```

**Example 5 (zero vector)**
```
v = [0, 0, 0]
p = 2
Output: [0, 0, 0]
```

## Constraints
- `1 <= len(v) <= 1000`
- `p >= 1` or `float('inf')`

## Rules
- Plain Python only. You can use `abs`, `max`, and `**`, but no NumPy and no `math.hypot`.
- `normalize` must call `lp_norm`.

## Bonus
1. Why does `p < 1` not give a valid norm? Which property fails? (Hint: try `v = [1, 0]`, `w = [0, 1]`, `p = 0.5` and check the triangle inequality.)
2. Where do L1 and L2 norms show up in ML? Name one use of each.

Send your code when ready.

'''

def lp_norm(v: list[float], p) -> float:
    if p == float('inf'):
        return round(float(max(abs(x) for x in v)), 4)
    
    return round(pow(sum([pow(abs(e), p) for e in v]), 1/p), 4)

def normalize(v: list[float], p = 2) -> list[float]:
    norm = lp_norm(v, p)

    if norm == 0:
        return v
    
    return [round(e / norm, 4) for e in v]

v = [3, -4]
p = 1

# v = [3, -4]
# p = 2

# v = [1, -5, 3]
# p = float('inf')

res = lp_norm(v, p)

# v = [3, 4]
# p = 2

# v = [0, 0, 0]
# p = 2

# res = normalize(v, p)

print(res)

'''

Bonus 1: why p < 1 isn't a norm

It fails the triangle inequality: ||v + w|| <= ||v|| + ||w||. Take p = 0.5, v = [1, 0], w = [0, 1]:

||v|| = 1 and ||w|| = 1, so the right side is 2.
v + w = [1, 1], and ||v + w|| = (1^0.5 + 1^0.5)^(1/0.5) = 2^2 = 4.
4 <= 2 is false.

Geometrically, the unit ball for p < 1 is non-convex (it curves inward like a star), and a norm's unit ball must be convex. That is also why p = 1 is the smallest p where the norm is valid, and why L1 is the closest convex stand-in for the "count non-zeros" idea.

Bonus 2: where L1 and L2 show up

Your answer is partly right. L2 normalization does give unit length, which is used in cosine similarity and in embedding normalization. But it doesn't bound values to a range like [0, 1]. The more concrete answers interviewers expect:

L1: Lasso regularization. The penalty λ·||w||_1 pushes many weights to exactly zero, which gives sparse models and feature selection.
L2: Ridge regression and weight decay. The penalty λ·||w||_2² shrinks weights smoothly without zeroing them. L2 is also the basis of Euclidean distance in k-NN and k-means.

Rule of thumb: L1 gives sparsity, L2 gives shrinkage.

'''