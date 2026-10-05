'''

# Problem 5: Cosine Similarity 🔥

**Difficulty:** Easy | **Topic:** Linear Algebra / Retrieval

## Task
Implement cosine similarity between two vectors, then use it to find the most similar vector in a list.

## Function signatures
```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    ...

def most_similar(query: list[float], vectors: list[list[float]]) -> int:
    ...
```

## Definition
`cos(a, b) = (a · b) / (||a||_2 * ||b||_2)`

## Input
- `a`, `b`, `query`: non-empty lists of numbers
- `vectors`: a non-empty list of vectors, each the same length as `query`

## Output
- `cosine_similarity`: a float rounded to 4 decimals. If either vector has zero norm, return `0.0`. If the lengths differ, return `-1`.
- `most_similar`: the **index** of the vector with the highest cosine similarity to `query`. On ties, return the smallest index.

## Examples

**Example 1**
```
a = [1, 2, 3]
b = [4, 5, 6]
Output: 0.9746
```

**Example 2 (orthogonal)**
```
a = [1, 0]
b = [0, 1]
Output: 0.0
```

**Example 3 (opposite direction)**
```
a = [1, 2]
b = [-1, -2]
Output: -1.0
```

**Example 4 (zero vector)**
```
a = [0, 0]
b = [1, 2]
Output: 0.0
```

**Example 5 (most_similar)**
```
query = [1, 0]
vectors = [[0, 1], [2, 0.1], [-1, 0]]
Output: 1
```

## Constraints
- `1 <= len(a) <= 1000`
- `1 <= len(vectors) <= 1000`
- Values are integers or floats

## Rules
- Plain Python only. No NumPy.
- Reuse your `dot` and `lp_norm` functions from the earlier problems, or rewrite them.
- Be careful with floating-point output: `-1.0` should not come out as `-0.9999999999999998` after rounding. Check it.

## Bonus
1. For **unit-length** vectors, how is `||a - b||²` related to `cos(a, b)`? Why does this mean k-NN with cosine and k-NN with Euclidean distance give the same neighbors once the vectors are normalized?
2. In `most_similar`, can you avoid recomputing `||query||` for every candidate? Does it change the answer?

Send your code when ready.

'''
from dot_matvec import dot
from vector_norms import lp_norm

def cosine_similarity(a: list[float], b: list[float]) -> float:
    m = len(a)
    n = len(b)

    if m != n:
        return -1
    
    dot_product = dot(a, b)

    norm_a = lp_norm(a, 2)
    norm_b = lp_norm(b, 2)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return round(dot_product / (norm_a * norm_b), 4)

def most_similar(query: list[float], vectors: [list[list[float]]]) -> int:

    # Approach 1 ->

    # max_dist = 0
    # most_sim_idx = 0
    # for idx, vec in enumerate(vectors):
    #     dist = cosine_similarity(query, vec)
    #     if dist > max_dist:
    #         max_dist = dist
    #         most_sim_idx = idx
    # return most_sim_idx

    # Approach 2 ->

    best_score = float('-inf')
    most_sim_idx = 0
    for idx, vec in enumerate(vectors):
        norm = lp_norm(vec, 2)
        score = dot(query, vec) / norm if norm != 0 else 0.0
        print(score)
        if score > best_score:
            best_score = score
            most_sim_idx = idx
    return most_sim_idx

# a = [1, 2, 3]
# b = [4, 5, 6]

# a = [1, 0]
# b = [0, 1]

# a = [1, 2]
# b = [-1, -2]

# a = [0, 0]
# b = [1, 2]

# result = cosine_similarity(a, b)

# query = [1, 0]
# vectors = [[0, 1], [2, 0.1], [-1, 0]]

query = [1, 0]
vectors = [[-1, 0], [-2, 0.1], [-1, -5]]

result = most_similar(query, vectors)
print(result)

'''

Bonus answers

Bonus 2: Your answer is right. ||query|| is a positive constant shared by all candidates, so dividing by it can't change which candidate wins.

Bonus 1: For unit vectors, ||a||² = ||b||² = 1, so
||a - b||² = ||a||² + ||b||² - 2(a·b) = 2 - 2cos(a, b).

Squared Euclidean distance is a decreasing function of cosine similarity. The smallest distance corresponds to the largest cosine, so both give the same neighbors. This is why vector databases normalize embeddings and then use fast Euclidean or dot-product search.

'''