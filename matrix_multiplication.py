import numpy as np

def matrix_multiply(A: list[list[float]], B:list[list[float]]):
    a = np.array(A)
    b = np.array(B)
    print(a.shape)
    print(b.shape)

A = [[1, 2],
     [3, 4]]
B = [[5, 6],
     [7, 8]]

matrix_multiply(A, B)