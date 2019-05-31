import numpy as np
import timeit
import os

# Xavier initializer
def Xavier(dimension):
    # checking if dimension is a tuple or not
    if not isinstance(dimension, tuple):
        sys.exit("Argument 'dimension' is not a tuple. Terminating....")

    n_in, n_out = dimension
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size = dimension)

# O(logn)
def get_index(start, end, arr):
    while True:
        if arr[start] == 1:
            return start
        if arr[end] == 1:
            return end
        start = start + 1
        end = end - 1

if __name__ == "__main__":
    arr = np.zeros(50000)
    arr2 = np.random.random((50000, 512))
    arr[33534] = 1
    start = timeit.default_timer()
    index = get_index(0,len(arr)-1,arr) # faster
    result = arr2[index]
    stop = timeit.default_timer()
    print("Result: ", result)
    print("Time: ", stop - start)

    start2 = timeit.default_timer()
    result = np.matmul(np.transpose(arr), arr2)
    stop2 = timeit.default_timer()
    print("Result2: ", result)
    print("Time2: ", stop2 - start2)
