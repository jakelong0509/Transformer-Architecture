import numpy as np

# Xavier initializer
def Xavier(dimension):
    # checking if dimension is a tuple or not
    if not isinstance(dimension, tuple):
        sys.exit("Argument 'dimension' is not a tuple. Terminating....")

    n_in, n_out = dimension
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size = dimension)

# Make Batches
