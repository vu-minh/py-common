import numpy as np
from scipy.spatial.distance import pdist, squareform


MACHINE_EPSILON = np.finfo(np.double).eps


def compute_Q(Y2d, degrees_of_freedom=1, return_squared_form=True):
    """ Matrix Q in t-sne, used to calculate the prob. that a point `j`
    being neighbor of a point `i` (the value of Q[i,j])
    Make sure to call squareform(Q) before using it.
    """
    dist = pdist(Y2d, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    return squareform(Q) if return_squared_form else Q
