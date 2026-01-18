import numpy as np


def quaternion_matrix(quaternion):
    """
    Return homogeneous rotation matrix from quaternion.
    Quaternion is expected in (w, x, y, z) format.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
