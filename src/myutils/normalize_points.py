import numpy as np


def rescale(points):
    """
    Rescale point set to [-1, 1].

    Args:
        points: array of size [n, 3]

    Returns:
        points_scaled
        scale
    """
    range_min = -1
    range_max = 1
    p_min = np.amin(points)
    p_max = np.amax(points)
    scale = (range_max - range_min) / (p_max - p_min)
    points_scaled = points.dot(scale) + range_min - (p_min * scale)

    return points_scaled, scale
