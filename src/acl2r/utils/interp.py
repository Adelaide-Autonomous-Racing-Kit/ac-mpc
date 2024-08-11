import numpy as np


def smooth_track_with_polyfit(track, num_points, degree=3):
    ynew = np.linspace(0, np.max(track[1]), num_points)
    coeffs = np.polyfit(track[1], track[0], degree)
    xnew = np.polyval(coeffs, ynew)
    return np.array([xnew, ynew])
