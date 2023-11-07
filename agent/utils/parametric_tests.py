import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


def long_force(data, cm1, cm2, cm3, cphys1, cphys2, cphys3):
    return (cm1 - cm2 * data[0] - cm3 * data[0] ** 2) * data[1] - cphys1 - cphys2 * data[0] - cphys3 * data[0] ** 2


def sigmoid(data, c1, c2, c3, c4, c5):
    return c1 / (c5 + np.exp(-c2 * (data - c3))) - c4


def hyperbolic_tangent(data, c1, c2, c3, c4):
    return c1 * np.tanh(c2 + data) + c3 + c4


def log_fit(data, c1, c2, c3, c4, c5):
    return c1 * np.log(c2 * data + c3) + c4 + c5


if __name__ == "__main__":
    coeffs = [5, 0, 1, 0]

    x = np.linspace(-2, 2, 100)
    y = coeffs[0] * x ** 3 + coeffs[2] * x
    xy = np.array((x, y)).T

    diff = np.diff(xy, axis=0)
    distances = np.sqrt(np.power(diff, 2).sum(axis=1))
    cumulative_sum = distances.sum()
    print(cumulative_sum)

    distances_along_curve = np.cumsum(distances)
    evenly_spaced_points = np.linspace(0, cumulative_sum, len(x))

    t_params, _ = curve_fit(log_fit, np.log(distances_along_curve), x[:-1])

    cs = CubicSpline(distances_along_curve, x[:-1])
    t_params = np.polyfit(distances_along_curve, x[:-1], 8)

    yt_params = np.polyfit(evenly_spaced_points, y, 2)

    xt_values = sigmoid(evenly_spaced_points, t_params[0], t_params[1], t_params[2], t_params[3], t_params[4])
    # xt_values = np.polyval(t_params, distances_along_curve)
    yt_values = np.polyval(coeffs, xt_values)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="original curve")
    ax.scatter(x[1::5], y[1::5], label="original points")
    ax.plot(xt_values, yt_values, label="parametised")
    # ax.scatter(xt_values[1::5], yt_values[1::5], label="parametised points")
    ax.set_aspect(1)
    ax.legend()

    x_vals = log_fig(distances_along_curve, t_params[0], t_params[1], t_params[2], t_params[3], t_params[4])
    # x_vals = np.polyval(t_params, distances_along_curve)
    # x_vals = cs(distances_along_curve)
    fig1, ax1 = plt.subplots()
    ax1.plot(distances_along_curve, np.log(x[:-1]), label="original")
    ax1.plot(distances_along_curve, x_vals + 1, label="predicted")
    ax1.legend()

    plt.show()
