import copy
import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LineString
from tqdm import tqdm


def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, ".", color="#999999", zorder=1)


def plot_bounds(ax, ob):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, ".", color="#000000", zorder=1)


def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(
        x, y, color="cyan", alpha=0.7, linewidth=3, solid_capstyle="round", zorder=2
    )


def print_border(ax, waypoints, inner_border_waypoints, outer_border_waypoints):
    line = LineString(waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)

    line = LineString(inner_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)

    line = LineString(outer_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)


def menger_curvature(pt1, pt2, pt3, atol=1e-3):
    vec21 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vec23 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])

    norm21 = np.linalg.norm(vec21)
    norm23 = np.linalg.norm(vec23)

    theta = np.arccos(np.dot(vec21, vec23) / (norm21 * norm23))
    if np.isclose(theta - np.pi, 0.0, atol=atol):
        theta = 0.0

    dist13 = np.linalg.norm(vec21 - vec23)

    return 2 * np.sin(theta) / dist13


def improve_race_line(old_line, inner_border, outer_border):
    """Use gradient descent, inspired by K1999, to find the racing line"""
    # start with the center line
    XI_ITERATIONS = 4

    new_line = copy.deepcopy(old_line)
    ls_inner_border = Polygon(inner_border)
    ls_outer_border = Polygon(outer_border)
    for i in range(0, len(new_line)):
        xi = new_line[i]
        npoints = len(new_line)
        prevprev = (i - 2 + npoints) % npoints
        prev = (i - 1 + npoints) % npoints
        nexxt = (i + 1 + npoints) % npoints
        nexxtnexxt = (i + 2 + npoints) % npoints
        # print("%d: %d %d %d %d %d" % (npoints, prevprev, prev, i, nexxt, nexxtnexxt))
        ci = menger_curvature(new_line[prev], xi, new_line[nexxt])
        c1 = menger_curvature(new_line[prevprev], new_line[prev], xi)
        c2 = menger_curvature(xi, new_line[nexxt], new_line[nexxtnexxt])
        target_ci = (c1 + c2) / 2
        # print("i %d ci %f target_ci %f c1 %f c2 %f" % (i, ci, target_ci, c1, c2))

        # Calculate prospective new track position, start at half-way (curvature zero)
        xi_bound1 = copy.deepcopy(xi)
        xi_bound2 = (
            (new_line[nexxt][0] + new_line[prev][0]) / 2.0,
            (new_line[nexxt][1] + new_line[prev][1]) / 2.0,
        )
        p_xi = copy.deepcopy(xi)
        for j in range(0, XI_ITERATIONS):
            p_ci = menger_curvature(new_line[prev], p_xi, new_line[nexxt])
            # print("i: {} iter {} p_ci {} p_xi {} b1 {} b2 {}".format(i,j,p_ci,p_xi,xi_bound1, xi_bound2))
            if np.isclose(p_ci, target_ci):
                break
            if p_ci < target_ci:
                # too flat, shrinking track too much
                xi_bound2 = copy.deepcopy(p_xi)
                new_p_xi = (
                    (xi_bound1[0] + p_xi[0]) / 2.0,
                    (xi_bound1[1] + p_xi[1]) / 2.0,
                )
                if Point(new_p_xi).within(ls_inner_border) or not Point(
                    new_p_xi
                ).within(ls_outer_border):
                    xi_bound1 = copy.deepcopy(new_p_xi)
                else:
                    p_xi = new_p_xi
            else:
                # too curved, flatten it out
                xi_bound1 = copy.deepcopy(p_xi)
                new_p_xi = (
                    (xi_bound2[0] + p_xi[0]) / 2.0,
                    (xi_bound2[1] + p_xi[1]) / 2.0,
                )

                # If iteration pushes the point beyond the border of the track,
                # just abandon the refinement at this point.  As adjacent
                # points are adjusted within the track the point should gradually
                # make its way to a new position.  A better way would be to use
                # a projection of the point on the border as the new bound.  Later.
                if Point(new_p_xi).within(ls_inner_border) or not Point(
                    new_p_xi
                ).within(ls_outer_border):
                    xi_bound2 = copy.deepcopy(new_p_xi)
                else:
                    p_xi = new_p_xi
        new_xi = p_xi
        # New point which has mid-curvature of prev and next points but may be outside of track
        # print((new_line[i], new_xi))
        new_line[i] = new_xi
    return new_line


def calculate_raceline(track_path):
    """
    Track should have centreline, left and right track points
    """
    # folder = track_path.split("/")[:-1]
    # folder = os.path.join(*folder)
    # with open(track_path, "rb") as handle:
    #    track_dict = json.load(handle)

    track_dict = np.load("track_maps/monza_verysmooth.npy", allow_pickle=True).item()

    left_track = track_dict["outside_track"][1::16]
    right_track = track_dict["inside_track"][1::16]
    centre_track = track_dict["centre_track"][1::16]
    l_center_line = LineString(centre_track)
    print("Is loop/ring? ", l_center_line.is_ring)

    fig = plt.figure(1, figsize=(16, 10))
    ax = fig.add_subplot(111, facecolor="black")
    plt.axis("equal")
    print_border(ax, centre_track, right_track, left_track)
    plt.title("Original")
    plt.pause(2)
    plt.close()

    race_line = copy.deepcopy(
        centre_track[:-1]
    )  # Use this for centerline being outer bound
    LINE_ITERATIONS = 1000

    for i in tqdm(range(LINE_ITERATIONS)):
        race_line = improve_race_line(race_line, right_track, left_track)
        if i % 100 == 0:
            # npy_fname = f"{folder}/Thruxton-{i}.npy"
            loop_race_line = np.append(race_line, [race_line[0]], axis=0)
            raceline_dictionary = {
                "centre": centre_track,
                "raceline": loop_race_line,
                "outside": left_track,
                "inside": right_track,
            }
            # np.save(npy_fname, raceline_dictionary)

    # need to put duplicate point race_line[0] at race_line[-1] to make a closed loops
    loop_race_line = np.append(race_line, [race_line[0]], axis=0)

    # These should be the same
    print("These should be the same: ", (centre_track.shape, loop_race_line.shape))
    print("Original centerline length: %0.2f" % l_center_line.length)
    print("New race line length: %0.2f" % LineString(loop_race_line).length)

    prefix = f"track_maps/monza_verysmooth-{LINE_ITERATIONS}"
    npy_fname = prefix + ".npy"

    print("Writing numpy binary to %s" % npy_fname)
    raceline_dictionary = {
        "centre": track_dict["centre_track"],
        "raceline": loop_race_line,
        "outside": track_dict["outside_track"],
        "inside": track_dict["inside_track"],
    }
    np.save(npy_fname, raceline_dictionary)

    # fig = plt.figure(1, figsize=(16, 10))
    # ax = fig.add_subplot(111, facecolor="black")
    # plt.axis("equal")
    # print_border(ax, loop_race_line, right_track, left_track)
    # plt.pause(2)

    return npy_fname


if __name__ == "__main__":
    file = calculate_raceline("agents/utils/racetracks/map_we_made.json")

    track = np.load(file, allow_pickle=True).item()

    fig = plt.figure(1, figsize=(16, 10))
    ax = fig.add_subplot(111, facecolor="black")
    plt.axis("equal")
    print_border(ax, track["raceline"], track["outside"], track["inside"])
    plt.title("Raceline")
    plt.show()
