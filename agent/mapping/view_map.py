import matplotlib.pyplot as plt
import numpy as np

from map_maker import MapMaker

if __name__ == "__main__":
    # track_dict = np.load("../../track_maps/monza.npy", allow_pickle=True).item()
    track_dict = np.load(
        "../../track_maps/spa_verysmooth.npy", allow_pickle=True
    ).item()

    outside = track_dict.get("outside_track")
    inside = track_dict.get("inside_track")
    centre = track_dict.get("centre_track")

    outside = MapMaker.upsample_track(outside)
    inside = MapMaker.upsample_track(inside)
    centre = MapMaker.upsample_track(centre)

    outside = np.stack(
        (
            MapMaker.smooth_boi(outside, 0, window_length=180, polyorder=3),
            MapMaker.smooth_boi(outside, 1, window_length=180, polyorder=3),
        ),
        axis=1,
    )

    inside = np.stack(
        (
            MapMaker.smooth_boi(inside, 0, window_length=360, polyorder=3),
            MapMaker.smooth_boi(inside, 1, window_length=360, polyorder=3),
        ),
        axis=1,
    )

    centre = np.stack(
        (
            MapMaker.smooth_boi(centre, 0, window_length=360, polyorder=3),
            MapMaker.smooth_boi(centre, 1, window_length=360, polyorder=3),
        ),
        axis=1,
    )

    # output_map = {
    #     "outside_track": outside,
    #     "inside_track": inside,
    #     "centre_track": centre,
    # }
    # np.save("track_maps/monza_badT1_verysmooth.npy", output_map, allow_pickle=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(
        outside[:, 0],
        outside[:, 1],
        color="red",
    )
    ax.scatter(
        inside[:, 0],
        inside[:, 1],
        color="blue",
    )
    ax.scatter(
        centre[:, 0],
        centre[:, 1],
        c=np.arange(len(centre)) / len(centre),
    )

    ax.set_aspect(1)
    plt.gray()
    plt.savefig("spa.png")
    plt.show()
