import matplotlib.pyplot as plt
import numpy as np

from mapping.map_maker import MapMaker

if __name__ == "__main__":
    track_dict = np.load(
        "agents/utils/our_racetracks/anglesly2.npy", allow_pickle=True
    ).item()

    outside = track_dict.get("outside_track")
    inside = track_dict.get("inside_track")
    centre = track_dict.get("centre_track")

    outside = MapMaker.upsample_track(outside)
    inside = MapMaker.upsample_track(inside)
    centre = MapMaker.upsample_track(centre)

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
    plt.show()
