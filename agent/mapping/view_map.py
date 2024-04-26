import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from map_maker import MapMaker


def main(args: argparse.Namespace):
    title = Path(args.input_path).stem
    track_dict = np.load(args.input_path, allow_pickle=True).item()

    outside = track_dict.get("outside_track")
    inside = track_dict.get("inside_track")
    centre = track_dict.get("centre_track")

    if args.smooth_map:
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
                MapMaker.smooth_boi(inside, 0, window_length=180, polyorder=3),
                MapMaker.smooth_boi(inside, 1, window_length=180, polyorder=3),
            ),
            axis=1,
        )

        centre = np.stack(
            (
                MapMaker.smooth_boi(centre, 0, window_length=180, polyorder=3),
                MapMaker.smooth_boi(centre, 1, window_length=180, polyorder=3),
            ),
            axis=1,
        )

        output_map = {
            "outside_track": outside,
            "inside_track": inside,
            "centre_track": centre,
        }
        if args.output_path:
            np.save(args.output_path, output_map, allow_pickle=True)
        title += " smoothed"

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(
        outside[:, 0],
        outside[:, 1],
        cmap=mpl.colormaps["autumn"],
        c=np.arange(len(outside)) / len(outside),
    )
    ax.scatter(
        inside[:, 0],
        inside[:, 1],
        cmap=mpl.colormaps["winter"],
        c=np.arange(len(inside)) / len(inside),
    )
    ax.scatter(
        centre[:, 0],
        centre[:, 1],
        c=np.arange(len(centre)) / len(centre),
    )

    ax.set_aspect(1)
    plt.gray()
    plt.savefig(f"track_maps/{title}.png")
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to map")
    parser.add_argument("--smooth-map", action="store_true", help="Smooth input map")
    parser.add_argument("--output-path", type=str, help="Path to save smoothed map")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
