import json as _json
from typing import Dict

from loguru import logger
import numpy as np
from ruamel.yaml import YAML


def track_map(path: str) -> Dict:
    track_dict = EXTENSION_TO_METHOD[path.split(".")[-1]](path)
    tracks = {
        "left": track_dict["outside_track"],
        "right": track_dict["inside_track"],
        "centre": track_dict["centre_track"],
    }
    logger.info(
        f"Loaded map with shapes: {tracks['left'].shape=}"
        + f"{tracks['right'].shape=}, {tracks['centre'].shape=}"
    )
    tracks["left"] = remove_near_duplicate_points(tracks["left"])
    tracks["right"] = remove_near_duplicate_points(tracks["right"])
    tracks["centre"] = remove_near_duplicate_points(tracks["centre"])
    logger.info(
        f"Removed duplicates from map, shapes: {tracks['left'].shape=}"
        + f"{tracks['right'].shape=}, {tracks['centre'].shape=}"
    )
    return tracks


def remove_near_duplicate_points(track: np.array) -> np.array:
    d = np.diff(track, axis=0)
    dists = np.hypot(d[:, 0], d[:, 1])
    is_not_duplicated = np.ones(dists.shape[0] + 1).astype(bool)
    is_not_duplicated[1:] = dists > 0.0001
    return track[is_not_duplicated]


def _load_json_track(path: str) -> Dict:
    data = json(path)
    data = {
        "centre_track": np.array(data["Centre"]),
        "outside_track": np.array(data["Outside"]),
        "inside_track": np.array(data["Inside"]),
    }
    return data


def yaml(path: str) -> Dict:
    _yaml = YAML()
    with open(path) as file:
        params = _yaml.load(file)
    return params


def json(filepath: str) -> Dict:
    with open(filepath) as file:
        data = _json.load(file)
    return data


def npy(filepath: str) -> Dict:
    return np.load(filepath, allow_pickle=True).item()


EXTENSION_TO_METHOD = {"npy": npy, "json": _load_json_track}
