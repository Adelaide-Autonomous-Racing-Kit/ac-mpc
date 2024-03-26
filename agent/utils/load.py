import json as _json
from typing import Dict

import numpy as np
from loguru import logger
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
    # Remove near duplicate centre points
    d = np.diff(tracks["centre"], axis=0)
    dists = np.hypot(d[:, 0], d[:, 1])
    is_not_duplicated = np.ones(dists.shape[0] + 1).astype(bool)
    is_not_duplicated[1:] = dists > 0.0001
    tracks["left"] = tracks["left"][is_not_duplicated]
    tracks["right"] = tracks["right"][is_not_duplicated]
    tracks["centre"] = tracks["centre"][is_not_duplicated]
    logger.info(
        f"Removed duplicates from map, shapes: {tracks['left'].shape=}"
        + f"{tracks['right'].shape=}, {tracks['centre'].shape=}"
    )
    return tracks


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
