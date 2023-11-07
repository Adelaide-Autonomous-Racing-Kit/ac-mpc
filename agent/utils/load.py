import json as _json
from typing import Dict

import numpy as np
from ruamel.yaml import YAML


def track_map(path: str) -> Dict:
    return EXTENSION_TO_METHOD[path.split(".")[-1]](path)


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
