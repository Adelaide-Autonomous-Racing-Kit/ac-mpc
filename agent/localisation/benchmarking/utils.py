from typing import Dict, List

from utils import load


class LocalisationRecording:
    def __init__(self, data_path: str):
        self.__setup(data_path)

    def __setup(self, data_path: str):
        self._data_path = data_path
        self._setup_recording()

    def _setup_recording(self):
        control = load.npy(f"{self._data_path}/control.npy")
        observations = load.npy(f"{self._data_path}/observations.npy")
        control = self._dict_to_list(control)
        observations = self._dict_to_list(observations)
        control.extend(observations)
        self._recording = sorted(control, key=lambda x: x["time"])

    def _dict_to_list(self, dictionary: Dict) -> List:
        return [dictionary[x] for x in dictionary.keys()]

    def __getitem__(self, index: int) -> Dict:
        return self._recording[index]

    def __len__(self) -> int:
        return len(self._recording)
