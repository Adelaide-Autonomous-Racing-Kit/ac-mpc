from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List

from PyQt6.QtCore import QObject, QTimer, pyqtProperty, pyqtSignal


@dataclass
class LapInfo:
    n_lap: int
    sector_times: List[int] = field(default_factory=lambda: [0, 0, 0])
    time: int = 0


class SessionInformationProvider(QObject):
    informationUpdated = pyqtSignal()

    def __init__(self, agent: ElTuarMPC):
        super().__init__()
        self._agent = agent
        self._setup_state()
        self._setup_timer()

    def _setup_timer(self):
        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_info)
        self._timer.start()

    def _setup_state(self):
        self._laps = [LapInfo(0)]
        self._current_laptime = 0
        self._last_laptime = 0
        self._best_laptime = 0
        self._best_lap = 0
        self._last_sector_time = 0
        self._n_laps_completed = 0
        self._current_sector = 0
        self._previous_lap = 0
        self._previous_sector = -1
        self._previous_best_laptime = 0

    def _update_info(self):
        self._current_laptime = self._agent.session_info.current_laptime
        self._last_laptime = self._agent.session_info.last_laptime
        self._best_laptime = self._agent.session_info.best_laptime
        self._last_sector_time = self._agent.session_info.last_sector_time
        self._n_laps_completed = self._agent.session_info.n_laps_completed
        self._current_sector = self._agent.session_info.current_sector
        if self._n_laps_completed > self._previous_lap:
            # Finalise lap details
            lap = self._laps[-1]
            lap.time = self._last_laptime
            lap.sector_times[-1] = self._last_sector_time
            # Create new lap
            self._laps.append(LapInfo(self._n_laps_completed))
            self._previous_lap = self._n_laps_completed
            self._previous_sector = self._current_sector
            if self._previous_best_laptime == 0:
                self._previous_best_laptime = self._last_laptime

            if self._previous_best_laptime > self._last_laptime:
                self._previous_best_laptime = self._last_laptime
                self._best_lap = self._n_laps_completed - 1

        if self._current_sector > self._previous_sector:
            lap = self._laps[-1]
            lap.sector_times[self._previous_sector] = self._last_sector_time
            self._previous_sector = self._current_sector

        lap = self._laps[-1]
        current_sector_time = (
            self._current_laptime
            - sum(lap.sector_times)
            + lap.sector_times[self._current_sector]
        )
        lap.sector_times[self._current_sector] = current_sector_time
        self.informationUpdated.emit()

    @pyqtProperty(str, notify=informationUpdated)
    def current_laptime(self) -> str:
        return format_laptime(self._current_laptime)

    @pyqtProperty(str, notify=informationUpdated)
    def best_laptime(self) -> str:
        return format_laptime(self._best_laptime)

    @pyqtProperty(str, notify=informationUpdated)
    def last_laptime(self) -> str:
        return format_laptime(self._last_laptime)

    @pyqtProperty(str, notify=informationUpdated)
    def last_sector_time(self) -> str:
        return format_laptime(self._last_sector_time)

    @pyqtProperty(int, notify=informationUpdated)
    def current_sector(self) -> int:
        return self._current_sector

    @pyqtProperty(int, notify=informationUpdated)
    def n_laps_completed(self) -> str:
        return self._n_laps_completed

    @pyqtProperty(str, notify=informationUpdated)
    def current_first_sector_time(self) -> str:
        return format_sector_time(self._current_first_sector_time)

    @property
    def _current_first_sector_time(self) -> int:
        return self._get_current_sector_time(0)

    @pyqtProperty(str, notify=informationUpdated)
    def current_second_sector_time(self) -> str:
        return format_sector_time(self._current_second_sector_time)

    @property
    def _current_second_sector_time(self) -> int:
        return self._get_current_sector_time(1)

    @pyqtProperty(str, notify=informationUpdated)
    def current_third_sector_time(self) -> str:
        return format_sector_time(self._current_third_sector_time)

    @property
    def _current_third_sector_time(self) -> int:
        return self._get_current_sector_time(2)

    def _get_current_sector_time(self, sector: int) -> int:
        return self._laps[-1].sector_times[sector]

    @pyqtProperty(str, notify=informationUpdated)
    def last_first_sector_time(self) -> str:
        return format_sector_time(self._last_first_sector_time)

    @property
    def _last_first_sector_time(self) -> int:
        return self._get_last_sector_time(0)

    @pyqtProperty(str, notify=informationUpdated)
    def last_second_sector_time(self) -> str:
        return format_sector_time(self._last_second_sector_time)

    @property
    def _last_second_sector_time(self) -> int:
        return self._get_last_sector_time(1)

    @pyqtProperty(str, notify=informationUpdated)
    def last_third_sector_time(self) -> str:
        return format_sector_time(self._last_third_sector_time)

    @property
    def _last_third_sector_time(self) -> int:
        return self._get_last_sector_time(2)

    @pyqtProperty(str, notify=informationUpdated)
    def best_first_sector_time(self) -> str:
        return format_sector_time(self._best_first_sector_time)

    @property
    def _best_first_sector_time(self) -> int:
        return self._get_best_sector_time(0)

    @pyqtProperty(str, notify=informationUpdated)
    def best_second_sector_time(self) -> str:
        return format_sector_time(self._best_second_sector_time)

    @property
    def _best_second_sector_time(self) -> int:
        return self._get_best_sector_time(1)

    @pyqtProperty(str, notify=informationUpdated)
    def best_third_sector_time(self) -> str:
        return format_sector_time(self._best_third_sector_time)

    @property
    def _best_third_sector_time(self) -> int:
        return self._get_best_sector_time(2)

    @pyqtProperty(str, notify=informationUpdated)
    def first_sector_colour(self) -> str:
        return self._get_current_sector_time_colour(0)

    @pyqtProperty(str, notify=informationUpdated)
    def second_sector_colour(self) -> str:
        return self._get_current_sector_time_colour(1)

    @pyqtProperty(str, notify=informationUpdated)
    def third_sector_colour(self) -> str:
        return self._get_current_sector_time_colour(2)

    def _get_current_sector_time_colour(self, sector: int) -> str:
        is_finished_sector = self._current_sector > sector
        if not is_finished_sector:
            return "white"
        sector_time = self._get_current_sector_time(sector)
        best_time = self._get_best_sector_time(sector)
        return self._get_sector_colour(sector_time, best_time)

    @pyqtProperty(str, notify=informationUpdated)
    def last_first_sector_colour(self) -> str:
        return self._get_last_sector_time_colour(0)

    @pyqtProperty(str, notify=informationUpdated)
    def last_second_sector_colour(self) -> str:
        return self._get_last_sector_time_colour(1)

    @pyqtProperty(str, notify=informationUpdated)
    def last_third_sector_colour(self) -> str:
        return self._get_last_sector_time_colour(2)

    def _get_last_sector_time_colour(self, sector: int) -> str:
        sector_time = self._get_last_sector_time(sector)
        best_time = self._get_best_sector_time(sector)
        return self._get_sector_colour(sector_time, best_time)

    def _get_sector_colour(self, sector_time: int, best_time: int) -> str:
        is_best_valid = best_time > 0
        if not is_best_valid:
            return "white"
        is_better = sector_time < best_time
        is_worse = sector_time > best_time + 100
        if is_better:
            return "purple"
        elif is_worse:
            return "yellow"
        return "green"

    def _get_best_sector_time(self, sector: int) -> int:
        if self._n_laps_completed == 0:
            best_sector_time = 0
        else:
            best_lap = self._best_lap
            best_sector_time = self._laps[best_lap].sector_times[sector]
        return best_sector_time

    def _get_last_sector_time(self, sector: int) -> int:
        if self._n_laps_completed == 0:
            last_sector_time = 0
        else:
            previous_lap = self._n_laps_completed - 1
            last_sector_time = self._laps[previous_lap].sector_times[sector]
        return last_sector_time


def format_sector_time(sector_time_ms: int) -> str:
    if sector_time_ms == 0:
        time_string = "-"
    else:
        time_string = format_laptime(sector_time_ms)
    return time_string


def format_laptime(laptime_ms: int) -> str:
    current_laptime = laptime_ms / 1000
    minutes = math.floor(current_laptime / 60)
    seconds = current_laptime % 60
    return f"{minutes:02d}:{seconds:06.3f}"
