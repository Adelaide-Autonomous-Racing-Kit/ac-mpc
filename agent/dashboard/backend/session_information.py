from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List

from PyQt6.QtCore import QObject, QTimer, pyqtProperty, pyqtSignal

MINIMUM_DELTA_MS = 50

GREEN = "#1ced23"
PURPLE = "#d23ce6"
YELLOW = "yellow"
WHITE = "white"
RED = "red"


@dataclass
class LapInfo:
    n_lap: int
    sector_times: List[int] = field(default_factory=lambda: [0, 0, 0])
    time: int = 0


class SessionInformationProvider(QObject):
    informationUpdated = pyqtSignal()
    lapCompleted = pyqtSignal()
    sectorCompleted = pyqtSignal()

    def __init__(self, agent: ElTuarMPC):
        super().__init__()
        self.__setup(agent)

    def _update_info(self):
        self._update_state()
        self._maybe_finalise_sector()
        self._maybe_finalise_lap()
        self._finalise_update()

    def _update_state(self):
        self._current_laptime = self._agent.session_info.current_laptime
        self._last_laptime = self._agent.session_info.last_laptime
        self._best_laptime = self._agent.session_info.best_laptime
        self._last_sector_time = self._agent.session_info.last_sector_time
        self._n_laps_completed = self._agent.session_info.n_laps_completed
        self._current_sector = self._agent.session_info.current_sector

    def _maybe_finalise_lap(self):
        if self._n_laps_completed > self._previous_lap:
            self._finalise_lap()

    def _finalise_lap(self):
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
        self.lapCompleted.emit()
        self.sectorCompleted.emit()

    def _maybe_finalise_sector(self):
        if self._current_sector != self._previous_sector:
            self._finalise_sector()

    def _finalise_sector(self):
        lap = self._laps[-1]
        lap.sector_times[self._previous_sector] = self._last_sector_time
        self._previous_sector = self._current_sector
        self.sectorCompleted.emit()

    def _finalise_update(self):
        self._update_current_sector_time()
        self.informationUpdated.emit()

    def _update_current_sector_time(self):
        lap = self._laps[-1]
        current_sector_time = lap.sector_times[self._current_sector]
        cum_sector_time = sum(lap.sector_times)
        new_sector_time = self._current_laptime - cum_sector_time + current_sector_time
        lap.sector_times[self._current_sector] = new_sector_time

    @pyqtProperty(int, notify=lapCompleted)
    def best_lap(self) -> int:
        return self._best_lap

    @pyqtProperty(int, notify=lapCompleted)
    def n_laps_completed(self) -> int:
        return self._n_laps_completed

    @pyqtProperty(int, notify=lapCompleted)
    def last_lap(self) -> int:
        return self._n_laps_completed - 1 if self._n_laps_completed > 0 else 0

    @pyqtProperty(str, notify=informationUpdated)
    def current_laptime(self) -> str:
        return format_laptime(self._current_laptime)

    @pyqtProperty(str, notify=lapCompleted)
    def best_laptime(self) -> str:
        return format_laptime(self._best_laptime)

    @pyqtProperty(str, notify=lapCompleted)
    def last_laptime(self) -> str:
        return format_laptime(self._last_laptime)

    @pyqtProperty(str, notify=sectorCompleted)
    def last_sector_time(self) -> str:
        return format_laptime(self._last_sector_time)

    @pyqtProperty(int, notify=sectorCompleted)
    def current_sector(self) -> int:
        return self._current_sector

    @pyqtProperty(str, notify=informationUpdated)
    def current_first_sector_time(self) -> str:
        return format_sector_time(self._current_first_sector_time)

    @pyqtProperty(str, notify=informationUpdated)
    def current_second_sector_time(self) -> str:
        return format_sector_time(self._current_second_sector_time)

    @pyqtProperty(str, notify=informationUpdated)
    def current_third_sector_time(self) -> str:
        return format_sector_time(self._current_third_sector_time)

    @pyqtProperty(str, notify=lapCompleted)
    def last_first_sector_time(self) -> str:
        return format_sector_time(self._last_first_sector_time)

    @pyqtProperty(str, notify=lapCompleted)
    def last_second_sector_time(self) -> str:
        return format_sector_time(self._last_second_sector_time)

    @pyqtProperty(str, notify=lapCompleted)
    def last_third_sector_time(self) -> str:
        return format_sector_time(self._last_third_sector_time)

    @pyqtProperty(str, notify=lapCompleted)
    def best_first_sector_time(self) -> str:
        return format_sector_time(self._best_first_sector_time)

    @pyqtProperty(str, notify=lapCompleted)
    def best_second_sector_time(self) -> str:
        return format_sector_time(self._best_second_sector_time)

    @pyqtProperty(str, notify=lapCompleted)
    def best_third_sector_time(self) -> str:
        return format_sector_time(self._best_third_sector_time)

    @pyqtProperty(str, notify=sectorCompleted)
    def first_sector_colour(self) -> str:
        return self._get_current_sector_time_colour(0)

    @pyqtProperty(str, notify=sectorCompleted)
    def second_sector_colour(self) -> str:
        return self._get_current_sector_time_colour(1)

    @pyqtProperty(str, notify=sectorCompleted)
    def third_sector_colour(self) -> str:
        return self._get_current_sector_time_colour(2)

    @pyqtProperty(str, notify=lapCompleted)
    def last_first_sector_colour(self) -> str:
        return self._get_last_sector_time_colour(0)

    @pyqtProperty(str, notify=lapCompleted)
    def last_second_sector_colour(self) -> str:
        return self._get_last_sector_time_colour(1)

    @pyqtProperty(str, notify=lapCompleted)
    def last_third_sector_colour(self) -> str:
        return self._get_last_sector_time_colour(2)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_first_sector_delta(self) -> str:
        delta = self._current_first_sector_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_second_sector_delta(self) -> str:
        delta = self._current_second_sector_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_third_sector_delta(self) -> str:
        delta = self._current_third_sector_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_first_sector_delta_colour(self) -> str:
        delta = self._current_first_sector_delta
        return self._get_delta_colour(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_second_sector_delta_colour(self) -> str:
        delta = self._current_second_sector_delta
        return self._get_delta_colour(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_third_sector_delta_colour(self) -> str:
        delta = self._current_third_sector_delta
        return self._get_delta_colour(delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_first_sector_delta(self) -> str:
        delta = self._last_first_sector_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_second_sector_delta(self) -> str:
        delta = self._last_second_sector_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_third_sector_delta(self) -> str:
        delta = self._last_third_sector_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_first_sector_delta_colour(self) -> str:
        delta = self._last_first_sector_delta
        return self._get_delta_colour(delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_second_sector_delta_colour(self) -> str:
        delta = self._last_second_sector_delta
        return self._get_delta_colour(delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_third_sector_delta_colour(self) -> str:
        delta = self._last_third_sector_delta
        return self._get_delta_colour(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_lap_delta(self) -> str:
        delta = self._current_lap_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_lap_delta_colour(self) -> str:
        delta = self._current_lap_delta
        return self._get_delta_colour(delta)

    @pyqtProperty(str, notify=sectorCompleted)
    def current_lap_colour(self) -> str:
        return self._get_lap_colour(self._current_lap_delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_lap_colour(self) -> str:
        return self._get_lap_colour(self._last_lap_delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_lap_delta(self) -> str:
        delta = self._last_lap_delta
        return format_delta_time(delta)

    @pyqtProperty(str, notify=lapCompleted)
    def last_lap_delta_colour(self) -> str:
        delta = self._last_lap_delta
        return self._get_delta_colour(delta)

    @property
    def _current_lap_delta(self) -> int:
        delta = 0
        delta += self._current_first_sector_delta
        delta += self._current_second_sector_delta
        delta += self._current_third_sector_delta
        return delta

    @property
    def _last_lap_delta(self) -> int:
        delta = 0
        delta += self._last_first_sector_delta
        delta += self._last_second_sector_delta
        delta += self._last_third_sector_delta
        return delta

    @property
    def _current_second_sector_time(self) -> int:
        return self._get_current_sector_time(1)

    @property
    def _current_first_sector_time(self) -> int:
        return self._get_current_sector_time(0)

    @property
    def _current_third_sector_time(self) -> int:
        return self._get_current_sector_time(2)

    @property
    def _last_first_sector_time(self) -> int:
        return self._get_last_sector_time(0)

    @property
    def _last_third_sector_time(self) -> int:
        return self._get_last_sector_time(2)

    @property
    def _last_second_sector_time(self) -> int:
        return self._get_last_sector_time(1)

    @property
    def _best_first_sector_time(self) -> int:
        return self._get_best_sector_time(0)

    @property
    def _best_second_sector_time(self) -> int:
        return self._get_best_sector_time(1)

    @property
    def _best_third_sector_time(self) -> int:
        return self._get_best_sector_time(2)

    @property
    def _current_first_sector_delta(self) -> int:
        return self._get_current_sector_delta(0)

    @property
    def _current_second_sector_delta(self) -> int:
        return self._get_current_sector_delta(1)

    @property
    def _current_third_sector_delta(self) -> int:
        return self._get_current_sector_delta(2)

    @property
    def _last_first_sector_delta(self) -> int:
        return self._get_last_sector_delta(0)

    @property
    def _last_second_sector_delta(self) -> int:
        return self._get_last_sector_delta(1)

    @property
    def _last_third_sector_delta(self) -> int:
        return self._get_last_sector_delta(2)

    def _get_current_sector_time(self, sector: int) -> int:
        return self._laps[-1].sector_times[sector]

    def _get_delta_colour(self, delta: int) -> str:
        return RED if delta > 0 else GREEN

    def _get_lap_colour(self, lap_delta: int) -> str:
        if lap_delta > MINIMUM_DELTA_MS:
            colour = YELLOW
        elif lap_delta < -MINIMUM_DELTA_MS:
            colour = PURPLE
        elif lap_delta == 0:
            colour = WHITE
        else:
            colour = GREEN
        return colour

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

    def _get_current_sector_delta(self, sector: int) -> int:
        if self._current_sector > sector:
            current_time = self._get_current_sector_time(sector)
            best_time = self._get_best_sector_time(sector)
            if best_time == 0:
                delta = 0
            else:
                delta = current_time - best_time
        else:
            delta = 0
        return delta

    def _get_last_sector_delta(self, sector: int) -> int:
        last_time = self._get_last_sector_time(sector)
        best_time = self._get_best_sector_time(sector)
        if best_time == 0:
            delta = 0
        else:
            delta = last_time - best_time
        return delta

    def _get_last_sector_time_colour(self, sector: int) -> str:
        sector_time = self._get_last_sector_time(sector)
        best_time = self._get_best_sector_time(sector)
        return self._get_sector_colour(sector_time, best_time)

    def _get_sector_colour(self, sector_time: int, best_time: int) -> str:
        is_best_valid = best_time > 0
        if not is_best_valid:
            return WHITE
        is_better = sector_time < best_time - MINIMUM_DELTA_MS
        is_worse = sector_time > best_time + MINIMUM_DELTA_MS
        is_equal = sector_time == best_time
        if is_better:
            return PURPLE
        elif is_worse:
            return YELLOW
        elif is_equal:
            return WHITE
        return GREEN

    def _get_current_sector_time_colour(self, sector: int) -> str:
        is_finished_sector = self._current_sector > sector
        if not is_finished_sector:
            return WHITE
        sector_time = self._get_current_sector_time(sector)
        best_time = self._get_best_sector_time(sector)
        return self._get_sector_colour(sector_time, best_time)

    def __setup(self, agent: ElTuarMPC):
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
        self._previous_sector = 0
        self._previous_best_laptime = 0


def format_sector_time(sector_time_ms: int) -> str:
    if sector_time_ms == 0:
        time_string = "-"
    else:
        time_string = format_laptime(sector_time_ms)
    return time_string


def format_delta_time(delta_ms: int) -> str:
    if delta_ms == 0:
        return ""
    if delta_ms < 0:
        prefix = "-"
    else:
        prefix = "+"
    return f"{prefix}{format_laptime(delta_ms)}"


def format_laptime(laptime_ms: int) -> str:
    current_laptime = abs(laptime_ms) / 1000
    minutes = math.floor(current_laptime / 60)
    seconds = current_laptime % 60
    return f"{minutes:02d}:{seconds:06.3f}"
