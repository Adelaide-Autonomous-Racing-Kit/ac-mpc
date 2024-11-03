import abc
import multiprocessing as mp
import signal
from typing import Dict


class WorkerProcess(mp.Process):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.daemon = True
        self.__setup(cfg)

    @property
    def is_running(self) -> bool:
        """
        Checks if the worker process is running

        :return: True if the process is running, false if it is not
        :rtype: bool
        """
        with self._is_running.get_lock():
            is_running = self._is_running.value
        return is_running

    @is_running.setter
    def is_running(self, is_running: bool):
        """
        Sets if the worker process is running

        :is_running: True if the process is running, false if it is not
        :type is_running: bool
        """
        with self._is_running.get_lock():
            self._is_running.value = is_running

    def run(self):
        """
        Called on WorkerProcess.start()
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._runtime_setup()
        while self.is_running:
            self._work()

    @abc.abstractmethod
    def _runtime_setup(self):
        """Implement any setup required before work loop is entered here"""
        pass

    @abc.abstractmethod
    def _work(self):
        """Implement core worker loop here"""
        pass

    def __setup(self, cfg: Dict):
        self._is_running = mp.Value("i", True)
        self._setup(cfg)

    @abc.abstractmethod
    def _setup(self, cfg: Dict):
        """
        Implement before runtime setup operations here
        """
        pass
