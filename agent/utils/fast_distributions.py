import numpy as np
from scipy.stats import norm
from loguru import logger


class FastNormalDistribution:
    def __init__(self, location: float, scale: float):
        self._location = location
        self._scale = scale
        self._distribution = norm

    def pdf(self, samples: np.array) -> np.array:
        samples -= self._location
        samples /= self._scale
        pdf_values = self._distribution._pdf(samples)
        pdf_values /= self._scale
        return pdf_values
