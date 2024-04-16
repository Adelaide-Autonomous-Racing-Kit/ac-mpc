import numpy as np
from loguru import logger

from utils import load
from localisation.benchmarking.benchmark_localisation import BenchmarkLocalisation


def main():
    cfg = load.yaml("agent/localisation/benchmarking/config.yaml")
    benchmarker = BenchmarkLocalisation(cfg)

    logger.info("Running localisation benchmark")
    benchmarker.run()

    logger.success(
        f"Percentage of time localised: {benchmarker._tracker.percentage_of_steps_localised_for()}%"
    )
    logger.success(
        f"Average position error: {benchmarker._tracker.average_position_error():.2f} meters"
    )
    logger.success(
        f"Average rotation error: {benchmarker._tracker.average_rotation_error() * 180/np.pi:.2f} degrees"
    )


if __name__ == "__main__":
    main()
