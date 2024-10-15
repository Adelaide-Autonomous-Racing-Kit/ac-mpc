import argparse

from acmpc.localisation.benchmarking.benchmark_localisation import BenchmarkLocalisation
from loguru import logger
import numpy as np
from acmpc.utils import load


def main():
    args = parse_arguments()
    cfg = load.yaml(args.config)
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to configuration")
    return parser.parse_args()


if __name__ == "__main__":
    main()
