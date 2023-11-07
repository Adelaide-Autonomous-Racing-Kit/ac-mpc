from localisation.localisation_bench import BenchmarkLocalisation
from utils import load


def main():
    cfg = load.yaml("agents/localisation/single_config.yaml")
    benchmark = BenchmarkLocalisation(cfg)

    benchmark.run()


if __name__ == "__main__":
    main()
