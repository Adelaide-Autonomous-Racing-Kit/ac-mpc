import argparse

from agent import ElTuarMPC


def main():
    args = parse_arguments()
    agent = ElTuarMPC(args.config)
    agent.run()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to configuration")
    return parser.parse_args()


if __name__ == "__main__":
    main()
