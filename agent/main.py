import multiprocessing as mp

from agent import ElTuarMPC


def main():
    agent = ElTuarMPC()
    agent.run()


if __name__ == "__main__":
    main()
