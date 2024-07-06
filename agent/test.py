from typing import Dict

import numpy as np
from loguru import logger

from aci.interface import AssettoCorsaInterface
from aci.utils.config import load_config


class TestAgent(AssettoCorsaInterface):
    def behaviour(self, observation: Dict) -> np.array:
        # Randomly generate an action [steering_angle, brake, throttle]
        action = np.random.rand(3)
        # Rescale steering angle to be between [-1., 1]
        action[0] = (action[0] - 0.5) * 2
        return action

    def teardown(self):
        pass

    def termination_condition(self, observation: Dict) -> bool:
        if observation["state"]["i_current_time"] > 12000:
            return True
        return False


if __name__ == "__main__":
    config = load_config()
    agent = TestAgent(config)
    agent.run()
    logger.info("Agent Terminated...")
    agent = TestAgent(config)
    agent.run()
    logger.info("Shutting Down...")
