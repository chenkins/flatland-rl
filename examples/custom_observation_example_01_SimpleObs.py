import random

import gymnasium as gym
import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from utils.decorators import candidate_for_deletion

random.seed(100)
np.random.seed(100)


@candidate_for_deletion
class SimpleObs(ObservationBuilder):
    """
    Simplest observation builder. The object returns observation vectors with 5 identical components,
    all equal to the ID of the respective agent.
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        return

    def get(self, handle: int = 0) -> np.ndarray:
        observation = handle * np.ones((5,))
        return observation

    def get_observation_space(self, handle: int = 0):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,))


@candidate_for_deletion
def create_env():
    nAgents = 3
    n_cities = 2
    max_rails_between_cities = 2
    max_rails_in_city = 4
    seed = 0
    env = RailEnv(
        width=20,
        height=30,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=nAgents,
        obs_builder_object=SimpleObs()
    )
    return env


@candidate_for_deletion
def main():
    env = create_env()
    env.reset()

    # Print the observation vector for each agents
    obs, all_rewards, done, _ = env.step({0: 0})
    for i in range(env.get_num_agents()):
        print("Agent ", i, "'s observation: ", obs[i])
        print("Agent ", i, "'s sampled observation: ", env.obs_builder.get_observation_space(i).sample())


if __name__ == '__main__':
    main()
