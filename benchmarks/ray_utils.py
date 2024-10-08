from typing import Optional

from flatland.envs.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_wrappers import ray_multi_agent_env_wrapper
from flatland.envs.rail_generators import sparse_rail_generator


def ray_env_creator(n_agents=10, x_dim=20, y_dim=30, n_cities=2, seed=None, obs_builder_object=None, render_mode: Optional[str] = None):
    max_rails_between_cities = 2
    max_rails_in_city = 4

    if obs_builder_object is None:
        obs_builder_object = FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))

    rail_env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        obs_builder_object=obs_builder_object
    )
    # install agents!
    rail_env.reset()
    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    env = ray_multi_agent_env_wrapper(wrap=rail_env, render_mode=render_mode)
    return env
