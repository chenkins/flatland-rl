from typing import Optional

from flatland.envs.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_wrappers import ray_multi_agent_env_wrapper
from flatland.envs.rail_generators import sparse_rail_generator


# defaults from Flatland 3 Round 2 Test_0
def ray_env_creator(n_agents=7,
                    x_dim=30,
                    y_dim=30,
                    n_cities=2,
                    max_rail_pairs_in_city=4,
                    # n_envs
                    # seed
                    grid_mode=False,
                    max_rails_between_cities=2,
                    malfunction_duration_min=20,
                    malfunction_duration_max=50,
                    malfunction_interval=540,
                    speed_ratios={1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25},
                    seed=42,
                    obs_builder_object=None,
                    render_mode: Optional[str] = None):
    if obs_builder_object is None:
        obs_builder_object = FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50))

    rail_env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city
        ),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(
            min_duration=malfunction_duration_min, max_duration=malfunction_duration_max, malfunction_rate=1.0 / malfunction_interval)),
        line_generator=sparse_line_generator(speed_ratio_map=speed_ratios, seed=seed),
        number_of_agents=n_agents,
        obs_builder_object=obs_builder_object,
        record_steps=True
    )
    # install agents!
    rail_env.reset()
    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    env = ray_multi_agent_env_wrapper(wrap=rail_env, render_mode=render_mode)
    return env
