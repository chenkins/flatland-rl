import pytest
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy

from flatland.envs.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_wrappers import ray_multi_agent_env_wrapper
from flatland.envs.rail_generators import sparse_rail_generator


@pytest.mark.parametrize(
    "obs_builder_object",
    [
        pytest.param(
            DummyObservationBuilder(), id="DummyObservationBuilder"
        ),
        pytest.param(
            GlobalObsForRailEnv(), id="GlobalObsForRailEnv"
        ),
        pytest.param(
            FlattenTreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()), id="FlattenTreeObsForRailEnv_max_depth_2"
        ),
        pytest.param(
            FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()), id="FlattenTreeObsForRailEnv_max_depth_3"
        ),
    ],
)
def test_rail_env_wrappers(obs_builder_object):
    number_of_agents = 1
    n_cities = 2
    max_rails_between_cities = 2
    max_rails_in_city = 4
    seed = 0
    rail_env = RailEnv(
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
        number_of_agents=number_of_agents,
        obs_builder_object=obs_builder_object
    )
    rail_env.reset()

    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    env = ray_multi_agent_env_wrapper(wrap=rail_env)
    worker = RolloutWorker(
        env_creator=lambda _: env,
        config=AlgorithmConfig().experimental(_disable_preprocessor_api=True, _disable_execution_plan_api=True).multi_agent(
            # TODO can we not have shared policy?
            policies={
                f"main{aid}": (RandomPolicy, env.observation_space[aid], env.action_space[aid], {})
                for aid in env.get_agent_ids()
            },
            policy_mapping_fn=(
                lambda aid, episode, **kwargs: f"main{aid}"
            )
        )
    )
    print(worker.sample())
