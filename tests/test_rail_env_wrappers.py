from typing import Union, List, Optional, Dict, Tuple

import pytest
from ray.rllib import RolloutWorker, Policy
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.tune import run_experiments, register_env

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_wrappers import ray_multi_agent_env_wrapper
from flatland.envs.rail_generators import sparse_rail_generator

# TODO shortest_path_deadlock_avoidance_policy run evaluation
class ShortestPathDeadlockAvoidanceAlgorithm(Policy):
    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType], state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None, info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None, explore: Optional[bool] = None, timestep: Optional[int] = None, **kwargs) -> Tuple[
        TensorType, List[TensorType], Dict[str, TensorType]]:
        pass


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
            FlattenTreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), id="FlattenTreeObsForRailEnv_max_depth_2_50"
        ),
        pytest.param(
            FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), id="FlattenTreeObsForRailEnv_max_depth_3_50"
        ),
    ],
)
def test_rail_env_wrappers_rollout(obs_builder_object):
    env = _get_env(obs_builder_object)
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


def _get_env(obs_builder_object):
    number_of_agents = 10
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
    return env


@pytest.mark.parametrize(
    "obs_builder_object,algo",
    [
        pytest.param(
            obs_builder_object, algo, id=f"{obid}_{algo}"
        )
        for obs_builder_object, obid in
        [
            (DummyObservationBuilder(), "DummyObservationBuilder"),
            (GlobalObsForRailEnv(), "GlobalObsForRailEnv"),
            (FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), "FlattenTreeObsForRailEnv_max_depth_3_50")
        ]
        for algo in ['A2C', 'DQN', 'PPO']
    ]
)
def test_rail_env_wrappers_training(obs_builder_object, algo):
    register_env("my_env", lambda _: _get_env(obs_builder_object))

    run_experiments({
        f'test-{algo}': {
            'run': algo,
            'config': {
                'env': 'my_env',
                'exploration_final_eps': 0,
                'exploration_fraction': 0},
            'stop': {"training_iteration": 1}}
    })

# TODO 0.6 bei 1000 Episode
# TODO whole adrian zoo of policies run for one training epoch for illustration
