from typing import Callable

import pytest
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import AlgorithmConfig
from ray.tune import run_experiments, register_env

from benchmarks.ray_utils import get_env
from core.env_observation_builder import ObservationBuilder
from flatland.contrib.policies.DeadlockAvoidancePolicy import DeadLockAvoidancePolicy
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv


@pytest.mark.parametrize(
    "obs_builder_object",
    [
        # pytest.param(DeadLockAvoidancePolicy(), id="DeadLockAvoidancePolicy"),
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
def test_rail_env_wrappers_random_rollout(obs_builder_object: ObservationBuilder):
    env = get_env(obs_builder_object=obs_builder_object)
    worker = RolloutWorker(
        env_creator=lambda _: env,
        config=AlgorithmConfig().experimental(_disable_preprocessor_api=True, _disable_execution_plan_api=True).multi_agent(
            policies={
                f"main": (DeadLockAvoidancePolicy, env.observation_space[aid], env.action_space[aid], {})
                for aid in env.get_agent_ids()
            },
            policy_mapping_fn=(
                lambda aid, episode, **kwargs: f"main"
            )
        )
    )
    # TODO very dirty....
    for p in worker.policy_map.values():
        p.env = worker.env
        p.action_size = 5
    worker.sample()


@pytest.mark.parametrize(
    "obs_builder,algo",
    [
        pytest.param(
            obs_builder, algo, id=f"{obid}_{algo}"
        )
        for obs_builder, obid in
        [
            (lambda: DummyObservationBuilder(), "DummyObservationBuilder"),
            (lambda: GlobalObsForRailEnv(), "GlobalObsForRailEnv"),
            (lambda: FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), "FlattenTreeObsForRailEnv_max_depth_3_50")
        ]
        for algo in ['A2C', 'DQN', 'PPO']
    ]
)
def test_rail_env_wrappers_training(obs_builder: Callable[[], ObservationBuilder], algo: str):
    register_env("my_env", lambda _: get_env(obs_builder()))

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
# TODO whole adrian zoo of contrib.policies run for one training epoch for illustration


# https://flatland.aicrowd.com/challenges/flatland3/envconfig.html
# https://flatland.aicrowd.com/challenges/flatland3/test-submissions-local.html
