from typing import Callable

import pytest
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.utils.test_utils import add_rllib_example_script_args

from benchmarks.ray_training import train
from benchmarks.ray_utils import ray_env_creator
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
    env = ray_env_creator(obs_builder_object=obs_builder_object)
    worker = RolloutWorker(
        env_creator=lambda _: env,
        config=AlgorithmConfig().experimental(_disable_preprocessor_api=True).multi_agent(
            policies={
                f"main": (DeadLockAvoidancePolicy, env.observation_space["0"], env.action_space["0"], {})
            },
            policy_mapping_fn=(
                lambda aid, episode, **kwargs: f"main"
            )
        )
    )
    # TODO very dirty.... deadlockavoidancepolicy specific
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
            # TODO training with all obs builders
            # (lambda: DummyObservationBuilder(), "DummyObservationBuilder"),
            # (lambda: GlobalObsForRailEnv(), "GlobalObsForRailEnv"),
            (lambda: FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), "FlattenTreeObsForRailEnv_max_depth_3_50")
        ]
        for algo in ["DQN", "PPO"]
    ]
)
def test_rail_env_wrappers_training(obs_builder: Callable[[], ObservationBuilder], algo: str):
    parser = add_rllib_example_script_args()
    train(parser.parse_args(["--algo", algo, "--num-agents", "2", "--enable-new-api-stack", "--stop-iters", "1"]))

# TODO 0.6 bei 1000 Episode
# TODO whole adrian zoo of contrib.policies run for one training epoch for illustration
# https://flatland.aicrowd.com/challenges/flatland3/envconfig.html
# https://flatland.aicrowd.com/challenges/flatland3/test-submissions-local.html
