import logging
import sys

import click
import numpy as np
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import AlgorithmConfig

from benchmarks.ray_utils import get_env
from flatland.contrib.policies.DeadlockAvoidancePolicy import DeadLockAvoidancePolicy

logger = logging.getLogger()


# https://flatland.aicrowd.com/challenges/flatland3/envconfig.html

def main(label="", n_agents=10, x_dim=20, y_dim=30, n_cities=2, n_envs_run=10, seed=None):
    total_reward = 0
    for i_envs_run in range(n_envs_run):
        logger.info(f"{label} start {i_envs_run + 1}/{n_envs_run}")
        env = get_env(n_agents=n_agents, x_dim=x_dim, y_dim=y_dim, n_cities=n_cities, seed=seed)
        worker = RolloutWorker(
            env_creator=lambda _: env,
            config=AlgorithmConfig()
            .experimental(_disable_preprocessor_api=True, _disable_execution_plan_api=True)
            .multi_agent(
                policies={
                    f"main": (DeadLockAvoidancePolicy, env.observation_space[aid], env.action_space[aid], {})
                    for aid in env.get_agent_ids()
                },
                policy_mapping_fn=(
                    lambda aid, episode, **kwargs: f"main"
                ),
            )
            .rollouts(num_envs_per_worker=1, batch_mode='complete_episodes')
            # .callbacks(RenderCallback)
        )
        # TODO very dirty....
        for p in worker.policy_map.values():
            p.env = worker.env
            p.action_size = 5
        batch = worker.sample()

        # get first complete episode
        # TODO verify either max_episode_steps reached or done (complete
        cumulative_reward = np.sum(batch['main']['rewards'][np.argwhere(batch['main']['eps_id'] == batch['main']['eps_id'][0])])
        logger.info(f"{label} end {i_envs_run + 1}/{n_envs_run}:")
        # https://flatland.aicrowd.com/challenges/flatland3/eval.html
        normalized_reward = (cumulative_reward / (env.wrap._max_episode_steps * env.wrap.get_num_agents())) + 1
        logger.info(f"{label}: normalized_reward={normalized_reward}")
        total_reward += normalized_reward
    return total_reward


# https://flatland.aicrowd.com/challenges/flatland3/envconfig.html
# TODO check exact specs: malfunctions etc. seeds? also max_rails_between_cities etc.
configs = {
    "F3:R2:Test_00": [10, 20, 30, 2],
    "F3:R2:Test_01": [10, 30, 30, 2],
    "F3:R2:Test_02": [50, 30, 30, 3],
    "F3:R2:Test_03": [50, 30, 35, 3],
    "F3:R2:Test_04": [80, 35, 30, 5],
    "F3:R2:Test_05": [80, 45, 35, 7],
    "F3:R2:Test_06": [80, 40, 60, 9],
    "F3:R2:Test_07": [80, 60, 40, 13],
    "F3:R2:Test_08": [80, 60, 60, 17],
    "F3:R2:Test_09": [100, 80, 120, 21],
    "F3:R2:Test_10": [100, 100, 80, 25],
    "F3:R2:Test_11": [200, 100, 100, 29],
    "F3:R2:Test_12": [200, 150, 150, 33],
    "F3:R2:Test_13": [400, 150, 150, 37],
    "F3:R2:Test_14": [425, 158, 158, 41]
}


@click.command()
@click.option('--benchmark',
              type=str,
              help="Name of the benchmark",
              required=True
              )
@click.option('--n_envs_run',
              type=int,
              help="Number of envs",
              default=1,
              required=False
              )
def benchmark(benchmark: str, n_envs_run: int):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s")
    main(benchmark, *configs[benchmark], n_envs_run=n_envs_run)

    # expectation:
    # https://www.aicrowd.com/challenges/flatland-3/leaderboards?challenge_leaderboard_extra_id=965&challenge_round_id=1083&post_challenge=true: adrian OR aka. deadlockavoidance: 30.134 	0.3620 	OR 	47


if __name__ == "__main__":
    sys.exit(benchmark())
