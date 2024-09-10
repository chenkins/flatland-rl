import logging

import numpy as np
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import AlgorithmConfig

from flatland.contrib.policies.DeadlockAvoidancePolicy import DeadLockAvoidancePolicy
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_wrappers import ray_multi_agent_env_wrapper
from flatland.envs.rail_generators import sparse_rail_generator

logger = logging.getLogger(__name__)


def _get_env(n_agents=10, x_dim=20, y_dim=30, n_cities=2):
    max_rails_between_cities = 2
    max_rails_in_city = 4
    seed = 0
    obs_builder_object = GlobalObsForRailEnv()

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
    # env = ray_multi_agent_env_wrapper(wrap=rail_env, render_mode="rgb_array")
    env = ray_multi_agent_env_wrapper(wrap=rail_env)
    return env


# class RenderCallback(DefaultCallbacks):
#     def __init__(self):
#         super().__init__()
#         self.env_renderer = None
#
#     def on_episode_step(
#         self,
#         **kwargs,
#     ) -> None:
#         if self.env_renderer is None:
#             # TODO dirty!
#             env = kwargs['base_env'].envs[0].wrap
#             self.env_renderer = RenderTool(env)
#
#         self.env_renderer.render_env(
#             show=True,
#             frames=False,
#             show_observations=False,
#             show_predictions=False
#         )


# https://flatland.aicrowd.com/challenges/flatland3/envconfig.html

def main(label="", n_agents=10, x_dim=20, y_dim=30, n_cities=2, n_envs_run=10):
    total_reward = 0
    for i_envs_run in range(n_envs_run):
        logger.info(f"{label} start {i_envs_run + 1}/{n_envs_run}")
        env = _get_env(n_agents=n_agents, x_dim=x_dim, y_dim=y_dim, n_cities=n_cities)
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
        logger.info(normalized_reward)
        total_reward += normalized_reward
    return total_reward


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(clientip)-15s %(user)-8s %(message)s')

    # https://flatland.aicrowd.com/challenges/flatland3/envconfig.html
    tot = 0
    n_envs_run = 10
    # tot += main("F3:R2:Test_00", n_agents=10, x_dim=20, y_dim=30, n_cities=2, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_01", n_agents=10, x_dim=30, y_dim=30, n_cities=2, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_02", n_agents=50, x_dim=30, y_dim=30, n_cities=3, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_03", n_agents=50, x_dim=30, y_dim=35, n_cities=3, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_04", n_agents=80, x_dim=35, y_dim=30, n_cities=5, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_05", n_agents=80, x_dim=45, y_dim=35, n_cities=7, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_06", n_agents=80, x_dim=40, y_dim=60, n_cities=9, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_07", n_agents=80, x_dim=60, y_dim=40, n_cities=13, n_envs_run=n_envs_run)
    # tot += main("F3:R2:Test_08", n_agents=80, x_dim=60, y_dim=60, n_cities=17, n_envs_run=n_envs_run)
    #tot += main("F3:R2:Test_09", n_agents=100, x_dim=80, y_dim=120, n_cities=21, n_envs_run=n_envs_run)
    tot += main("F3:R2:Test_10", n_agents=100, x_dim=100, y_dim=80, n_cities=25, n_envs_run=n_envs_run)
    tot += main("F3:R2:Test_11", n_agents=200, x_dim=100, y_dim=100, n_cities=29, n_envs_run=n_envs_run)
    tot += main("F3:R2:Test_12", n_agents=200, x_dim=150, y_dim=150, n_cities=33, n_envs_run=n_envs_run)
    tot += main("F3:R2:Test_13", n_agents=400, x_dim=150, y_dim=150, n_cities=37, n_envs_run=n_envs_run)
    tot += main("F3:R2:Test_14", n_agents=425, x_dim=158, y_dim=158, n_cities=41, n_envs_run=n_envs_run)
    logger.info("===============================================================")
    logger.info(tot / (15 * n_envs_run))

    # expectation:
    # https://www.aicrowd.com/challenges/flatland-3/leaderboards?challenge_leaderboard_extra_id=965&challenge_round_id=1083&post_challenge=true: adrian OR aka. deadlockavoidance: 30.134 	0.3620 	OR 	47

