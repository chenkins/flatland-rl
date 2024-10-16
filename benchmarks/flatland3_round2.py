import ast
import datetime
import json
import logging
import os
import sys

import click
import pandas as pd
from ray.rllib import RolloutWorker, BaseEnv
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import flatland
from benchmarks.ray_utils import ray_env_creator
from flatland.contrib.policies.DeadlockAvoidancePolicy import DeadLockAvoidancePolicy
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_wrappers import ray_multi_agent_env_wrapper

# TODO grrr Flatland must be installed as package...
flatland.__version__ = None

from flatland.evaluators.service import FlatlandRemoteEvaluationService, RANDOM_SEED

logger = logging.getLogger()


class LightEvaluationService(DefaultCallbacks, FlatlandRemoteEvaluationService):
    def __init__(self, results_output_path: str, test_env_folder: str):
        super(LightEvaluationService, self).__init__(result_output_path=results_output_path, test_env_folder=test_env_folder)
        self.first_episode_done = False

    def on_episode_step(
        self,
        *,
        base_env: BaseEnv,
        **kwargs,
    ):
        print(f"on_episode_step {self.current_test, self.current_level, self.current_step} [{self.first_episode_done}]")
        # get action from ray rollout
        _raw_rail_env: RailEnv = base_env._unwrapped_env.wrap
        action = _raw_rail_env.list_actions[-1]  # record_steps=True

        print(_raw_rail_env.agents)

        # TODO how to run only one episode with ray?
        if not self.first_episode_done:
            self.handle_env_step({"payload": {
                "action": action,
                "inference_time": 0
            }})
            assert len(_raw_rail_env.agents) == len(self.env.agents)

            # TODO ray does an additional reset without seed.... Grr!
            for a, b in zip(_raw_rail_env.agents, self.env.agents):
                assert a.handle == b.handle, (a, b)
                assert a.initial_position == b.initial_position, (a, b)
                assert a.initial_direction == b.initial_direction, (a, b)
                assert a.target == b.target, (a, b)
                assert a.earliest_departure == b.earliest_departure, (a, b)
                assert a.latest_arrival == b.latest_arrival, (a, b)
                assert a.latest_arrival == b.latest_arrival, (a, b)
                assert a.direction == b.direction, (a, b)
                assert a.position == b.position, (a, b)
                assert a.state == b.state, (a, b)

            if self.env.dones['__all__']:
                print(f"--> done {self.current_test, self.current_level, self.current_step}")
                self.first_episode_done = True

    def send_response(self, _command_response, command, suppress_logs=False):
        pass


# https://flatland.aicrowd.com/challenges/flatland3/envconfig.html: n_agents, x_dim, y_dim, n_cities
@click.command()
@click.option('--tests',
              type=click.Path(exists=True),
              help="Path to folder containing Flatland tests",
              required=True
              )
@click.option('--gen_pkl',
              type=bool,
              help="Name of the benchmark",
              default=False,
              required=False
              )
@click.option('--results_path',
              type=click.Path(exists=True),
              default=None,
              help="Path where the evaluator should write the results metadata.",
              required=False
              )
def benchmark(tests, gen_pkl: bool, results_path):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s")
    test_env_folder = tests

    results_output_path_json = None
    if results_path:
        results_name = f"results_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.csv"

        results_output_path_json = results_name.replace(".csv", ".json")
        results_output_path_json = os.path.join(
            results_path,
            results_output_path_json
        )
        results_path = os.path.join(
            results_path,
            results_name
        )

    evaluator = LightEvaluationService(results_output_path=results_path, test_env_folder=test_env_folder)

    evaluator.instantiate_evaluation_metadata()
    metadata: pd.DataFrame = evaluator.evaluation_metadata_df
    if gen_pkl:
        for _, row in metadata.iterrows():
            test_id = row['test_id']
            env_id = row['env_id']
            n_agents = row['n_agents']
            x_dim = row['x_dim']
            y_dim = row['y_dim']
            n_cities = row['n_cities']
            max_rail_pairs_in_city = row['max_rail_pairs_in_city']
            grid_mode = row['grid_mode']
            max_rails_between_cities = row['max_rails_between_cities']
            malfunction_duration_min = row['malfunction_duration_min']
            malfunction_duration_max = row['malfunction_duration_max']
            malfunction_interval = row['malfunction_interval']
            speed_ratios = ast.literal_eval(row['speed_ratios'])
            seed = row['seed']
            env = ray_env_creator(
                n_agents=n_agents,
                x_dim=x_dim,
                y_dim=y_dim,
                n_cities=n_cities,
                max_rail_pairs_in_city=max_rail_pairs_in_city,
                grid_mode=grid_mode,
                max_rails_between_cities=max_rails_between_cities,
                malfunction_duration_min=malfunction_duration_min,
                malfunction_duration_max=malfunction_duration_max,
                malfunction_interval=malfunction_interval,
                speed_ratios=speed_ratios,
                seed=seed
            )
            os.makedirs(test_env_folder, exist_ok=True)
            os.makedirs(os.path.join(test_env_folder, test_id), exist_ok=True)
            pkl = os.path.join(test_env_folder, test_id, f"{env_id}.pkl")
            if os.path.exists(pkl):
                print(f"{pkl} already exists. Remove manually and re-run.")
                sys.exit(2)
            print(f"Gen pkl {pkl}")
            RailEnvPersister.save(env.wrap, pkl)
        return
    for meta in metadata.iterrows():
        print("=================================================")
        print("=================================================")
        print("=================================================")
        print(f"meta {meta[0]}")
        print("=================================================")
        print("=================================================")
        print("=================================================")
        evaluator.handle_env_create(command={})

        if evaluator.evaluation_done:
            break

        # TODO hacky todonow
        evaluator.first_episode_done = False

        test_env_file_path = os.path.join(
            evaluator.test_env_folder,
            evaluator.simulation_env_file_paths[-1]
        )

        raw_env, _ = RailEnvPersister.load_new(test_env_file_path)
        _ = raw_env.reset(
            regenerate_rail=True,
            regenerate_schedule=True,
            random_seed=RANDOM_SEED
        )

        assert len(raw_env.agents) == len(evaluator.env.agents)

        # TODO remove?
        for a, b in zip(raw_env.agents, evaluator.env.agents):
            assert a.handle == b.handle, (a, b)
            assert a.initial_position == b.initial_position, (a, b)
            assert a.initial_direction == b.initial_direction, (a, b)
            assert a.target == b.target, (a, b)
            assert a.earliest_departure == b.earliest_departure, (a, b)
            assert a.latest_arrival == b.latest_arrival, (a, b)
            assert a.latest_arrival == b.latest_arrival, (a, b)
            assert a.direction == b.direction, (a, b)
            assert a.position == b.position, (a, b)
            assert a.state == b.state, (a, b)

        assert evaluator.get_env_test_and_level(test_env_file_path) == (evaluator.current_test, evaluator.current_level)

        env = ray_multi_agent_env_wrapper(wrap=raw_env)

        # TODO old api stack
        worker = RolloutWorker(
            env_creator=lambda _: env,
            config=AlgorithmConfig()
            .multi_agent(
                policies={
                    f"main": (DeadLockAvoidancePolicy, env.observation_space[aid], env.action_space[aid], {})
                    for aid in env.get_agent_ids()
                },
                policy_mapping_fn=(
                    lambda aid, episode, **kwargs: f"main"
                ),
            )
            # TODO deprecated api
            .rollouts(num_envs_per_worker=1, batch_mode='complete_episodes')
            .callbacks(
                # TODO inject dir
                lambda: evaluator
            )
        )
        # TODO very dirty....
        for p in worker.policy_map.values():
            p.env = worker.env
            p.action_size = 5
        worker.sample()

        if evaluator.evaluation_done:
            break
        return

    results = evaluator.handle_env_submit({"payload": {}})
    if results_output_path_json:
        with open(results_output_path_json, "w") as f:
            f.write(json.dumps(results, indent=4))
            print(f"Wrote results to {results_output_path_json}")


# TODO cleanup documentation
# expectation:
# https://www.aicrowd.com/challenges/flatland-3/leaderboards?challenge_leaderboard_extra_id=965&challenge_round_id=1083&post_challenge=true: adrian OR aka. deadlockavoidance: 30.134 	0.3620 	OR 	47
# https://www.aicrowd.com/challenges/flatland-3/submissions/169916
#
#
# Total Reward (primary score) -> sum_normalized_reward
# 30.134
# % Done  (secondary score)-> mean_percentage_complete
# 0.362
# normalized_reward -> mean_normalized_reward
# 0.75335
# reward -> mean reward????
# -2222.0
# percentage_complete -> -> mean_percentage_complete
# 0.362


if __name__ == "__main__":
    # blup = RailEnvPersister.load_new("/Users/che/Documents/job/FLATland/Flatland3_SBB/ENV_PKLS/Test_0/Level_0.pkl")
    sys.exit(benchmark())
