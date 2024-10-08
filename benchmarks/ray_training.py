import logging
from argparse import Namespace

import ray
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

from benchmarks.ray_utils import ray_env_creator


def setup_func():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s")


parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=1000000,
    default_reward=0.0,
)


def train(args: Namespace):
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"
    setup_func()
    ray.init(runtime_env={
        "conda": "environment.yaml",
        "working_dir": ".",
        "excludes": ["notebooks/", ".git/", ".tox/", ".venv/", "docs/", ".idea", "tmp"],
        "env_vars": {
            "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1",
        },
        "worker_process_setup_hook": setup_func
    })
    try:
        env_name = "flatland_10_20_30_2"
        register_env(env_name, lambda _: ray_env_creator(n_agents=args.num_agents))
        # TODO is this what we want?
        # Policies are called just like the agents (exact 1:1 mapping).
        policies = {str(i) for i in range(args.num_agents)}
        base_config = (
            get_trainable_cls(args.algo)
            .get_default_config()
            .environment("flatland_10_20_30_2")
            .multi_agent(
                policies=policies,
                # Exact 1:1 mapping from AgentID to ModuleID.
                policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
            )
            .training(
                # vf_loss_coeff=0.005,
            )
            .rl_module(
                model_config_dict={"vf_share_layers": True},
                rl_module_spec=MultiRLModuleSpec(
                    module_specs={p: RLModuleSpec() for p in policies},
                ),
            )

        )
        run_rllib_example_script_experiment(base_config, args)
    except Exception as e:
        ray.shutdown()
        raise e


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
