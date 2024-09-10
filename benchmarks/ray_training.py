from ray.tune import register_env, run_experiments

from benchmarks.ray_utils import get_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune

# TODO add click click cli
if __name__ == '__main__':
    register_env("flatland_10_20_30_2", lambda _: get_env())

    config = (
        PPOConfig()
        # Set the config object's env.
        .environment(env="flatland_10_20_30_2")
        # Update the config object's training parameters.
        .training(
            lr=0.001, clip_param=0.2
        )
    )

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(stop={"training_iteration": 100}),
        param_space=config,
    ).fit()
