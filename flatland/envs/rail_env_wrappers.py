import copy
from typing import Tuple, Set, Optional

import gymnasium as gym
from gymnasium.vector.utils import spaces
from ray.rllib import RolloutWorker, MultiAgentEnv
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict, AgentID

from core.env import Environment
from envs.rail_env import RailEnv
from envs.rail_env_action import RailEnvActions
from policy.random_policy import RandomPolicy


# TODO petting zoo wrapper
# TODO training with policy etc.? sample policies whole adrian zoo?


# TODO generalize to wrapping Environment instead of RailEnv?
class RayMultiAgentWrapper(MultiAgentEnv, Environment):

    def __init__(self, wrap: RailEnv):
        self.wrap: RailEnv = wrap
        self.action_space: gym.spaces.Dict = spaces.Dict({
            i: gym.spaces.Discrete(5)
            for i in range(self.wrap.number_of_agents)
        })

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(spaces={
            handle: self.wrap.obs_builder.get_observation_space(handle)
            for handle in self.get_agent_handles()
        })
        super().__init__()

        # Provide full (preferred format) observation- and action-spaces as Dicts
        # mapping agent IDs to the individual agents' spaces.
        self._spaces_in_preferred_format = True
        self._action_space_in_preferred_format = True

    @override(Environment)
    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:

        prev_dones = copy.deepcopy(self.wrap.dones)

        action_dict = {k: RailEnvActions(v) for k, v in action_dict.items()}
        obs, rewards, terminateds, infos = self.wrap.step(action_dict=action_dict)

        infos = {
            i:
                {
                    'action_required': infos['action_required'][i],
                    'malfunction': infos['malfunction'][i],
                    'speed': infos['speed'][i],
                    'state': infos['state'][i]
                } for i in self.get_agent_ids()
        }

        # report obs/done/info only once per agent per episode,
        # see https://github.com/ray-project/ray/issues/10761
        terminateds = copy.deepcopy(terminateds)
        for i in self.get_agent_ids():
            if prev_dones[i] is True:
                del obs[i]
                del terminateds[i]
                del infos[i]

        truncateds = {"__all__": False}
        return obs, rewards, terminateds, truncateds, infos

    @override(Environment)
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        return self.wrap.reset()

    @override(MultiAgentEnv)
    def get_agent_ids(self) -> Set[AgentID]:
        return set(self.get_agent_handles())

    @override(Environment)
    def get_agent_handles(self):
        return self.wrap.get_agent_handles()


def ray_multi_agent_env_wrapper(wrap: RailEnv) -> RayMultiAgentWrapper:
    return RayMultiAgentWrapper(wrap)


if __name__ == '__main__':
    number_of_agents = 5
    rail_env = RailEnv(30, 30, number_of_agents=number_of_agents)
    rail_env.reset()

    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    env = ray_multi_agent_env_wrapper(wrap=rail_env)
    worker = RolloutWorker(
        env_creator=lambda _: env,
        config=AlgorithmConfig().multi_agent(
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
