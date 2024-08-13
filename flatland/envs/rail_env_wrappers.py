import copy
from typing import Tuple, Set, Optional

import gymnasium as gym
from ray.rllib import RolloutWorker, MultiAgentEnv
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.utils.typing import MultiAgentDict, AgentID

from core.env import Environment
from envs.rail_env import RailEnv
from envs.rail_env_action import RailEnvActions
from policy.random_policy import RandomPolicy


class RayMultiAgentWrapper(MultiAgentEnv, Environment):

    # TODO generalize to wrapping Environment instead of RailEnv?
    def __init__(self, wrap: RailEnv):
        self.wrap = wrap
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

        infos = {i:
            {
                'action_required': infos['action_required'][i],
                'malfunction': infos['malfunction'][i],
                'speed': infos['speed'][i],
                'state': infos['state'][i]
            } for i in range(self.wrap.get_num_agents())
        }

        # report obs/done/info only once per agent per episode,
        # see https://github.com/ray-project/ray/issues/10761
        terminateds = copy.deepcopy(terminateds)
        # TODO us get agent ids instead?
        for i in range(self.wrap.get_num_agents()):
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
        # TODO typing
        return self.wrap.reset()

    @property
    @override(Environment)
    def observation_space(self) -> gym.spaces.Dict:
        return self.wrap.observation_space

    @property
    @override(Environment)
    def action_space(self) -> gym.spaces.Dict:
        return self.wrap.action_space

    @override(MultiAgentEnv)
    def get_agent_ids(self) -> Set[AgentID]:
        return set(range(self.wrap.get_num_agents()))

    @override(MultiAgentEnv)
    def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        obs = self.observation_space.sample()
        return obs

    @override(MultiAgentEnv)
    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        # TODO use get_agent_ids?
        actions = self.action_space.sample()
        return actions

    @override(MultiAgentEnv)
    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        # TODO cleanup
        return True  # all(self.action_space.contains(val) for val in x.values())

    @override(MultiAgentEnv)
    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return True
        # TODO cleanup
        # return all(self.observation_space.contains(val) for val in x.values())


def ray_multi_agent_env_wrapper(env):
    return RayMultiAgentWrapper(env)


if __name__ == '__main__':
    number_of_agents = 5
    env = RailEnv(30, 30, number_of_agents=number_of_agents)
    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    worker = RolloutWorker(
        env_creator=lambda _: ray_multi_agent_env_wrapper(env=env),
        config=AlgorithmConfig().multi_agent(
            # TODO can we not have shared policy?
            policies={
                f"main{aid}": (RandomPolicy, env.observation_space[aid], env.action_space[aid], {})
                for aid in range(number_of_agents)
            },
            policy_mapping_fn=(
                lambda aid, episode, **kwargs: f"main{aid}"
            )
        )
    )
    print(worker.sample())
