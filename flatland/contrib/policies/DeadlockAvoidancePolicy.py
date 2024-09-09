from typing import Union, List, Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from ray.rllib import Policy, SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType, AlgorithmConfigDict

from flatland.contrib.policies.utils.deadlock_avoidance_policy import DeadlockAvoidanceShortestDistanceWalker
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState


# TODO do we need a policy abstraction ray/pettingzoo?
class DeadLockAvoidancePolicy(Policy):
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 config: AlgorithmConfigDict,
                 min_free_cell=1,
                 enable_eps=False,
                 show_debug_plot=False):

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )
        # self.env: RailEnv = env

        self.loss = 0
        # self.action_size = action_size
        self.agent_can_move = {}
        self.show_debug_plot = show_debug_plot
        self.enable_eps = enable_eps
        self.shortest_distance_walker: Union[DeadlockAvoidanceShortestDistanceWalker, None] = None
        self.min_free_cell = min_free_cell
        self.agent_positions = None

    def compute_actions_from_input_dict(
        self,
        input_dict: Union[SampleBatch, Dict[str, TensorStructType]],
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        episodes: Optional[List["Episode"]] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        # pre-compute deadlock avoidance data
        self.start_step()
        return super().compute_actions_from_input_dict(
            input_dict,
            explore,
            timestep,
            episodes,
            agent_index=input_dict['agent_index'].tolist(),
            **kwargs
        )

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None, **kwargs
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        # only active agents
        return [self.act(x) for x in kwargs['agent_index']], [], {}

    def act(self, handle, eps=0.):
        # Epsilon-greedy action selection
        if self.enable_eps:
            if np.random.random() < eps:
                return np.random.choice(np.arange(self.action_size))

        check = self.agent_can_move.get(handle, None)
        act = RailEnvActions.STOP_MOVING
        if check is not None:
            act = check[3]
        return act

    # TODO reset?
    # def reset(self, env: Environment):
    #     self.env = env.get_raw_env()
    #     if self.shortest_distance_walker is not None:
    #         self.shortest_distance_walker.reset(self.env)
    #     self.shortest_distance_walker = None
    #     self.agent_positions = None
    #     self.shortest_distance_walker = None

    def start_step(self):
        self._build_agent_position_map()
        self._shortest_distance_mapper()
        self._extract_agent_can_move()

    def _build_agent_position_map(self):
        # build map with agent positions (only active agents)
        # TODO dirty hack - can we assume wrapper? streamline type hints at least
        self.agent_positions = np.zeros((self.env.wrap.height, self.env.wrap.width), dtype=int) - 1
        for handle in range(self.env.wrap.get_num_agents()):
            agent = self.env.wrap.agents[handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                if agent.position is not None:
                    self.agent_positions[agent.position] = handle

    def _shortest_distance_mapper(self):
        if self.shortest_distance_walker is None:
            self.shortest_distance_walker = DeadlockAvoidanceShortestDistanceWalker(self.env.wrap)
        self.shortest_distance_walker.clear(self.agent_positions)
        for handle in range(self.env.wrap.get_num_agents()):
            agent = self.env.wrap.agents[handle]
            if agent.state <= TrainState.MALFUNCTION:
                self.shortest_distance_walker.walk_to_target(handle)

    def _extract_agent_can_move(self):
        self.agent_can_move = {}
        shortest_distance_agent_map, full_shortest_distance_agent_map = self.shortest_distance_walker.getData()
        for handle in range(self.env.wrap.get_num_agents()):
            agent = self.env.wrap.agents[handle]
            if agent.state < TrainState.DONE:
                if self._check_agent_can_move(handle,
                                              shortest_distance_agent_map[handle],
                                              self.shortest_distance_walker.same_agent_map.get(handle, []),
                                              self.shortest_distance_walker.opp_agent_map.get(handle, []),
                                              full_shortest_distance_agent_map):
                    next_position, next_direction, action, _ = self.shortest_distance_walker.walk_one_step(handle)
                    self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})

        if self.show_debug_plot:
            a = np.floor(np.sqrt(self.env.wrap.get_num_agents()))
            b = np.ceil(self.env.wrap.get_num_agents() / a)
            for handle in range(self.env.wrap.get_num_agents()):
                plt.subplot(a, b, handle + 1)
                plt.imshow(full_shortest_distance_agent_map[handle] + shortest_distance_agent_map[handle])
            plt.show(block=False)
            plt.pause(0.01)

    def _check_agent_can_move(self,
                              handle,
                              my_shortest_walking_path,
                              same_agents,
                              opp_agents,
                              full_shortest_distance_agent_map):
        agent_positions_map = (self.agent_positions > -1).astype(int)
        len_opp_agents = len(opp_agents)
        for opp_a in opp_agents:
            opp = full_shortest_distance_agent_map[opp_a]
            delta = ((my_shortest_walking_path - opp - agent_positions_map) > 0).astype(int)
            sum_delta = np.sum(delta)
            if sum_delta < (self.min_free_cell + len_opp_agents):
                return False
        return True
