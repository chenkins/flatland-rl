# Copied from https://github.com/aiAdrian/flatland_solver_policy/blob/main/policy/heuristic_policy/shortest_path_deadlock_avoidance_policy/deadlock_avoidance_policy.py

from functools import lru_cache

import numpy as np

from flatland.contrib.policies.utils.shortest_distance_walker import ShortestDistanceWalker
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.rail_env import RailEnv

# activate LRU caching
flatland_deadlock_avoidance_policy_lru_cache_functions = []


def _enable_flatland_deadlock_avoidance_policy_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        flatland_deadlock_avoidance_policy_lru_cache_functions.append(func)
        return func

    return decorator


def _send_flatland_deadlock_avoidance_policy_data_change_signal_to_reset_lru_cache():
    for func in flatland_deadlock_avoidance_policy_lru_cache_functions:
        func.cache_clear()


class DeadlockAvoidanceShortestDistanceWalker(ShortestDistanceWalker):
    def __init__(self, env: RailEnv):
        super().__init__(env)
        self.shortest_distance_agent_map = None
        self.full_shortest_distance_agent_map = None
        self.agent_positions = None
        self.opp_agent_map = {}
        self.same_agent_map = {}

    def reset(self, env: RailEnv):
        super(DeadlockAvoidanceShortestDistanceWalker, self).reset(env)
        self.shortest_distance_agent_map = None
        self.full_shortest_distance_agent_map = None
        self.agent_positions = None
        self.opp_agent_map = {}
        self.same_agent_map = {}
        _send_flatland_deadlock_avoidance_policy_data_change_signal_to_reset_lru_cache()

    def clear(self, agent_positions):
        self.shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                     self.env.height,
                                                     self.env.width),
                                                    dtype=int) - 1

        self.full_shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                          self.env.height,
                                                          self.env.width),
                                                         dtype=int) - 1

        self.agent_positions = agent_positions

        self.opp_agent_map = {}
        self.same_agent_map = {}

    def getData(self):
        return self.shortest_distance_agent_map, self.full_shortest_distance_agent_map

    def callback(self, handle, agent, position, direction, action, possible_transitions) -> bool:
        opp_a = self.agent_positions[position]
        if opp_a != -1 and opp_a != handle:
            if self.env.agents[opp_a].direction != direction:
                d = self.opp_agent_map.get(handle, [])
                if opp_a not in d:
                    d.append(opp_a)
                self.opp_agent_map.update({handle: d})
            else:
                if len(self.opp_agent_map.get(handle, [])) == 0:
                    d = self.same_agent_map.get(handle, [])
                    if opp_a not in d:
                        d.append(opp_a)
                    self.same_agent_map.update({handle: d})

        if len(self.opp_agent_map.get(handle, [])) == 0:
            if self._is_no_switch_cell(position):
                self.shortest_distance_agent_map[(handle, position[0], position[1])] = 1
        self.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 1
        return True

    @_enable_flatland_deadlock_avoidance_policy_lru_cache(maxsize=100000)
    def _is_no_switch_cell(self, position) -> bool:
        for new_dir in range(4):
            possible_transitions = self.env.rail.get_transitions(*position, new_dir)
            num_transitions = fast_count_nonzero(possible_transitions)
            if num_transitions > 1:
                return False
        return True


# define Python user-defined exceptions
class InvalidRawEnvironmentException(Exception):
    def __init__(self, env, message="This policy works only with a RailEnv or its specialized version. "
                                    "Please check the raw_env . "):
        self.env = env
        self.message = message
        super().__init__(self.message)
