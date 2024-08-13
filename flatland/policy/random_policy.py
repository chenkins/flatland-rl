from typing import Any, Union, List, Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
import tree
from ray.rllib.policy import Policy
from ray.rllib.utils import override
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorStructType, TensorType


# TODO ray only?
class RandomPolicy(Policy):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: AlgorithmConfigDict,
    ):
        super().__init__(observation_space, action_space, config)

    @override
    def compute_single_action(self, obs: Any = None, state: Any = None, **kwargs) -> Any:
        return np.random.randint(0, 5)

    # @override
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        obs_batch_size = len(tree.flatten(obs_batch)[0])
        sample = [gym.spaces.Discrete(5).sample() for _ in range(obs_batch_size)]
        return (
            sample,
            [],
            {},
        )
