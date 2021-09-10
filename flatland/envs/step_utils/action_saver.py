from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

class ActionSaver:
    def __init__(self):
        self.saved_action = None

    @property
    def is_action_saved(self):
        return self.saved_action is not None
    
    def __repr__(self):
        return f"is_action_saved: {self.is_action_saved}, saved_action: {self.saved_action}"


    def save_action_if_allowed(self, action, state):
        if action.is_moving_action() and \
               not self.is_action_saved and \
               not state.is_malfunction_state() and \
               not state == TrainState.DONE:
            self.saved_action = action

    def clear_saved_action(self):
        self.saved_action = None

    def to_dict(self):
        return {"saved_action": self.saved_action}
    
    def from_dict(self, load_dict):
        self.saved_action = load_dict['saved_action']


