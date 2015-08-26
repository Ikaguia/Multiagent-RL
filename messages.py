STATE = 'State'
ACTION = 'Action'

class BaseMessage(object):
    def __init__(self, msg_type=None):
        self.msg_type = msg_type


class StateMessage(BaseMessage):
    def __init__(self, msg_type=None, index=None, pacman_position=None,
        ghost_positions=None, food_positions=None, legal_actions=None,
        reward=None):
        super(StateMessage, self).__init__(msg_type=msg_type)
        self.index = index
        self.pacman_position = pacman_position
        self.ghost_positions = ghost_positions
        self.food_positions = food_positions
        self.legal_actions = legal_actions
        self.reward = reward


class ActionMessage(BaseMessage):
    def __init__(self, msg_type=None, index=None, action=None):
        super(ActionMessage, self).__init__(msg_type=msg_type)
        self.index = index
        self.action = action