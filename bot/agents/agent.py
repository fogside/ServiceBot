from copy import deepcopy


class Agent:
    def __init__(self, content_manager):
        self.content_manager = content_manager

    def next(self):
        pass


class EchoAgent(Agent):
    def __init__(self, content_manager):
        super().__init__(content_manager)

    def next(self):
        return 'привет от агента'


class RuleAgent(Agent):
    def __init__(self, content_manager):
        super().__init__(content_manager)

    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """
        self.state = {
            'inform_slots': dict(),
            'request_slots': set(),
            'proposed_slots': dict(),  # Значение: [значение слота, был ли Negate, был ли ReqAlts]
            'agent_request_slots': dict(),
        }
        self.history = []

    def _next(self):
        if self.turn_count == 0:
            return [['welcomemsg', None, None]]

        return [['welcomemsg', None, None]]

    def next(self):
        actions = self._next()
        self._update_state_agent(actions)
        return actions
        
    @property
    def turn_count(self):
        return len(self.history)

    def _update_state_agent(self, agent_actions, nl=None):
        """
        :param agent_actions: list of triples: (agent action, slot_name, slot_value)
        :param nl: Natural language string from agent
        """

        for action, slot_name, slot_value in agent_actions:
            if action == 'welcomemsg':
                continue

        self.history.append({
            'state_agent': deepcopy(self.state),
            'agent_action': agent_actions,
            'user_action': None,
            'state_user': None,
            'agent_nl': nl,
            'user_nl': None
        })

    def update_state_user(self, user_actions, nl=None):
        """
        :param user_actions: list of triples: (agent action, slot_name, slot_value)
        :param nl: Natural language string from user
        """

        self.history[-1]['user_action'] = user_actions
        self.history[-1]['user_action'] = deepcopy(self.state)
        self.history[-1]['user_nl'] = nl
