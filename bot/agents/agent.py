from copy import deepcopy


class Agent:
    def __init__(self, content_manager):
        self.content_manager = content_manager
        self.history = []

    def next(self):
        pass

    def _update_state_agent(self, agent_actions, nl=None):
        """
        :param agent_actions: list of triples: (agent action, slot_name, slot_value)
        :param nl: Natural language string from agent
        """

        for action, slot_name, slot_value in agent_actions:
            if action == 'request':
                self.state['agent_request_slots'].append(slot_name)
            elif action=='inform':
                self.state['proposed_slots'][slot_name] = [slot_value, False, False]

        self.history.append({
            'state_agent': deepcopy(self.state),
            'agent_action': agent_actions,
            'user_action': None,
            'state_user': None,
            'agent_nl': nl,
            'user_nl': None
        })

    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """
        self.state = {
            'inform_slots': dict(),
            'request_slots': set(),
            'proposed_slots': dict(),  # Значение: [значение слота, был ли Negate, был ли ReqAlts]
            'agent_request_slots': [],
        }

    def update_state_user(self, user_actions, nl=None):
        """
        :param user_actions: list of triples: (agent action, slot_name, slot_value)
        :param nl: Natural language string from user
        """

        self.history[-1]['user_action'] = user_actions
        self.history[-1]['state_user'] = deepcopy(self.state)
        self.history[-1]['user_nl'] = nl

        # updating state
        for action, slot_name, slot_value in user_actions:
            if action=='inform':
                self.inform_slots[slot_name] = slot_value
            elif action=='dontcare':
                for slot in self.state['agent_request_slots']:
                    self.inform_slots[slot] = 'dontcare'

                self.state['agent_request_slots'] = []
    
    @property
    def inform_slots(self):
        return self.state['inform_slots']
    
    @property
    def last_user_action(self):
        return self.history[-1]['user_action']

class EchoAgent(Agent):
    def __init__(self, content_manager):
        super().__init__(content_manager)

    def next(self):
        return None, 'hello'


class RuleAgent(Agent):
    def __init__(self, content_manager):
        super().__init__(content_manager)
        self.required_slots = ['food', 'area', 'pricerange']

    def _next(self):
        if self.turn_count == 0:
            return [['welcomemsg', None, None]]

        for slot in self.required_slots:
            if slot not in self.inform_slots and slot not in self.state['request_slots']:
                return [['request', 'food', None]]

        variants = self.content_manager.available_results(self.inform_slots)
        if len(variants)==0:
            variants = self.content_manager.available_results({'food': self.inform_slots['food']})
            if len(variants)==0:
                return [['canthelp', 'food', self.inform_slots['food']]]

            variants = self.content_manager.available_results({'food': self.inform_slots['food'], 'area': self.inform_slots['area']})
            if len(variants) == 0:
                return [['canthelp', 'food', self.inform_slots['food']], ['canthelp', 'area', self.inform_slots['area']]]

            return [['canthelp', 'food', self.inform_slots['food']],
                    ['canthelp', 'area', self.inform_slots['area']], ['canthelp', 'pricerange',self.inform_slots['pricerange']]]

        return [
            ['inform', 'food', self.inform_slots['food']],
            ['inform', 'name', variants[0]['name']],
            ['inform', 'pricerange', self.inform_slots['pricerange']],
        ]

    def next(self):
        actions = self._next()
        self._update_state_agent(actions)
        return actions

    @property
    def turn_count(self):
        return len(self.history)
