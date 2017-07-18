from copy import deepcopy


class Agent:
    def __init__(self, content_manager):
        self.content_manager = content_manager
        self.history = []

    def next(self):
        pass

    def update_state_agent(self, agent_actions, nl=None):
        """
        :param agent_actions: list of triples: (agent action, slot_name, slot_value)
        :param nl: Natural language string from agent
        """

        self.history.append({
            'agent_action': agent_actions,
            #'agent_state': deepcopy(self.state),
            'user_action': None,
            'agent_nl': nl,
            'user_nl': None
        })

        for action, slot_name, slot_value in agent_actions:
            if action == 'request':
                self.state['agent_request_slots'].append(slot_name)
            elif action=='inform':
                self.state['proposed_slots'][slot_name] = [slot_value, set()]
                if slot_name in self.request_slots:
                    self.request_slots.remove(slot_name)

    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """
        self.state = {
            'inform_slots': dict(),
            'request_slots': set(),
            'proposed_slots': dict(),  # Значение: [значение слота, множество Negate + ReqAlts]
            'agent_request_slots': [],
        }
        self.history = []

    def update_state_user(self, user_actions, nl=None, goal=None):
        """
        :param user_actions: list of triples: (agent action, slot_name, slot_value)
        :param nl: Natural language string from user
        """

        self.history[-1]['user_action'] = user_actions
        self.history[-1]['user_nl'] = nl

        # updating state
        for action, slot_name, slot_value in user_actions:
            if action=='inform':
                self.inform_slots[slot_name] = slot_value
            elif action=='dontcare':
                for slot in self.state['agent_request_slots']:
                    self.inform_slots[slot] = 'dontcare'

                self.state['agent_request_slots'] = []
            elif action=='request':
                self.request_slots.add(slot_name)
            elif action=='reqalts':
                slot_for_reqalts = None
                slot_value_for_reqalts = None
                for history in reversed(self.history):
                    if history['agent_action'] is None:
                        continue

                    for action, slot_name, slot_value in history['agent_action']:
                        if action=='inform':
                            slot_for_reqalts = slot_name
                            slot_value_for_reqalts = slot_value
                            break
                    if slot_for_reqalts is not None:
                        break

                if slot_for_reqalts is not None and slot_for_reqalts in self.state['proposed_slots']:
                    self.state['proposed_slots'][slot_for_reqalts][1].add(slot_value_for_reqalts)

    def was_user_action_last_turn(self, user_action):
        if len(self.history)<2:
            return False
        for action, slot_name, slot_value in self.history[-2]['user_action']:
            if action== user_action:
                return True

        return False

    @property
    def turn_count(self):
        if len(self.history)==0:
            return 0
        if self.history[-1]['user_action'] is None:
            return len(self.history)
        return len(self.history) + 1

    @property
    def slot_restrictions(self):
        return {key:value[1] for key,value in self.proposed_slots.items() if len(value[1])>0}

    @property
    def request_slots(self):
        return self.state['request_slots']

    @property
    def agent_request_slots(self):
        return self.state['agent_request_slots']

    @property
    def proposed_slots(self):
        return self.state['proposed_slots']

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

    def next(self):
        if self.turn_count == 0:
            return [['welcomemsg', None, None]]

        for slot in self.required_slots:
            if slot not in self.inform_slots and slot not in self.state['request_slots'] and slot not in self.inform_slots:
                return [['request', slot, None]]

        variants = self.content_manager.available_results(self.inform_slots, self.slot_restrictions)
        if len(variants)==0:
            variants = self.content_manager.available_results({'food': self.inform_slots['food']}, self.slot_restrictions)
            if len(variants)==0:
                return [['canthelp', 'food', self.inform_slots['food']]]

            variants = self.content_manager.available_results({'food': self.inform_slots['food'], 'area': self.inform_slots['area']}, self.slot_restrictions)
            if len(variants) == 0:
                return [['canthelp', 'area', self.inform_slots['area']]]

            return [['canthelp', 'pricerange',self.inform_slots['pricerange']]]

        valid_variant = variants[0]
        if 'name' not in self.state['proposed_slots']:
            return [
                ['inform', 'food', valid_variant['food']],
                ['inform', 'name', valid_variant['name']],
                ['inform', 'pricerange', valid_variant['pricerange']],
            ]

        if 'addr' in self.request_slots:
            return [
                ['inform', 'name', valid_variant['name']],
                ['inform', 'addr', valid_variant['addr']]
            ]

        if 'phone' in self.request_slots:
            return [
                ['inform', 'name', valid_variant['name']],
                ['inform', 'phone', valid_variant['phone']]
            ]

        if 'area' in self.request_slots:
            return [
                ['inform', 'name', valid_variant['name']],
                ['inform', 'area', valid_variant['area']]
            ]

        if 'postcode' in self.request_slots:
            return [
                ['inform', 'name', valid_variant['name']],
                ['inform', 'postcode', valid_variant['postcode']]
            ]

        return [
            ['reqmore', None, None]
        ]


