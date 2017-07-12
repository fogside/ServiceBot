from itertools import permutations

class NlgPattern:
    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.patterns = [
            ['Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?', [['welcomemsg', None]]],
            ['What kind of food would you like?', [['request', 'food']]],
            ['{} is a nice place in the {} of town serving {} food', [['inform', 'name'], ['inform', 'area'], ['inform', 'food'] ] ],
            ['The phone number of {} is {} and its postcode is {}', [['inform', 'name'], ['inform', 'phone'], ['inform', 'postcode']] ],
            ['{} is a great restaurant in the {} of town and the prices are {}', [['inform', 'name'], ['inform', 'area'], ['inform', 'pricerange']] ],
            ['{} is a nice restaurant serving {} food and it is in the {} price range', [['inform', 'name'], ['inform', 'food'], ['inform', 'pricerange']]],
            ['The phone number of {} is {}', [['inform', 'name'], ['inform', 'phone'] ]],
            ['{} is a nice restaurant in the {} of town', [['inform', 'name'], ['inform', 'area']]],
            ['Sure , {} is on {}', [['inform', 'name'], ['inform', 'addr']]],
            ['The phone number of the {} is {} and it is on {}', [['inform', 'name'], ['inform', 'phone'], ['inform', 'addr'] ]],
            ['{} is in the {} price range', [['inform', 'name'], ['inform', 'pricerange']]],
            ['{} serves {} food', [['inform', 'name'], ['inform', 'food']]],
            ['The post code of {} is {}', [['inform', 'name'], ['inform', 'postcode']]],
            ['{} is a great restaurant in the {} of town in the {} price range', [['inform', 'name'], ['inform', 'area'], ['inform', 'pricerange']]],
            ['{} is in the {} part of town', [['inform', 'name'], ['inform', 'area']]],
            ['The price range at {} is {}', [['inform', 'name'], ['inform', 'pricerange']]],
            ['What part of town do you have in mind?', [['request', 'area']]],
            ["I'm sorry but there is no restaurant serving {} food", [['canthelp', 'food']]],
            ["I'm sorry but there is no {} restaurant in the {} of town", [['canthelp', 'food'], ['canthelp', 'area']]],
            ["Sorry there is no {} restaurant in the {} of town serving $area$ american {}", [['canthelp', 'pricerange'], ['canthelp', 'area'], ['canthelp', 'food'] ]],
        ]

        self.patterns_indexes = dict()
        for i, (pattern, actions) in enumerate(self.patterns):
            for permutation in permutations(actions):
                key = '+'.join('{}={}'.format(a,s) for a,s in permutation)
                self.patterns_indexes[key] = i

    def agent_action_nl(self, agent_actions):
        key = '+'.join(['{}={}'.format(action, slot_name) for action, slot_name, slot_value in agent_actions])
        if key not in self.patterns_indexes:
            raise Exception('Unknown agent_actions = {}'.format(agent_actions))
        pattern, actions = self.patterns[self.patterns_indexes[key]]
        format_args = []
        for action, slot in actions:
            for agent_action, agent_slot_name, agent_slot_value in agent_actions:
                if agent_action==action and slot==agent_slot_name:
                    if agent_slot_value is not None:
                        format_args.append(agent_slot_value)
                    break

        return pattern.format(*format_args)