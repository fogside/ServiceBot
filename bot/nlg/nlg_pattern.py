from itertools import permutations

class NlgPattern:
    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.patterns = [
            ['Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?', [['welcomemsg', None]]],
            ['What kind of food would you like?', [['request', 'food']]],
            ['{} is a nice place in the {} of town serving {} food', [['inform', 'name'], ['inform', 'area'], ['inform', 'food'] ] ],
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
            ["I'm sorry but there is no {} restaurant in the {} of town", [['canthelp', 'area']] ],
            ["Sorry there is no {} restaurant in the {} of town serving {} food", [['canthelp', 'food'] ]],
            ['Sorry I am a bit confused ; please tell me again what you are looking for .', [['repeat', None]]],
            ['Can I help you with anything else?', [['reqmore', None]]],
            ['Would you like something in the cheap , moderate , or expensive price range?', [['request', 'pricerange']]],
            ['Sure , {} is on {}, the phone number is {}', [['inform', 'name'], ['inform', 'addr'], ['inform', 'phone']]],
            ['{} is on {} , and it is in the {} price range', [['inform', 'name'], ['inform', 'addr'], ['inform', 'pricerange']]],
            ['The phone number of the {} is {} and its postcode is {}', [['inform', 'name'], ['inform', 'phone'], ['inform', 'postcode']]],
            ['{} is on {} and serves tasty {} food', [['inform', 'name'], ['inform', 'addr'], ['inform', 'food']]],
            ['Sorry would you like something in the {} price range or in the {} price range', [['select', 'pricerange'], ['select', 'pricerange']] ],
            ['Sorry would you like something in the {} or in the {}', [['select', 'area'], ['select', 'area']]],
            ['Sorry would you like {} food or {}', [['select', 'food'], ['select', 'food']]],
            ['Let me confirm , You are looking for a restaurant in the {} price range right?', [['expl-conf', 'pricerange']]],
            ['You are looking for a restaurant serving {} kind of food right?', [['expl-conf', 'food']]],
            ['Ok , a restaurant in {} part of town is that right?', [['expl-conf', 'area']]],
            ['The phone number of {} restaurant is {} and it is in the {} part of town', [['inform', 'name'], ['inform', 'phone'], ['inform', 'area'], ]],
            ['{} is in the {} price range , and their post code is {}', [['inform', 'name'], ['inform', 'pricerange'], ['inform', 'postcode'], ]],
            ['The phone number of {} restaurant is {} and it is in the {} price range', [['inform', 'name'], ['inform', 'phone'], ['inform', 'pricerange'], ]],
            ['{} is a great restaurant in the {} of town and their post code is {}', [['inform', 'name'], ['inform', 'area'], ['inform', 'postcode']]],
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