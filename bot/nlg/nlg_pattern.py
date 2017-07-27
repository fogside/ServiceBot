from itertools import permutations


class NlgPattern:
    def __init__(self, content_manager):
        self.content_manager = content_manager

        self.patterns_indexes = dict()
        for i, (pattern, actions) in enumerate(self.content_manager.nlg_patterns):
            for permutation in permutations(actions):
                key = '+'.join('{}={}'.format(a, s) for a, s in permutation)
                self.patterns_indexes[key] = i

    def agent_action_nl(self, agent_actions):
        key = '+'.join(['{}={}'.format(action, slot_name) for action, slot_name, slot_value in agent_actions])
        if key not in self.patterns_indexes:
            raise Exception('Unknown agent_actions = {}'.format(agent_actions))
        pattern, actions = self.content_manager.nlg_patterns[self.patterns_indexes[key]]
        format_args = []
        for action, slot in actions:
            for agent_action, agent_slot_name, agent_slot_value in agent_actions:
                if agent_action == action and slot == agent_slot_name:
                    if agent_slot_value is not None:
                        format_args.append(agent_slot_value)
                    break

        try:
            return pattern.format(*format_args)
        except Exception:
            raise Exception('Not enough args for pattern {} given. Agent actions: {}'.format(pattern, agent_actions))
