import json
from fuzzywuzzy import fuzz
from collections import defaultdict
import numpy as np


class ContentManager:
    def __init__(self, act_set, slot_set, restaurant_dict, goals, random_state):
        self.random_state = random_state
        self.goals = goals
        self.restaurant_dict = restaurant_dict
        self.slot_set = slot_set
        self.act_set = act_set
        self._make_cache_by_slot()

    def random_slot_value(self, slot_name):
        try:
            return self.random_state.choice(self.possible_slot_values[slot_name])
        except Exception:
            raise Exception(slot_name)

    def _make_cache_by_slot(self):
        self.cache_by_slot = defaultdict(lambda: defaultdict(set))
        self.possible_slot_values = defaultdict(set)

        for key, value in self.restaurant_dict.items():
            for slot_name, slot_value in value.items():
                self.cache_by_slot[slot_name][slot_value].add(key)
                self.cache_by_slot[slot_name]['dontcare'].add(key)

                if slot_name in ['area', 'postcode', 'addr', 'postcode']:
                    self.possible_slot_values[slot_name].add(slot_value)

        for goal in self.goals:
            for slot_name, slot_value in goal['constraints']:
                self.possible_slot_values[slot_name].add(slot_value)

        for key in self.possible_slot_values:
            self.possible_slot_values[key].add('dontcare')
            self.possible_slot_values[key] = sorted(list(self.possible_slot_values[key]))

    @staticmethod
    def from_settings(settings, random_state):
        act_set = set(open(settings.ACT_SET_PATH).readlines())
        slot_set = set(open(settings.SLOTS_SET_PATH).readlines())
        restaurant_dict = json.load(open(settings.RESTAURANT_PATH))
        goals = json.load(open(settings.GOALS_PATH))
        return ContentManager(act_set, slot_set, restaurant_dict, goals, random_state)

    def available_results(self, slots, restrictions):
        result = set(self.restaurant_dict.keys())
        for slot_name, slot_value in slots.items():
            result = result.intersection(self.cache_by_slot[slot_name].get(slot_value, set()))
            if len(result) == 0:
                break
        result = [self.restaurant_dict[r] for r in result]
        for i in range(len(result)-1, -1, -1):
            for restr_key, restr_set in restrictions.items():
                if result[i][restr_key] in restr_set:
                    del result[i]
                    break
        return result

    def random_goal(self):
        result = self.random_state.choice(self.goals)
        result['constraints_dict'] = {key: value for key,value in result['constraints']}
        return result