import json
from fuzzywuzzy import fuzz
from collections import defaultdict
import numpy as np


class ContentManager:
    def __init__(self, act_set, slot_set, restaurant_dict, goals):
        self.goals = goals
        self.restaurant_dict = restaurant_dict
        self.slot_set = slot_set
        self.act_set = act_set
        self._make_cache_by_slot()

    def _make_cache_by_slot(self):
        self.cache_by_slot = defaultdict(lambda: defaultdict(set))
        for key, value in self.restaurant_dict.items():
            for slot_name, slot_value in value.items():
                self.cache_by_slot[slot_name][slot_value].add(key)
                self.cache_by_slot[slot_name]['dontcare'].add(key)

    @staticmethod
    def from_settings(settings):
        act_set = set(open(settings.ACT_SET_PATH).readlines())
        slot_set = set(open(settings.SLOTS_SET_PATH).readlines())
        restaurant_dict = json.load(open(settings.RESTAURANT_PATH))
        goals = json.load(open(settings.GOALS_PATH))
        return ContentManager(act_set, slot_set, restaurant_dict, goals)

    def available_results(self, slots):
        result = set(self.restaurant_dict.keys())
        for slot_name, slot_value in slots.items():
            result = result.intersection(self.cache_by_slot[slot_name].get(slot_value, set()))
            if len(result) == 0:
                break
        return [self.restaurant_dict[r] for r in result]

    def random_goal(self):
        return np.random.choice(self.goals)