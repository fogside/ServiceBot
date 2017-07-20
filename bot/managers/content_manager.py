import json
from fuzzywuzzy import fuzz
from collections import defaultdict
import numpy as np
import glob
import os
from bot.dstc_helper import dstc_turns_to_triplets
import random


class ContentManager:
    def __init__(self, act_set, slot_set, restaurant_dict, goals, path_to_dstc, path_to_save, random_state):
        self.path_to_save = path_to_save
        self.path_to_dstc = path_to_dstc
        self.random_state = random_state
        self.goals = goals
        self.restaurant_dict = restaurant_dict
        self.slot_set = slot_set
        self.act_set = act_set
        self._make_cache_by_slot()
        self._text_to_actions = dict()

    @property
    def text_to_actions(self):
        if len(self._text_to_actions)==0:
            self.init_text_to_actions()

        return self._text_to_actions

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

    def init_text_to_actions(self):
        self._text_to_actions = json.load(open(self.path_to_save)) if os.path.exists(self.path_to_save) else defaultdict(list)
        if len(self._text_to_actions) > 0:
            return

        for label_path in glob.glob('{}/**/label.json'.format(self.path_to_dstc), recursive=True):
            label = json.load(open(label_path))
            log = json.load(open(label_path.replace('label', 'log')))

            log_turns = log['turns']
            label_turns = label['turns']

            label_transcripts = [t['transcription'] for t in label_turns]
            log_transcripts = [t['output']['transcript'] for t in log_turns]

            dstc_turns_to_triplets(label_turns, log_turns)
            for text, actions in zip(label_transcripts, label_turns):
                self._text_to_actions[text].append(actions)

            for text, actions in zip(log_transcripts, log_turns):
                self._text_to_actions[text].append(actions)

        for key, value in self._text_to_actions.items():
            self._text_to_actions[key] = random.choice(value)

        json.dump(self._text_to_actions, open(self.path_to_save, 'w'))


    @staticmethod
    def from_settings(settings, random_state):
        act_set = set(open(settings.ACT_SET_PATH).readlines())
        slot_set = set(open(settings.SLOTS_SET_PATH).readlines())
        restaurant_dict = json.load(open(settings.RESTAURANT_PATH))
        goals = json.load(open(settings.GOALS_PATH))
        return ContentManager(act_set, slot_set, restaurant_dict, goals, settings.DSTC_PATH, settings.CONTENT_MANAGER_SAVE_PATH, random_state)

    def available_results(self, slots, restrictions):
        result = set(self.restaurant_dict.keys())
        for slot_name, slot_value in slots.items():
            result = result.intersection(self.cache_by_slot[slot_name].get(slot_value, set()))
            if len(result) == 0:
                break
        result = [self.restaurant_dict[r] for r in result]
        if len(restrictions)>0:
            for i in range(len(result)-1, -1, -1):
                for restr_key, restr_set in restrictions.items():
                    if result[i][restr_key] in restr_set:
                        del result[i]
                        break
        return result

    def random_goal(self):
        result = self.random_state.choice(self.goals)
        result['constraints_dict'] = {key: value for key,value in result['constraints']}
        return result.copy()