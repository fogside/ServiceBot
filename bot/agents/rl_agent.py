from .agent import Agent
import os
import pickle
import glob
import json
from dstc_helper import *
from .dqn import DoubleDQN
import numpy as np
from collections import defaultdict
import random

class RLAgent(Agent):
    def __init__(self, content_manager, path_to_train, path_to_model,
                 path_to_dstc, binarizers, need_sl=True, sl_rounds=1000, rounds=100, batch_size=100):
        super().__init__(content_manager)
        self.batch_size = batch_size
        self.binarizers = binarizers
        self.rounds = rounds
        self.sl_rounds = sl_rounds
        self.path_to_model = path_to_model
        self.path_to_train = path_to_train
        self.path_to_dstc = path_to_dstc
        self.model = DoubleDQN.load(path_to_model) if os.path.exists(path_to_model) \
            else DoubleDQN(n_actions=len(binarizers['agent_act_full']), n_features=len(self.state2vec()))
        self.agent_act_full_revers = {value: key for key,value in binarizers['agent_act_full'].items()}
        if need_sl:
            if os.path.exists(path_to_train):
                self.train_data = pickle.load(open(path_to_train, 'rb'))
            else:
                self.train_data = self.make_sl_data()

            self.train(sl_rounds)

    def initialize_episode(self):
        super(RLAgent, self).initialize_episode()
        self.user_act_slot_dict = defaultdict(int)

    def update_state_user(self, user_actions, nl=None):
        super(RLAgent, self).update_state_user(user_actions, nl)
        for action, slot_name, slot_value in user_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            self.user_act_slot_dict[act_slot] += 1

    def state2vec(self):
        result = []
        last_history = None
        if len(self.history) > 0:
            last_history = self.history[-1]

        last_user_actions = [] if last_history is None else last_history['user_action']
        user_acts = [action for action, slot_name, slot_value in last_user_actions]
        user_act_slots = [action +('_'+slot_name if slot_name is not None else '') for action, slot_name, slot_value in last_user_actions]

        if len(user_acts) == 0:
            user_acts.append('empty')

        if len(user_act_slots) == 0:
            user_act_slots.append('empty')

        result.extend(np.max(self.binarizers['user_act'].transform(user_acts), axis=0))
        result.extend(np.max(self.binarizers['user_act_slots'].transform(user_act_slots), axis=0))
        return np.array(result)

    def action_index(self, agent_actions):
        act_slots = []
        for action, slot_name, slot_value in agent_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            act_slots.append(act_slot)
        act_slots = sorted(act_slots)
        if len(act_slots) > 3 and 'inform_pricerange' in act_slots:
            act_slots.remove('inform_pricerange')

        act_slots = '__'.join(act_slots)
        if act_slots in ['', 'inform_name', 'inform_area__inform_name__inform_phone__inform_postcode']:
            return -1

        return self.binarizers['agent_act_full'][act_slots]

    def reward(self, user_actions):
        result = 0
        #reqalts_exists = any([a[0] == 'reqalts' for a in user_actions])
        #if reqalts_exists:
         #   return result

        for action, slot_name, slot_value in user_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            result -= self.user_act_slot_dict[act_slot]

        return result

    def make_sl_data(self):
        result = []
        for file_index, label_path in enumerate(glob.glob('{}/**/label.json'.format(self.path_to_dstc), recursive=True)):
            self.initialize_episode()

            if file_index > 0 and file_index % 500 == 0:
                print('{} files processed'.format(file_index))
                break
            label = json.load(open(label_path))
            log = json.load(open(label_path.replace('label', 'log')))
            log_turns = log['turns']
            label_turns = label['turns']
            dstc_turns_to_triplets(label_turns, log_turns)

            dialog_results = []
            valid_dialog = True
            for i in range(len(log_turns)):
                agent_acts = log_turns[i]
                user_acts = label_turns[i]

                s = self.state2vec()
                self.update_state_agent(agent_acts)
                r = self.reward(user_acts)

                self.update_state_user(user_acts)

                s_ = self.state2vec()
                a = self.action_index(agent_acts)
                if a==-1:
                    valid_dialog=False
                    break

                dialog_results.append(np.hstack([s, a, r, s_]))

            if valid_dialog:
                result.extend(dialog_results)

        return result

    def train(self, rounds):
        for _ in range(rounds):
            iter_count = min(200, len(self.train_data) // self.batch_size)
            cur_bellman_err = 0
            for i in range(iter_count):
                batch = [random.choice(self.train_data) for __ in range(self.batch_size)]
                batch_struct = self.model.learn(batch)
                cur_bellman_err += batch_struct['cost']['total_cost']

            self.model.replace_target_params()
            print('Cost = {}'.format(cur_bellman_err))

    def next(self):
        if self.turn_count == 0:
            return [['welcomemsg', None, None]]

        index = self.model.choose_action(self.state2vec())
        act_full = self.agent_act_full_revers[index]
        result = []
        for part in act_full.split('__'):
            part_parts = part.split('_')
            action = part_parts[0]
            slot_name = None if len(part_parts) == 1 else part_parts[1]

            result.append([action, slot_name, ''])

        return result