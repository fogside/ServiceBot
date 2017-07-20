from .agent import Agent
import os
import pickle
import json
from dstc_helper import *
from .dqn import DoubleDQN
import numpy as np
from collections import defaultdict
import random
import glob
from usersims.sl_usersim import fill_goal_service_fields, update_goal_fields


class RLAgent(Agent):
    def __init__(self,content_manager, path_to_model,
                 path_to_dstc, binarizers, max_turn, need_sl=False, rounds=1, batch_size=32,
                 send_to_user = None
                 ):
        super().__init__(content_manager)
        self.send_to_user = send_to_user
        self.path_to_dstc = path_to_dstc
        self.max_turn = max_turn
        self.batch_size = batch_size
        self.binarizers = binarizers
        self.rounds = rounds
        self.path_to_model = path_to_model
        self.initialize_episode()
        self.debug_parts = []
        #binarizers['agent_act_full'] = {'welcomemsg':0, 'request_food': 2, 'request_area':1}
        self.model = DoubleDQN.load(path_to_model) if os.path.exists(path_to_model) \
            else DoubleDQN(n_actions=len(binarizers['agent_act_full']), n_features=len(self.state2vec()),
                           memory_size = 1000, batch_size=batch_size, random_state=content_manager.random_state,
                           replace_target_iter=10)
        self.agent_act_full_revers = {value: key for key,value in binarizers['agent_act_full'].items()}

        if need_sl:
            self.sl_train()

    def sl_train(self):

        for file_index, label_path in enumerate(glob.glob('{}/**/label.json'.format(self.path_to_dstc), recursive=True)):
            self.initialize_episode()

            if file_index > 0 and file_index % 10 == 0:
                print('{} files processed'.format(file_index))

            #if file_index>=500:
                #break

            label = json.load(open(label_path))
            log = json.load(open(label_path.replace('label', 'log')))
            log_turns = log['turns']
            label_turns = label['turns']
            dstc_turns_to_triplets(label_turns, log_turns)
            goal = label['task-information']['goal']
            add_alt_constraints(goal)
            fill_goal_service_fields(goal)

            dialog_results = []
            valid_dialog = True
            for i in range(len(log_turns)):
                agent_acts = log_turns[i]
                user_acts = label_turns[i]

                s = self.state2vec()
                self.update_state_agent(agent_acts)
                update_goal_fields(agent_acts, goal, self.turn_count, self.max_turn)
                r = self.reward(user_acts, goal)

                self.update_state_user(user_acts)

                s_ = self.state2vec()
                a = self.action_index(agent_acts)
                if a == -1:
                    valid_dialog = False
                    break

                if self.turn_count>0:
                    dialog_results.append(np.hstack([s, a, r, s_]))

            if valid_dialog:
                for r in dialog_results:
                    self.model.store_transition(r)
                    self.model.learn()

    def initialize_episode(self):
        super(RLAgent, self).initialize_episode()
        self.user_act_slot_dict = defaultdict(int)
        self.statevec_before = None
        self.debug_parts = []

    def update_state_agent(self, agent_actions, nl=None):
        self.statevec_before = self.state2vec()
        super(RLAgent, self).update_state_agent(agent_actions, nl)

    def update_state_user(self, user_actions, nl=None, goal=None):
        super(RLAgent, self).update_state_user(user_actions, nl)
        r = self.reward(user_actions, goal) if goal is not None else 0

        for action, slot_name, slot_value in user_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            self.user_act_slot_dict[act_slot] += 1

        if goal is None or self.turn_count==0:
            return

        statevec_after = self.state2vec()
        if self.send_to_user is not None:
            self.debug_parts.insert(0, 'Reward = {}'.format(r))
            self.send_to_user('Debug: ' + ' '.join(self.debug_parts))

        a = self.action_index(self.history[-1]['agent_action'])
        self.model.store_transition(np.hstack([self.statevec_before, a, r, statevec_after]))
        self.model.learn()
        self.debug_parts = []

    def state2vec(self, variants = None):
        if variants is None:
            variants = self.content_manager.available_results(self.inform_slots, self.slot_restrictions)

        result = []
        # last_history = None
        # if len(self.history) > 0:
        #     last_history = self.history[-1]
        #
        # last_user_actions = [] if last_history is None else last_history['user_action']
        # last_agent_actions = [] if last_history is None else last_history['agent_action']
        #
        # user_acts = [[action for action, slot_name, slot_value in last_user_actions], 'user_act']
        # user_act_slots = [[action +('_'+slot_name if slot_name is not None else '') for action, slot_name, slot_value in last_user_actions], 'user_act_slots']
        # agent_acts = [[action for action, slot_name, slot_value in last_agent_actions], 'agent_act']
        # agent_act_slots = [[action +('_'+slot_name if slot_name is not None else '') for action, slot_name, slot_value in last_agent_actions], 'agent_act_slots']
        #
        # constraint_slots = [list(self.inform_slots.keys()), 'user_constraint_slots']
        # user_request_slots = [list(self.request_slots), 'user_request_slots']
        # proposed_slots = [list(self.proposed_slots.keys()), 'all_slots']
        # request_slots = [list(self.agent_request_slots), 'all_slots']
        #
        # for data, binarizer_name in [user_acts, user_act_slots, constraint_slots, user_request_slots,
        #                              proposed_slots, request_slots, agent_acts, agent_act_slots]:
        #     if len(data)==0:
        #         data.append('empty')
        #     result.extend(np.max(self.binarizers[binarizer_name].transform(data), axis=0))
        #
        # turn_onehot_rep = np.zeros(self.max_turn)
        # turn_onehot_rep[min(self.turn_count, self.max_turn-1)] = 1.0
        #
        # result.extend(turn_onehot_rep)
        result.append(float(len(variants)>0))
        result.append(float(len(variants)))

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

    def reward(self, user_actions, goal):
        if goal['done']:
            if goal['failed']:
                return -80
            return 80

        result = 0
        result += len(goal['current_filled_slots']) * 10

        #reqalts_exists = any([a[0] == 'reqalts' for a in user_actions])
        #if reqalts_exists:
         #   return result

        for action, slot_name, slot_value in user_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            result -= self.user_act_slot_dict[act_slot]

        return result

    def not_valid_action_indexes(self, variants):
        result = []
        informables = ['food', 'area', 'pricerange']
        actions_with_informable = ['canthelp', 'inform', 'expl-conf', 'select']
        for act_full, index in self.binarizers['agent_act_full'].items():
            if len(variants)==0 and 'inform_name' in act_full:
                result.append(index)
                continue

            parts = act_full.split('__')
            for p in parts:
                sub_parts = p.split('_')
                action = sub_parts[0]
                slot_name = None if len(sub_parts) == 1 else sub_parts[1]
                if slot_name in informables and action in actions_with_informable and slot_name not in self.inform_slots:
                    result.append(index)
                    break

        return result

    def next(self):
        if self.turn_count == 0:
            return [['welcomemsg', None, None]]

        variants = self.content_manager.available_results(self.inform_slots, self.slot_restrictions)
        act_indexes_to_ignore = self.not_valid_action_indexes(variants)
        index, was_random = self.model.choose_action(self.state2vec(variants), act_indexes_to_ignore)
        self.debug_parts.append('Was random = {}'.format(was_random))
        act_full = self.agent_act_full_revers[index]
        result = []

        for part in act_full.split('__'):
            part_parts = part.split('_')
            action = part_parts[0]
            slot_name = None if len(part_parts) == 1 else part_parts[1]
            slot_value = '' if slot_name is not None else None
            if len(variants)>0 and 'inform_name' in act_full:
                variant = variants[0]
                if slot_name in variant:
                    slot_value = variant[slot_name]
            else:
                slot_value = self.inform_slots.get(slot_name, '' if slot_name is not None and action!='request' else None)

            result.append([action, slot_name, slot_value])

        return result

