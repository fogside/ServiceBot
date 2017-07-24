from .agent import Agent
import os
import pickle
import json
from dstc_helper import *
from .dqn import DoubleDQN
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


class DebugSituation:
    def __init__(self, agent,plot_every=30):
        self.plot_every = plot_every
        self.agent = agent
        self.df = None
        inform_name_indexes = [agent.binarizers['agent_act_full'][n] for n in agent.inform_action_names]
        request_food_index = [agent.binarizers['agent_act_full']['request_food']]
        repeat_index = [agent.binarizers['agent_act_full']['repeat']]
        reqmore_index = [agent.binarizers['agent_act_full']['reqmore']]

        self.situations = [
            #[self.inform_name_situation_ok, 'Inform name OK', inform_name_indexes, 0],
           # [self.inform_name_situation_wrong, 'Inform name Wrong', inform_name_indexes, 0],
            [self.request_food, 'Request food', request_food_index, 0],
            [self.request_food, 'Inform name', inform_name_indexes, 0],
            [self.request_food, 'Repeat', repeat_index, 0],
            [self.request_food, 'Req more', reqmore_index, 0],
        ]

    def request_food(self):
        agent = self.agent
        if agent.turn_count != 1:
            return False

        if len(agent.user_action)!=1 or agent.user_action[0][:2]!=['inform', 'pricerange']:
            return False

        return True


    def inform_name_situation_ok(self):
        agent = self.agent
        if agent.turn_count != 2:
            return False

        return agent.statevec_before[-1] == 1

    def inform_name_situation_wrong(self):
        agent = self.agent
        if agent.turn_count != 2:
            return False

        return agent.statevec_before[-1]>20

    def add_point(self, reward):
        if len(self.situations)==0:
            return

        point = dict()
        for i, (is_valid, name, action_indexes, current_q_value) in enumerate(self.situations):
            q_value = current_q_value
            if is_valid():
                q_value = self.agent.model.get_q_value(self.agent.statevec_before, action_indexes)
            point[name+' Q value'] = q_value
            self.situations[i][3] = q_value

        frames = [pd.DataFrame([point])]
        if self.df is not None:
            frames.insert(0, self.df)

        self.df = pd.concat(frames, ignore_index=True)
        self.plot()

    def is_valid_state(self):
        pass

    def plot(self):
        if self.df.shape[0]==0 or self.df.shape[0]%self.plot_every>0:
            return

        ax = self.df.plot()
        fig = ax.get_figure()
        try:
            fig.savefig('RL.png')
            plt.close(fig)
        except Exception:
            pass



class RLAgent(Agent):
    def __init__(self,content_manager, path_to_model,
                 path_to_dstc, binarizers, max_turn, rounds=1, batch_size=32,
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
        self.last_variants = []
        self.was_random = False
        #binarizers['agent_act_full'] = {'welcomemsg':0, 'inform_name__inform_postcode__inform_pricerange': 1}
        self.model = DoubleDQN.load(path_to_model) if os.path.exists(path_to_model) \
            else DoubleDQN(n_actions=len(binarizers['agent_act_full']), n_features=len(self.state2vec()),
                           random_state=content_manager.random_state)
        self.agent_act_full_revers = {value: key for key,value in binarizers['agent_act_full'].items()}
        self.inform_action_names = [a for a in self.binarizers['agent_act_full'] if 'inform_name' in a]
        self.debug_situation = DebugSituation(self)


    def initialize_episode(self):
        super(RLAgent, self).initialize_episode()
        self.user_act_slot_dict = defaultdict(int)
        self.statevec_before = None
        self.debug_parts = []
        self.total_reward = 0
        self.last_variants = []

    def update_state_agent(self, agent_actions, nl=None):
        self.statevec_before = self.state2vec()
        super(RLAgent, self).update_state_agent(agent_actions, nl)

    def update_state_user(self, user_actions, nl=None, user_state=None):
        turn_count = self.turn_count
        super(RLAgent, self).update_state_user(user_actions, nl)
        r = self.reward(user_actions, user_state) if user_state is not None else 0

        for action, slot_name, slot_value in user_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            self.user_act_slot_dict[act_slot] += 1

        if user_state is None:
            return

        statevec_after = self.state2vec()
        if self.send_to_user is not None:
            self.debug_parts.insert(0, 'Reward = {}'.format(r))
            self.send_to_user('Debug: ' + ' '.join(self.debug_parts))

        a = self.action_index(self.history[-1]['agent_action'])
        self.total_reward += r
        self.debug_situation.add_point(r)

        if turn_count>0:
            self.model.store_transition(np.hstack([self.statevec_before, a, r, statevec_after]))
            self.model.learn()
        self.debug_parts = []

    def state2vec(self, variants = None):
        if variants is None:
            variants = self.content_manager.available_results(self.inform_slots, self.slot_restrictions)

        result = []
        # for i in range(1, 2):
        #     history = self.history[-i] if len(self.history)>=i else None
        #
        #     last_user_actions = [] if history is None else history['user_action']
        #     last_agent_actions = [] if history is None else history['agent_action']
        #
        #     user_acts = [[action for action, slot_name, slot_value in last_user_actions], 'user_act']
        #     user_act_slots = [[action +('_'+slot_name if slot_name is not None else '') for action, slot_name, slot_value in last_user_actions], 'user_act_slots']
        #     agent_acts = [[action for action, slot_name, slot_value in last_agent_actions], 'agent_act']
        #     agent_act_slots = [[action +('_'+slot_name if slot_name is not None else '') for action, slot_name, slot_value in last_agent_actions], 'agent_act_slots']
        #
        #     for data, binarizer_name in [user_acts, user_act_slots, agent_acts, agent_act_slots]:
        #         if len(data)==0:
        #             data.append('empty')
        #         result.extend(np.max(self.binarizers[binarizer_name].transform(data), axis=0))
        #
        #     # Last agent action full
        #     if i==1:
        #         action_full = '__'.join(agent_act_slots[0])
        #         action_full_features = list(range(len(self.binarizers['agent_act_full'])))
        #         if action_full in self.binarizers['agent_act_full']:
        #             action_full_features[self.binarizers['agent_act_full'][action_full]] = 1
        #
        #         result.extend(action_full_features)
        #
        constraint_slots = [list(self.inform_slots.keys()), 'user_constraint_slots']
        user_request_slots = [list(self.request_slots), 'user_request_slots']
        proposed_slots = [list(self.proposed_slots.keys()), 'all_slots']
        request_slots = [list(self.agent_request_slots), 'all_slots']

        for data, binarizer_name in [constraint_slots, user_request_slots,
                                     proposed_slots, request_slots]:
            if len(data) == 0:
                data.append('empty')
            result.extend(np.max(self.binarizers[binarizer_name].transform(data), axis=0))
            result.append(len(data))

        # turn_onehot_rep = np.zeros(self.max_turn)
        # turn_onehot_rep[min(self.turn_count, self.max_turn-1)] = 1.0
        #
        # result.extend(turn_onehot_rep)

        variant_variation = self.content_manager.explain_variation(variants)

        result.append(int(variant_variation) ==1)
        result.append(variant_variation)

        return np.array(result)

    def action_index(self, agent_actions):
        act_slots = []
        for action, slot_name, slot_value in agent_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            if 'canthelp' in act_slot:
                act_slot = action

            act_slots.append(act_slot)
        act_slots = sorted(act_slots)
        if len(act_slots) > 3 and 'inform_pricerange' in act_slots:
            act_slots.remove('inform_pricerange')

        act_slots = '__'.join(act_slots)
        if act_slots in ['', 'inform_name', 'inform_area__inform_name__inform_phone__inform_postcode']:
            return -1

        return self.binarizers['agent_act_full'][act_slots]

    def last_inform_slots(self):
        if self.previous_action is None:
            return []

        return [s for a,s,v in self.previous_action if a=='inform']

    def reward(self, user_actions, user_state):
        if user_state['done']:
            if user_state['failed']:
                return -100

            return 30

        state_turn_begin = self.history[-1]['agent_state']
        user_requested_slot_failed = len([s for s in state_turn_begin['request_slots'] if s not in self.last_inform_slots()])>0 and 'name' in state_turn_begin['proposed_slots']
        result = 0
        result += len(user_state['current_filled_slots']) * 5
        if user_requested_slot_failed:
            result = -5
            #print('user_requested_slot_failed')

        #result -= len(user_state['error_slots']) * 20

        #reqalts_exists = any([a[0] == 'reqalts' for a in user_actions])
        #if reqalts_exists:
         #   return result

        for action, slot_name, slot_value in user_actions:
            act_slot = action + ('_' + slot_name if slot_name is not None else '')
            result -= self.user_act_slot_dict[act_slot]*5

        return result

    def not_valid_action_indexes(self, variants):
        result = []
        actions_with_informable = ['canthelp', 'inform', 'expl-conf', 'select']
        for act_full, index in self.binarizers['agent_act_full'].items():
            if len(variants)==0 and 'inform_name' in act_full:
                result.append(index)
                continue

            if 'select' in act_full:
                result.append(index)
                continue

            if 'expl-conf' in act_full:
                result.append(index)
                continue

            last_user_request = self.user_action_last_turn('request')
            if last_user_request is not None:
                if 'request' in act_full and 'name' in self.proposed_slots:
                    result.append(index)
                    continue

                if 'inform_name' in act_full and 'inform_'+last_user_request[1] not in act_full:
                    result.append(index)
                    continue

            parts = act_full.split('__')
            for p in parts:
                sub_parts = p.split('_')
                action = sub_parts[0]
                slot_name = None if len(sub_parts) == 1 else sub_parts[1]
                if slot_name in self.content_manager.informable_slots and action in actions_with_informable and slot_name not in self.inform_slots:
                    result.append(index)
                    break

                if len(variants)>0 and action=='canthelp':
                    result.append(index)
                    break

                if action=='request' and slot_name in self.inform_slots:
                    result.append(index)
                    break

        return result

    def next(self):
        if self.turn_count == 0:
            return [['welcomemsg', None, None]]

        variants = self.content_manager.available_results(self.inform_slots, self.slot_restrictions)
        self.last_variants = variants
        act_indexes_to_ignore = self.not_valid_action_indexes(variants)
        index, was_random = self.model.choose_action(self.state2vec(variants), act_indexes_to_ignore)
        self.was_random = was_random
        self.debug_parts.append('Was random = {} Epsilon = {}'.format(was_random, round(self.model.epsilon, 2)))
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

            if action=='canthelp':
                for slot in self.content_manager.informable_slots:
                    inform_slots = self.inform_slots.copy()
                    inform_slots[slot] = 'dontcare'
                    _v = self.content_manager.available_results(inform_slots, self.slot_restrictions)
                    if len(_v)>0:
                        slot_name = slot
                        slot_value = self.inform_slots[slot_name]
                    break

                if slot_name is None and len(self.inform_slots)>0:
                    slot_name = list(self.inform_slots)[0]
                    slot_value = self.inform_slots[slot_name]

            result.append([action, slot_name, slot_value])

        return result

