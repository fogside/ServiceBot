import xgboost as xgb
import pickle
import numpy as np
from copy import deepcopy
from scipy.sparse import hstack
from sklearn.preprocessing import normalize

class UserSimulator:
    def __init__(self, content_manager, nlg):
        self.nlg = nlg
        self.content_manager = content_manager

    def send_to_user(self, message):
        pass

    def inform_user(self, agent_actions):
        self.send_to_user(self.agent_message(agent_actions))

    def next(self):
        pass

    def initialize_episode(self):
        pass

    def agent_message(self, agent_actions):
        return self.nlg.agent_action_nl(agent_actions)


class EchoUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg):
        super().__init__(content_manager, nlg)

    def send_to_user(self, message):
        print(message)

    def next(self):
        answer = 'hello'
        print(answer)
        return None, answer


class ConsoleUserSimulator(UserSimulator):
    def send_to_user(self, message):
        print(message)

    def next(self):
        answer = input()
        return None, answer


class TelegramUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg, update):
        super().__init__(content_manager, nlg)
        self.update = update
        self.message = None

    def send_to_user(self, message):
        self.update.message.reply_text(message)

    def next(self):
        return None, self.message

    def new_message(self, message):
        self.message = message


class SupervisedUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg, model_path, binarizers_path, print_dialog=True):
        super().__init__(content_manager, nlg)
        self.print_dialog = print_dialog
        self.model = pickle.load(open(model_path, 'rb'))
        self.binarizers = pickle.load(open(binarizers_path, 'rb'))

    def initialize_episode(self):
        self.goal = self.content_manager.random_goal()
        self.state = {
            'user_informed': set(),
            'user_requested': set(),
            'no_data_requests': []
        }
        self.history = []

    def send_to_user(self, message):
        print(message)

    def inform_user(self, agent_actions):
        if self.print_dialog:
            self.send_to_user(self.agent_message(agent_actions))

        self._update_state_agent(agent_actions)

    def _update_state_agent(self, agent_actions):
        self.history.append({
            'agent_action': agent_actions,
            'user_action': None,
            'state_user': None
        })

        no_data_parts = dict()
        for action, slot_name, slot_value in agent_actions:
            if action=='canthelp':
                no_data_parts[slot_name] = slot_value

        if len(no_data_parts)>0:
            self.state['no_data_requests'].append(no_data_parts)


    def _update_state_user(self, user_actions):
        for action, slot_name, slot_value in user_actions:
            if action=='inform':
                self.state['user_informed'].add(slot_name)
            elif action=='request':
                self.state['user_requested'].add(slot_name)

        self.history[-1]['user_action'] = user_actions
        self.history[-1]['state_user'] = deepcopy(self.state)

    def create_features_for_turn(self):
        features = []

        # What user/agent did in the last N turns
        for j in range(4):
            index = len(self.history) - j - 1
            agent_acts_for_binary = []
            agent_acts_slots_for_binary = []

            agent_acts_history = self.history[index]['agent_action'] if index >= 0 else []
            if len(agent_acts_history) == 0:
                agent_acts_for_binary.append('empty')
                agent_acts_slots_for_binary.append('empty')
            else:
                for act, slot_name, slot_value in agent_acts_history:
                    agent_acts_for_binary.append(act)
                    agent_acts_slots_for_binary.append(act + ('_' + slot_name if slot_name is not None else ''))

            features.extend(np.max(self.binarizers['agent_act'].transform(agent_acts_for_binary), axis=0))
            features.extend(np.max(self.binarizers['agent_act_slots'].transform(agent_acts_slots_for_binary), axis=0))

            if j > 0:
                user_acts_for_binary = []
                user_acts_slots_for_binary = []

                user_acts_history = self.history[index]['user_action'] if index > 0 else []
                if len(user_acts_history) == 0:
                    user_acts_for_binary.append('empty')
                    user_acts_slots_for_binary.append('empty')
                else:
                    for act, slot_name, slot_value in user_acts_history:
                        user_acts_for_binary.append(act)
                        user_acts_slots_for_binary.append(act + ('_' + slot_name if slot_name is not None else ''))

                features.extend(np.max(self.binarizers['user_act'].transform(user_acts_for_binary), axis=0))
                features.extend(np.max(self.binarizers['user_act_slots'].transform(user_acts_slots_for_binary), axis=0))

        # User goals
        constraints = self.goal['constraints']
        requests = self.goal['request-slots']
        user_constraint_slots = ['empty'] if len(constraints) == 0 else [c[0] for c in constraints]
        user_request_slots = ['empty'] if len(requests) == 0 else requests
        features.extend(np.max(self.binarizers['user_request_slots'].transform(user_request_slots), axis=0))
        features.extend(np.max(self.binarizers['user_constraint_slots'].transform(user_constraint_slots), axis=0))

        state = deepcopy(self.state)
        if len(state['user_informed']) == 0:
            state['user_informed'].add('empty')

        if len(state['user_requested']) == 0:
            state['user_requested'].add('empty')

        # Not filled slots(from goals)
        not_filled_constraints = [c[0] for c in constraints if c[0] not in state['user_informed']]
        not_filled_requests = [c for c in requests if c not in state['user_requested']]
        if len(not_filled_constraints) == 0:
            not_filled_constraints.append('empty')

        if len(not_filled_requests) == 0:
            not_filled_requests.append('empty')

        features.extend(np.max(self.binarizers['user_constraint_slots'].transform(list(state['user_informed'])), axis=0))
        features.extend(np.max(self.binarizers['user_request_slots'].transform(list(state['user_requested'])), axis=0))

        features.extend(np.max(self.binarizers['user_constraint_slots'].transform(not_filled_constraints), axis=0))
        features.extend(np.max(self.binarizers['user_request_slots'].transform(not_filled_requests), axis=0))

        features.append(len(not_filled_constraints))
        features.append(len(not_filled_requests))
        features.append(len(not_filled_requests) + len(not_filled_constraints))

        features = hstack(features)
        return features

    def next(self):
        features = self.create_features_for_turn()
        pred = self.model.predict(xgb.DMatrix(features.tocsc()))[0]
        pred_index = self.content_manager.random_state.choice(list(range(pred.shape[0])), p=pred)
        pred_string = self.binarizers['user_act_all'].classes_[pred_index]
        parts = pred_string.split('__')
        user_actions = []
        for p in parts:
            if '_' in p:
                action, slot_name = p.split('_')
            else:
                action, slot_name = p, None

            if p=='inform_name':
                continue

            if p=='reqalts' and 'inform_name' in parts:
                continue

            slot_value = None
            if action=='inform' and slot_name is not None:
                if slot_name!='dontcare':
                    slot_value = self.goal['constraints_dict'][slot_name] if slot_name in self.goal['constraints_dict'] else self.content_manager.random_slot_value(slot_name)
                    if slot_name=='food' and any('food' in d for d in self.state['no_data_requests']) and 'alt_constraints' in self.goal:
                        slot_value = self.goal['alt_constraints'][1]

                else:
                    action = 'dontcare'
                    slot_name = None
            user_actions.append([action, slot_name, slot_value])

        if len(user_actions)==0:
            user_actions = [['empty', None, None]]

        if self.print_dialog:
            print(user_actions)

        self._update_state_user(user_actions)
        return user_actions, None
