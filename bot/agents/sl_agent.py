from .agent import Agent
import numpy as np
import xgboost as xgb


def create_features_for_turn(binarizers, agent_actions, user_actions, i, state, content_manager):
    features = []

    # What user/agent did in the last N turns
    for j in range(1, 4):
        index = i - j
        agent_acts = []
        agent_acts_slots = []
        agent_slots = []

        agent_acts_history = agent_actions[index] if index >= 0 else []
        if len(agent_acts_history) == 0:
            agent_acts.append('empty')
            agent_acts_slots.append('empty')
        else:
            for act in agent_acts_history:
                agent_acts.append(act[0])
                if act[1] is not None:
                    agent_acts_slots.append(act[0] + '_' + act[1])
                    agent_slots.append(act[1])

        if len(agent_slots) == 0:
            agent_slots.append('empty')
            agent_acts_slots.append('empty')

        features.extend(np.max(binarizers['agent_act'].transform(agent_acts), axis=0))
        features.extend(np.max(binarizers['agent_act_slots'].transform(agent_acts_slots), axis=0))
        features.extend(np.max(binarizers['all_slots'].transform(agent_slots), axis=0))

        if j > 0:
            user_acts = []
            user_acts_slots = []
            user_slots = []

            user_acts_history = user_actions[index] if index > 0 else []
            if len(user_acts_history) == 0:
                user_acts.append('empty')
                user_acts_slots.append('empty')
            else:
                for act in user_acts_history:
                    user_acts.append(act[0])
                    if act[1] is not None:
                        user_acts_slots.append(act[0] + '_' + act[1])
                        user_slots.append(act[1])

            if len(user_slots) == 0:
                user_slots.append('empty')
                user_acts_slots.append('empty')

            features.extend(np.max(binarizers['user_act'].transform(user_acts), axis=0))
            features.extend(np.max(binarizers['user_act_slots'].transform(user_acts_slots), axis=0))
            features.extend(np.max(binarizers['all_slots'].transform(user_slots), axis=0))

    features.append(i)

    variant_count = len(content_manager.available_results(state['inform_slots'], state['proposed_slots']))
    if variant_count>0 and len(agent_actions[i])>0 and agent_actions[i][0][0]=='canthelp':
        variant_count = 0

    features.append(variant_count)
    features.append(variant_count==0)

    user_inform = set(state['inform_slots'])
    user_request = set(state['request_slots'])

    if len(user_inform) == 0:
        user_inform.add('empty')

    if len(user_request) == 0:
        user_request.add('empty')

    features.extend(np.max(binarizers['user_constraint_slots'].transform(list(user_inform)), axis=0))
    features.extend(np.max(binarizers['user_request_slots'].transform(list(user_request)), axis=0))

    features.append(len(user_inform))
    features.append(len(user_request))
    features.append(len(state['proposed_slots']))

    return np.array(features)


class SLAgent(Agent):
    def __init__(self, content_manager, model, binarizers):
        super().__init__(content_manager)
        self.model = model
        self.binarizers = binarizers
        self.agent_acts = [k for k,v in sorted(list(self.binarizers['agent_act_full'].items()), key=lambda x: x[1])]

    def create_features_for_turn(self):
        features = []
        actions = []
        proposed_rest = None
        variants = self.content_manager.available_results(self.inform_slots, self.proposed_slots)
        if 'name' in self.proposed_slots and self.proposed_slots['name'][0] not in self.proposed_slots['name'][1]:
            if self.proposed_slots['name'][0] in [v['name'] for v in variants]:
                proposed_name = self.proposed_slots['name'][0]
                proposed_rest = self.content_manager.restaurant_dict_by_name[proposed_name]

        if len(variants) > 0:
            proposed_rest = variants[0]

        user_actions = [h['user_action'] for h in self.history if h['user_action'] is not None]
        agent_actions = [h['agent_action'] for h in self.history if h['agent_action'] is not None]

        for act in self.agent_acts:
            parts = act.split('__')
            current_action = []
            invalid = False

            for p in parts:
                intent, slot = p.split('_') if '_' in p else (p, None)
                value = None
                if intent == 'inform':
                    if proposed_rest is None:
                        invalid = True
                        break

                    value = proposed_rest[slot]
                elif intent == 'canthelp':
                    if proposed_rest is not None:
                        invalid = True
                        break

                    if 'food' in self.inform_slots:
                        variants = self.content_manager.available_results({'food': self.inform_slots['food']},
                                                                          self.slot_restrictions)
                        if len(variants) == 0:
                            slot = 'food'
                            value = self.inform_slots['food']

                    if slot is None and 'area' in self.inform_slots and 'food' in self.inform_slots:
                        variants = self.content_manager.available_results(
                            {'food': self.inform_slots['food'], 'area': self.inform_slots['area']}, self.slot_restrictions)

                        if len(variants) == 0:
                            slot = 'area'
                            value = self.inform_slots['area']

                    if slot is None and 'pricerange' in self.inform_slots:
                        slot = 'pricerange'
                        value = self.inform_slots['pricerange']
                    elif slot is None:
                        invalid = True

                elif intent == 'expl-conf':
                    if slot not in self.state['inform_slots']:
                        invalid = True
                        break

                    value = self.state['inform_slots'][slot]

                current_action.append([intent, slot, value])

            # Если не добавляем новой информации - не делаем действие
            if all(a[0] == 'inform' and a[2] == self.proposed_slots.get(a[1], [None, None])[0] for a in
                   current_action) and agent_actions[self.turn_count-1]==current_action:
                invalid = True

            if invalid or len(current_action)==0:
                actions.append([])
                continue

            actions.append(current_action)

        user_actions.append([])
        agent_actions.append([])

        features.append(
            create_features_for_turn(self.binarizers, agent_actions, user_actions, self.turn_count,
                                           self.state, self.content_manager)
        )
        return {i: triplets for i, triplets in enumerate(actions)}, features

    def next(self):
        if self.turn_count == 0:
            return [['welcomemsg', None, None]]

        actions, features = self.create_features_for_turn()
        pred = self.model.predict(xgb.DMatrix(features))[0]
        pred = sorted(list(enumerate(pred)), key=lambda x:x[1], reverse=True)
        for i, _ in pred:
            if len(actions[i])>0:
                return actions[i]

        return [['welcomemsg', None, None]]
