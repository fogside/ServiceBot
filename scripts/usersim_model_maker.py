import glob
import json
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pickle
from scipy.sparse import hstack, vstack
import xgboost as xgb
from sklearn.cross_validation import KFold
from collections import defaultdict
import re
from bot.dstc_helper import *


def usersim_binarizers():
    result = dict()
    agent_act = {'empty'}
    user_act = {'empty'}
    agent_act_slots = {'empty'}
    user_act_slots = {'empty'}
    user_act_all = {'empty'}
    user_request_slots = {'empty'}
    user_constraint_slots = {'empty'}
    agent_act_full = set()

    for label_path in glob.glob('../data/dstc2_traindev/**/label.json', recursive=True):
        label = json.load(open(label_path))
        log = json.load(open(label_path.replace('label', 'log')))
        log_turns = log['turns']
        label_turns = label['turns']
        filter_all_acts(label_turns, log_turns)

        if len(label_turns) != len(log_turns):
            return result
        for i in range(len(log_turns)):
            agent_acts = log_turns[i]['output']['dialog-acts']
            user_acts = label_turns[i]['semantics']['json']

            turn_act_slots = []
            for act in agent_acts:
                agent_act.add(act['act'])
                # if act['act']=='offer':
                # continue
                slot_index = 1 if act['act']=='request' else 0
                act_slot = act['act'] + '_' + act['slots'][0][slot_index] if len(act['slots']) > 0 else act['act']
                agent_act_slots.add(act_slot)
                turn_act_slots.append(act_slot)
            turn_act_slots = sorted(turn_act_slots)
            if len(turn_act_slots)>3 and 'inform_pricerange' in turn_act_slots:
                turn_act_slots.remove('inform_pricerange')

            turn_act_slots_str = '__'.join(turn_act_slots)
            if 'inform_name__inform_postcode__inform_pricerange' in turn_act_slots_str:
                print('{}, {}'.format(log_turns[i]['output']['transcript'], turn_act_slots))

            agent_act_full.add(turn_act_slots_str)

            user_act_all_key = []
            for act in user_acts:
                user_act.add(act['act'])
                key = act['act'] + '_' + act['slots'][0][0] if len(act['slots']) > 0 else act['act']
                # if key== 'inform_area__negate':
                # continue
                user_act_slots.add(key)
                user_act_all_key.append(key)

            if len(user_act_all_key) > 0:
                key = '__'.join(user_act_all_key)
                # if key=='inform_area__negate':
                # continue
                if not 'bye' in key:
                    user_act_all.add(key)

        # User goals and constraints
        goal = label['task-information']['goal']
        constraints = goal['constraints']
        requests = goal['request-slots']
        for c in constraints:
            user_constraint_slots.add(c[0])
        for c in requests:
            user_request_slots.add(c)

    act_to_remove = ['', 'inform_name', 'inform_area__inform_name__inform_phone__inform_postcode']
    for act in act_to_remove:
        if act in agent_act_full:
            agent_act_full.remove(act)

    user_act.remove('bye')
    user_act_slots.remove('bye')

    result['agent_act'] = agent_act
    result['user_act'] = user_act
    result['agent_act_slots'] = agent_act_slots
    result['user_act_slots'] = user_act_slots
    result['user_act_all'] = user_act_all
    result['user_request_slots'] = user_request_slots
    result['user_constraint_slots'] = user_constraint_slots
    result['all_slots'] = user_request_slots|user_constraint_slots

    for key, value in result.items():
        lb = LabelBinarizer() if key != 'user_act_all' else LabelEncoder()
        lb.fit(list(value))
        result[key] = lb

    agent_act_full = sorted(list(agent_act_full))
    result['agent_act_full'] = {a: i for i, a in enumerate(agent_act_full)}
    pickle.dump(result, open('supervised_user_simulator_binarizers.p', 'wb'))


def create_features_for_turn(binarizers, goal, log_turns, label_turns, i, state):
    features = []

    # What user/agent did in the last N turns
    for j in range(4):
        index = i - j
        agent_acts = []
        agent_acts_slots = []

        agent_acts_history = log_turns[index]['output']['dialog-acts'] if index >= 0 else []
        if len(agent_acts_history) == 0:
            agent_acts.append('empty')
            agent_acts_slots.append('empty')
        else:
            for act in agent_acts_history:
                agent_acts.append(act['act'])
                slot_index = 1 if act['act'] == 'request' else 0
                agent_acts_slots.append(act['act'] + ('_' + act['slots'][0][slot_index] if len(act['slots']) > 0 else ''))

        features.extend(np.max(binarizers['agent_act'].transform(agent_acts), axis=0))
        features.extend(np.max(binarizers['agent_act_slots'].transform(agent_acts_slots), axis=0))

        if j > 0:
            user_acts = []
            user_acts_slots = []

            user_acts_history = label_turns[index]['semantics']['json'] if index > 0 else []
            if len(user_acts_history) == 0:
                user_acts.append('empty')
                user_acts_slots.append('empty')
            else:
                for act in user_acts_history:
                    user_acts.append(act['act'])
                    user_acts_slots.append(act['act'] + ('_' + act['slots'][0][0] if len(act['slots']) > 0 else ''))

            features.extend(np.max(binarizers['user_act'].transform(user_acts), axis=0))
            features.extend(np.max(binarizers['user_act_slots'].transform(user_acts_slots), axis=0))

    # User goals
    constraints = goal['constraints']
    requests = goal['request-slots']
    user_constraint_slots = ['empty'] if len(constraints) == 0 else [c[0] for c in constraints]
    user_request_slots = ['empty'] if len(requests) == 0 else requests
    features.extend(np.max(binarizers['user_request_slots'].transform(user_request_slots), axis=0))
    features.extend(np.max(binarizers['user_constraint_slots'].transform(user_constraint_slots), axis=0))

    if len(state['user_informed']) == 0:
        state['user_informed'].add('empty')

    if len(state['user_requested']) == 0:
        state['user_requested'].add('empty')

    # Not filled slots(from goals)
    not_filled_constraints = [c[0] for c in constraints if c[0] not in state['user_informed']]
    not_filled_requests = [c for c in requests if c not in state['user_requested']]
    if len(not_filled_constraints)==0:
        not_filled_constraints.append('empty')

    if len(not_filled_requests) == 0:
        not_filled_requests.append('empty')

    features.extend(np.max(binarizers['user_constraint_slots'].transform(list(state['user_informed'])), axis=0))
    features.extend(np.max(binarizers['user_request_slots'].transform(list(state['user_requested'])), axis=0))

    features.extend(np.max(binarizers['user_constraint_slots'].transform(not_filled_constraints), axis=0))
    features.extend(np.max(binarizers['user_request_slots'].transform(not_filled_requests), axis=0))

    features.append(len(not_filled_constraints))
    features.append(len(not_filled_requests))
    features.append(len(not_filled_requests)+len(not_filled_constraints))

    features = hstack(features)

    user_acts = label_turns[i]['semantics']['json']
    user_act_all = []
    for act in user_acts:
        first_slot_value = act['slots'][0][0] if len(act['slots']) > 0 else None
        key = act['act'] + '_' + first_slot_value if len(act['slots']) > 0 else act['act']
        user_act_all.append(key)

        # What user already did(inform/request)
        if first_slot_value is not None:
            if act['act'] == 'inform' and first_slot_value in binarizers['user_constraint_slots'].classes_:
                state['user_informed'].add(first_slot_value)
            elif act['act'] == 'request' and first_slot_value in binarizers['user_request_slots'].classes_:
                state['user_requested'].add(first_slot_value)

    key = '__'.join(user_act_all) if len(user_act_all) > 0 else 'empty'
    if 'bye' in key:
        return None, None

    return features, binarizers['user_act_all'].transform([key])[0]


def process_dialog(label_path):
    result_x = []
    result_y = []
    label = json.load(open(label_path))
    log = json.load(open(label_path.replace('label', 'log')))
    log_turns = log['turns']
    label_turns = label['turns']
    filter_all_acts(label_turns, log_turns)

    binarizers = pickle.load(open('supervised_user_simulator_binarizers.p', 'rb'))
    if len(label_turns) != len(log_turns):
        return result_x, result_y

    goal = fill_goal_from_label(label)
    state = defaultdict(set)
    for i in range(len(log_turns)):
        x, y = create_features_for_turn(binarizers, goal, log_turns, label_turns, i, state)
        if x is None or y is None:
            continue

        result_x.append(x)
        result_y.append(y)

    return vstack(result_x), result_y


def process_all_dialogs():
    X = []
    Y = []
    for i, label_path in enumerate(glob.glob('../data/dstc2_traindev/**/label.json', recursive=True)):
        if i > 0 and i % 10 == 0:
            print(i)
            #if i==300:
             #   break

        x, y = process_dialog(label_path)
        if len(y) > 0:
            X.append(x)
            Y.extend(y)
    X = vstack(X)
    Y = np.array(Y)
    pickle.dump(X, open('X.p', 'wb'))
    pickle.dump(Y, open('Y.p', 'wb'))


def train():
    X = pickle.load(open('X.p', 'rb')).tocsc()
    Y = pickle.load(open('Y.p', 'rb'))

    params = {}
    params['objective'] = 'multi:softprob'
    params['eval_metric'] = 'mlogloss'
    params['eta'] = 0.05
    params['max_depth'] = 6
    params['silent'] = 1
    params['num_class'] = max(Y) + 1

    kfolds = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=39)
    result = np.zeros(X.shape[0])
    num_boost_round = 300
    for i, (train_index, test_index) in enumerate(kfolds):
        print('Fold = {}'.format(str(i)))

        d_train = xgb.DMatrix(X[train_index], label=Y[train_index])
        d_valid = xgb.DMatrix(X[test_index], label=Y[test_index])

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(params, d_train, num_boost_round, watchlist, early_stopping_rounds=2, verbose_eval=10)
        pickle.dump(model, open('supervised_user_simulator_model.p', 'wb'))
        #predicted = model.predict(d_valid)
        #result[test_index] = predicted
        break


def filter_all_acts(label_turns, log_turns):
    for i in range(len(label_turns)):
        label_turns[i]['semantics']['json'] = sort_filter_acts(label_turns[i]['semantics']['json'], True)

    for i in range(len(log_turns)):
        log_turns[i]['output']['dialog-acts'] = sort_filter_acts(log_turns[i]['output']['dialog-acts'], False)


def predict_for_dialog(label_path):
    model = pickle.load(open('supervised_user_simulator_model.p', 'rb'))
    label = json.load(open(label_path))
    log = json.load(open(label_path.replace('label', 'log')))
    log_turns = log['turns']
    label_turns = label['turns']
    filter_all_acts(label_turns, log_turns)
    binarizers = pickle.load(open('supervised_user_simulator_binarizers.p', 'rb'))
    goal = fill_goal_from_label(label)
    state = defaultdict(set)

    for i in range(len(log_turns)):
        user_acts = label_turns[i]['semantics']['json']
        x, y = create_features_for_turn(binarizers, goal, log_turns, label_turns, i, state)
        y_pred = model.predict(xgb.DMatrix(x.tocsc()))[0]
        y_pred_max = y_pred.argmax()
        print('Turn = {} Real = {} Predicted = {}'.format(i, user_acts, binarizers['user_act_all'].classes_[y_pred_max]))


def fill_goal_from_label(label):
    """
    Not all slots present in original goal
    """
    goal, label_turns = label['task-information']['goal'], label['turns']
    constraints = set()
    requests = set()
    for turn in label_turns:
        for act in turn['semantics']['json']:
            if len(act['slots'])==0:
                continue

            if act['act']=='request':
                requests.add(act['slots'][0][0])
            elif act['act']=='inform':
                constraints.add(act['slots'][0][0])

    goal['constraints'] = [c for c in goal['constraints'] if c[0] in constraints]
    goal['request-slots'] = [c for c in goal['request-slots'] if c in requests]

    return goal

def create_goals_file():
    all_goals = []
    for i, label_path in enumerate(glob.glob('../data/dstc2_traindev/**/label.json', recursive=True)):
        label = json.load(open(label_path))
        goal = label['task-information']['goal']

        all_goals.append(goal)

    json.dump(all_goals, open('../data/goals.json', 'w'))



#usersim_binarizers()
process_all_dialogs()
train()


# result = process_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-db80a9e6df-20130328_230811\\label.json')
#predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-db80a9e6df-20130328_230811\\label.json')
#predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S0A0\\voip-14cb91bc48-20130328_161626\\label.json')
#predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-e0035cc31b-20130323_211354\\label.json')
#predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S0A0\\voip-ad40cf5489-20130325_181825\\label.json')
#predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-e54437a6f0-20130325_133942\\label.json')
#predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S0A1\\voip-597cfafdee-20130328_231524\\label.json')# Empty первым действием у юзера


