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
from bot.usersims.sl_usersim import create_features_for_turn


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
        dstc_turns_to_triplets(label_turns, log_turns)

        if len(label_turns) != len(log_turns):
            return result
        for i in range(len(log_turns)):
            agent_acts = log_turns[i]
            user_acts = label_turns[i]

            turn_act_slots = []
            for act in agent_acts:
                agent_act.add(act[0])
                # if act[0]=='offer':
                # continue

                act_slot = act[0] +  ('_' + act[1] if act[1] is not None else '')
                if act[1] is not None:
                    agent_act_slots.add(act_slot)

                turn_act_slots.append(act_slot)
            turn_act_slots = sorted(turn_act_slots)
            if len(turn_act_slots)>3 and 'inform_pricerange' in turn_act_slots:
                turn_act_slots.remove('inform_pricerange')

            turn_act_slots_str = '__'.join(turn_act_slots)
            #if 'inform_name__inform_postcode__inform_pricerange' in turn_act_slots_str:
               # print('{}, {}'.format(log_turns[i]['output']['transcript'], turn_act_slots))

            agent_act_full.add(turn_act_slots_str)

            user_act_all_key = []
            for act in user_acts:
                user_act.add(act[0])

                key = act[0] + ('_' + act[1] if act[1] is not None else '')
                # if key== 'inform_area__negate':
                # continue
                if act[1] is not None:
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

    act_to_remove = ['', 'inform_name', 'inform_area__inform_name__inform_phone__inform_postcode',
                     'canthelp_pricerange', 'canthelp_area', 'canthelp_food'
                     ]
    for act in act_to_remove:
        if act in agent_act_full:
            agent_act_full.remove(act)

    agent_act_full.add('canthelp')
    user_act.remove('bye')

    result['agent_act'] = agent_act
    result['user_act'] = user_act
    result['agent_act_slots'] = agent_act_slots
    result['user_act_slots'] = user_act_slots
    result['user_act_all'] = user_act_all
    result['user_request_slots'] = user_request_slots
    result['user_constraint_slots'] = user_constraint_slots
    result['all_slots'] = user_request_slots|user_constraint_slots

    for key, value in list(result.items()):
        lb = LabelBinarizer() if key != 'user_act_all' else LabelEncoder()
        lb.fit(list(value))
        result[key] = lb
        result[key+'_reverse'] = {name: i for i, name in enumerate(lb.classes_)}

    agent_act_full = sorted(list(agent_act_full))
    result['agent_act_full'] = {a: i for i, a in enumerate(agent_act_full)}
    pickle.dump(result, open('../bot/models/supervised_user_simulator_binarizers.p', 'wb'))


def process_dialog(label_path):
    result_x = []
    result_y = []
    label = json.load(open(label_path))
    log = json.load(open(label_path.replace('label', 'log')))
    log_turns = log['turns']
    label_turns = label['turns']
    filter_all_acts(label_turns, log_turns)

    binarizers = pickle.load(open('../bot/models/supervised_user_simulator_binarizers.p', 'rb'))
    if len(label_turns) != len(log_turns):
        return result_x, result_y

    goal = fill_goal_from_label(label)
    state = defaultdict(set)
    dstc_turns_to_triplets(label_turns, log_turns)

    for i in range(len(log_turns)):
        x = create_features_for_turn(binarizers, goal, log_turns, label_turns, i, state)
        if x is None:
            continue

        user_act_all = fill_state(binarizers, state, goal, log_turns[i], label_turns[i])
        key = '__'.join(user_act_all) if len(user_act_all) > 0 else 'empty'
        if 'bye' in key:
            continue

        y = binarizers['user_act_all'].transform([key])[0]
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
        pickle.dump(model, open('../bot/models/supervised_user_simulator_model.p', 'wb'))
        #predicted = model.predict(d_valid)
        #result[test_index] = predicted
        break


def filter_all_acts(label_turns, log_turns):
    for i in range(len(label_turns)):
        label_turns[i]['semantics']['json'] = sort_filter_acts(label_turns[i]['semantics']['json'], True)

    for i in range(len(log_turns)):
        log_turns[i]['output']['dialog-acts'] = sort_filter_acts(log_turns[i]['output']['dialog-acts'], False)


def fill_state(binarizers, state, goal, agent_acts, user_acts):
    # Agent action
    agent_act_all = []
    for act in agent_acts:
        key = act[0] + '_' + act[1] if act[1] is not None else act[0]
        if act[0]=='canthelp':
            key = 'canthelp'

        agent_act_all.append(key)

        if act[1] is not None:
            if act[0] == 'inform' and act[1]:
                state['proposed_slots'][act[1]] = [act[2], set()]
            elif act[0] == 'request' and act[1]:
                state['agent_request_slots'].add(act[1])

    # User action
    for act in user_acts:
        # What user already did(inform/request)
        if act[1] is not None:
            if act[0] == 'inform' and act[1] in binarizers['user_constraint_slots'].classes_:
                state['inform_slots'][act[1]] = act[2]
            elif act[0] == 'request' and act[1] in binarizers['user_request_slots'].classes_:
                state['request_slots'].add(act[1])
            elif act[0] == 'dontcare':
                if len(agent_acts) == 1 and agent_acts[0][0] == 'request':
                    state['inform_slots'][agent_acts[0][1]] = 'dontcare'

    if len(agent_act_all)>3 and 'inform_name' in agent_act_all:
        for i in range(len(agent_act_all)-1, -1, -1):
            if agent_act_all[i]!='inform_name':
                del agent_act_all[i]
                break

    if agent_act_all == ['inform_name']:
        if 'food' in state['inform_slots']:
            agent_act_all.insert(0, 'inform_food')
        elif 'area' in state['inform_slots']:
            agent_act_all.insert(0, 'inform_area')

    return agent_act_all


def predict_for_dialog(label_path):
    model = pickle.load(open('../bot/models/supervised_user_simulator_model.p', 'rb'))
    label = json.load(open(label_path))
    log = json.load(open(label_path.replace('label', 'log')))
    log_turns = log['turns']
    label_turns = label['turns']
    filter_all_acts(label_turns, log_turns)
    binarizers = pickle.load(open('../bot/models/supervised_user_simulator_binarizers.p', 'rb'))
    goal = fill_goal_from_label(label)
    state = defaultdict(set)
    dstc_turns_to_triplets(label_turns, log_turns)
    for i in range(len(log_turns)):
        x = create_features_for_turn(binarizers, goal, log_turns, label_turns, i, state)
        fill_state(binarizers, state, goal, log_turns[i], label_turns[i])
        y_pred = model.predict(xgb.DMatrix(x.tocsc()))[0]
        y_pred_max = y_pred.argmax()
        print('Turn = {} Real = {} Predicted = {}'.format(i, label_turns[i], binarizers['user_act_all'].classes_[y_pred_max]))


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
    goal['reqalts'] = 'ask if there is anything' in goal['text']

    return goal


def create_goals_file():
    all_goals = []
    for i, label_path in enumerate(glob.glob('../data/dstc2_traindev/**/label.json', recursive=True)):
        label = json.load(open(label_path))
        goal = label['task-information']['goal']

        all_goals.append(goal)

    json.dump(all_goals, open('../data/goals.json', 'w'))

if  __name__=="__main__":
    usersim_binarizers()
    #process_all_dialogs()
    #train()


    # result = process_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-db80a9e6df-20130328_230811\\label.json')
    #predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-db80a9e6df-20130328_230811\\label.json')
    #predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S0A0\\voip-14cb91bc48-20130328_161626\\label.json')
    #predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-e0035cc31b-20130323_211354\\label.json')
    #predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S0A0\\voip-ad40cf5489-20130325_181825\\label.json')
    #predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S1A1\\voip-e54437a6f0-20130325_133942\\label.json')
    #predict_for_dialog('..\data\dstc2_traindev\data\Mar13_S0A1\\voip-597cfafdee-20130328_231524\\label.json')# Empty первым действием у юзера


