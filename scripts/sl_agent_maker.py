from managers import ContentManager
from scripts.usersim_model_maker import *
from copy import deepcopy
import numpy as np
import settings
from numpy.random import RandomState
from catboost import Pool, CatBoostRegressor
from sklearn.cross_validation import train_test_split
from bot.agents.sl_agent import create_features_for_turn
import xgboost as xgb


def goal_constraints(goal):
    return {a: b for a, b in goal['constraints']}


def estimate_reward(binarizers, goal, agent_actions, user_actions, i, state, content_manager):
    state = deepcopy(state)

    max_index = min(len(agent_actions) - 1, i + 3)

    def score_state(state_before, state_after, turn):
        score = 0

        for slot, _ in state_after['proposed_slots'].values():
            if slot not in state_before and slot in goal['request-slots']:
                score += 5

        for intent, slot, value in agent_actions[turn]:
            if score <= 0:
                if intent == 'request' and slot in state_before['agent_request_slots']:
                    score -= 1

                elif intent == 'inform' and slot in state_before['proposed_slots'].values():
                    score -= 1

            if slot == 'name' and value is not None and slot not in state_before:
                rest = content_manager.restaurant_dict_by_name.get(value)
                if rest is not None:
                    constraints_dict = goal_constraints(goal)
                    if content_manager.valid_rest_for_constraints(rest, constraints_dict):
                        score += 5
                    else:
                        score -= 3

        return score

    scores = []
    for turn in range(i, max_index):
        state_before = deepcopy(state)
        fill_state(binarizers, state, goal, agent_actions[turn], user_actions[turn])
        state_after = deepcopy(state)
        scores.append(score_state(state_before, state_after, turn))

    return np.mean(scores) if len(scores) > 0 else 0


def process_dialog(label_path, content_manager):
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
    state = {
        'inform_slots': dict(),
        'request_slots': set(),
        'proposed_slots': dict(),  # Значение: [значение слота, множество Negate + ReqAlts]
        'agent_request_slots': set(),
    }
    dstc_turns_to_triplets(label_turns, log_turns)

    fill_state(binarizers, state, goal, log_turns[0], label_turns[0])
    for i in range(1, len(log_turns)):
        x = create_features_for_turn(binarizers, log_turns, label_turns, i, state, content_manager)
        if x is None:
            continue

        agent_act_all = fill_state(binarizers, state, goal, log_turns[i], label_turns[i])
        key = '__'.join(agent_act_all) if len(agent_act_all) > 0 else 'empty'

        if key not in binarizers['agent_act_full']:
            print('Agent action = {} not found in agent_act_full'.format(key))
            continue

        y = binarizers['agent_act_full'][key]

        result_x.append(x)
        result_y.append(y)

    return result_x, result_y


def process_all_dialogs():
    X = []
    Y = []
    random_state = RandomState(12)
    content_manager = ContentManager.from_settings(settings, random_state)

    for i, label_path in enumerate(glob.glob('../data/dstc2_traindev/**/label.json', recursive=True)):
        if i > 0 and i % 10 == 0:
            print(i)
            #if i==200:
              #break

        x, y = process_dialog(label_path, content_manager)
        if len(y) > 0:
            X.extend(x)
            Y.extend(y)

    X = np.array(X)
    Y = np.array(Y)
    pickle.dump(X, open('X.p', 'wb'))
    pickle.dump(Y, open('Y.p', 'wb'))


def train():
    X = pickle.load(open('X.p', 'rb'))
    Y = pickle.load(open('Y.p', 'rb'))

    print('X shape = {}'.format(X.shape))
    binarizers = pickle.load(open('../bot/models/supervised_user_simulator_binarizers.p', 'rb'))
    params = {'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'eta': 0.05, 'max_depth': 6, 'silent': 1,
              'num_class': len(binarizers['agent_act_full'])}

    #xgb.cv(params, xgb.DMatrix(X,Y), num_boost_round=100, nfold=4, verbose_eval=1)
    #return

    model = xgb.train(params, xgb.DMatrix(X, Y), 200, verbose_eval=1)
    pickle.dump(model, open('sl_agent.cat', 'wb'))


if __name__ == "__main__":
    process_all_dialogs()
    train()
