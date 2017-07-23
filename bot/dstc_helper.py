import re

def filter_all_acts(label_turns, log_turns):
    for i in range(len(label_turns)):
        label_turns[i]['semantics']['json'] = sort_filter_acts(label_turns[i]['semantics']['json'], True)

    for i in range(len(log_turns)):
        log_turns[i]['output']['dialog-acts'] = sort_filter_acts(log_turns[i]['output']['dialog-acts'], False)


def dstc_turns_to_triplets(label_turns, log_turns):
    filter_all_acts(label_turns, log_turns)
    for i in range(len(label_turns)):
        turn = label_turns[i]
        actions = []
        for act in turn['semantics']['json']:
            action = act['act']
            slot_name = None
            slot_value = None
            slots = act['slots']
            if len(slots)>0:
                slots = slots[0]
                if slots[0]== 'slot':
                    slot_name = slots[1]
                else:
                    slot_name = slots[0]
                    if len(slots)>1:
                        slot_value = slots[1]

            if action=='inform' and slot_name=='dontcare':
                action='dontcare'
                slot_name = None

            if action=='inform':
                was_inform = True

            if action=='reqalts':
                was_reqalt = True

            actions.append([action, slot_name, slot_value])

        label_turns[i] = actions

    for i in range(len(log_turns)):
        turn = log_turns[i]
        actions = []
        for act in turn['output']['dialog-acts']:
            action = act['act']
            slot_name = None
            slot_value = None
            slots = act['slots']
            if len(slots) > 0:
                slots = slots[0]
                if slots[0] == 'slot':
                    slot_name = slots[1]
                else:
                    slot_name = slots[0]
                    if len(slots) > 1:
                        slot_value = slots[1]

            actions.append([action, slot_name, slot_value])

        log_turns[i] = actions


def sort_filter_acts(acts, is_user):
    acts = filter_acts_user(acts) if is_user else filter_acts_agent(acts)
    return sorted(acts, key=lambda x: x['act'] + '_' + ('' if len(x['slots']) == 0 else x['slots'][0][0]))


def filter_acts_agent(acts):
    for i in range(len(acts) - 1, -1, -1):
        act = acts[i]
        if act['act'].strip() == '' or act['act'] in ['canthelp.exception', 'confirm-domain', 'impl-conf']:
            del acts[i]
            # if act['act'] in ['deny']:
            # act['act'] = 'negate'
        if act['act'] in ['offer']:
            act['act'] = 'inform'

    return acts


def filter_acts_user(acts):
    for i in range(len(acts) - 1, -1, -1):
        act = acts[i]
        if act['act'].strip() == '' or act['act'] in ['thankyou', 'restart', 'repeat']:
            del acts[i]
        if act['act'] in ['deny']:
            act['act'] = 'negate'

        if act['act'] == 'ack':
            act['act'] = 'affirm'

        if act['act'] == 'reqmore':
            act['act'] = 'reqalts'

        for slot in act['slots']:
            if 'this' in slot:
                slot.remove('this')

            if 'slot' in slot:
                slot.remove('slot')

    return acts

food_pattern = re.compile('.*it should serve ([^.]+?)(?: food)?[.]', re.IGNORECASE)


def add_alt_constraints(goal):
    if 'If there is no such venue' in goal['text']:
        match = food_pattern.match(goal['text'])
        if match is None:
            raise Exception('No alt constraint in {}'.format(goal['text']))
        for i, (slot_name, slot_value) in enumerate(list(goal['constraints'])):
            if slot_name == 'food':
                goal['alt_constraints'] = ['food', slot_value]
                goal['constraints'][i][1] = match.group(1)