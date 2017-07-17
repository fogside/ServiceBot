def filter_all_acts(label_turns, log_turns):
    for i in range(len(label_turns)):
        label_turns[i]['semantics']['json'] = sort_filter_acts(label_turns[i]['semantics']['json'], True)

    for i in range(len(log_turns)):
        log_turns[i]['output']['dialog-acts'] = sort_filter_acts(log_turns[i]['output']['dialog-acts'], False)


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