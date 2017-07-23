from collections import defaultdict
import re
import json
from numpy.random import choice

def add_restaurant_for_goals():
    import bot.settings as settings
    from bot.managers import ContentManager
    from numpy.random import RandomState
    content_manager = ContentManager.from_settings(settings, RandomState(1))

    goals = json.load(open(settings.GOALS_PATH))
    restaurant_dict = content_manager.restaurant_dict

    for goal in goals:
        goal['reqalts'] = 'ask if there is anything' in goal['text']
        goal['constraints_dict'] = {key: value for key, value in goal['constraints']}

    keys = list(restaurant_dict[list(restaurant_dict.keys())[0]].keys())
    max_retaurant_index = max(map(int, restaurant_dict.keys()))
    for goal in goals:
        if len(content_manager.available_results(goal['constraints_dict'], []))==0:
            exists = True
            constraints_dict = goal['constraints_dict'].copy()
            if 'alt_constraints' not in goal:
                exists = False
            else:
                constraints_dict['food'] = goal['alt_constraints'][1]
                if len(content_manager.available_results(constraints_dict, [])) == 0:
                    exists = False


            if exists:
                continue

            print('No restaurant for goal = {}'.format(str(goal)))
            restaurant = constraints_dict
            for key in keys:
                if not key in restaurant:
                    restaurant[key] = content_manager.random_slot_value(key)

            max_retaurant_index+=1
            restaurant_dict[str(max_retaurant_index)] = restaurant
            content_manager._make_cache_by_slot()

    json.dump(goals, open(settings.GOALS_PATH, 'w'))
    json.dump(restaurant_dict, open(settings.RESTAURANT_PATH, 'w'))

def creat_restaurants():
    restaurant_dict = defaultdict(dict)
    ontology = json.load(open('../data/ontology_dstc2.json', 'r'))
    text = open('../data/train/dialog_raw.txt', 'r').read()
    text += '\n' + open('../data/dev/dialog_raw.txt', 'r').read()

    valid_names = set(ontology['informable']['name'])
    available_slots = {'area', 'phone', 'food', 'pricerange', 'postcode', 'addr'}

    patterns = [
        (re.compile('^(.+) is a (?:nice|great) (?:place|restaurant) in the (.+) of town serving (.+) food[\n .]*$', re.MULTILINE), ['name', 'area', 'food']),
        (re.compile('^The phone number of (.+) is (.+) and its postcode is ([^,]+)', re.MULTILINE), ['name', 'phone', 'postcode']),
        (re.compile('^(.+) is a (?:nice|great) (?:place|restaurant) in the (.+) of town and the prices are (.+)[\n .]*$', re.MULTILINE), ['name', 'area', 'pricerange']),
        (re.compile('^(.+) is a (?:nice|great) (?:place|restaurant) serving (.+) food and it is in the (.+) price range[\n .]*$', re.MULTILINE), ['name', 'food', 'pricerange']),
        (re.compile('^The phone number of (.+) is (.+) [.][\n .]*$', re.MULTILINE), ['name', 'phone']),
        (re.compile('^(.+) is a (?:nice|great) (?:place|restaurant) in the (.+) of town[\n .]*$', re.MULTILINE), ['name', 'area']),
        (re.compile('^Sure , (.+) is on (.+)[\n .]*$', re.MULTILINE), ['name', 'addr']),
        (re.compile('^The phone number of the (.+) is (.+) and it is on (.+) [.][\n .]*$', re.MULTILINE), ['name', 'phone', 'addr']),
        (re.compile('^(.+) is in the (.+) price range[\n .]*$', re.MULTILINE), ['name', 'pricerange']),
        (re.compile('^(.+) serves (.+) food[\n .]*$', re.MULTILINE), ['name', 'food']),
        (re.compile('^The post code of (.+) is ([^,]+)', re.MULTILINE), ['name', 'postcode']),
        (re.compile('^(.+) is a (?:nice|great) (?:place|restaurant) in the (.+) of town in the (.+) price range[\n .]*$', re.MULTILINE), ['name', 'area', 'pricerange']),
        (re.compile('^(.+) is in the (.+) part of town .+[\n .]*$', re.MULTILINE), ['name', 'area']),
        (re.compile('^The price range at (.+) is (.+)[\n .]*$', re.MULTILINE), ['name', 'pricerange']),
    ]

    for pattern, slots in patterns:
        for match in pattern.finditer(text):
            slot_values = {slot: slot_value for slot, slot_value in zip(slots, match.groups())}
            name = slot_values['name']
            if name.lower() not in valid_names:
                continue

            for key,value in slot_values.items():
                if key=='name':
                    continue
                value = value.strip()
                if key =='food' and 'tasty 'in value:
                    value = value.replace('tasty ', '')

                if value.lower() not in ontology['informable'][key]:
                    raise Exception("Unknown value = {} for Key = {}".format(value, key))
                restaurant_dict[name][key] = value


    for name in restaurant_dict:
        if 'postcode' not in restaurant_dict[name]:
            restaurant_dict[name]['postcode'] = choice(ontology['informable']['postcode'])

        if 'phone' not in restaurant_dict[name]:
            restaurant_dict[name]['phone'] = choice(ontology['informable']['phone'])

        if 'addr' not in restaurant_dict[name]:
            restaurant_dict[name]['addr'] = choice(ontology['informable']['addr'])

    not_filled_restaurant = list(key for key in restaurant_dict if len(restaurant_dict[key])<6)
    print('Restaurant count = {} Not filled count = {}'.format(len(restaurant_dict), len(not_filled_restaurant)))
    for name in not_filled_restaurant:
        print('Name = {} No slots = {}'.format(name, {s for s in available_slots if s not in restaurant_dict[name]}))

    result_dict = defaultdict(dict)
    for i, key in enumerate(restaurant_dict):
        result_dict[i] = restaurant_dict[key]
        result_dict[i]['name'] = key

    json.dump(result_dict, open('restaurant_dict.json', 'w'))

add_restaurant_for_goals()