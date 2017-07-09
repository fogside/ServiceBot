from collections import defaultdict
import re
import json
from numpy.random import choice

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