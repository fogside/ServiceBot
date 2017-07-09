import json

class ContentManager:
    def __init__(self, act_set, slot_set, restaurant_dict):
        self.restaurant_dict = restaurant_dict
        self.slot_set = slot_set
        self.act_set = act_set

    @staticmethod
    def from_settings(settings):
        act_set = set(open(settings.ACT_SET_PATH).readlines())
        slot_set = set(open(settings.SLOTS_SET_PATH).readlines())
        restaurant_dict = json.load(open(settings.RESTAURANT_PATH))
        return ContentManager(act_set, slot_set, restaurant_dict)