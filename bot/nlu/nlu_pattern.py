from collections import defaultdict
import glob
import json
from dstc_helper import dstc_turns_to_triplets
import random
import os


class Nlu:
    def parse_user_actions(self, string):
        pass

class NluPattern(Nlu):
    def __init__(self, content_manager):
        self.content_manager = content_manager

    def parse_user_actions(self, string):
        if string.startswith('['):
            return eval(string)

        string = string.strip()
        if string in self.content_manager.text_to_actions:
            return self.content_manager.text_to_actions[string]

        raise Exception('NLU: Unknown string = {}'.format(string))