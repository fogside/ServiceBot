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
    def __init__(self, path_to_dstc, path_to_save):
        self.path_to_save = path_to_save
        self.path_to_dstc = path_to_dstc
        self.init()

    def init(self):
        self.text_to_actions = json.load(open(self.path_to_save)) if os.path.exists(self.path_to_save) else defaultdict(list)
        if len(self.text_to_actions)>0:
            return

        for label_path in glob.glob('{}/**/label.json'.format(self.path_to_dstc), recursive=True):
            label = json.load(open(label_path))
            log = json.load(open(label_path.replace('label', 'log')))

            log_turns = log['turns']
            label_turns = label['turns']

            label_transcripts = [t['transcription'] for t in label_turns]

            dstc_turns_to_triplets(label_turns, log_turns)
            for text, actions in zip(label_transcripts, label_turns):
                self.text_to_actions[text].append(actions)

        for key, value in self.text_to_actions.items():
            self.text_to_actions[key] = random.choice(value)

        json.dump(self.text_to_actions, open(self.path_to_save, 'w'))

    def parse_user_actions(self, string):
        if string in self.text_to_actions:
            return self.text_to_actions[string]

        raise Exception('NLU: Unknown string = {}'.format(string))
