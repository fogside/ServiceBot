from .agent import Agent
import os
import pickle
import glob
import json
from dstc_helper import *


class RLAgent(Agent):
    def __init__(self, content_manager, path_to_train, path_to_model, path_to_dstc, need_sl, sl_rounds, rounds):
        super().__init__(content_manager)
        self.path_to_model = path_to_model
        self.path_to_train = path_to_train
        self.path_to_dstc = path_to_dstc

        if need_sl:
            if os.path.exists(path_to_train):
                self.sl_data = pickle.load(open(path_to_train, 'rb'))
            else:
                self.sl_data = self.make_sl_data()

            self.sl()

    def make_sl_data(self):
        for i, label_path in enumerate(glob.glob('{}/**/label.json'.format(self.path_to_dstc), recursive=True)):
            label = json.load(open(label_path))
            log = json.load(open(label_path.replace('label', 'log')))
            log_turns = log['turns']
            label_turns = label['turns']
            filter_all_acts(label_turns, log_turns)

    def sl(self):
        pass
