import numpy as np
import pandas as pd
import json
import copy
import random
from itertools import cycle

def read_json(path):
    return json.load(open(path, 'r'))

class NLUDataGenerator:
    def __init__(self, path_to_template, path_to_dict, path_to_slot, seq_len=64, batch_size=32, time_major = True):

        self.time_major = time_major
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dict = read_json(path_to_dict)  # for random choices and slot filling;
        self.slots = list(np.array(pd.read_csv(path_to_slot, sep='\n')).reshape(-1))
        self.spec_no_slots = ['bye', 'hello', 'reqalts', 'doncare']  # all template is marked always;
        self.spec_with_slots = ['affirm', 'negate']  # if there's no slots all template is marked;
        templates = pd.read_csv(path_to_template).fillna(-1)

        self.templates = []  # for templates with blanks;
        self.vocab = set()  # all the words in all templates;
        self.slots_encode = set() # all slots
        self.acts_encode = set() # all acts

        for index, row in templates.iterrows():
            user_line = row["nl"]
            acts = [row["act" + str(i)] for i in range(1, 4) if row["act" + str(i)] != -1]
            bio_slots, bio_acts, fill_slots = self.template_to_BIO(user_line, acts)
            self.templates.append((user_line, bio_slots, bio_acts, fill_slots))

        for type_ in ['informable', 'requestable']:
            for tag in self.dict[type_]:
                for filling in self.dict[type_][tag]:
                    for w in filling.split():  # Конечно, это вызывает вопросы, т.к имена будут явно разнесены(
                        self.vocab.add(w)

        self.vocab = dict(zip(self.vocab, range(3, len(self.vocab) + 3))) # + 'O' + PAD + EOF
        self.slots_encode = dict(zip(self.slots_encode, range(3, len(self.slots_encode) + 3))) # + 'O' + PAD + EOF
        self.acts_encode = dict(zip(self.acts_encode, range(3, len(self.acts_encode) + 3))) # + 'O' + PAD + EOF

        for dic in [self.vocab, self.slots_encode, self.acts_encode]:
            dic["PAD"] = 0
            dic["EOF"] = 1
            dic["O"] = 2

        self.inverse_vocab = {val: key for key, val in self.vocab.items()}
        self.inverse_slots = {val: key for key, val in self.slots_encode.items()}
        self.inverse_acts = {val: key for key, val in self.acts_encode.items()}
        self.vocab_size = len(self.vocab)

    def template_to_BIO(self, template, acts):

        bio_slots = []
        bio_acts = []
        acts_iter = cycle(acts)
        fill_slots = True

        for w in template.split():
            if w[0] == w[-1] == "$":
                continue
            self.vocab.add(w)

        # special case:
        if len(acts) == 1:
            if (acts[0] in self.spec_no_slots) or \
                    (('$' not in template) and (acts[0] in self.spec_with_slots)):
                nl_len = len(template.split())
                bio_slots.extend(['B-' + acts[0]] * nl_len)  # slots have the same value as act;
                bio_acts.extend(['B-' + acts[0]] * nl_len)
                self.acts_encode.add('B-' + acts[0])
                self.acts_encode.add('I-' + acts[0])
                self.slots_encode.add('B-' + acts[0])
                self.slots_encode.add('I-' + acts[0])

                fill_slots = False
                return bio_slots, bio_acts, fill_slots
        # else:
        for token in template.split():
            if token[0] == token[-1] == "$":
                slot = token[1:-1]
                bio_slots.append('B-' + slot)
                act_ = 'B-' + next(acts_iter)
                bio_acts.append(act_)

                self.acts_encode.add(act_)
                self.acts_encode.add(act_.replace('B-', 'I-'))
                self.slots_encode.add('B-' + slot)
                self.slots_encode.add('I-' + slot)
            else:
                bio_slots.append("O")
                bio_acts.append("O")
        return bio_slots, bio_acts, fill_slots

    def __next__(self):
        batch = random.sample(self.templates, self.batch_size)
        filled_batch = []
        for nl, slot_arr, act_arr, fill_slots in batch:
            nl_arr = nl.split()
            input_ = []
            target_slot = []
            target_acts = []

            if not fill_slots:
                input_.extend(nl_arr)
                target_slot.extend(slot_arr)
                target_acts.extend(act_arr)
            else:
                for i in range(len(nl_arr)):
                    if slot_arr[i] == 'O':
                        input_.append(nl_arr[i])
                        target_slot.append(slot_arr[i])
                        target_acts.append(act_arr[i])
                    else:  # slot_arr[i] is B-smth
                        slot = slot_arr[i].split("-")[1]
                        act = act_arr[i].split("-")[1]
                        if act == 'request':
                            filler = random.choice(self.dict['requestable'][slot]).split()
                        else:
                            filler = random.choice(self.dict['informable'][slot]).split()
                        input_.append(filler[0])
                        target_slot.append(slot_arr[i])
                        target_acts.append(act_arr[i])
                        for f in filler[1:]:
                            target_slot.append("I-" + slot)
                            target_acts.append("I-" + act)

                            input_.append(f)
            filled_batch.append((copy.deepcopy(input_), copy.deepcopy(target_slot), copy.deepcopy(target_acts)))
        return self.digitaze(filled_batch)

    def digitaze(self, batch):

        max_size = max([len(item[0]) for item in batch])
        targets_slots = np.zeros(shape=[self.batch_size, max_size], dtype=np.int32)  # == PAD
        targets_acts = np.zeros(shape=[self.batch_size, max_size], dtype=np.int32)  # == PAD
        dseqs = np.zeros(shape=[self.batch_size, max_size], dtype=np.int32)  # == PAD
        actual_lengths = []

        for i, row in enumerate(batch):
            actual_lengths.append(len(row[0]))
            for j, elements in enumerate(zip(row[0], row[1], row[2])):
                dseqs[i, j] = self.vocab[elements[0]]
                targets_slots[i, j] = self.slots_encode[elements[1]]
                targets_acts[i, j] = self.acts_encode[elements[2]]

        if self.time_major:
            dseqs = dseqs.swapaxes(0,1)
        return dseqs, targets_slots, targets_acts, actual_lengths

    def decode_sentence(self, seq):
        """
        Expected array of digits

        """
        return [self.inverse_vocab[i] for i in seq if self.inverse_vocab[i] != 'PAD']

    def decode_slots(self, seq):
        """
        Expected array of digits

        """
        return [self.inverse_slots[i] for i in seq if self.inverse_slots[i] != 'PAD']

    def decode_acts(self, seq):
        """
        Expected array of digits

        """
        return [self.inverse_acts[i] for i in seq if self.inverse_acts[i] != 'PAD']



