import json
from fuzzywuzzy import process
from tqdm import tqdm


def search_ontology_for_acts(word, act_ontology, req=True, thresh=80):
    """
    return True if word or similar word in ontology
    else return False;
    
    // ontology = {'request':requests_list, 'inform':informs_list}
    
    """
    if len(word) < 4:
        return False

    name = 'request'
    if not req:
        name = 'inform'

    if word in act_ontology[name]:
        return True
    else:
        matches = process.extractOne(word, act_ontology[name])
        if matches[1] > thresh:
            # print("{} :acts match: {}".format(word, matches))
            return True
        else:
            return False


def search_ontology_for_slots(word, slot_ontology, thresh=80):
    """
    
    
    """

    if len(word) < 4:
        return {}

    matches = {}
    for key, arr in slot_ontology.items():
        val = process.extractOne(word, arr)
        matches[key] = 1 if val[1] > thresh else 0
        # if matches[key] == 1:
            # print("{} :slots match: {}".format(word, val))

    return matches


class FeatureGen:
    def __init__(self, ontology_path):
        with open(ontology_path, 'r') as fs:
            ont = json.load(fs)

        requests = ont['requestable'].values()
        requests = [v for m in requests for v in m]
        informs = ont['informable']['area'] + ont['informable']['food'] + ont['informable']['pricerange']
        self.ont_acts = {'request': requests, 'inform': informs}

        self.ont_slots = {
            'area': ont['informable']['area'] + ont['requestable']['area'],
            'food': ont['informable']['food'] + ont['requestable']['food'],
            'pricerange': ont['informable']['pricerange'] + ont['requestable']['pricerange'],
            'name': ont['requestable']['name'],
            'phone': ont['requestable']['phone'],
            'postcode': ont['requestable']['postcode'],
            'addr': ont['requestable']['addr']
        }

    def word2features(self, sent, i, for_acts=True):

        word = sent[i]
        features = {
            'bias': 1.0,
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[:3]': word[:3],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }

        if for_acts:
            features.update({
                'word.request': search_ontology_for_acts(word.lower(), act_ontology=self.ont_acts, req=True),
                'word.inform': search_ontology_for_acts(word.lower(), act_ontology=self.ont_acts, req=False)})
        else:
            matches = search_ontology_for_slots(word, self.ont_slots)
            features.update(matches)

        if i > 0:
            word1 = sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word[-3:]': word1[-3:],
            })
        else:
            features['BOS'] = True  # begin of sentence

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update({
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word[-3:]': word1[-3:],
            })

        else:
            features['EOS'] = True  # end of sentence

        return features

    def sent2features(self, sent, for_acts=True):
        """
        input looks like:
            sent = ['okay', 'and', 'can', 'i', 'get', 'their', 'phone']
            acts = ['O', 'O', 'O', 'O', 'O', 'O', 'B-request']
            slots = ['O', 'O', 'O', 'O', 'O', 'O', 'B-phone']

        here for_acts=True means that you would like to make features to predict acts;
        if you want predict slots you should pass for_acts=False;

        """
        if isinstance(sent, str):
            sent = sent.split()
        return [self.word2features(sent, i, for_acts=for_acts) for i in range(len(sent))]

    def batch2features(self, x_words, DataGen, for_acts=True):

        feats = []
        for x_ in tqdm(x_words):
            sent = DataGen.decode_sentence(x_)
            feats.append(self.sent2features(sent, for_acts=for_acts))
        return feats
