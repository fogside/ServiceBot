from .feature_extractor import FeatureGen
import joblib as jb
import sys
import re


class NLU_crf:
    def __init__(self, ontology_path, model_slot_path, model_acts_path):
        self.slots_crf = jb.load(model_slot_path)
        self.acts_crf = jb.load(model_acts_path)
        self.fGen = FeatureGen(ontology_path)
        self.no_slots_acts = set(('affirm', 'reqalts', 'bye', 'hello', 'dontcare', 'negate'))
        self.req_acts = ['request', 'dontcare_slot']  # when the value is not important and transformed to None

    @staticmethod
    def make_clean_tokens(sent):
        """
        :param sent: it's a string like: "I dont;#$% 5675 like cucumbers!"
        :return: clean tokens like: ['I', 'dont', 'like', 'cucumbers']
        
        """
        return re.findall(r'[^\s!,.?"%$#:;0-9]+', sent)

    def parse_user_actions(self, sent):
        """
        :param sent: it's a string
        :return: triples like these:
            "moderately priced restaurant south part of town": 
            [["inform", "area", "south"], 
            ["inform", "pricerange", "moderate"]]
        """
        triples = []
        tokens = self.make_clean_tokens(sent)

        # TODO: add returning correct word form according to ontology
        sent_feats_slots = self.fGen.sent2features(tokens, for_acts=False)
        sent_feats_acts = self.fGen.sent2features(tokens, for_acts=True)

        p_slots = self.slots_crf.predict_single(sent_feats_slots)
        p_acts = self.acts_crf.predict_single(sent_feats_acts)

        act = list(set([p[2:] for p in p_acts]) & self.no_slots_acts)

        if len(act) == 1:
            print("act: ", act[0])
            return [[act[0], None, None]]

        prev_slot = ''
        for t, slot, act in zip(tokens, p_slots, p_acts):
            if act != 'O':

                if (prev_slot.replace('B', 'I') == slot) or (prev_slot == slot):
                    triples[-1] = [triples[-1][0], slot, triples[-1][2] + ' ' + t]
                else:
                    triples.append([act, slot, t])

                prev_slot = slot

        if len(triples) == 0:
            print("empty")
            triples.append(['empty', None, None])
            return triples
        else:
            # remove I-, B-:
            triples = [[act[2:], slot[2:], t] if act[2:] not in self.req_acts
                       else [act[2:], slot[2:], None] for act, slot, t in triples]
            print("triples: ", triples)
            sys.stdout.flush()
            return triples


if __name__ == "__main__":
    model_slot_path = "./crf_slots_new.model"
    model_acts_path = "./crf_acts_new.model"
    ontology_path = "../data/ontology_new.json"

    sent1 = "I don't care about price range. What about rusian food restaurant?"
    sent2 = "Moderately priced restaurant south part of town"
    sent3 = "Yes, I want it"
    sent4 = "I'm looking for delicious food in the center of town."
    sent5 = "No italian food"
    sent6 = "I dontcare pricerange, what about danish food?"
    sent7 = 'italian'
    sent8 = 'cheap'
    sent9 = 'fuck it, no!'
    sent10 = 'Yes, correct'
    sent11 = 'No, I dont like that'
    sent12 = 'is it in the south part of town'
    sent13 = 'alright good bye thank you'
    sent14 = 'hi!'
    sent15 = 'fuck creative'
    sent16 = 'is there anything else, please'
    sent17 = 'can i have the addres and phone number please'
    sent18 = 'no what about danish food'
    sent19 = 'no danish food'
    sent20 = 'no italian food'
    sent21 = 'wrong'
    sent22 = 'doesnt matter'

    sents = [sent1, sent2, sent3, sent4, sent5, sent6, sent7,
             sent8, sent9, sent10, sent11, sent12, sent13, sent14,
             sent15, sent16, sent17, sent18, sent19, sent20, sent21, sent22]

    nlu = NLU_crf(ontology_path, model_slot_path, model_acts_path)

    for s in sents:
        print("test_sent: ", s)
        triples = nlu.parse_user_actions(s)
        print(triples)
        print('-------')
