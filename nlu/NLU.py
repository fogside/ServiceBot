from feature_extractor import FeatureGen
import joblib as jb
import re

class NLU_crf:
    def __init__(self, ontology_path, model_slot_path, model_acts_path):
        self.slots_crf = jb.load(model_slot_path)
        self.acts_crf = jb.load(model_acts_path)
        self.fGen = FeatureGen(ontology_path)

    def make_clean_tokens(self, sent):
        """
        :param sent: it's a string like: "I dont;#$% 5675 like cucumbers!"
        :return: clean tokens like: ['I', 'dont', 'like', 'cucumbers']
        
        """
        return re.findall(r'[^\s!,.?"%$#:;0-9]+', sent)

    def make_triples_for_sent(self, sent):
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

        p_slots = self.slots_crf.predict([sent_feats_slots])[0]
        p_acts = self.acts_crf.predict([sent_feats_acts])[0]

        ## debug:
        # print(p_slots)
        # print(p_acts)
        # print('<debug>----')

        # TODO: add processing B-slot, I-slot into one token
        for t, slot, act in zip(tokens, p_slots, p_acts):
            if act!='O':
                t = None if act == 'request' else t
                slot = None if slot == 'O' else slot
                triples.append([act, slot, t])
        return triples

if __name__ == "__main__":
    model_slot_path = "./crf_slots_new.model"
    model_acts_path = "./crf_acts_new.model"
    ontology_path = "../data/ontology_new.json"

    test_sent = "I don't care about price range. What about rusian food restaurant?"
    test_sent2 = "moderately priced restaurant south part of town"

    nlu = NLU_crf(ontology_path, model_slot_path, model_acts_path)

    print("test_sent: ", test_sent)
    triples = nlu.make_triples_for_sent(test_sent)
    print(triples)

    print("test_sent: ", test_sent2)
    triples = nlu.make_triples_for_sent(test_sent2)
    print(triples)