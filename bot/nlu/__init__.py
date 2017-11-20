from .feature_extractor import FeatureGen
import joblib as jb
import re

word_pattern = re.compile('\w+')

class Nlu:
    def parse_user_actions(self, string):
        pass


class NluMl(Nlu):
    def __init__(self, content_manager, act_model_path, slot_model_path, ontology_path):
        self.slot_model = jb.load(slot_model_path)
        self.act_model = jb.load(act_model_path)
        self.content_manager = content_manager
        self.fgen = FeatureGen(ontology_path)

    def parse_user_actions(self, string):
        string = string.strip().lower()
        if string in self.content_manager.text_to_actions:
            return self.content_manager.text_to_actions[string]

        string = string.replace("'s", ' is')
        words = word_pattern.findall(string)

        acts = self.act_model.predict([self.fgen.sent2features(words, True)])[0]
        slots = self.slot_model.predict([self.fgen.sent2features(words, False)])[0]
        
        act_spans = []
        act_keys = []
        start = None
        for i in range(len(words)):
            if acts[i].startswith('B-'):
                if start is not None:
                    act_spans.append((start, i-1))
                    act_keys.append(acts[start])
                start = i
                    
            if acts[i]=='O':
                if start is not None:
                    act_spans.append((start, i-1))
                    act_keys.append(acts[start])
                
                start = None
                
        if start is not None:
            act_spans.append((start, len(words)-1))
            act_keys.append(acts[start])
        
        slot_keys = []
        slot_values = []
        slot_spans = []

        start = None
        for i in range(len(words)):
            if slots[i].startswith('B-'):
                if start is not None:
                    slot_spans.append((start, i-1))
                    slot_keys.append(slots[start].replace('B-', ''))
                    slot_values.append(' '.join(words[start:i]))
                start = i

            if slots[i] == 'O':
                if start is not None:
                    slot_spans.append((start, i-1))
                    slot_keys.append(slots[start])
                    slot_values.append(' '.join(words[start:i]))

                start = None

        if start is not None:
            slot_spans.append((start, len(words) - 1))
            slot_values.append(' '.join(words[start: len(words)]))
            slot_keys.append(slots[start])

        for i in range(len(act_keys)-len(slot_keys)):
            for j in range(len(act_keys)):
                if not any(act_spans[j][0]<=s<=act_spans[j][1] or act_spans[j][0] <= e <= act_spans[j][1] for s,e in slot_spans):
                    slot_spans.insert(j, act_spans[j])
                    slot_keys.insert(j, None)
                    slot_values.insert(j, None)
                    break

        result = [[act_keys[i].replace('B-', ''), slot_keys[i].replace('B-', '') if slot_keys[i] is not None else None, slot_values[i]] for i in range(len(act_keys))]
        result = list(filter(lambda x: x[0]!='inform' or x[1] is not None, result))
        print(result)
        return result


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
