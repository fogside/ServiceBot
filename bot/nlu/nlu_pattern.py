class Nlu:
    def parse_user_actions(self, string):
        pass


class NluPattern(Nlu):
    def parse_user_actions(self, string):
        if string in ['hello']:
            return [
                ['hello', None, None]
            ]
        if string in ['i would like to find an expensive restaurant in the south part']:
            return [
                ['inform', 'area', 'south'],
                ['inform', 'pricerange', 'expensive']
            ]
        if string in ['does not matter', 'any type of food is okay']:
            return [
                ['dontcare', None, None]
            ]
        if string in ['what is the address']:
            return [
                ['request', 'addr', None]
            ]

        if string in ['what is the phone number']:
            return [
                ['request', 'phone', None]
            ]

        if string in ['what type of food']:
            return [
                ['request', 'food', None]
            ]

        if string in ['bye']:
            return [
                ['bye', None, None]
            ]

        if string in ['russian']:
            return [
                ['inform', 'food', 'russian']
            ]

        if string in ['indian']:
            return [
                ['inform', 'food', 'indian']
            ]
        if string in ['how about asian oriental food']:
            return [
                ['inform', 'food', 'asian oriental']
            ]

        raise Exception('NLU: Unknown string = {}'.format(string))