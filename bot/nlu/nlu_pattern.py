class NluPattern:
    def parse_user_actions(self, string):
        if string =='hello':
            return [
                ['hello', None, None]
            ]

        raise Exception('NLU: Unknown string = {}'.format(string))