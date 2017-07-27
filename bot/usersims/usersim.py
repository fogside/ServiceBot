class UserSimulator:
    def __init__(self, content_manager, nlg):
        self.nlg = nlg
        self.content_manager = content_manager

    def send_to_user(self, message):
        pass

    def inform_person(self, agent_actions):
        self.send_to_user(self.agent_message_nl(agent_actions))

    def next(self):
        pass

    def initialize_episode(self):
        pass

    def agent_message_nl(self, agent_actions):
        return self.nlg.agent_action_nl(agent_actions)


class EchoUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg):
        super().__init__(content_manager, nlg)

    def send_to_user(self, message):
        print(message)

    def next(self):
        answer = 'hello'
        print(answer)
        return None, answer

class ConsoleUserSimulator(UserSimulator):
    def send_to_user(self, message):
        print(message)

    def next(self):
        answer = input()
        return None, answer

class TelegramUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg, update):
        super().__init__(content_manager, nlg)
        self.update = update
        self.message = None

    def send_to_user(self, message):
        self.update.message.reply_text(message)

    def next(self):
        return None, self.message

    def new_message(self, message):
        self.message = message


