class UserSimulator:
    def __init__(self, content_manager, nlg):
        self.nlg = nlg
        self.content_manager = content_manager

    def inform_user(self, agent_actions):
        pass

    def next(self):
        pass

    def initialize_episode(self):
        pass

    def agent_message(self, agent_actions):
        return self.nlg.agent_action_nl(agent_actions)


class EchoUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg):
        super().__init__(content_manager, nlg)

    def inform_user(self, agent_actions):
        print(self.agent_message(agent_actions))

    def next(self):
        answer = 'hello'
        print(answer)
        return None, answer


class ConsoleUserSimulator(UserSimulator):
    def inform_user(self, agent_actions):
        print(self.agent_message(agent_actions))

    def next(self):
        answer = input()
        return None, answer


class TelegramUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg, update):
        super().__init__(content_manager, nlg)
        self.update = update
        self.message = None

    def inform_user(self, agent_actions):
        self.update.message.reply_text(self.agent_message(agent_actions))

    def next(self):
        return None, self.message

    def new_message(self, message):
        self.message = message


class SupervisedUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg, print_dialog=True):
        super().__init__(content_manager, nlg)
        self.print_dialog = print_dialog

    def initialize_episode(self):
        self.goal = self.content_manager.random_goal()

    def inform_user(self, agent_actions):
        if self.print_dialog:
            print(self.agent_message(agent_actions))

    def next(self):
        pass