from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import traceback


class UserSimulator:
    def __init__(self, content_manager, nlg):
        self.nlg = nlg
        self.content_manager = content_manager

    def next(self, agent_actions):
        pass

    def initialize_episode(self):
        pass

    def agent_message(self, agent_actions):
        return self.nlg.agent_action_nl(agent_actions)


class EchoUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg):
        super().__init__(content_manager, nlg)

    def next(self, agent_actions):
        print(self.agent_message(agent_actions))
        answer = 'hello'
        print(answer)
        return None, answer


class ConsoleUserSimulator(UserSimulator):
    def next(self, agent_actions):
        print(self.agent_message(agent_actions))
        answer = input()
        return None, answer


class TelegramUserSimulator(UserSimulator):
    def __init__(self, content_manager, nlg, api_key):
        super().__init__(content_manager, nlg)
        self.api_key = api_key

        self.updater = Updater(api_key)

        dp = self.updater.dispatcher
        dp.add_error_handler(self.error)

        self.updater.start_polling()

    def set_newchat_handler(self, handler):
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler("start", handler))

    def set_message_handler(self, handler):
        dp = self.updater.dispatcher
        dp.add_handler(MessageHandler(Filters.text, handler))

    @staticmethod
    def from_settings(settings):
        return TelegramUserSimulator(settings.TELEGRAM_BOT_API_KEY)

    def error(bot, update, error):
        print(traceback.format_exc())

    def next(self, agent_message):
        print(agent_message)
        answer = input()
        return answer
