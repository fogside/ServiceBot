from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import traceback
from usersims import TelegramUserSimulator
import logging
import os


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    def __init__(self, agent, user, content_manager, nlu):
        self.nlu = nlu
        self.content_manager = content_manager
        self.user = user
        self.agent = agent
        self.dialog_number = 0
        self.turn_number = 0
        self.initialize_episode()

    def initialize_episode(self):
        self.agent.initialize_episode()
        self.user.initialize_episode()

        self.dialog_number += 1
        self.user.send_to_user('>'+str(self.dialog_number))

    def agent_action(self):
        self.agent_actions = self.agent.next()
        self.agent.update_state_agent(self.agent_actions)

        self.user.inform_user(self.agent_actions)

    def user_action(self):
        user_actions, user_message = self.user.next()
        if user_message is not None and user_actions is None:
            user_actions = self.nlu.parse_user_actions(user_message)

        goal = None
        if 'goal' in dir(self.user):
            goal = self.user.goal
        self.agent.update_state_user(user_actions, goal=goal)

        if 'bye' in [ua[0] for ua in user_actions]:
            self.initialize_episode()

    def next_turn(self, reverse=False):
        if reverse:
            self.user_action()
            self.agent_action()
        else:
            self.agent_action()
            self.user_action()

        self.turn_number += 1

    def start(self):
        while True:
            self.next_turn()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()]
                    )

logger = logging.getLogger(__name__)


class TelegramDialogManager:
    def __init__(self, agent_constructor, nlu, nlg, content_manager, api_key):
        self.nlg = nlg
        self.nlu = nlu
        self.content_manager = content_manager
        self.agent_constructor = agent_constructor
        self.chats = dict()
        self.updater = Updater(api_key)

        dp = self.updater.dispatcher
        dp.add_error_handler(self.error)
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(MessageHandler(Filters.text, self.message))

        self.updater.start_polling()

    def start(self, bot, update):
        self.new_chat(bot, update)

    def error(self, bot, update, error):
        print(traceback.format_exc())

    def new_chat(self, bot, update):
        user = TelegramUserSimulator(self.content_manager, self.nlg, update)
        inner_manager = DialogManager(self.agent_constructor(), user, self.content_manager, self.nlu)
        self.chats[update.message.chat_id] = inner_manager
        inner_manager.agent_action()

    def message(self, bot, update):
        if update.message.chat_id not in self.chats:
            self.new_chat(bot, update)
            return

        self.chats[update.message.chat_id].user.new_message(update.message.text)
        self.chats[update.message.chat_id].next_turn(reverse=True)

    def idle(self):
        self.updater.idle()
