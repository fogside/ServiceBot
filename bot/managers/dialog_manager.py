class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    def __init__(self, agent, user, content_manager, nlu):
        self.nlu = nlu
        self.content_manager = content_manager
        self.user = user
        self.agent = agent

    def initialize_episode(self):
        self.agent.initialize_episode()
        self.user.initialize_episode()

    def next_turn(self):
        agent_actions = self.agent.next()
        user_actions, user_message = self.user.next(agent_actions)
        if user_message is not None and user_actions is None:
            user_actions = self.nlu.parse_user_actions(user_message)

        self.agent.update_state_user(user_actions)


class TelegramDialogContext:
    def __init__(self, agent, context_manager):
        self.context_manager = context_manager
        self.agent = agent
        self.dialog_number = 0

    def new_dialog_message(self):
        return '>{}\r\nHello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?'.format(self.dialog_number)

    def next_agent_action(self, user_message):
        return 'привет'


class TelegramDialogManager:
    def __init__(self, agent, telegram_user, content_manager):
        self.content_manager = content_manager
        self.telegram_user = telegram_user
        self.agent = agent
        self.chats = dict()
        self.telegram_user.set_newchat_handler(self.new_chat)
        self.telegram_user.set_message_handler(self.message)

    def new_chat(self, bot, update):
        context = TelegramDialogContext(self.agent, self.content_manager)
        self.chats[update.message.chat_id] = context
        update.message.reply_text(context.new_dialog_message())

    def message(self, bot, update):
        if update.message.chat_id not in self.chats:
            self.new_chat(bot, update)
            return

        update.message.reply_text(self.chats[update.message.chat_id].next_agent_action(update.message.text))

    def idle(self):
        self.telegram_user.updater.idle()
