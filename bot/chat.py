from usersims import *
from agents import *
from managers import *
from nlg import *
from nlu import *
import settings

content_manager = ContentManager.from_settings(settings)
nlg = NlgPattern(content_manager)
nlu = NluPattern()
user = ConsoleUserSimulator(content_manager, nlg)
agent = RuleAgent(content_manager)
if isinstance(user, TelegramUserSimulator):
    dialog_manager = TelegramDialogManager(agent, user, content_manager)
    dialog_manager.idle()
else:
    dialog_manager = DialogManager(agent, user, content_manager, nlu)
    dialog_manager.initialize_episode()

    while True:
        dialog_manager.next_turn()