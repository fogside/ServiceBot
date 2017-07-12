from usersims import *
from agents import *
from managers import *
from nlg import *
from nlu import *
import settings

USE_TELEGRAM = False

content_manager = ContentManager.from_settings(settings)
nlg = NlgPattern(content_manager)
nlu = NluPattern()
if USE_TELEGRAM:
    dialog_manager = TelegramDialogManager(lambda : RuleAgent(content_manager), nlu,nlg, content_manager, settings.TELEGRAM_BOT_API_KEY)
    dialog_manager.idle()
else:
    agent = RuleAgent(content_manager)
    user = ConsoleUserSimulator(content_manager, nlg)
    dialog_manager = DialogManager(agent, user, content_manager, nlu)
    dialog_manager.start()