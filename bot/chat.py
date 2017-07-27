from usersims import *
from agents import *
from managers import *
from nlg import *
from nlu import *
import settings
from numpy.random import RandomState

USE_TELEGRAM = False

random_state = RandomState(12)
content_manager = ContentManager.from_settings(settings, random_state)
nlg = NlgPattern(content_manager)
nlu = NLU_crf(settings.ONTOLOGY_PATH, settings.MODEL_SLOTS_PATH, settings.MODEL_ACTS_PATH)
if USE_TELEGRAM:
    dialog_manager = TelegramDialogManager(lambda: RuleAgent(content_manager), nlu, nlg, content_manager,
                                           settings.TELEGRAM_BOT_API_KEY)
    dialog_manager.idle()
else:
    agent = RuleAgent(content_manager)
    user = ConsoleUserSimulator(content_manager, nlg)
    dialog_manager = DialogManager(agent, user, content_manager, nlu)
    dialog_manager.start()
