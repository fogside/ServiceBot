from usersims import *
from agents import *
from managers import *
from nlg import *
from nlu import *
import settings
from numpy.random import RandomState
import xgboost as xgb
import pickle

USE_TELEGRAM = True

random_state = RandomState(12)
content_manager = ContentManager.from_settings(settings, random_state)
nlg = NlgPattern(content_manager)
nlu = NluMl(content_manager, settings.NLU_ACTON_MODEL_PATH, settings.NLU_SLOT_MODEL_PATH, settings.ONTOLOGY_PATH)
binarizers = pickle.load(open(settings.SL_BINARIZERS_PATH, 'rb'))
model = pickle.load(open('../scripts/sl_agent.cat', 'rb'))

agent = SLAgent(content_manager, model, binarizers)
if USE_TELEGRAM:
    dialog_manager = TelegramDialogManager(lambda : RuleAgent(content_manager), nlu,nlg, content_manager, settings.TELEGRAM_BOT_API_KEY)
    dialog_manager.idle()
else:
    #agent = RuleAgent(content_manager)
    #agent = ConsoleAgent(content_manager)
    user = ConsoleUserSimulator(content_manager, nlg)
    dialog_manager = DialogManager(agent, user, content_manager, nlu)
    dialog_manager.start()