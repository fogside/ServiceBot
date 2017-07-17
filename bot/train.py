from usersims import *
from agents import *
from managers import *
from nlg import *
from nlu import *
import settings
from numpy.random import RandomState

random_state = RandomState(13)
content_manager = ContentManager.from_settings(settings, random_state)
nlg = NlgPattern(content_manager)
nlu = NluPattern()

agent = RuleAgent(content_manager)
user = SupervisedUserSimulator(content_manager, nlg, settings.SL_US_MODEL_PATH, settings.SL_BINARIZERS_PATH)
dialog_manager = DialogManager(agent, user, content_manager, nlu)
dialog_manager.start()