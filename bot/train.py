from usersims import *
from agents import *
from managers import *
from nlg import *
from nlu import *
import settings
from numpy.random import RandomState
import pickle

random_state = RandomState(13)
content_manager = ContentManager.from_settings(settings, random_state)
nlg = NlgPattern(content_manager)
nlu = NluPattern(settings.DSTC_PATH, settings.NLU_PATTERN_SAVE_PATH)
binarizers = pickle.load(open(settings.SL_BINARIZERS_PATH, 'rb'))

agent = RLAgent(content_manager=content_manager, path_to_model= settings.RLAGENT_MODEL,
                path_to_dstc=settings.DSTC_PATH, binarizers=binarizers, max_turn=settings.MAX_TURN)
user = SupervisedUserSimulator(content_manager, nlg, settings.SL_US_MODEL_PATH, binarizers, settings.MAX_TURN)

#user = ConsoleUserSimulator(content_manager, nlg)
dialog_manager = DialogManager(agent, user, content_manager, nlu)
dialog_manager.start()