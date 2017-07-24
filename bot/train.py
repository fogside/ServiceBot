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
nlu = NluPattern(settings.DSTC_PATH)
binarizers = pickle.load(open(settings.SL_BINARIZERS_PATH, 'rb'))

max_turn = 10
user = SupervisedUserSimulator(content_manager, nlg, settings.SL_US_MODEL_PATH, binarizers, max_turn,
                               print_goal=True)
#user = ConsoleUserSimulator(content_manager, nlg)
agent = RLAgent(content_manager=content_manager, path_to_model=settings.RLAGENT_MODEL,
                path_to_dstc=settings.DSTC_PATH, binarizers=binarizers, max_turn=max_turn,
                send_to_user=user.send_to_user)

#agent = RuleAgent(content_manager)
#agent = ConsoleAgent(content_manager)

dialog_manager = DialogManager(agent, user, content_manager, nlu, print_every_n=1, stats_every=50, max_turn=None)
dialog_manager.start()