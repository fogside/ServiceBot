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
nlu = NluPattern()
binarizers = pickle.load(open(settings.SL_BINARIZERS_PATH, 'rb'))

agent = RLAgent(content_manager, settings.RLAGENT_SL_TRAIN_DATA_PATH, settings.RLAGENT_MODEL,
                settings.DSTC_PATH, binarizers=binarizers, need_sl=True, sl_rounds=100, rounds=100,
                batch_size=100)


user = SupervisedUserSimulator(content_manager, nlg, settings.SL_US_MODEL_PATH, binarizers)
dialog_manager = DialogManager(agent, user, content_manager, nlu)
dialog_manager.start()