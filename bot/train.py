from usersims import *
from agents import *
from managers import *
from nlg import *
from nlu import *
import settings


content_manager = ContentManager.from_settings(settings)
nlg = NlgPattern(content_manager)
nlu = NluPattern()

agent = RuleAgent(content_manager)
user = SupervisedUserSimulator(content_manager, nlg)
dialog_manager = DialogManager(agent, user, content_manager, nlu)
dialog_manager.start()