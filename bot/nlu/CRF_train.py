## it's a draft version without argparse and with COPY-PASTE style /
## Remove this message when optimize code /

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from feature_extractor import FeatureGen
from data_generator import NLUDataGenerator
import joblib

batch_size = 3000
TimeMajor = False

DataGenDev = NLUDataGenerator('../data/dev/usr_df_final.csv',
                              '../data/ontology_new.json', None,
                              batch_size=int(batch_size * 0.3), time_major=TimeMajor, digitize=False)

DataGen = NLUDataGenerator('../data/train/usr_df_final.csv',
                           '../data/ontology_new.json', DataGenDev.vocab,
                           batch_size=batch_size, time_major=TimeMajor, digitize=False)

fGen = FeatureGen("../data/ontology_new.json")

x, m1, m2 = next(DataGen)  # train
x_dev, m1_dev, m2_dev = next(DataGenDev)  # test

########### training for slots ###########################################################
## --  it takes approx ~ 4 min to extract features from 3000 sentences due to fuzzy search
##########################################################################################

X_train_slots = fGen.batch2features(x, for_acts=False)
y_train_slots = m1

X_test_slots = fGen.batch2features(x_dev, for_acts=False)
y_test_slots = m1_dev

crf_slots = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.005,
    c2=0.04,
    max_iterations=100,
    all_possible_transitions=True
)
crf_slots.fit(X_train_slots, y_train_slots)

# --------------------- test slots -------------------------------------------------------

labels_slots = list(crf_slots.classes_)
labels_slots.remove('O')

print("TRAIN:")
y_pred = crf_slots.predict(X_train_slots)
slots_f1_res = metrics.flat_f1_score(y_train_slots, y_pred,
                                     average='weighted', labels=labels_slots)

print("slots f1: {}\n More detailed:\n".format(slots_f1_res))
sorted_labels = sorted(
    labels_slots,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_train_slots, y_pred, labels=sorted_labels, digits=3
))


print("TEST:")
y_pred = crf_slots.predict(X_test_slots)
slots_f1_res = metrics.flat_f1_score(y_test_slots, y_pred,
                                     average='weighted', labels=labels_slots)

print("slots f1: {}\n More detailed:\n".format(slots_f1_res))
sorted_labels = sorted(
    labels_slots,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test_slots, y_pred, labels=sorted_labels, digits=3
))

# ------------------------- save ----------------------------------------------------------

joblib.dump(crf_slots, "./crf_slots_new.model")

########### training for acts ############################################################
## --  it takes approx ~ 4 min to extract features from 3000 sentences due to fuzzy search
##########################################################################################

X_train_acts = fGen.batch2features(x, for_acts=True)
y_train_acts = m2

X_test_acts = fGen.batch2features(x_dev, for_acts=True)
y_test_acts = m2_dev

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.003,
    c2=0.04,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train_acts, y_train_acts)

labels = list(crf.classes_)
labels.remove('O')

# --------------------- test acts ----------------------------------------------------------


print("TRAIN_acts:")
y_pred = crf.predict(X_train_acts)
slots_f1_res = metrics.flat_f1_score(y_train_acts, y_pred,
                                     average='weighted', labels=labels)

print("slots f1: {}\n More detailed:\n".format(slots_f1_res))
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_train_acts, y_pred, labels=sorted_labels, digits=3
))

print("TEST_acts:")
y_pred = crf.predict(X_test_acts)
slots_f1_res = metrics.flat_f1_score(y_test_acts, y_pred,
                                     average='weighted', labels=labels)

print("slots f1: {}\n More detailed:\n".format(slots_f1_res))
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test_acts, y_pred, labels=sorted_labels, digits=3
))

# ------------------------- save -----------------------------------------------------------

joblib.dump(crf, "./crf_acts_new.model")
