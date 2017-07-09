import numpy as np
from nlu.data_generator import NLUDataGenerator

PAD = 0

TimeMajor = False
batch_size = 32
encoder_max_time = 64

data_gen = NLUDataGenerator('../data/train/usr_df_final.csv',
                           '../data/ontology_dstc2.json',
                           '../data/slots.txt', None, seq_len = encoder_max_time,
                           batch_size = batch_size, time_major=TimeMajor)

x, m1, m2, _lens = next(data_gen)
print(x)