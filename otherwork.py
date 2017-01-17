# coding=utf-8

# import re
# from setting import *
# import matplotlib.pyplot as plt
# import numpy as np
# import os
from utils import *
from models.RBM import *

# import cPickle as pickle
#
# dic = {}
# with open("data/log_log.txt", "rb") as f:
#     context = f.readlines()
#     for line in context:
#         sentence = line.split(" ")
#         for word in sentence:
#             if word in dic:
#                 dic[word] += 1
#             else:
#                 dic[word] = 1
#
# sorted_dic = sorted(dic.items(), key=lambda e: e[1], reverse=True)
#
# f = open("outfile", 'wb')
# pickle.dump(sorted_dic, f, pickle.HIGHEST_PROTOCOL)


# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import random

this_path = pjoin(m_path, "cut500/all/")

# file_names = []
# for dir_name in os.listdir(this_path):
#     for file_name in os.listdir(pjoin(this_path, dir_name)):
#         file_names.append(".".join([dir_name, file_name]))
#
# file_num = len(file_names)
# file_names_idx = range(file_num)
# random.shuffle(file_names_idx)
#
# train_len = 1000
# test_len = 200
# train_idxs = file_names_idx[:train_len]
# test_idxs = file_names_idx[train_len:train_len + test_len]
#
# model_w2v = load_model_onehot(m_path, m_model_w2v_name, 100)
# model_rbm = RBM()
# model_rbm.load_model(pjoin(m_path, "rbm_model_2000_800"))
#
# this_idxs = train_idxs#[i * batch_size: (i + 1) * batch_size]
# trainx, trainy = get_batchdata_sent_onehot(this_path, file_names, this_idxs, model_w2v, model_rbm, sent_len=15,
#                                            word_dim=100, text_size=500)
#
# this_idxs = test_idxs#[i * batch_size: (i + 1) * batch_size]
# testx, testy = get_batchdata_sent_onehot(this_path, file_names, this_idxs, model_w2v, model_rbm, sent_len=15,
#                                            word_dim=100, text_size=500)
#
# f = open(pjoin(m_path, "rbm_input"), "wb")
# pickle.dump([trainx, trainy], f, 1)
# pickle.dump([testx, testy], f, 1)

f = open(pjoin(m_path, "rbm_input"), "rb")
[trainx, trainy] = pickle.load(f)
[testx, testy] = pickle.load(f)


print('Building model...')
model = Sequential()
model.add(Dense(400, input_shape=(400000,)))
model.add(Activation('relu'))
model.add(Dense(40, input_shape=(400,)))
model.add(Activation('relu'))
# model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Activation('softmax'))

# for ep in xrange(epoch):
#     print
#     "epoch = ", ep
#     costep = 0
#
#     for i in xrange(train_num):



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(trainx.reshape(1000, 400000), np_utils.to_categorical(trainy, 4),
                    nb_epoch=10, batch_size=50,
                    verbose=1, validation_split=0.1)

print(model.get_weights()[-1])

history = model.fit(trainx.reshape(1000, 400000), np_utils.to_categorical(trainy, 4),
                    nb_epoch=10, batch_size=50,
                    verbose=1, validation_split=0.1)

print(model.get_weights()[-1])

score = model.evaluate(testx.reshape(200, 400000), np_utils.to_categorical(testy, 4),
                       batch_size=20, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])