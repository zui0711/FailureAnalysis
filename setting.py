from utils import *

dic_size = 100  # 99+1
hidden_dim = 72
sent_len = 18

E = np.eye(dic_size + 1, dic_size + 1).astype(theano.config.floatX)

learning_rate = 0.05

idx2word, word2idx = load_dic("data/dic.txt", dic_size)

# print idx2word
