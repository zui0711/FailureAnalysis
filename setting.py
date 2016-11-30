#from utils import *
from gensim.models import Word2Vec

m_dic_size = 100  # 99+1
m_hidden_dim = 72

m_sent_len = 20
m_word_dim = 50

m_path = "data/network_diagnosis_data/cut1000/"
#m_path =
m_model_w2v_name = "word_emb_size50wind100count100iter30"
m_epoch = 5
#model_w2v = Word2Vec.load(path + "word_emb_size50wind100count100iter30")

#print model_w2v
#E = np.eye(dic_size + 1, dic_size + 1).astype(theano.config.floatX)

_learning_rate = 0.05

#idx2word, word2idx = load_dic("data/dic.txt", dic_size)

# print idx2word
