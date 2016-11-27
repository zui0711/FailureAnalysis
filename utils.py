import numpy as np
import theano
from gensim.models import Word2Vec

unknown_word = "UNKNOWN"


def load_dic(dic_file, dic_size=-1):
    idx2word = []
    word2idx = {}
    i = 0
    with open(dic_file, "rb") as f:
        for line in f:
            word = line.strip().split(",")[0]
            idx2word.append(word)
            word2idx[word] = i
            i += 1
            if dic_size != -1 and i == dic_size:
                idx2word.append(unknown_word)
                word2idx[unknown_word] = i
                break

    return idx2word, word2idx


def load_model(path, modelname):
    return Word2Vec.load(path + modelname)


def sent2vector(sentence, model_w2v, sent_len, word_dim):
    retvector = []
    thislen = len(sentence.split())
    for word in sentence.strip().split(" "):
        retvector.append(model_w2v[word] if word in model_w2v)
    for i in xrange(sent_len - thislen):
        retvector.append(np.zeros(word_dim))
    return np.array(retvector)

"""
def vector2sent(vector, idx2word):
    sentence = [idx2word[word] for word in vector]
    return " ".join(sentence)
"""

def format_sent(sent, word2idx, sent_len):
    # retvector = np.zeros(sent_len)
    # for i, word in enumerate(sent.strip().split(" ")):
    #    retvector[i] = (word2idx[word] if word in word2idx else word2idx[unknown_word])
    retvector = np.array(sent2vector(sent, word2idx)).astype("int32")
    return retvector


def format_sent_cnn(sent, model_w2v, sent_len):
    retvector = np.zeros(sent_len, dtype="int32")
    for i, word in enumerate(sent.strip().split(" ")):
        retvector[i] = (word2idx[word] if word in word2idx else word2idx[unknown_word])
    # retvector = np.array(sent2vector(sent, word2idx)).astype("int32")
    return retvector


if __name__ == "__main__":
    dic_size = 100
    idx2word, word2idx = load_dic("data/dic.txt", dic_size)
    print idx2word, len(idx2word)
    print word2idx
    f = open("data/all.txt", "rb")
    context = f.readlines()[:10]
    for line in context:
        # print vector2sent(sent2vector(line, word2idx),idx2word)
        print format_sent(line, word2idx, 20)

