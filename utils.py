from setting import *
import numpy as np
import theano
import theano.tensor as T
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
    if thislen > sent_len:
        thislen = sent_len
    for i, word in enumerate(sentence.strip().split(" ")):
        if i == sent_len:
            break
        if word in model_w2v:
            retvector.append(model_w2v[word])
        else:
            thislen -= 1
    for i in xrange(sent_len - thislen):
        retvector.extend([[0 for i in xrange(word_dim)]])
    return retvector


# TODO
def format_sent(sent, word2idx, sent_len):
    retvector = np.array(sent2vector(sent, word2idx))
    return retvector


# TODO
def format_sent_cnn(sent, model_w2v, sent_len):
    retvector = np.zeros(sent_len, dtype="int32")
    for i, word in enumerate(sent.strip().split(" ")):
        retvector[i] = (word2idx[word] if word in word2idx else word2idx[unknown_word])
    return retvector


def get_batchdata(path, file_names, idxs, model_w2v, sent_len, word_dim):
    def a(s):
        print s

    def switch(label, y):
        try:
            {"NORMAL": lambda: y.append(0),
             "GTPC_TUNNEL_PATH_BROKEN": lambda: y.append(1),
             "Paging": lambda: y.append(2),
             "UeAbnormal": lambda: y.append(3)
             }[label]()
        except KeyError:
            a("Key not Found")

    rety = []
    for idx in idxs:
        name = file_names[idx]
        arr = name.split(".")
        this_file_name = ".".join([arr[1], arr[2]])
        switch(arr[0], rety)

        count = 0
        with open(path + name, "rb") as f:
            for line in f:
                if locals().has_key("retx"):
                    retx.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    retx = sent2vector(line, model_w2v, sent_len, word_dim)
                count += 1

    return np.array(retx, dtype=theano.config.floatX), np.array(rety, dtype="int32")


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

