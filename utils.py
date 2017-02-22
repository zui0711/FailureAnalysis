# coding=utf-8
from setting import *
import numpy as np
import theano
import theano.tensor as T
from gensim.models import Word2Vec
import os
import cPickle as pickle
from os.path import join as pjoin
import time

unknown_word = "UNKNOWN"


def load_model(path, modelname):
    # load word2vec模型
    return Word2Vec.load(path + modelname)


def sent2vector(sentence, model_w2v, sent_len, word_dim):
    # 句子转为向量
    retvector = []
    thislen = len(sentence.split())
    if thislen > sent_len:
        thislen = sent_len
    for i, word in enumerate(sentence.strip().split()):
        if i == sent_len:
            break
        if word in model_w2v:
            retvector.append(model_w2v[word])
        else:
            thislen -= 1
    retvector.extend([[0 for i in xrange(word_dim)] for i in xrange(sent_len - thislen)])
    return retvector


def text2vec_one(con, model_w2v, sent_len_o, word_dim):
    # 文本转为向量，一个text为一个句子即一个向量
    retvec = []
    count = 0
    outflag = False
    for line in con:
        for word in line.split():
            if word in model_w2v:#.index2word:
                retvec.append(model_w2v[word])
                count += 1
                if count == sent_len_o:
                    outflag = True
                    break
        if outflag:
            break
    if sent_len_o > count:
        retvec.extend([[0 for i in xrange(word_dim)] for i in xrange(sent_len_o - count)])
    return retvec


def get_batchdata(path, file_names, idxs, model_w2v, sent_len, word_dim):
    # 一个text为一个句子

    def switch(label, y):
        try:
            {"NORMAL": lambda: y.append(0),
             "GTPC_TUNNEL_PATH_BROKEN": lambda: y.append(1),
             "Paging": lambda: y.append(2),
             "UeAbnormal": lambda: y.append(3)
             }[label]()
        except KeyError:
            print("Key not Found")
        # try:
        #     {"neg": lambda: y.append(0),
        #      "pos": lambda: y.append(1),
        #      }[label]()
        # except KeyError:
        #     a("Key not Found")

    rety = []
    for idx in idxs:
        name = file_names[idx]
        arr = name.split(".")
        this_file_name = ".".join([arr[1], arr[2]])
        switch(arr[0], rety)

        with open(os.path.join(path, arr[0], this_file_name), "rb") as f:
            context = f.readlines()
            line_num = len(context)

            if locals().has_key("retx"):
                retx.extend(text2vec_one(context, model_w2v, sent_len, word_dim))
            else:
                retx = text2vec_one(context, model_w2v, sent_len, word_dim)

    return np.array(retx, dtype=theano.config.floatX), np.array(rety, dtype="int32")


# TODO
def get_batchdata_sent(path, file_names, idxs, model_w2v, sent_len, word_dim, text_size):
    """
    text_size个句子，分为词读入
    :param path:
    :param file_names:
    :param idxs:
    :param model_w2v:
    :param sent_len:
    :param word_dim:
    :param text_size:
    :return:
    """
    def switch(label, y):
        try:
            {"NORMAL": lambda: y.append(0),
             "GTPC_TUNNEL_PATH_BROKEN": lambda: y.append(1),
             "Paging": lambda: y.append(2),
             "UeAbnormal": lambda: y.append(3)
             }[label]()
        except KeyError:
            print("Key not Found")

    rety = []
    for idx in idxs:
        name = file_names[idx]
        arr = name.split(".")
        this_file_name = ".".join([arr[1], arr[2]])
        switch(arr[0], rety)

        with open(os.path.join(path, arr[0], this_file_name), "rb") as f:
            con = f.readlines()
            line_num = len(con)
            if line_num > text_size:
                flag = text_size
                #print line_num
            else:
                flag = line_num

            for line in con[:flag]:
                if locals().has_key("retx"):
                    retx.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    retx = sent2vector(line, model_w2v, sent_len, word_dim)
            if line_num < text_size:
               retx.extend(np.zeros([(text_size-line_num) * sent_len, word_dim]))
    return np.array(retx, dtype=theano.config.floatX), np.array(rety, dtype="int32")


def load_dic(dic_file, dic_size=-1):
    # 目前没用，手动进行onehot转换操作
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


def load_model_onehot(path, modelname, dicsize):
    # 借用word2vec模型，简建立onehot向量
    # dic_size => word_dim

    #dicfile = open(pjoin(path, modelname), "rb")
    model = {}
    mat = np.eye(dicsize)
    #for i, w in enumerate(pickle.load(dicfile)[:dicsize]):
     #   model[w[0]] = mat[i]

    dicfile = Word2Vec.load(pjoin(path, modelname))
    for i, w in enumerate(dicfile.index2word[:dicsize]):
        model[w] = mat[i]
    return model


# TODO
def sent2vec_rbm(sentence, model_w2v, model_rbm, sent_len=15, word_dim=100):
    # 利用rbm进行句子转向量
    retvector = []
    thislen = len(sentence.split())
    if thislen > sent_len:
        thislen = sent_len
    for i, word in enumerate(sentence.strip().split()):
        if i == sent_len:
            break
        if word in model_w2v:
            retvector.append(model_w2v[word])
        else:
            thislen -= 1
    retvector.extend([[0 for i in xrange(word_dim)] for i in xrange(sent_len - thislen)])

    #return model_rbm.propup(np.array(retvector, dtype=theano.config.floatX).reshape(1, sent_len*word_dim))[1].eval().tolist()
    def sigmoid(pre):
        return 1. / (1. + np.exp(-1. *pre))
    ret = np.dot(np.array(retvector).flatten(), model_rbm.W.get_value()) + model_rbm.hbias.get_value()
    return sigmoid(ret).tolist()


def get_batchdata_sent_onehot(path, file_names, idxs, model_w2v, model_rbm, sent_len, word_dim, text_size):
    """
    batch 分rbm转换后的句子进行读入，其中使用onehot进行句子到向量转换
    :param path:
    :param file_names:
    :param idxs:
    :param model_w2v:
    :param sent_len:
    :param word_dim:
    :param text_size:
    :return:
    """
    def switch(label, y):
        try:
            {"NORMAL": lambda: y.append(0),
             "GTPC_TUNNEL_PATH_BROKEN": lambda: y.append(1),
             "Paging": lambda: y.append(2),
             "UeAbnormal": lambda: y.append(3)
             }[label]()
        except KeyError:
            print("Key not Found")

    rety = []
    for i, idx in enumerate(idxs):
        t1 = time.clock()
        print(i)
        name = file_names[idx]
        arr = name.split(".")
        this_file_name = ".".join([arr[1], arr[2]])
        switch(arr[0], rety)

        with open(os.path.join(path, arr[0], this_file_name), "rb") as f:
            con = f.readlines()
            line_num = len(con)
            if line_num > text_size:
                flag = text_size
                #print line_num
            else:
                flag = line_num

            for line in con[:flag]:
                if locals().has_key("retx"):
                    retx.extend(sent2vec_rbm(line, model_w2v, model_rbm, sent_len, word_dim))
                else:
                    retx = sent2vec_rbm(line, model_w2v, model_rbm, sent_len, word_dim)
            if line_num < text_size:
               retx.extend(np.zeros([(text_size-line_num) * 800]))
        t2 = time.clock()
        #print(t2 - t1)
    return np.array(retx, dtype=theano.config.floatX), np.array(rety, dtype="int32")


if __name__ == "__main__":
    model = load_model_onehot(m_path, m_model_w2v_name, 100)
    for i in model:
        print(i, model[i])