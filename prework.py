# coding=utf-8
import os
import shutil
import theano
import theano.tensor as T
from setting import *
from utils import *
import random
import cPickle as pickle

FORMATLETTER = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'


# 数据分割
def seperate_data(path, name, partnum=10):
    """
    names = ["BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1",
            "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
            "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"]

    path = "data/network_diagnosis_data/"
    """
    f = open(path + name + ".log", "rb")
    con = f.readlines()

    linelen = len(con)
    batch_size = linelen / partnum

    if not os.path.exists(path + name):
        os.mkdir(path + name)
    for idx in xrange(partnum - 1):
        wf = open(path + name + "/part" + str(idx) + ".txt", "wb")
        for line in con[idx * batch_size: (idx + 1) * batch_size]:
            wf.write(line.strip())
            wf.write("\n")
        wf.close()

    idx = partnum - 1
    wf = open(path + name + "/part" + str(idx) + ".txt", "wb")
    for line in con[idx * batch_size:]:
        wf.write(line.strip())
        wf.write("\n")
    wf.close()

    f.close()
    print name + "  DONE!"


# 单句去符号
def format_string(string):
    for c in string:
        if c not in FORMATLETTER:
            string = string.replace(c, ' ')
    retstring = ' '.join(string.split())
    return retstring


# log去符号
def get_text(filename, part=True, partnum=10):
    if part:
        for part in xrange(partnum):
            f = open(filename + "/part" + str(part)+ ".txt", "rb")
            context = f.readlines()
            wf = open(filename + "/clean." + "part" + str(part) + ".txt", "wb")
            for line in context:
                s = format_string(line)
                if s != "":
                    wf.write(s + "\n")
            wf.close()
            f.close()
            print filename + " part" + str(part) + " get_text_part  DONE!"

    else:
        f = open(filename + ".log", "rb")
        context = f.readlines()

        if not os.path.exists(filename):
            os.mkdir(filename)

        wf = open(filename + "/clean" + ".txt", "wb")
        for line in context:
            s = format_string(line)
            if s != "":
                wf.write(s + "\n")
        wf.close()
        f.close()
    print filename + " get_text  DONE!"


# 获得字典
def get_dic(filename, part=True, partnum=10):
    wf = open(filename + "/dic.txt", "wb")
    dic = {}

    if part:
        for part in xrange(partnum):
            f = open(filename + "/clean." + "part" + str(part) + ".txt", "rb")
            context = f.readlines()
            for line in context:
                sentence = line.strip().split(" ")
                for word in sentence:
                    if word in dic:
                        dic[word] += 1
                    else:
                        dic[word] = 1
            f.close()
            print filename + " part" + str(part) + " get_dic_part  DONE!"

    else:
        f = open(filename + "/clean" + ".txt", "rb")
        context = f.readlines()
        for line in context:
            sentence = line.strip().split(" ")
            for word in sentence:
                if word in dic:
                    dic[word] += 1
                else:
                    dic[word] = 1
        f.close()

    okdic = sorted(dic.items(), key=lambda e: e[1], reverse=True)
    # print len(okdic)
    for item in okdic:
        wf.write("%s,%d\n" % (item[0], item[1]))

    wf.close()
    print filename + " get_dic  DONE!\n"


# 样本分割
def cut_data(path, name, label, iflabel=True):
    if iflabel:
        f = open(path + name + "/clean.txt", "rb")
        contxt = f.readlines()[:10000]

        num = [] # 包含标签的行
        for i, line in enumerate(contxt):
            arr = line.split()
            if label in arr:
                num.append(i)
            # print num, "\n"
        nbegin = 0
        nnew = [nbegin]
        for n in num:
            if n - nbegin > 10:
                #print n - nbegin
                nnew.append(n)
                nbegin = n

        filename = path + "cut1000/labeled/"
        if not os.path.exists(filename):
            os.mkdir(filename)

        newname = filename + label + "."

        idx = 0
        wwf = open(newname + str(idx) + ".txt", "wb")
        for i, line in enumerate(contxt):
            if i - 10 in nnew:
                wwf.close()
                wwf = open(newname + str(idx) + ".txt", "wb")
                idx += 1
            if i not in num:
                wwf.write(line)

        print nnew, len(nnew)

    else:
        ERRORNAME = ["USER_CONGESTION",
                     "GTPC_TUNNEL_PATH_BROKEN",
                     "PROCESS_CPU",
                     "SYSTEM_FLOW_CTRL",
                     "EPU_PORT_CONGESTION"]

        f = open(path + name + "/clean.txt", "rb")
        contxt = f.readlines()

        filename = path + "cut1000/unlabeled/"
        if not os.path.exists(filename):
            os.mkdir(filename)

        count = 0
        line_count = 0
        file_empty = True

        wwf = open(filename + ".".join([label, str(count), "txt"]), "wb")
        for line in contxt:
            if line_count % 1000 == 0 and not file_empty:
                wwf.close()
                count += 1
                wwf = open(filename + ".".join([label, str(count), "txt"]), "wb")
                file_empty = True

            arr = line.split()
            write = True
            for word in arr:
                if word in ERRORNAME:
                    write = False
                    break
            if write:
                line_count += 1
                file_empty = False
                wwf.write(line)
        print count

        for num in ["0", str(count)]:
            nn = filename + ".".join([label, num, "txt"])
            if os.path.exists(nn):
                print "remove   " + nn
                shutil.copy(nn, filename+"../delete/"+".".join([label, num, "txt"]))
                os.remove(nn)



# TODO
def save_embdding_data(path, model_w2v, sent_len, word_dim):
    file_names = os.listdir(path)
    file_num = len(file_names)
    file_names_idx = range(file_num)
    random.shuffle(file_names_idx)

    file_names_idx = file_names_idx[:100]
    file_num = 100

    train_idx = file_names_idx[: 5 * file_num / 10]
    valid_idx = file_names_idx[5 * file_num / 10: 8 * file_num / 10]
    test_idx = file_names_idx[8 * file_num / 10:]

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

    train_y = []
    for idx in train_idx:
        name = file_names[idx]
        arr = name.split(".")
        if arr[1] == "-1" or arr[1] == "0":
            continue
        switch(arr[0], train_y)

        with open(path + name, "rb") as f:
            for line in f:
                if locals().has_key("train_x"):
                    train_x.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    train_x = sent2vector(line, model_w2v, sent_len, word_dim)

    valid_y = []
    for idx in valid_idx:
        name = file_names[idx]
        arr = name.split(".")
        if arr[1] == "-1" or arr[1] == "0":
            continue
        switch(arr[0], valid_y)

        with open(path + name, "rb") as f:
            for line in f:
                if locals().has_key("valid_x"):
                    valid_x.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    valid_x = sent2vector(line, model_w2v, sent_len, word_dim)

    test_y = []
    for idx in test_idx:
        name = file_names[idx]
        arr = name.split(".")
        if arr[1] == "-1" or arr[1] == "0":
            continue
        switch(arr[0], test_y)

        with open(path + name, "rb") as f:
            for line in f:
                if locals().has_key("test_x"):
                    test_x.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    test_x = sent2vector(line, model_w2v, sent_len, word_dim)

    with open(path+"../embdding_data.pk", "wb") as f:
        pickle.dump(train_x, f)
        pickle.dump(train_y, f)
        pickle.dump(valid_y, f)
        pickle.dump(train_x, f)
        pickle.dump(test_x, f)
        pickle.dump(test_y, f)

"""
    return theano.shared(np.array(train_x, dtype=theano.config.floatX), borrow=True), \
           T.cast(theano.shared(np.asarray(train_y, dtype=theano.config.floatX), borrow=True), "int32"), \
           theano.shared(np.array(valid_x, dtype=theano.config.floatX), borrow=True), \
           T.cast(theano.shared(np.asarray(valid_y, dtype=theano.config.floatX), borrow=True), "int32"), \
           theano.shared(np.array(test_x, dtype=theano.config.floatX), borrow=True), \
           T.cast(theano.shared(np.asarray(test_y, dtype=theano.config.floatX), borrow=True), "int32")
"""


if __name__ == "__main__":
    #path = "data/network_diagnosis_data/"

    # name = "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag"
    # seperate_data(path, name)
    # cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1", "GTPC_TUNNEL_PATH_BROKEN")
    # cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1", "NORMAL", True)
    # cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1", "NORMAL", True)

    #for name in names:
    #    get_text_part(path + name, 10)
    #    get_dic_part(path + name, 10)
    #name = "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag"
    #name = "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1"
    #name = "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1"
    #name = "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"
    #get_text(path + name, part=False)
    #get_dic(path + name, part=False)

    #cut_data(path, name, "Paging", iflabel=False)
    #model_w2v = load_model(m_path+"../../", m_model_w2v_name)
    #save_embdding_data(m_path, model_w2v, m_sent_len, m_word_dim)
    #for i, name in enumerate(m_names):
    #    get_text(m_path + name, part=False)
    #    cut_data(m_path, name, m_labels[i], False)
    name = "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1"
    #"BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"
    #seperate_data(m_path, name, 10)
    get_text(m_path+name, False)
    cut_data(m_path, name, "UeAbnormal", False)
