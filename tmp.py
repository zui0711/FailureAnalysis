# coding=utf-8

import os

"""from setting import *
from autoencoder import *
import numpy
import theano

model = autoencoder(dic_size+1, hidden_dim, sent_len)

model.load_model("data/model-dic_size100-hidden_dim72-sent_len18.npz")

#dic = []
#f = open("data/dic.txt")
#context = f.readlines()

num = numpy.zeros(dic_size+1, dtype="int32")

for i in xrange(dic_size+1):
    num[i] = i
print num
print idx2word[100]

ans = model.get_encode(E, num)
print ans

f = open("data/word_emb.txt", "wb")
for line in ans:
    for nn in line:
        f.write(str(nn)+" ")
    f.write("\n")
"""

def seperate_data(path, name):
    # 数据分割
    """
    names = ["BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1",
            "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
            "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"]

    path = "data/network_diagnosis_data/"
    """
    for name in names:
        f = open(path + name + ".log", "rb")
        con = f.readlines()

        linelen = len(con)
        batch_size = linelen / 10


        if not os.path.exists(path + name):
            os.mkdir(path + name)
        for idx in xrange(9):
            wf = open(path + name + "/part" + str(idx) + ".txt", "wb")
            for line in con[idx * batch_size: (idx + 1) * batch_size]:
                wf.write(line.strip())
                wf.write("\n")
            wf.close()

        idx = 9
        wf = open(path + name + "/part" + str(idx) + ".txt", "wb")
        for line in con[idx * batch_size:]:
            wf.write(line.strip())
            wf.write("\n")
        wf.close()

        f.close()
        print name + "  DONE!"


def cut_data(path, name, label, part=False):
    if not part:
        f = open(path + name + "/clean.txt", "rb")
        contxt = f.readlines()

        num = []
        for i, line in enumerate(contxt):
            arr = line.strip().split()
            if label in arr:
                num.append(i)
        #print num, "\n"

        nbegin = num[0]
        nnew = [nbegin]
        for n in num:
            if n - nbegin > 10:
                print n - nbegin
                nnew.append(n)
                nbegin = n

        newname = path + "cut/" + label+"."
        i = 0
        wwf = open(newname + str(i) + ".txt", "wb")

        idx = 0
        for i, line in enumerate(contxt):
            if i-10 in nnew:
                wwf.close()
                wwf = open(newname + str(idx) + ".txt", "wb")
                idx += 1
            wwf.write(line)

        print nnew, len(nnew)

    else:
        count = -1
        for part in xrange(10):
            f = open(path + name + "/clean.part" + str(part) + ".txt", "rb")
            contxt = f.readlines()

            newname = path + "cut/" + label + "."
            wwf = open(newname + str(count) + ".txt", "wb")
            for i, line in enumerate(contxt):
                if i%10000 == 0:
                    wwf.close()
                    count += 1
                    wwf = open(newname + str(count) + ".txt", "wb")
                wwf.write(line)
            print count



if __name__ == "__main__":
    #cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1", "GTPC_TUNNEL_PATH_BROKEN")
    cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1", "NORMAL", True)