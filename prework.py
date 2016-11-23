# coding=utf-8
import os

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
        contxt = f.readlines()

        num = []
        for i, line in enumerate(contxt):
            arr = line.strip().split()
            if label in arr:
                num.append(i)
            # print num, "\n"
        nbegin = num[0]
        nnew = [nbegin]
        for n in num:
            if n - nbegin > 10:
                print n - nbegin
                nnew.append(n)
                nbegin = n

        filename = path + "cut/labeled/"
        if not os.path.exists(filename):
            os.mkdir(filename)

        newname = filename + label + "."
        i = 0
        wwf = open(newname + str(i) + ".txt", "wb")

        idx = 0
        for i, line in enumerate(contxt):
            if i - 10 in nnew:
                wwf.close()
                wwf = open(newname + str(idx) + ".txt", "wb")
                idx += 1
            if i not in nnew:
                wwf.write(line)

        print nnew, len(nnew)

    else:
        count = -1
        f = open(path + name + "/clean.txt", "rb")
        contxt = f.readlines()

        filename = path + "cut/unlabeled/"
        if not os.path.exists(filename):
            os.mkdir(filename)

        newname = filename + label + "."
        wwf = open(newname + str(count) + ".txt", "wb")
        for i, line in enumerate(contxt):
            if i % 14000 == 0:
                wwf.close()
                count += 1
                wwf = open(newname + str(count) + ".txt", "wb")
            wwf.write(line)
        print count


if __name__ == "__main__":
    path = "data/network_diagnosis_data/"

    # path = "data/network_diagnosis_data/"
    # name = "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag"
    # seperate_data(path, name)
    # cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1", "GTPC_TUNNEL_PATH_BROKEN")
    # cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1", "NORMAL", True)
    # cut_data("data/network_diagnosis_data/", "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1", "NORMAL", True)

    #names = ["BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1",
    #         "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
    #         "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"]

    #for name in names:
    #    get_text_part(path + name, 10)
    #    get_dic_part(path + name, 10)
    name = "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag"
    get_text(path + name, part=False)
    get_dic(path + name, part=False)
    cut_data(path + name, )