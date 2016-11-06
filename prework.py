# coding=utf-8
import os

FORMATLETTER = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'


# 单句去符号
def format_string(string):
    for c in string:
        if c not in FORMATLETTER:
            string = string.replace(c, ' ')
    retstring = ' '.join(string.split())
    return retstring


# log去符号(不分part)
def get_text(filename):
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


# log去符号(分part)
def get_text_part(filename, partnum):
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

    print filename + " get_text_part  DONE!\n"


# 获得字典(不分part)
def get_dic(filename):
    wf = open(filename + "/dic.txt", "wb")
    dic = {}

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


# 获得字典(分part)
def get_dic_part(filename, partnum):
    wf = open(filename + "/dic.txt", "wb")
    dic = {}

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

    okdic = sorted(dic.items(), key=lambda e: e[1], reverse=True)
    #print len(okdic)
    for item in okdic:
        wf.write("%s,%d\n" % (item[0], item[1]))

    wf.close()
    print filename + " get_dic_part  DONE!\n"


if __name__ == "__main__":
    path = "data/network_diagnosis_data/"

    #names = ["BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1",
    #         "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
    #         "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"]

    #for name in names:
    #    get_text_part(path + name, 10)
    #    get_dic_part(path + name, 10)
    get_text(path + "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1")
    get_dic(path + "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1")