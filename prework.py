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


# log去符号
def get_text(filename, partnum):
    #if not os.path.exists(filename):
    #    os.mkdir(filename)

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
        print filename + " part" + str(part) + " get_text  DONE!"

    print filename + " get_text  DONE!\n"

# 获得字典
def get_dic(filename, partnum):
    #if not os.path.exists(filename):
    #    os.mkdir(filename)
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
        print filename + " part" + str(part) + " get_dic  DONE!"

    okdic = sorted(dic.items(), key=lambda e: e[1], reverse=True)
    #print len(okdic)
    for item in okdic:
        wf.write("%s,%d\n" % (item[0], item[1]))

    wf.close()
    print filename + " get_dic  DONE!\n"


if __name__ == "__main__":
    names = ["BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1",
             "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
             "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"]
    path = "data/network_diagnosis_data/"
    for name in names:
        get_text(path + name, 10)
        get_dic(path + name, 10)
