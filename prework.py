FORMATLETTER = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'


def format_string(string):
    for c in string:
        if c not in FORMATLETTER:
            string = string.replace(c, ' ')
    retstring = ' '.join(string.split())
    return retstring


def get_text():
    f = open("data/network_diagnosis_data/BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1.log", "rb")
    context = f.readlines()[220:]

    wf = open("data/all_ueabnormal.txt", "wb")

    # [220:1725]+[1731:13425]

    for line in context:
        s = format_string(line)
        if s != "":
            wf.write(s + "\n")


def get_dic():
    f = open("data/all_ueabnormal.txt", "rb")
    wf = open("data/dic_ueabnormal.txt", "wb")
    context = f.readlines()

    dic = {}

    for line in context:
        sentence = line.strip().split(" ")
        for word in sentence:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

    okdic = sorted(dic.items(), key=lambda e: e[1], reverse=True)
    print len(okdic)
    for item in okdic:
        wf.write("%s,%d\n" % (item[0], item[1]))


if __name__ == "__main__":
    get_text()
    get_dic()
