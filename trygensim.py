from gensim.models import Word2Vec,LdaModel
from gensim import corpora

import time

class MyCorpus(object):
    def __init__(self,docs):
        self.docs=docs
    def __iter__(self):
        for line in self.docs:
            yield line.split()

if __name__=="__main__":
    path = "data/network_diagnosis_data/"
    name = "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1"

    doc = []
    for part in xrange(2):
        f = open(path + name + "/clean.part"  + str(part) + ".txt", "rb")
        doc += f.readlines()[0:10000]
        f.close()

    name1 = "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1"
    f = open(path + name1 + "/clean.txt", "rb")
    doc += f.readlines()[0:10000]
    f.close()

    corpuss=MyCorpus(doc)

    t1 = time.clock()
    model = Word2Vec(corpuss, size=15, window=10, min_count=100)
    t2 = time.clock()
    print t2 - t1
    wf = open(path + "dic_emb.txt", "wb")

    print len(model.index2word)
    for word in model.index2word:
        wf.write(str(word) + ","+ str(model[str(word)]) + "\n")
        print model[str(word)]

    wf.close()