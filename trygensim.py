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
    for part in xrange(1):
        f = open(path + name + "/clean.part"  + str(part) + ".txt", "rb")
        doc += f.readlines()[0:10000]
        f.close()

    name1 = "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1"
    f = open(path + name1 + "/clean.txt", "rb")
    doc += f.readlines()[0:10000]
    f.close()

    corpuss=MyCorpus(doc)

    t1 = time.clock()
    model = Word2Vec(corpuss, size=50, window=5, min_count=100, alpha=0.0001, iter=1000)
    t2 = time.clock()
    print t2 - t1

    print len(model.index2word)

    maxv = 0
    for name in model.index2word:
        if abs(max(model[name])) > maxv:
            maxv = abs(max(model[name]))
        print name, maxv
    #model.save(path+"word_emb")