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
        doc += f.readlines()[0:1000]
        f.close()

    name1 = "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1"
    f = open(path + name1 + "/clean.txt", "rb")
    doc += f.readlines()[0:1000]
    f.close()

    corpuss=MyCorpus(doc)

    for iteration in [10, 50, 100, 1000, 2000]:
        print "iteration = ", iteration
        print time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
        model = Word2Vec(corpuss, size=36, window=5, min_count=100, iter=iteration, alpha=0.0005, min_alpha=0.00001)
        print time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())

        #print len(model.index2word)

        maxv = 0
        for name in model.index2word:
            if abs(max(model[name])) > maxv:
                maxv = abs(max(model[name]))
        print maxv, "\n"
    #model.save(path+"word_emb")