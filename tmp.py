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
name = "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1"
f = open("data/network_diagnosis_data/" + name + ".log", "rb")
con = f.readlines()[0:100]

linelen = len(con)
batch_size = linelen / 10

for idx in xrange(9):
    wf = open("data/network_diagnosis_data/" + name + ".part" + str(idx) + ".txt", "wb")
    for line in con[idx * batch_size: (idx + 1) * batch_size]:
        wf.write(line.strip())
    wf.close()

idx = 9
wf = open("data/network_diagnosis_data/" + name + ".part" + str(idx) + ".txt", "wb")
for line in con[idx * batch_size:]:
    wf.write(line.strip())
wf.close()
