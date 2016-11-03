from autoencoder import autoencoder

from setting import *

model = autoencoder(dic_size + 1, hidden_dim, sent_len)

model.load_model("data/model-dic_size100-hidden_dim72-sent_len18.npz")
arr = model.params["W1"].get_value()

print arr

f = open("data/all.txt", "rb")
context = f.readlines()

loss = 0
all = 0
mistake = 0
error = np.zeros(dic_size + 1)
for line in context[2005000:2010000]:
    x = format_sent(line, word2idx, dic_size)
    arr = model.forword_prop(E, x)
    for index, i in enumerate(x - np.argmax(arr, axis=1)):
        all += 1
        if i != 0:
            error[x[index]] += 1
            mistake += 1
    # print arr
    loss += model.calculate_loss(E, x)
print loss
print error
print all, mistake, mistake / float(all)
