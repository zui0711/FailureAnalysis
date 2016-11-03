import time
import numpy as np
import theano as T
from autoencoder import autoencoder
from utils import *
from setting import *


def train_model(model, dataset, dic_size, learning_rate, E):
    f = open(dataset, "rb")
    context = f.readlines()

    for epoch in xrange(3):
        for i, line in enumerate(context[:2000000]):
            x = format_sent(line, word2idx, dic_size)
            if i % 100000 == 0:
                # print line
                print i, line
                test_model(model, dataset, dic_size, E)  # model.calculate_loss(x)
            model.step(E, x, learning_rate)

    tt = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    # model.save_model("data/model-"+tt+".npz")
    model_name = "dic_size" + str(dic_size) + "-hidden_dim" + str(hidden_dim) + "-sent_len" + str(sent_len)
    model.save_model("data/model-" + model_name + ".npz")


def test_model(model, dataset, dic_size, E):
    f = open(dataset, "rb")
    context = f.readlines()

    loss = 0
    for line in context[2000000:2005000]:
        x = format_sent(line, word2idx, dic_size)
        loss += model.calculate_loss(E, x)
    print loss / 5000


if __name__ == "__main__":
    model = autoencoder(word_dim=dic_size + 1, hidden_dim=hidden_dim, sent_len=sent_len)  # maxlen = 17

    train_model(model, "data/all.txt", dic_size, learning_rate, E)
