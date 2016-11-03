# coding=utf-8
import numpy as np
import theano
import theano.tensor as T


class autoencoder:
    def __init__(self, word_dim, hidden_dim, sent_len):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_len = sent_len

        params = {}
        params["W1"] = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),
                                         (word_dim, hidden_dim))
        params["W2"] = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),
                                         (hidden_dim, word_dim))
        params["B1"] = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), hidden_dim)
        params["B2"] = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), word_dim)

        self.param_names = ["W1", "W2", "B1", "B2"]

        self.params = {}
        for name in self.param_names:
            self.params[name] = theano.shared(value=params[name].astype(theano.config.floatX), name=name)

        self.__theano_build__()

    def __theano_build__(self):
        params = self.params
        param_names = self.param_names

        learning_rate = T.scalar("learning_rate")

        x = T.ivector('x')
        E = T.fmatrix("E")

        Emb = E[x, :]

        encode = T.nnet.sigmoid(Emb.dot(params["W1"]) + params["B1"])
        decode = T.nnet.sigmoid(encode.dot(params["W2"]) + params["B2"])

        # loss = - T.sum(Emb * T.log(decode) + (1 - Emb) * T.log(1 - decode), axis=1).mean()
        loss = T.sum(T.sqr(Emb - decode))

        grads = [T.grad(loss, params[name]) for name in param_names]
        updates = [(params[name], params[name] - learning_rate * grads[i]) for i, name in enumerate(param_names)]

        self.get_encode = theano.function([E, x], encode)
        self.forword_prop = theano.function([E, x], decode)
        self.calculate_loss = theano.function([E, x], loss)
        self.step = theano.function([E, x, learning_rate], updates=updates)

    def save_model(self, outfile):
        np.savez(outfile,
                 hidden_dim=self.hidden_dim,
                 word_dim=self.word_dim,
                 sent_len=self.sent_len,
                 W1=self.params["W1"].get_value(),
                 W2=self.params["W2"].get_value(),
                 B1=self.params["B1"].get_value(),
                 B2=self.params["B2"].get_value())
        print "Saved parameters to %s." % outfile

    def load_model(self, path):
        npzfile = np.load(path)
        print("Loading model from %s.\nParamenters: \n\t|hidden_dim=%d \n\t|word_dim=%d \n\t|sent_len=%d"
              % (path, npzfile["hidden_dim"], npzfile["word_dim"], npzfile["sent_len"]))
        self.params["W1"].set_value(npzfile["W1"])
        self.params["W2"].set_value(npzfile["W2"])
        self.params["B1"].set_value(npzfile["B1"])
        self.params["B2"].set_value(npzfile["B2"])
