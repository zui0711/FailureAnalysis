# coding=utf-8

from mlp import *

from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv
from utils import *
import cPickle as pickle

# image_shape:(batch size, num input feature maps, image height, image width)
#   (batch_size, featuremap_num, sent_len, word_dim)
# filter_shape:(number of filters, num input feature maps, filter height, filter width)
#   (filter_num, featuremap_num, filter_len, word_dim)
# poolsize: (#rows, #cols)
#   (batch_size, filter_num, output row, output col)

class LeNetConvPoolLayer_sent:
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(5, 1)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize)

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W_value = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)
        b_value = numpy.zeros(filter_shape[0], dtype=theano.config.floatX)

        self.W = theano.shared(value=W_value, name="W", borrow=True)
        self.b = theano.shared(value=b_value, name="b", borrow=True)
        self.params = [self.W, self.b]

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=False
        )# (10, 1, 200, 128)

        self.output = T.tanh(pooled_out + self.b.dimshuffle("x", 0, "x", "x"))


# TODO
class LeNetConvPoolLayer_sent_g:
    def __init__(self, rng, input, filter_shape, image_shape, group_size=5, poolsize=(1, 2)):
        assert image_shape[1] == filter_shape[1]
        assert image_shape[3] == filter_shape[3]
        assert poolsize[1] == group_size
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        #fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize)

        #W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W_bound = numpy.sqrt(6. / (fan_in * 2))

        W_value = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)
        b_value = numpy.zeros(filter_shape[0], dtype=theano.config.floatX)

        self.W = theano.shared(value=W_value, name="W", borrow=True)
        self.b = theano.shared(value=b_value, name="b", borrow=True)
        self.params = [self.W, self.b]

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pool_input = conv_out.reshape((image_shape[0]/group_size,
                                       image_shape[1],
                                       image_shape[2]-filter_shape[2]+1,
                                       group_size))

        pooled_out = pool_2d(
            input=pool_input,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle("x", 0, "x", "x"))


class FourConvCNN:
    # 默认输入 batch_size, 1, 500, 50
    def __init__(self, rng, sent_len, word_dim, batch_size, learning_rate):
        x = T.matrix("x")
        y = T.ivector("y")

        layer0_input = x.reshape((batch_size, 1, sent_len, word_dim))
        layer0 = LeNetConvPoolLayer_sent(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, sent_len, word_dim),
            filter_shape=(128, 1, 5, word_dim),
            poolsize=(5, 1)
        )

        layer1 = LeNetConvPoolLayer_sent(
            rng,
            input=layer0.output,
            image_shape=(batch_size, 128, 1000, 1),
            filter_shape=(128, 128, 5, 1),
            poolsize=(5, 1)
        )

        layer2 = LeNetConvPoolLayer_sent(
            rng,
            input=layer1.output,
            image_shape=(batch_size, 128, 200, 1),
            filter_shape=(128, 128, 5, 1),
            poolsize=(5, 1)
        )

        layer3 = LeNetConvPoolLayer_sent(
            rng,
            input=layer2.output,
            image_shape=(batch_size, 128, 40, 1),
            filter_shape=(128, 128, 5, 1),
            poolsize=(40, 1)
        )

        layer4_input = layer3.output.reshape((batch_size, 1, 1, 128))
        layer4_input = layer4_input.flatten(2)
        layer4 = HiddenLayer(
            rng,
            input=layer4_input,
            n_in=1 * 1 * 128,
            n_out=20,
            activation=T.tanh
        )

        layer5 = LogisticRegression(input=layer4.output, n_in=20, n_out=4)

        self.cost = layer5.negative_log_likelihood(y)
        self.layers = [layer5, layer4, layer3, layer2, layer1, layer0]
        self.params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

        grads = T.grad(self.cost, self.params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
            ]

        self.train = theano.function([x, y], self.cost, updates=updates)
        self.test = theano.function([x, y], layer5.errors(y))
        self.pred = theano.function([x], layer5.y_pred)

    def save_model(self, outfile):
        f = open(outfile, 'wb')
        print("saving model... ")
        for param in self.params:
            pickle.dump(param.get_value(borrow=True), f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, infile):
        f = open(infile, "rb")
        print("loading model...")
        for layer in self.layers:
            for p in layer.params:
                p.set_value(pickle.load(f), borrow=True)




class FourConvCNN_rbm:
    # 默认输入 batch_size, 1, 500*8, 100
    def __init__(self, rng, sent_len, word_dim, batch_size, text_size, learning_rate):
        x = T.matrix("x")
        y = T.ivector("y")

        layer0_input = x.reshape((batch_size, 1, sent_len*text_size, word_dim))
        layer0 = LeNetConvPoolLayer_sent(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, sent_len*text_size, word_dim),
            filter_shape=(128, 1, 5, word_dim),
            poolsize=(5, 1)
        )

        layer1 = LeNetConvPoolLayer_sent(
            rng,
            input=layer0.output,
            image_shape=(batch_size, 128, 800, 1),
            filter_shape=(128, 128, 5, 1),
            poolsize=(5, 1)
        )

        layer2 = LeNetConvPoolLayer_sent(
            rng,
            input=layer1.output,
            image_shape=(batch_size, 128, 160, 1),
            filter_shape=(128, 128, 5, 1),
            poolsize=(4, 1)
        )

        layer3 = LeNetConvPoolLayer_sent(
            rng,
            input=layer2.output,
            image_shape=(batch_size, 128, 40, 1),
            filter_shape=(128, 128, 5, 1),
            poolsize=(40, 1)
        )

        layer4_input = layer3.output.reshape((batch_size, 1, 1, 128))
        layer4_input = layer4_input.flatten(2)
        layer4 = HiddenLayer(
            rng,
            input=layer4_input,
            n_in=1 * 1 * 128,
            n_out=20,
            activation=T.tanh
        )

        layer5 = LogisticRegression(input=layer4.output, n_in=20, n_out=4)

        self.cost = layer5.negative_log_likelihood(y)
        self.layers = [layer5, layer4, layer3, layer2, layer1, layer0]
        self.params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

        grads = T.grad(self.cost, self.params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
            ]

        self.train = theano.function([x, y], self.cost, updates=updates)
        self.test = theano.function([x, y], layer5.errors(y))
        self.pred = theano.function([x], layer5.y_pred)

    def save_model(self, outfile):
        f = open(outfile, 'wb')
        print("saving model... ")
        for param in self.params:
            pickle.dump(param.get_value(borrow=True), f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, infile):
        f = open(infile, "rb")
        print("loading model...")
        for layer in self.layers:
            for p in layer.params:
                p.set_value(pickle.load(f), borrow=True)


