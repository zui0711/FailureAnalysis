from mlp import *
import sys, os, time

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from setting import *

# image_shape:(batch size, num input feature maps, image height, image width)
#   (batch_size, featuremap_num, sent_len, word_dim)
# filter_shape:(number of filters, num input feature maps, filter height, filter width)
#   (filter_num, featuremap_num, filter_len, word_dim)
# poolsize: (#rows, #cols)
#   (batch_size, filter_num, output row, output col)

class LeNetConvPoolLayer_sent:
    def __init__(self, rng, input, filter_shape, image_shape, group_size=5, poolsize=(2, 2)):
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

        pooled_out = downsample.max_pool_2d(
            input=pool_input,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle("x", 0, "x", "x"))

class ConvLayer:
    def __int__(self, rng, input, filter_shape, image_shape):
        assert image_shape[1] == filter_shape[1]
        self.input = input


def mycnn(rng, learning_rate=0.1, batch_size=5*200*10):
    index = T.iscalar()
    x = T.matrix("x")
    y = T.ivector("y")



    print "... building the model"
    # data 1000 sentences * 10
    # word_dim = 50, sent_len = 20, batch_size = 5*200*10
    # filter(3, 50) => (18, 1)
    # get 5 sentences  => (18, 5)
    # pool(1, 5) => (18, 1)
    # activation
    # => (10, 1, 200, 18)

    layer0_input = x.reshape((batch_size, 1, 20, 50))

    layer0 = LeNetConvPoolLayer_sent(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 20, 50),
        filter_shape=(1, 1, 2, 50),
        poolsize=(1, 5)
    )

    layer1_input = layer0.output.reshape((10, 1, 200, 18))

    layer1_input = layer1_input.flatten(2)

    layer1 = HiddenLayer(
        rng,
        input=layer1_input,
        n_in=1 * 200 * 18,
        n_out=10,
        activation=T.tanh
    )

    layer2 = LogisticRegression(input=layer1.output, n_in=10, n_out=4)

    cost = layer2.negative_log_likelihood(y)

    params = layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

