from setting import *
from mlp import *
import sys, os, time, random

import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from utils import *

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
        batch_size = 5 * 200 * 10

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



def load_data(path, model_w2v, sent_len, word_dim):
    file_names = os.listdir(path)
    file_num = len(file_names)
    file_names_idx = range(file_num)
    random.shuffle(file_names_idx)

    file_names_idx = file_names_idx[:200]
    file_num = 200

    train_idx = file_names_idx#[: 5 * file_num / 10]
    valid_idx = file_names_idx[5 * file_num / 10 : 8 * file_num / 10]
    test_idx = file_names_idx[8 * file_num / 10:]

    def a(s):
        print s

    def switch(label, y):
        try:
            {"NORMAL": lambda: y.append(0),
             "GTPC_TUNNEL_PATH_BROKEN": lambda: y.append(1),
             "Paging": lambda: y.append(2),
             "UeAbnormal": lambda: y.append(3)
             }[label]()
        except KeyError:
            a("Key not Found")

    train_y = []
    for idx in train_idx:
        name = file_names[idx]
        arr = name.split(".")
        if arr[1] == "-1" or arr[1] == "0":
            continue
        switch(arr[0], train_y)

        with open(path + name, "rb") as f:
            for line in f:
                if locals().has_key("train_x"):
                    train_x.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    train_x = sent2vector(line, model_w2v, sent_len, word_dim)

    valid_y = []
    for idx in valid_idx:
        name = file_names[idx]
        arr = name.split(".")
        if arr[1] == "-1" or arr[1] == "0":
            continue
        switch(arr[0], valid_y)

        with open(path + name, "rb") as f:
            for line in f:
                if locals().has_key("valid_x"):
                    valid_x.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    valid_x = sent2vector(line, model_w2v, sent_len, word_dim)

    test_y = []
    for idx in test_idx:
        name = file_names[idx]
        arr = name.split(".")
        if arr[1] == "-1" or arr[1] == "0":
            continue
        switch(arr[0], test_y)

        with open(path + name, "rb") as f:
            for line in f:
                if locals().has_key("test_x"):
                    test_x.extend(sent2vector(line, model_w2v, sent_len, word_dim))
                else:
                    test_x = sent2vector(line, model_w2v, sent_len, word_dim)

    return theano.shared(np.array(train_x, dtype=theano.config.floatX), borrow=True), \
           T.cast(theano.shared(np.asarray(train_y, dtype=theano.config.floatX), borrow=True), "int32"), \
           theano.shared(np.array(valid_x, dtype=theano.config.floatX), borrow=True), \
           T.cast(theano.shared(np.asarray(valid_y, dtype=theano.config.floatX), borrow=True), "int32"), \
           theano.shared(np.array(test_x, dtype=theano.config.floatX), borrow=True), \
           T.cast(theano.shared(np.asarray(test_y, dtype=theano.config.floatX), borrow=True), "int32")


def mycnn(path, model_w2v, sent_len, word_dim, epoch, learning_rate=0.01, batch_size=5*200*10):
    rng = np.random.RandomState(123)
    index = T.iscalar()
    x = T.matrix("x")
    y = T.ivector("y")

    t1 = time.clock()
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(path, model_w2v, sent_len, word_dim)

    file_names = os.listdir(path)
    file_num = len(file_names)
    file_names_idx = range(file_num)
    random.shuffle(file_names_idx)

    t2 = time.clock()

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
        filter_shape=(1, 1, 3, 50),
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
            x: train_x[index * sent_len*1000*10: (index + 1) * sent_len*1000*10],
            y: train_y[index * 10: (index + 1) * 10]#get_batchdata(path, [index, index+9], file_names, train_idx, model_w2v, sent_len, word_dim)[1]
        }
    )

    print "...training"


    #train_num = batch_size/sent_len
    for ep in xrange(epoch):
        print "epoch = ", ep
        for idx in xrange(10):
            t3 = time.clock()
            cost = train_model(idx)
            t4 = time.clock()
            print idx, " cost = ", cost
        print "\n"


if __name__ == "__main__":
    model_w2v = load_model(m_path, m_model_w2v_name)
    mycnn(m_path+"cut1000/unlabeled/", model_w2v, m_sent_len, m_word_dim,  m_epoch, m_learning_rate)

