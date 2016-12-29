import numpy
import theano
import theano.tensor as T


class HiddenLayer:
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):

        self.input = input

        if W is None:
            W_value = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                W_value *= 4
        else:
            W_value = W
        W = theano.shared(value=W_value, name="W", borrow=True)

        # b_values = (numpy.zeros(n_out, dtype=theano.config.floatX) if b is None else b)
        if b is None:
            b_values = numpy.zeros(n_out, dtype=theano.config.floatX)
        else:
            b_values = b

        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))


class LogisticRegression:
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            self.input = input

            W_value = numpy.zeros((n_in, n_out), dtype=theano.config.floatX) if W is None else W
            b_value = numpy.zeros(n_out, dtype=theano.config.floatX) if b is None else b

            W = theano.shared(value=W_value, name="W", borrow=True)
            b = theano.shared(value=b_value, name="b", borrow=True)

            self.W = W
            self.b = b
            self.params = [self.W, self.b]

            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class MLP:
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.input = input

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


if __name__ == "__main__":
    data = numpy.array([[1, 2, 3], [2, 3, 4], [2, 3, 4], [3, 4, 6]], dtype=theano.config.floatX)
    model = MLP(numpy.random.RandomState(1234), data, 3, 4, 2)
    print model.errors
