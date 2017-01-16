"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""

from __future__ import print_function

import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import cPickle as pickle
import os

from theano.tensor.shared_randomstreams import RandomStreams

from rbmutils import tile_raster_images
from logistic_sgd import load_data


# start-snippet-1
class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
            self,
            n_visible=784,
            n_hidden=500,
            numpy_rng=None,
            theano_rng=None,
            batch_size=10,
            learning_rate=0.01
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        initial_W = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                                                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                                                    size=(n_visible, n_hidden)),
                                  dtype=theano.config.floatX)
        W = theano.shared(value=initial_W, name='W', borrow=True)

        hbias = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX),
                              name='hbias',
                              borrow=True)

        vbias = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX),
                              name='vbias',
                              borrow=True)

        self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

        x = T.matrix('x')  # the data is presented as rasterized images
        self.persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                     dtype=theano.config.floatX),
                                         borrow=True)

        self.cost, self.updates = self.get_cost_updates(lr=learning_rate,
                                              persistent=self.persistent_chain,
                                              k=3)

        self.train = theano.function([self.input], self.cost, updates=self.updates)
        # self.prop = theano.function([self.input], self.propup())

        n_chains = T.iscalar("n_chains")

        number_of_test_samples = self.input.shape[0]

        #test_idx = numpy.random.RandomState(123).randint(number_of_test_samples - n_chains)
        # test_idx = 0
        # persistent_vis_chain = theano.shared(
        #     numpy.asarray(
        #         self.input[test_idx:test_idx + n_chains],
        #         dtype=theano.config.floatX
        #     )
        # )
        #
        # plot_every = 1000
        # (
        #     [
        #         presig_hids,
        #         hid_mfs,
        #         hid_samples,
        #         presig_vis,
        #         vis_mfs,
        #         vis_samples
        #     ],
        #     updates
        # ) = theano.scan(
        #     self.gibbs_vhv,
        #     outputs_info=[None, None, None, None, None, persistent_vis_chain],
        #     n_steps=plot_every,
        #     name="gibbs_vhv"
        # )
        #
        # updates.update({persistent_vis_chain: vis_samples[-1]})
        # sample_fn = theano.function(
        #     [],
        #     [
        #         vis_mfs[-1],
        #         vis_samples[-1]
        #     ],
        #     updates=updates,
        #     name='sample_fn'
        # )
        # self.reconstruct = theano.function([self.input], persistent_vis_chain)

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        # end-snippet-4

    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        xi = T.round(self.input)

        fe_xi = self.free_energy(xi)

        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        fe_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    def save_model(self, outfile):
        f = open(outfile, "wb")
        print("saving model...")
        for param in self.params:
            pickle.dump(param.get_value(borrow=True), f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, infile):
        f = open(infile, "rb")
        print("loading model...")
        for param in self.params:
            param.set_value(pickle.load(f), borrow=True)



#
# def test_rbm(learning_rate=0.1, training_epochs=1,
#              dataset='mnist.pkl.gz', batch_size=20,
#              n_chains=20, n_samples=10, output_folder='rbm_plots',
#              n_hidden=500):
#     """
#     Demonstrate how to train and afterwards sample from it using Theano.
#
#     This is demonstrated on MNIST.
#
#     :param learning_rate: learning rate used for training the RBM
#
#     :param training_epochs: number of epochs used for training
#
#     :param dataset: path the the pickled dataset
#
#     :param batch_size: size of a batch used to train the RBM
#
#     :param n_chains: number of parallel Gibbs chains to be used for sampling
#
#     :param n_samples: number of samples to plot for each chain
#
#     """
#     datasets = load_data(dataset)
#
#     train_set_x, train_set_y = datasets[0]
#     test_set_x, test_set_y = datasets[2]
#
#     n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
#
#     index = T.lscalar()    # index to a [mini]batch
#     x = T.matrix('x')  # the data is presented as rasterized images
#
#     rng = numpy.random.RandomState(123)
#     theano_rng = RandomStreams(rng.randint(2 ** 30))
#
#     persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
#                                                  dtype=theano.config.floatX),
#                                      borrow=True)
#
#     rbm = RBM(input=x, n_visible=28 * 28,
#               n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)
#
#     cost, updates = rbm.get_cost_updates(lr=learning_rate,
#                                          persistent=persistent_chain, k=15)
#
#     #################################
#     #     Training the RBM          #
#     #################################
#     if not os.path.isdir(output_folder):
#         os.makedirs(output_folder)
#     os.chdir(output_folder)
#
#     image_data0 = numpy.zeros(
#         (29 * n_samples + 1, 29 * n_chains - 1),
#         dtype='uint8'
#     )
#     for idx in range(n_samples):
#         print(' ... plotting origin %d' % idx)
#         image_data0[29 * idx:29 * idx + 28, :] = tile_raster_images(
#             X=train_set_x.get_value(),
#             img_shape=(28, 28),
#             tile_shape=(1, n_chains),
#             tile_spacing=(1, 1)
#         )
#
#     image = Image.fromarray(image_data0)
#     image.save('origin.png')
#
#     train_rbm = theano.function(
#         [index],
#         cost,
#         updates=updates,
#         givens={
#             x: train_set_x[index * batch_size: (index + 1) * batch_size]
#         },
#         name='train_rbm'
#     )
#
#     plotting_time = 0.
#     start_time = timeit.default_timer()
#
#     for epoch in range(training_epochs):
#         mean_cost = []
#         for batch_index in range(n_train_batches):
#             mean_cost += [train_rbm(batch_index)]
#
#         print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))
#
#         plotting_start = timeit.default_timer()
#         image = Image.fromarray(
#             tile_raster_images(
#                 X=rbm.W.get_value(borrow=True).T,
#                 img_shape=(28, 28),
#                 tile_shape=(10, 10),
#                 tile_spacing=(1, 1)
#             )
#         )
#         image.save('filters_at_epoch_%i.png' % epoch)
#         plotting_stop = timeit.default_timer()
#         plotting_time += (plotting_stop - plotting_start)
#
#     end_time = timeit.default_timer()
#
#     pretraining_time = (end_time - start_time) - plotting_time
#
#     print ('Training took %f minutes' % (pretraining_time / 60.))
#
#     #################################
#     #     Sampling from the RBM     #
#     #################################
#
#     number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
#
#     test_idx = rng.randint(number_of_test_samples - n_chains)
#     persistent_vis_chain = theano.shared(
#         numpy.asarray(
#             test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
#             dtype=theano.config.floatX
#         )
#     )
#
#     plot_every = 1000
#     (
#         [
#             presig_hids,
#             hid_mfs,
#             hid_samples,
#             presig_vis,
#             vis_mfs,
#             vis_samples
#         ],
#         updates
#     ) = theano.scan(
#         rbm.gibbs_vhv,
#         outputs_info=[None, None, None, None, None, persistent_vis_chain],
#         n_steps=plot_every,
#         name="gibbs_vhv"
#     )
#
#     updates.update({persistent_vis_chain: vis_samples[-1]})
#     sample_fn = theano.function(
#         [],
#         [
#             vis_mfs[-1],
#             vis_samples[-1]
#         ],
#         updates=updates,
#         name='sample_fn'
#     )
#
#     image_data = numpy.zeros(
#         (29 * n_samples + 1, 29 * n_chains - 1),
#         dtype='uint8'
#     )
#     for idx in range(n_samples):
#         vis_mf, vis_sample = sample_fn()
#         print(' ... plotting sample %d' % idx)
#         image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
#             X=vis_mf,
#             img_shape=(28, 28),
#             tile_shape=(1, n_chains),
#             tile_spacing=(1, 1)
#         )
#
#     image = Image.fromarray(image_data)
#     image.save('samples.png')
#     os.chdir('../')
#
# if __name__ == '__main__':
#     test_rbm()

class thisrbm(object):
    def __init__(self, n_v, n_h):
        self.n_v = n_v
        self.n_h = n_h

        self.numpy_rng = numpy.random.RandomState(1234)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        initial_W = numpy.asarray(
            self.numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_h + n_v)),
                high=4 * numpy.sqrt(6. / (n_h + n_v)),
                size=(n_v, n_h)
            ),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(value=initial_W, name='W', borrow=True)

        self.vbias = theano.shared(value=numpy.zeros(n_v, dtype=theano.config.floatX), name='vbias', borrow=True)
        self.hbias = theano.shared(value=numpy.zeros(n_h, dtype=theano.config.floatX), name='hbias', borrow=True)

        self.params = [self.W, self.vbias, self.hbias]

        self.input = T.matrix("input")

    # def free_energy(self):

    def propup(self, vis):
        return T.nnet.sigmoid(T.dot(vis, self.W) + self.hbias)

    def sample_h_given_v(self, vis):
        h_probs = self.propup(vis)
        return self.theano_rng.binomial(size=h_probs.shape, n=1, p=h_probs, dtype=theano.config.floatX)

    def propdown(self, hid):
        return T.nnet.sigmoid(T.dot(hid, self.W.T) + self.vbias)

    def sample_v_given_h(self, hid):
        v_probs= self.propdown(hid)
        return self.theano_rng.binomial(size=v_probs.shape, n=1, p=v_probs, dtype=theano.config.floatX)

    def gibbs_hvh(self, h0):
        v1 = self.sample_v_given_h(h0)
        h1 = self.sample_h_given_v(v1)
        return [v1, h1]

    def gibbs_vhv(self, v0):
        h1 = self.sample_h_given_v(v0)
        v1 = self.sample_v_given_h(h1)
        return [v1, h1]

    # def get_cost_update(self, lr, k=1):
    #     ph = self.sample_h_given_v(self.input)

    def contrastive_divergence_1(self, v1):
        '''Determine the weight updates according to CD-1'''
        h1 = self.sample_h_given_v(v1)
        v2 = self.sample_v_given_h(h1)
        h2p = self.propup(v2)
        return (T.outer(v1, h1) - T.outer(v2, h2p),
                v1 - v2,
                h1 - h2p)

    def cd1_fun(self, vis, learning_rate=0.01):
        (dW, dvbias, dhbias) = self.contrastive_divergence_1(vis)
        return theano.function(
            [vis],
            dvbias,
            updates=[(self.W, T.cast(self.W + dW*learning_rate, 'floatX')),
                     (self.vbias, T.cast(self.vbias + dvbias*learning_rate, 'floatX')),
                     (self.hbias, T.cast(self.hbias + dhbias*learning_rate, 'floatX'))]
        )

    def getcost(self, vis):
        vn = self.gibbs_vhv(vis)
        cost = vis * T.log(vn[0]) #+ (1 - vis) * T.log(1 - vn[0])
        #cost = T.log(self.propup(vis))
        out = T.sum(cost)
        return theano.function([vis], cost)