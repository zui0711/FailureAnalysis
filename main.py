# coding=utf-8
from CNN_txt import *
from rbm import *
import time, random
from os.path import join as pjoin


def mycnn(path, path_add, model_w2v, sent_len, word_dim, epoch,
          learning_rate, batch_size):
    """

    :param path: 数据路径
    :param path_add: 进一层路径
    :param model_w2v: embedding model
    :param sent_len: 句子长度
    :param word_dim: embedding维度
    :param epoch: 迭代次数
    :param learning_rate: 学习步长
    :param batch_size: 批大小
    :param text_size: 样本中句子数
    :return:
    """

    file_names = []
    if path_add:
        this_path = pjoin(path, path_add)
    else:
        this_path = path
    for dir_name in os.listdir(this_path):
        for file_name in os.listdir(pjoin(this_path, dir_name)):
            file_names.append(".".join([dir_name, file_name]))
    file_num = len(file_names)
    file_names_idx = range(file_num)
    random.shuffle(file_names_idx)

    train_len = 20000
    test_len = 1000
    train_idxs = file_names_idx[:train_len]
    test_idxs = file_names_idx[train_len:train_len + test_len]

    train_num = train_len / batch_size
    test_num = test_len / batch_size

    cnn_model = FourConvCNN(np.random.RandomState(123), sent_len, word_dim, batch_size, learning_rate)

    # cnn_model.load_model(path+"cnnmodel")

    def test():
        wrong_rate = 0

        for i in xrange(test_len / batch_size):
            this_idxs = test_idxs[i * batch_size: (i + 1) * batch_size]
            testx, testy = get_batchdata(this_path, file_names, this_idxs, model_w2v, sent_len, word_dim)
            wrong_rate += cnn_model.test(testx, testy)
            #print testy[:20]
            #print cnn_model.pred(testx)[:20], "\n"
        print "right_rate = ", 1. - wrong_rate / test_num

    print "...training"

    for ep in xrange(epoch):
        print "epoch = ", ep
        costep = 0

        for i in xrange(train_num):
            t1 = time.clock()

            this_idxs = train_idxs[i * batch_size: (i + 1) * batch_size]
            trainx, trainy = get_batchdata(this_path, file_names, this_idxs, model_w2v, sent_len, word_dim)

            cost = cnn_model.train(trainx, trainy)
            t2 = time.clock()
            print i, t2 - t1, "       cost = ", cost
            costep += cost

            if (i + 1) % 20 == 0:
                test()
                print "costep = ", costep / 20.
                print "\n"
                costep = 0

    cnn_model.save_model(path+"cnnmodel")


def test_rbm(learning_rate=0.1, training_epochs=1,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    rbm = RBM(input=x, n_visible=28 * 28,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    image_data0 = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in range(n_samples):
        print(' ... plotting origin %d' % idx)
        image_data0[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=train_set_x.get_value(),
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    image = Image.fromarray(image_data0)
    image.save('origin.png')

    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

        plotting_start = timeit.default_timer()
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    #################################
    #     Sampling from the RBM     #
    #################################

    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

    plot_every = 1000
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )

    updates.update({persistent_vis_chain: vis_samples[-1]})
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    image_data = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in range(n_samples):
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    image = Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')


if __name__ == "__main__":
    model_w2v = load_model(m_path, m_model_w2v_name)
    mycnn(m_path, "cut500/all/", model_w2v, m_sent_len_o, m_word_dim, m_epoch, m_learning_rate,
          m_batch_size)

