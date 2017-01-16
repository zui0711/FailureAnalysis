# coding=utf-8
import random
import time

from models.CNN import *
from models.RBM import *


def mycnn(path, path_add, model_w2v, sent_len, word_dim, epoch,
          learning_rate, batch_size):
    """

    :param path: 数据路径
    :param path_add: 进一层路径
    :param model_w2v: embedding model
    :param sent_len: 句子长度，全文本用一个句子表示
    :param word_dim: embedding维度
    :param epoch: 迭代次数
    :param learning_rate: 学习步长
    :param batch_size: 批大小
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


def test_cnn_rbm(path, path_add, model_w2v, model_rbm, sent_len, word_dim, epoch, learning_rate, batch_size, text_size):
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

    train_len = 40
    test_len = 1000
    train_idxs = file_names_idx[:train_len]
    test_idxs = file_names_idx[train_len:train_len + test_len]

    train_num = train_len / batch_size
    test_num = test_len / batch_size

    cnn_model = FourConvCNN_rbm(np.random.RandomState(123), 8, 100, batch_size, text_size, learning_rate)

    # cnn_model.load_model(path+"cnnmodel")

    def test():
        wrong_rate = 0

        for i in xrange(test_len / batch_size):
            this_idxs = test_idxs[i * batch_size: (i + 1) * batch_size]
            testx, testy = get_batchdata_sent_onehot(this_path, file_names, this_idxs, model_w2v, model_rbm, sent_len, word_dim, text_size)
            wrong_rate += cnn_model.test(testx, testy)
            # print testy[:20]
            # print cnn_model.pred(testx)[:20], "\n"
        print "right_rate = ", 1. - wrong_rate / test_num

    print "...training"

    for ep in xrange(epoch):
        print "epoch = ", ep
        costep = 0

        for i in xrange(train_num):
            t1 = time.clock()

            this_idxs = train_idxs[i * batch_size: (i + 1) * batch_size]
            trainx, trainy = get_batchdata_sent_onehot(this_path, file_names, this_idxs, model_w2v, model_rbm, sent_len, word_dim, text_size)

            cost = cnn_model.train(trainx, trainy)
            t2 = time.clock()
            print i, t2 - t1, "       cost = ", cost
            costep += cost

            if (i + 1) % 1 == 0:
                test()
                print "costep = ", costep / 1.
                print "\n"
                costep = 0

    cnn_model.save_model(path + "cnnrbmmodel")


def test_rbm(path, path_add, model_w2v, sent_len, word_dim, epoch, learning_rate,
             batch_size=50):
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
    os.chdir(path)
    file_names = []
    if path_add:
        this_path = pjoin(path, path_add)
    else:
        this_path = path

    for dir_name in os.listdir(path_add):
        for file_name in os.listdir(pjoin(path_add, dir_name)):
            file_names.append(".".join([dir_name, file_name]))
    file_num = len(file_names)
    file_names_idx = range(file_num)
    print(file_num)
    random.shuffle(file_names_idx)

    train_len = 4000
    test_len = 1000
    train_idxs = file_names_idx[:train_len]
    test_idxs = file_names_idx[train_len:train_len + test_len]

    train_num = train_len / batch_size
    test_num = test_len / batch_size

    rbm_model = RBM(n_visible=word_dim*sent_len,
                    n_hidden=800,
                    numpy_rng=None,
                    theano_rng=None,
                    batch_size=batch_size,
                    learning_rate=learning_rate)

    # datasets = load_data(dataset)
    #
    # train_set_x, train_set_y = datasets[0]
    # test_set_x, test_set_y = datasets[2]

    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    #################################
    #     Training the RBM          #
    #################################


    #plotting_time = 0.
    start_time = timeit.default_timer()

    for ep in range(epoch):
        mean_cost = []
        t1 = timeit.default_timer()
        for i in range(train_num):
            this_idxs = train_idxs[i * batch_size: (i + 1) * batch_size]
            trainx, trainy = get_batchdata_sent(this_path, file_names, this_idxs, model_w2v, sent_len, word_dim, 500)

            cost = rbm_model.train(trainx.reshape(batch_size*500, word_dim*sent_len))
            #print("%d, %d" % (i, cost))
            mean_cost += [cost]
        t2 = timeit.default_timer()
        print('Training epoch %d, cost is ' % ep, np.mean(mean_cost), (t2-t1)/60.)

        # plotting_start = timeit.default_timer()
        # image = Image.fromarray(
        #     tile_raster_images(
        #         X=rbm.W.get_value(borrow=True).T,
        #         img_shape=(28, 28),
        #         tile_shape=(10, 10),
        #         tile_spacing=(1, 1)
        #     )
        # )
        # image.save('filters_at_epoch_%i.png' % epoch)
        # plotting_stop = timeit.default_timer()
        # plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = end_time - start_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    rbm_model.save_model(pjoin(path, "rbm_model"))


    #################################
    #     Sampling from the RBM     #
    #################################

    # number_of_test_samples = trainx.reshape(batch_size, word_dim*sent_len).shape[0]

    #test_idx = rng.randint(number_of_test_samples - n_chains)
    # test_idx = 0
    # persistent_vis_chain = theano.shared(
    #     numpy.asarray(
    #         trainx.reshape(batch_size, word_dim * sent_len)[test_idx:test_idx + n_chains],
    #         dtype=theano.config.floatX
    #     )
    # )

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
    #     rbm.gibbs_vhv,
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
    #
    # image_data = numpy.zeros(
    #     (29 * n_samples + 1, 29 * n_chains - 1),
    #     dtype='uint8'
    # )
    # for idx in range(n_samples):
    #     vis_mf, vis_sample = sample_fn()
    #     print(' ... plotting sample %d' % idx)
    #     image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
    #         X=vis_mf,
    #         img_shape=(28, 28),
    #         tile_shape=(1, n_chains),
    #         tile_spacing=(1, 1)
    #     )
    #
    # image = Image.fromarray(image_data)
    # image.save('samples.png')

    os.chdir('../')

def test_rbm_r(model, path="/media/zui/work/NETWORK/aclImdb/my/", learning_rate=0.01, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20, sent_len=30,word_dim=50,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=1000):
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
    os.chdir(path)
    file_names = []

    for dir_name in os.listdir(path):
        for file_name in os.listdir(pjoin(path, dir_name)):
            file_names.append(".".join([dir_name, file_name]))
    file_num = len(file_names)
    file_names_idx = range(file_num)
    print(file_num)
    random.shuffle(file_names_idx)

    train_len = 5000
    test_len = 1000
    train_idxs = file_names_idx[:train_len]
    test_idxs = file_names_idx[train_len:train_len + test_len]

    train_num = train_len / batch_size
    test_num = test_len / batch_size

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    rbm = RBM(input=x, n_visible=word_dim * sent_len,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=1)

    #################################
    #     Training the RBM          #
    #################################

    train_rbm = theano.function(
        [x],
        cost,
        updates=updates,
        name='train_rbm'
    )

    start_time = timeit.default_timer()

    for epoch in range(training_epochs):

        mean_cost = []
        for i in range(train_num):
            this_idxs = train_idxs[i * batch_size: (i + 1) * batch_size]
            trainx, trainy = get_batchdata(path, file_names, this_idxs, model_w2v, sent_len, word_dim)

            thiscost = train_rbm(trainx)
            #print('thiscost = ', thiscost)
            mean_cost += [thiscost]

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time)

    print ('Training took %f minutes' % (pretraining_time / 60.))

    os.chdir('../')

def test(model, path="/media/zui/work/NETWORK/aclImdb/my/", learning_rate=0.01, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20, sent_len=30,word_dim=50,
             n_hidden=1000):
    n_visible = sent_len * word_dim
    num_hidden = 30

    os.chdir(path)
    file_names = []

    for dir_name in os.listdir(path):
        for file_name in os.listdir(pjoin(path, dir_name)):
            file_names.append(".".join([dir_name, file_name]))
    file_num = len(file_names)
    file_names_idx = range(file_num)
    print(file_num)
    random.shuffle(file_names_idx)

    train_len = 2000
    test_len = 1000
    train_idxs = file_names_idx[:train_len]
    test_idxs = file_names_idx[train_len:train_len + test_len]

    train_num = train_len / batch_size
    test_num = test_len / batch_size

    rbm = thisrbm(n_visible, n_hidden)

    #xvis = T.fvector('xvis')
    #h1samples = rbm.sample_h_given_v(xvis)
    #v2samples = rbm.sample_v_given_h(h1samples)
    #sample_vhv = theano.function([xvis], v2samples)

    #example_indices = numpy.random.randint(low=0, high=num_data, size=num_examples)

    """
    def show_examples():
        for example in example_indices:
            dat = encoded[example]
            v2samples = sample_vhv(dat)
            print('input words:',
                  [(t + 1, words[idx])
                   for t in range(tuplesize)
                   for idx in range(num_words)
                   if encoded[example, t * num_words + idx]])
            print('reconstructed words:',
                  [(t + 1, words[idx])
                   for t in range(tuplesize)
                   for idx in range(num_words)
                   if v2samples[t * num_words + idx]])
            print('')

    def report_hidden():
        weights = rbm.weights.get_value()
        for h in range(num_hidden):
            print('hidden ', h)
            for block in range(tuplesize):
                for word in range(num_words):
                    w = weights[block * num_words + word, h]
                    if w > 0.5:
                        print('   %2i %8s  %4.1f' % (block, words[word], w))
    """

    vis = T.fvector('vis')
    train = rbm.cd1_fun(vis, 0.1)
    getcost = rbm.getcost(vis)
    #input_data = numpy.reshape(encoded[2], num_visible)
    #train(input_data)

    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        all_vdiffs = numpy.zeros(n_visible)
        print('epoch ', epoch)
        for i in range(train_num):
            aa = numpy.zeros(n_visible)
            this_idxs = train_idxs[i * batch_size: (i + 1) * batch_size]
            trainx, trainy = get_batchdata(path, file_names, this_idxs, model_w2v, sent_len, word_dim)
            cc = []
            for b in range(batch_size):
                vdiffs = train(trainx[b])
                all_vdiffs = all_vdiffs + numpy.abs(vdiffs)
                #aa = aa + numpy.abs(vdiffs)
                #thiscost = getcost(np.array(trainx[b]))
                #print(thiscost)
                #cc += [thiscost]
            #print('reconstruction error: ', numpy.sum(aa))

        print('reconstruction error: ', numpy.mean(all_vdiffs))
        #print(T.cast(rbm.W.get_value() * 100, 'int32').eval())


    end_time = timeit.default_timer()

    print(end_time - start_time)


if __name__ == "__main__":
    #pa = "/media/zui/work/NETWORK/aclImdb/"
    # model_w2v = load_model(pa, "imdb_model_count100size50")
    # mycnn(m_path, "cut500/all/", model_w2v, m_sent_len_o, m_word_dim, m_epoch, m_learning_rate,
    #       m_batch_size)
    #test_rbm(pa, "my/", model_w2v, 100, m_word_dim, 3, m_learning_rate,
          # m_batch_size)
    # test_rbm_r(model_w2v)
    # test(model_w2v)
    model_w2v = load_model_onehot(m_path, m_model_w2v_name, 100)
    model_rbm = RBM()
    model_rbm.load_model(pjoin(m_path, "rbm_model_2000_800"))
    test_cnn_rbm(path=m_path, path_add="cut500/all/", model_w2v=model_w2v, model_rbm=model_rbm, sent_len=15, word_dim=100, epoch=10, learning_rate=0.05,
          batch_size=20, text_size=500)

    # rbm_model = RBM()
    # rbm_model.load_model(pjoin(m_path, "rbm_model"))
    # print(rbm_model.W.get_value(borrow=True))