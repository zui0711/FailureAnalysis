# coding=utf-8
from CNN_txt import *
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

if __name__ == "__main__":
    model_w2v = load_model(m_path, m_model_w2v_name)
    mycnn(m_path, "cut500/all/", model_w2v, imdb_sent_len, m_word_dim, m_epoch, m_learning_rate,
          imdb_batch_size_o)

