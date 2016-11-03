# coding=utf-8
import cPickle, gzip
import numpy as np
import matplotlib.pyplot as plt


def display(data):  # 显示图片
    x = np.arange(0, 28)
    y = np.arange(0, 28)
    X, Y = np.meshgrid(x, y)
    plt.imshow(data.reshape(28, 28), interpolation='nearest', cmap='bone')
    plt.colorbar()
    plt.show()
    return


def save(data, name):  # 保存图片
    x = np.arange(0, 28)
    y = np.arange(0, 28)
    X, Y = np.meshgrid(x, y)
    plt.imshow(data.reshape(28, 28), interpolation='nearest', cmap='bone')
    plt.savefig(name)
    return


f = gzip.open('data/mnist.pkl.gz', 'rb')  # 读取数据
train_set, valid_set, test_set = cPickle.load(f)  # 分类
f.close()
train_set_image, train_set_num = train_set

token = 1  # 需要显示的图片个数
for i in range(0, token):
    save(train_set_image[i], "data/pic/" + str(i) + "-" + str(train_set_num[i]))
