# coding=utf-8

import numpy as np
import theano
from theano import tensor as T
import time
from datetime import datetime
import myRNN

vocabulary_size = 100 #词典词数
word_dim = vocabulary_size #输入维数
hidden_dim = 50 #隐藏层
class_dim = 2 #分类数
bptt_truncate = 4 #时间步数
nepoch = 200 #迭代次数
learning_rate = 0.005

'''
[o, s], updates = theano.scan(
    forward_prop_step,
    sequences=x,
    outputs_info=[None, dict(initial=T.zeros(hidden_dim))],
    non_sequences=[U, V, W],
    truncate_gradient=bptt_truncate,
    strict=True
    )

prediction = T.argmax(o)
o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

dU = T.grad(o_error, U)
dV = T.grad(o_error, V)
dW = T.grad(o_error, W)

forward_propagation = theano.function([x], o)
predict = theano.function([x], prediction)
ce_error = theano.function([x, y], o_error)
bptt = theano.function([x, y], [dU, dV, dW])


learning_rate = T.scalar('learning_rate')
sgd_step = theano.function([x,y,learning_rate], [],
                      updates=[(U, U - learning_rate * dU),
                              (V, V - learning_rate * dV),
                              (W, W - learning_rate * dW)])

def calculate_total_loss(X, Y):
    return np.sum([ce_error(x,y) for x,y in zip(X, Y)])

def calculate_loss(X, Y):
    num_words = np.sum([len(y) for y in Y])
    return calculate_total_loss(X,Y)/float(num_words)


f = open('data/reddit-comments-2015-08.csv', 'rb')
context = f.readlines()
for i in xrange(10):
    print context[i]
    print 'AAA'
print len(context)
f.close()



unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

with open('data/X_train.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size

print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

t1 = time.time()
mylearning_rate = 0.0005
sgd_step(X_train[10], y_train[10], mylearning_rate)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

losses = []
num_examples_seen = 0
evaluate_loss_after=10
for epoch in range(nepoch):
    # Optionally evaluate the loss
    if (epoch % evaluate_loss_after == 0):
        loss = calculate_loss(X_train[:100], y_train[:100])
        losses.append((num_examples_seen, loss))
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
        # Adjust the learning rate if loss increases
        if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
            mylearning_rate = mylearning_rate * 0.5
            print "Setting learning rate to %f" % mylearning_rate
        sys.stdout.flush()
        # ADDED! Saving model oarameters
        #save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
    # For each training example...
    for i in range(len(y_train[:100])):
        # One SGD step
        sgd_step(X_train[i], y_train[i], mylearning_rate)
        num_examples_seen += 1

print predict(X_train[0])


pp = [index_to_word[xp] for xp in predict(X_train[0])]
print pp
'''
X_train = myRNN.get_xtrain('data/X_train.csv', vocabulary_size)
y_train = myRNN.get_ytrain('data/y_train.txt')

print X_train[0]
print y_train


model = myRNN.RNN(word_dim, hidden_dim, class_dim, bptt_truncate)
model.sgd_step(X_train[10], y_train[10], learning_rate)
#[o, s] = model.forward_prop_step(X_train[0], np.zeros(50))
#model.U[:,X_train[0]].eval().shape
#T.nnet.ultra_fast_sigmoid(model.V.dot(model.U[:,X_train[0]])).eval()

#model.U[:,X_train[0]] + model.W.dot(s_t_prev)
print o, s