# coding=utf-8
import numpy as np

f = open('data/log_log.txt', 'rb')
context = f.readlines()

FORMATLETTER = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

lines = []
lens = 0
for line in context[0:100]:
    sp = line.split()
    line = ' '.join(sp[:len(sp) - 1])
    for c in line:
        if c not in FORMATLETTER:
            line = line.replace(c, ' ')
    line = ' '.join(line.split())
    lines.append(line)
    lens += len(line)

llines = len(lines)
tmpy = [len(ll) > float(lens)/llines for ll in lines]
y = []
#print float(lens)/llines
#print 'llines', llines
#print lens

ff = open('data/X_train.csv', 'wb')
for line in lines:
    for xx in line.split():
        ff.write('%s '%(xx))
    ff.write('\n')
ff.close()

ff2 = open('data/y_train.txt', 'wb')
y_train = np.zeros(llines)
for i, y in enumerate(tmpy):
    y_train[i] = 1 if y else 0
#print sum(y_train)
for y in y_train:
    ff2.write('%d '%(y))