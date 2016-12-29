# coding=utf-8

import re
from setting import *
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join as pjoin

ERRORNAME = ["USER_CONGESTION",
             "GTPC_TUNNEL_PATH_BROKEN",
             "PROCESS_CPU",
             "SYSTEM_FLOW_CTRL",
             "EPU_PORT_CONGESTION"]

# alltime = {}
# with open(m_path+"BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1.log", "rb") as f:
    # con = f.readlines()
    # for line in con:
    #     write = False
    #     arr = line.split()
    #     for word in arr:
    #         if word in ERRORNAME:
    #             write = True
    #             break
        # if write:
        #     time = re.findall(r"\[\d+\.\d+\]", line)
        #     #s = re.findall(r"\d+\.", time[0])
        #     thistime = int(re.findall(r"\d+", time[0])[0])
        #     if thistime in alltime:
        #         alltime[thistime] += 1
        #     else:
        #         alltime[thistime] = 1

# alltime = sorted(alltime.iteritems(), key=lambda x:x[0], reverse=False)
# for time in alltime:
#     print time[0], time[1]


#X = np.linspace(-np.pi,np.pi,256,endpoint=True)
#X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

#(C,S)=np.cos(X),np.sin(X)

# for name in [os.listdir(pjoin(m_path, "ERRORINFO"))[4]]:
#     arr = name.split("-")
#     print arr
#     x_time = []
#     y_num = []
#     with open(pjoin(m_path, "ERRORINFO", name), "rb") as f:
#         con = f.readlines()
#         for line in con[:50]:
#             [x, y] = line.split()
#             x_time.append(x)
#             y_num.append(y)
#     plt.plot(np.array(x_time), np.array(y_num))
#     #plt.xticks(np.linspace(21, 41, 5))
#     #plt.yticks(np.linspace(0, 6, 3))
#
#     plt.xlabel("time/s")
#     plt.ylabel("count")
#     plt.show()


#plt.plot(X,C)
#plt.plot(X,S)
#plt.show()
x1 = np.linspace(0, 55, 56)
x2 = np.linspace(21, 51, 6)
y = np.zeros(56)
for i in xrange(56):
    if i in x2:
        y[i] = 5

print x1
print x2
print y

plt.plot(x1[20:], y[20:])
#plt.xticks(np.linspace(20, 60, 41))
#plt.yticks(np.linspace(0, 6, 4))

plt.xlabel("time/s")
plt.ylabel("count")
plt.show()