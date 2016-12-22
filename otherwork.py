import re
from setting import *

ERRORNAME = ["USER_CONGESTION",
             "GTPC_TUNNEL_PATH_BROKEN",
             "PROCESS_CPU",
             "SYSTEM_FLOW_CTRL",
             "EPU_PORT_CONGESTION"]

alltime = {}
with open(m_path+"BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1.log", "rb") as f:
    con = f.readlines()
    for line in con:
        write = False
        arr = line.split()
        for word in arr:
            if word in ERRORNAME:
                write = True
                break
        if write:
            time = re.findall(r"\[\d+\.\d+\]", line)
            #s = re.findall(r"\d+\.", time[0])
            thistime = int(re.findall(r"\d+", time[0])[0])
            if thistime in alltime:
                alltime[thistime] += 1
            else:
                alltime[thistime] = 1

alltime = sorted(alltime.iteritems(), key=lambda x:x[0], reverse=False)
for time in alltime:
    print time[0], time[1]

# for (k, v) in alltime.items():
#     print "dict[%s]=" % k, v

#string = "A1.4.5, B5, 6.45, 8.82 [23.45]"
#print re.findall(r"\[+\d+\.\d+\]", string)
#print re.findall(r"\d+\.?\d*", string)
