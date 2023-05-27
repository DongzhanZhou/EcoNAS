import json
import numpy as np
import os
import sys

path=sys.argv[1]
record={}
for arch in os.listdir(path):
    json_path=os.path.join(path,arch,'record.json')
    if os.path.exists(json_path):
        try:
            record[arch]=json.load(open(json_path))
        except:
            print(json_path)
    else:
        continue


net2acc={}
x=[]
y=[]
xx=[]
yy=[]
for k,v in record.items():
    if 'e%s'%sys.argv[2] in v.keys():
        net2acc[k]=v['e%s'%sys.argv[2]][1]
        x.append(k)
        xx.append(int(k[1:]))
        y.append(v['e%s'%sys.argv[2]][1])
        yy.append(v['e%s'%sys.argv[2]][1])

index=np.argsort(y)[::-1]
print([x[i] for i in index])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.scatter(xx,yy)
plt.title("Epoch%d"%(int(sys.argv[2])*20))
plt.xlabel('network index')
plt.ylabel('accuracy')
plt.show()
plt.savefig('im.jpg')

