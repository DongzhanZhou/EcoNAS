import json
import numpy as np
import os
import sys

import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--save', type=str, default='', help='location of the searched models')
parser.add_argument('--group', type=str, default='e3', choices=['e1', 'e2', 'e3'], help='scan the specific group to find the top-k models')
parser.add_argument('--topN', type=int, default=5, help='show the best N models')
args = parser.parse_args()

record={}
for arch in os.listdir(args.save):
    json_path=os.path.join(args.save,arch,'record.json')
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

for k,v in record.items():
    if args.group in v.keys():
        net2acc[k]=v[args.group][1]
        x.append(k)
        y.append(v[args.group][1])

index=np.argsort(y)[::-1]
for i in range(args.topN):
    print("Arch %s, Acc %.3f" %(x[i], y[i]))

