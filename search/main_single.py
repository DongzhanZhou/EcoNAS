import os
import argparse
import importlib
import json
import network
import random
import numpy as np

def train_parallel(l,iter):
    with open('train_parallel.sh','w') as f:
        for i in l:
            f.write('python train_single.py --save=nets/%s --iter=%d &\n'%(i,iter))
        f.write('wait')
    os.system('sh train_parallel.sh')
    return 0

def arch_write(a,target):
    Head="from collections import namedtuple;Genotype=namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')"
    with open(target,'w') as f:
        f.write(Head+'\n')
        f.write('arch='+str(a))
    return 1

def init_write(target):
    open(target, "w").close()

def new_net(name,gene,root='nets'):
    os.makedirs(os.path.join(root,name))
    arch_write(gene,os.path.join(root,name,'arch.py'))
    init_write(os.path.join(root, name, '__init__.py'))

def arch_load(target):
    target=importlib.import_module(target.replace('/','.')+'.arch')
    arch=target.arch
    return arch

def result_scan(dir):
    result={}
    for net in os.listdir(dir):
        if os.path.exists(os.path.join(dir,net,'record.json')):
            result[net]=json.load(open(os.path.join(dir,net,'record.json')))
    return result

def sample(N,S,T):
    # N:total choices,S:samples,T:sample times
    # sample S nets from N nets, take the best one of the samples, repeat T times.
    return np.min((np.random.random([S,T])*N).astype('int'),axis=0)

def preprocess(dir):
    result=result_scan(dir)
    nets_index=[int(k[1:]) for k in os.listdir(dir) if k.startswith('a')]
    history={k:{kk:vv[0] for kk,vv in v.items()} for k,v in result.items()}
    accuracy={k:{kk:vv[1] for kk,vv in v.items()} for k,v in result.items()}

    last_updated={k:max(v.values()) for k,v in history.items()}
    best_acc={k:max(v.values()) for k,v in accuracy.items()}
    epochs={k:len(v) for k,v in accuracy.items()}
    e1_nets=[k for k,v in epochs.items() if v==1]
    return history,accuracy,last_updated,best_acc,epochs,rank(e1_nets,best_acc), nets_index

def rank(nets,net2acc):
    accs=[net2acc[net] for net in nets]
    index=np.argsort(accs)[::-1]
    nets=[nets[i] for i in index]
    return nets

def main():

    '''
    ITERS=100

    multiplier=4
    total_cards=multiplier*7
    e1=multiplier*4
    e2=multiplier*2
    e3=multiplier*1
    e1_net=[]
    e2_net=[]
    e3_net=[]
    for iter in ITERS
    '''
    for i in range(16):
        new_net('a%d'%i,network.randomarch(4))
    train_parallel(['a%d'%i for i in range(16)],0)

    def continue_choose(e1_nets,e2_nets):
        return e1_nets[:8]+e2_nets[:4]

    def mother_choose(e1_nets):
        l1 =len(e1_nets)
        choice1=[]
        if l1:choice1=sample(l1,l1//4+1,16)
        return [e1_nets[i] for i in choice1]

    def mother2children(e1_nets,start):
        mothers=mother_choose(e1_nets)
        mother_genes=[arch_load('nets/'+mother) for mother in mothers]
        children_genes=[network.mutate(mother_gene) for mother_gene in mother_genes]
        childrens=[]
        for i,gene in enumerate(children_genes):
            new_net('a%d'%(start+1+i),gene)
            childrens.append('a%d'%(start+1+i))
        return mothers,childrens

    def dead_filter(e1_nets,history,iter):
        e1_nets=[net for net in e1_nets if iter-history[net]<10]
        return e1_nets
    f_record=open('evo_record','w')

    for iter in range(100):
        history,accuracy,last_updated,best_acc,epochs,e1_nets,nets_index=preprocess('nets')
        e1_nets = dead_filter(e1_nets,last_updated,iter)
        mothers,children=mother2children(e1_nets,max(nets_index))
        f_record.write('iteration: %d:\mutate:\n%s->%s\n'%(iter,str(mothers),str(children)))
        train_parallel(children,iter)


if __name__=='__main__':
    main()
