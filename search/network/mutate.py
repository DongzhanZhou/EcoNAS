import random
import os
import sys
import numpy as np
from .genotypes import Genotype,PRIMITIVES

def reconcat(gene):
    n=int(len(gene[0])/2)
    normal_contain=set(i[1] for i in gene[0])
    reduce_contain=set(i[1] for i in gene[2])
    normal_concat=set(range(n+2))-normal_contain
    reduce_concat=set(range(n+2))-reduce_contain
    new_gene=Genotype(
        normal=gene[0],
        normal_concat=list(normal_concat),
        reduce=gene[2],
        reduce_concat=list(reduce_concat)
    )
    return new_gene

def mutate_connection(gene):
    n=len(gene[0])
    block_type=random.choice([0,2])
    connection_index=random.choice(range(n))
    raw=gene[block_type][connection_index][1]
    i=int(connection_index/2)
    connection_choices=set(range(i+2))
    connection_choices.remove(raw)
    connection_choices=list(connection_choices)
    new=random.choice(connection_choices)
    new_block=gene[block_type]
    new_block[connection_index]=(new_block[connection_index][0],new)
    if block_type:
        new_gene=Genotype(
            normal=gene.normal,
            normal_concat=gene.normal_concat,
            reduce=new_block,
            reduce_concat=gene.reduce_concat
        )
    else:
        new_gene=Genotype(
            normal=new_block,
            normal_concat=gene.normal_concat,
            reduce=gene.reduce,
            reduce_concat=gene.reduce_concat
        )
    new_gene=reconcat(new_gene)
    return new_gene

def mutate_operation(gene):
    n=len(gene[0])
    block_type=random.choice([0,2])
    connection_index=random.choice(range(n))
    raw=gene[block_type][connection_index][0]
    choices= [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
    ]
    choices.remove(raw)
    new=random.choice(choices)
    new_block=gene[block_type]
    new_block[connection_index]=(new,new_block[connection_index][1])
    if block_type:
        new_gene=Genotype(
            normal=gene.normal,
            normal_concat=gene.normal_concat,
            reduce=new_block,
            reduce_concat=gene.reduce_concat
        )
    else:
        new_gene=Genotype(
            normal=new_block,
            normal_concat=gene.normal_concat,
            reduce=gene.reduce,
            reduce_concat=gene.reduce_concat
        )
    return new_gene

def mutate(gene):
    approach=random.choice([0,1])
    if approach:
        new_gene=mutate_connection(gene)
    else:
        new_gene=mutate_operation(gene)
    return new_gene

def randomarch(l):
    normal=[]
    reduce=[]
    normal_concat=range(2,l)
    reduce_concat=range(2,l)
    for i in range(l):
        normal.append((random.choice(PRIMITIVES),random.choice(range(i+2))))
        normal.append((random.choice(PRIMITIVES),random.choice(range(i+2))))
        reduce.append((random.choice(PRIMITIVES),random.choice(range(i+2))))
        reduce.append((random.choice(PRIMITIVES),random.choice(range(i+2))))
    gene=Genotype(
        normal=normal,
        normal_concat=normal_concat,
        reduce=reduce,
        reduce_concat=reduce_concat
    )
    gene=reconcat(gene)
    return gene

class NetworkManager(object):
    def __init__(self,logpath):
        self.logpath=logpath
        self.logs=[i for i in os.listdir(self.logpath) if '.log' in i]
        self.chpts=[i for i in os.listdir(self.logpath) if '.ckpt' in i]
        self.arch=['arch.py']
    def analyse_log(self):
        pass
    @staticmethod
    def dump(a,target,name='arch'):
        Head="from collections import namedtuple;Genotype=namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')"
        with open(target,'w') as f:
            f.write(Head+'\n')
            f.write(name+'='+str(a))
        return 1
    @staticmethod
    def load(target):
        target=importlib.import_module(target)
        arch=target.arch
        return arch
