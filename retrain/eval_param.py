import numpy as np
import time
import os
import sys
import argparse

import torch
import torchvision
import numpy as np
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import collections
import torch.nn as nn
from model import NetworkCIFAR

from collections import namedtuple
Genotype=namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
#arch=Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=[2, 4, 5])
arch = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0)], reduce_concat=[3, 4, 5])

def flops_to_string(flops):
    if flops // 10**9 > 0:
        return str(round(flops / 10.**9, 2)) + 'GMac'
    elif flops // 10**6 > 0:
        return str(round(flops / 10.**6, 2)) + 'MMac'
    elif flops // 10**3 > 0:
        return str(round(flops / 10.**3, 2)) + 'KMac'
    return str(flops) + 'Mac'

def count_parameters_in_MB(model):
  return np.sum(np.fromiter((np.prod(v.size()) for v in model.parameters()),float))/1e6

def main():
    torch.backends.cudnn.benchmark = True
    model = NetworkCIFAR(10, 36, 20, arch)
    model = model.cuda()
    print(count_parameters_in_MB(model))

if __name__ == '__main__':
    main()
