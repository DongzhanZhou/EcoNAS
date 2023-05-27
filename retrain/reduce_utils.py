from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import os

class AverageMeter(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0
  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

class ReduceSampler(Sampler):
    def __init__(self, data_source, ratio):
        self.data_source = data_source
        self.ratio=ratio
        self.raw_length = len(self.data_source)
        self.red_length = self.__len__()
        self.indices = random.sample(range(self.raw_length),self.red_length)
    def __iter__(self):
        return iter(self.indices)
    def set_epoch(self):
        random.shuffle(self.indices)
    def __len__(self):
        return int(len(self.data_source)*self.ratio)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def transforms_cifar10_reduction(res, cutout_length):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize([res,res]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if cutout_length:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
    transforms.Resize([res,res]),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x