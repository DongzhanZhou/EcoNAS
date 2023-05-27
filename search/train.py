import os
import sys
import time
import json
import numpy as np
import torch


import logging
import argparse
import torch.nn as nn

import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import network
import importlib

import network.utils as utils

from network.model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=20, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=384, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--sampler_ratio', type=float, default=1.0, help='sampler_ratio')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=9, help='num of init channels')
parser.add_argument('--res', type=int, default=8, help='resolution')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='a1', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--iter', type=int, default=5, help='iter')


args = parser.parse_args()


def main():
  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  
  genotype=importlib.import_module(args.save.replace('/','.')+'.arch')
  genotype=genotype.arch

  if os.path.exists(os.path.join(args.save,'record.json')):
      record=json.load(open(os.path.join(args.save,'record.json')))
  else:
      record={}

  epochs_min=len(record)*args.epochs
  epochs_max=(len(record)+1)*args.epochs

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)

  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  logging.info("args = %s", args)

  model = Network(10,args.init_channels, args.layers,  genotype)

  model = model.cuda()
  if len(record):
    model.load_state_dict(torch.load(os.path.join(args.save, str(epochs_min)+'_weights.pt')))

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay = args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_sampler = utils.ReduceSampler(train_data, args.sampler_ratio)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=1, sampler=train_sampler)
  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size//2, shuffle=False, pin_memory=False, num_workers=1)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs*3))
  
  best_acc=0
  for epoch in range(args.epochs*3):

    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch /(3*args.epochs)

    if epoch < epochs_min:continue
    if epoch >= epochs_max:break

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    if valid_acc>best_acc:best_acc=valid_acc
  
  utils.save(model, os.path.join(args.save, str(epochs_max)+'_weights.pt'))
  record['e%d'%(len(record)+1)]=[args.iter,float(best_acc.cpu().numpy())]
  json.dump(record,open(os.path.join(args.save,'record.json'),'w'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()
    optimizer.zero_grad()
    logits= model(input,)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
