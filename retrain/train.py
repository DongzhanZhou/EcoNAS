import os
import sys
from collections import namedtuple
import time
import glob
import numpy as np
import torch
import logging
import argparse
import json
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import reduce_utils
from model import NetworkCIFAR
import importlib


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--batch_size_val', type=int, default=48, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--flip', type=float, default=0.5, help='flip probability')
parser.add_argument('--res', type=int, default=32, help='resolution of images')
parser.add_argument('--padding', type=int, default=4, help='add padding to images')
parser.add_argument('--sampler_ratio', type=float, default=1.0, help='sampler ratio')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='', help='log dir')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

def main():

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10

  genotype = importlib.import_module(args.save.replace('/','.')+'.arch')
  genotype = genotype.arch

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  logging.info("genotypes = %s", genotype)

  model = NetworkCIFAR(CIFAR_CLASSES, args.init_channels, args.layers, genotype)
  model = model.cuda()
  logging.info("param size = %fMB", reduce_utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = reduce_utils.transforms_cifar10_reduction(args.res, args.cutout_length)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size_val, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  #best_acc = 0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    logging.info("drop path prob %f", model.drop_path_prob)
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    if (epoch + 1) % 20 == 0:
        save_dict = {'state_dict': model.state_dict(), 'epoch': epoch + 1}
        torch.save(save_dict, os.path.join(args.save, 'ckpt.pth'))

def train(train_queue, model, criterion, optimizer):
  objs = reduce_utils.AverageMeter()
  top1 = reduce_utils.AverageMeter()
  top5 = reduce_utils.AverageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = reduce_utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = reduce_utils.AverageMeter()
  top1 = reduce_utils.AverageMeter()
  top5 = reduce_utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = reduce_utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  main()
