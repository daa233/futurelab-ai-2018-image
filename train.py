# Training code for FUTURELAB.AI 2018
# Adapted from https://github.com/macaodha/inat_comp_2018/blob/master/train_inat.py
# Author: Du Ang
# Create date: May 7, 2018
# Last update: May 18, 2018

from __future__ import print_function

import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pretrainedmodels
import models.torchvision_models as models

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

import futurelab_loader

parser = argparse.ArgumentParser(description='Training code for Futurelab.AI 2018')
parser.add_argument('--model', default='resnet50', type=str,
                    help='model type')
parser.add_argument('--pretrained', action='store_true',
                    help='fine-tuned on the pretrained model')
parser.add_argument('--num_classes', default=20, type=int,
                    help='the number of classes (default: 20)')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size for training (default: 224)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=0.0045, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual start epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str,
                    help='path to the checkpoint you want to resume (default: none)')
parser.add_argument('--save_every_checkpoint',
                    help='save every epoch checkpoint', action='store_true')
parser.add_argument('--expname', default='', type=str,
                    help='name of experiment')
parser.add_argument('--workers', default='4', type=int, help='number of workers')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2.')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--tensorboard',
                    help='log progress to TensorBoard', action='store_true')
parser.add_argument('--data_root', type=str, help='the training and validation data root')
parser.add_argument('--train_file_list', type=str, help='the train csv file list')
parser.add_argument('--val_file_list', type=str, help='the val csv file list')
parser.add_argument('--evaluate', help='evaluate the trained model', action='store_true')
parser.add_argument('--save_pred', help='save predicted labels', action='store_true')
parser.add_argument('--save_logits', help='save predicted logits', action='store_true')
parser.add_argument('--softmax_logits', help='save predicted logits', action='store_true')
parser.add_argument('--pred_filename', default='eval_pred.csv', type=str, help='the name of pred csv file')
parser.add_argument('--logits_filename', default='eval_logits.csv', type=str, help='the name of logits csv file')


best_prec1 = 0.0
best_prec3 = 0.0  # store current best top 3


def main():
    global args, best_prec1, best_prec3
    args = parser.parse_args()

    checkpoint_path = os.path.join('checkpoints', args.model + '_' + args.expname)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if not args.evaluate:
        with open(os.path.join(checkpoint_path, 'info.txt'), 'w') as info_file:
            info_file.write('The configuration to train the model:\n')
            for key, value in vars(args).items():
                info_file.write('\t{:>25}\t:\t{}\n'.format(key, value))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)

    # load pretrained model
    if args.model in pretrainedmodels.model_names:
        if args.pretrained:
            model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
            dim_features = model.last_linear.in_features
            model.last_linear = nn.Linear(dim_features, args.num_classes)
            print("Using pre-trained {}".format(args.model))
        else:
            model = models.__dict__[args.model](num_classes=1000)
            dim_features = model.last_linear.in_features
            model.last_linear = nn.Linear(dim_features, args.num_classes)
            print("Using {} without being pre-trained".format(args.model))
    else:
        raise NotImplementedError('Unsupport model: {}. Option: {}'
                        .format(args.model, pretrainedmodels.model_names))

    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if isinstance(checkpoint, dict):
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                if 'best_prec1' in checkpoint.keys():
                    best_prec1 = checkpoint['best_prec1']
                if 'best_prec3' in checkpoint.keys():
                    best_prec3 = checkpoint['best_prec3']
                if 'optimizer' in checkpoint.keys():
                    optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                model = checkpoint
        else:
            raise NotImplementedError("=> no checkpoint found at '{}'".format(args.resume))

    
    cudnn.benchmark = True

    # data loading code
    train_dataset = futurelab_loader.DATA(args.data_root,
                                          args.train_file_list,
                                          args.image_size,
                                          is_train=True)
    val_dataset = futurelab_loader.DATA(args.data_root,
                                        args.val_file_list,
                                        args.image_size,
                                        is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                   shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    
    # TensorBoard configure
    if args.tensorboard:
        configure(checkpoint_path)

    if args.evaluate:
            # write predictions to file
            if args.save_pred:
                prec1, prec3, preds, im_ids = validate(val_loader, model, criterion, epoch=1, save_pred=True)
                pred_filename = os.path.join(checkpoint_path, args.pred_filename)
                with open(pred_filename, 'w') as opfile:
                    opfile.write('FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2\n')
                    for ii in range(len(im_ids)):
                        opfile.write(str(im_ids[ii]) + ',' + ','.join(str(x) for x in preds[ii,:])+'\n')
                print("The pred csv file on validation set has been saved to {}".format(pred_filename))
            elif args.save_logits:
                prec1, prec3, logits, preds, im_ids = validate_logits(val_loader, model, criterion, epoch=1, save_logits=True)
                if args.softmax_logits:
                    args.logits_filename = 'softmax_' + args.logits_filename
                logits_filename = os.path.join(checkpoint_path, args.logits_filename)
                with open(logits_filename, 'w') as opfile:
                    opfile.write('FILE_ID')
                    for i in range(args.num_classes):
                        opfile.write(',CATEGORY_ID' + str(i))
                    opfile.write('\n')
                    for ii in range(len(im_ids)):
                        descent_indices = [int(index) for index in preds[ii, :]]
                        # print(descent_indices)
                        # print(logits[ii, :])
                        sorted_logits = [0 for i in range(len(descent_indices))]
                        for j in range(len(descent_indices)):
                            sorted_logits[descent_indices[j]] = logits[ii, j]
                        # print(sorted_logits)
                        opfile.write(str(im_ids[ii]) + ',' + ','.join(str(x) for x in sorted_logits)+'\n')
                print("The logits csv file on validation set has been saved to {}".format(logits_filename))
            return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec3 = validate(val_loader, model, criterion, epoch, False)

        # remember best prec@3 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec1 = max(prec1, best_prec1)
        best_prec3 = max(prec3, best_prec3)
        
        save_dict = {
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec3': best_prec3,
            'optimizer' : optimizer.state_dict(),
        }

        if args.save_every_checkpoint:
            save_checkpoint(save_dict, is_best, checkpoint_path, epoch+1)
        else:
            save_checkpoint(save_dict, is_best, checkpoint_path)

        print('Current best Prec@1 = {} \t best Pre@3 = {}'.format(best_prec1, best_prec3))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('Epoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tData\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, (input, im_id, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # For Inception-v3, the output may be a tuple
        if type(output) is tuple:
            output = output[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                '{data_time.val:.2f} ({data_time.avg:.2f})\t'
                '{loss.val:.3f} ({loss.avg:.3f})\t'
                '{top1.val:.2f} ({top1.avg:.2f})\t'
                '{top3.val:.2f} ({top3.avg:.2f})'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))

    # Log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_top1_accuracy', top1.avg, epoch)
        log_value('train_top3_accuracy', top3.avg, epoch)


def validate(val_loader, model, criterion, epoch, save_pred=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pred = []
    im_ids = []

    print('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, (inputs, im_id, target) in enumerate(val_loader):

        inputs = inputs.cuda()
        target = target.cuda(async=True)
        batch_size, n_crops, c, h, w = inputs.size()
        input_var = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        output = output.view(batch_size, n_crops, -1).mean(1)
        loss = criterion(output, target_var)

        if save_pred:
            # store the top K classes for the prediction
            im_ids.append(im_id)
            _, pred_inds = output.data.topk(3,1,True,True)
            pred.append(pred_inds.cpu().numpy().astype(np.int))

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))

    # Log to TensorBoard
    if args.tensorboard and not args.evaluate:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_top1_accuracy', top1.avg, epoch)
        log_value('val_top3_accuracy', top3.avg, epoch)    

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    if save_pred:
        return top1.avg, top3.avg, np.vstack(pred), np.hstack(im_ids)
    else:
        return top1.avg, top3.avg


def validate_logits(val_loader, model, criterion, epoch, save_logits=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    logit = []
    pred = []
    im_ids = []

    print('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, (inputs, im_id, target) in enumerate(val_loader):

        inputs = inputs.cuda()
        target = target.cuda(async=True)
        batch_size, n_crops, c, h, w = inputs.size()
        input_var = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        if args.softmax_logits:
            output = model(input_var)
            output = output.view(batch_size, n_crops, -1).mean(1)
            output = torch.nn.Softmax(dim=-1)(output)
        else:
            output = model(input_var)
            output = output.view(batch_size, n_crops, -1).mean(1)
        loss = criterion(output, target_var)

        if save_logits:
            # store the top K classes for the prediction
            im_ids.append(im_id)
            logit_inds, pred_inds = output.data.topk(args.num_classes,1,True,True)
            logit.append(logit_inds.cpu().numpy().astype(np.float))
            pred.append(pred_inds.cpu().numpy().astype(np.int))

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))   

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    if save_logits:
        return top1.avg, top3.avg, np.vstack(logit), np.vstack(pred), np.hstack(im_ids)
    else:
        return top1.avg, top3.avg


def save_checkpoint(state, is_best, checkpoint_path, epoch=-1, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_path, filename))
    if epoch >= 0:
        print("\tSave the epoch {} model".format(epoch))
        shutil.copyfile(os.path.join(checkpoint_path, filename),
                        os.path.join(checkpoint_path, 'epoch_{}_model.pth.tar'.format(epoch)))
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(os.path.join(checkpoint_path, filename),
                        os.path.join(checkpoint_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    log_value('lr', lr, epoch)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
