# Test code for FUTURELAB.AI 2018
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


parser = argparse.ArgumentParser(description='Testing code for Futurelab.AI 2018')
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
parser.add_argument('--resume', default='', type=str,
                    help='path to the checkpoint you want to resume (default: none)')
parser.add_argument('--expname', default='', type=str,
                    help='name of experiment')
parser.add_argument('--workers', default='4', type=int, help='number of workers')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2.')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--test_data_root', type=str, help='the test data root')
parser.add_argument('--test_file_list', type=str, help='the test csv file list')
parser.add_argument('--pred_filename', default='test_pred.csv', type=str, help='the name of pred csv file')
parser.add_argument('--save_logits', help='save predicted logits', action='store_true')
parser.add_argument('--logits_filename', default='test_logits.csv', type=str, help='the name of logits csv file')


def main():
    global args
    args = parser.parse_args()

    checkpoint_path = os.path.join('checkpoints', args.model + '_' + args.expname)

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

    # model = model.cuda()  # on single gpu
    model = torch.nn.DataParallel(model).cuda()     # on multiple gpus

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if isinstance(checkpoint, dict):
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                model = checkpoint
        else:
            raise NotImplementedError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loading code
    test_dataset = futurelab_loader.DATA(args.test_data_root,
                                          args.test_file_list,
                                          args.image_size,
                                          is_train=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    if not args.save_logits:
        preds, im_ids = test(test_loader, model)
        # write predictions to file
        pred_filename = os.path.join(checkpoint_path, args.pred_filename)
        with open(pred_filename, 'w') as opfile:
            opfile.write('FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2\n')
            for ii in range(len(im_ids)):
                opfile.write(str(im_ids[ii]) + ',' + ','.join(str(x) for x in preds[ii,:])+'\n')
        print("The pred csv file on test set has been saved to {}".format(pred_filename))
    else:
        logits, preds, im_ids = test_logits(test_loader, model)
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
        print("The logits csv file on test set has been saved to {}".format(logits_filename))
    return


def test(test_loader, model):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pred = []
    im_ids = []

    print('Test:\tTime')
    for i, (inputs, im_id, target) in enumerate(test_loader):

        inputs = inputs.cuda()
        target = target.cuda(async=True)    # fake targets, not used
        batch_size, n_crops, c, h, w = inputs.size()
        input_var = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)

        # compute output
        output = model(input_var)
        output = output.view(batch_size, n_crops, -1).mean(1)

        # store the top K classes for the prediction
        im_ids.append(im_id)
        _, pred_inds = output.data.topk(3,1,True,True)
        pred.append(pred_inds.cpu().numpy().astype(np.int))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                   i, len(test_loader), batch_time=batch_time))

    return np.vstack(pred), np.hstack(im_ids)


def test_logits(test_loader, model):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    logit = []
    pred = []
    im_ids = []

    print('Test:\tTime')
    for i, (inputs, im_id, target) in enumerate(test_loader):

        inputs = inputs.cuda()
        target = target.cuda(async=True)
        batch_size, n_crops, c, h, w = inputs.size()
        input_var = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)

        # compute output
        # output = torch.nn.Softmax(dim=-1)(model(input_var))
        output = model(input_var)
        output = output.view(batch_size, n_crops, -1).mean(1)

        # store the top K classes for the prediction
        im_ids.append(im_id)
        logit_inds, pred_inds = output.data.topk(args.num_classes,1,True,True)
        logit.append(logit_inds.cpu().numpy().astype(np.float))
        pred.append(pred_inds.cpu().numpy().astype(np.int))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                   i, len(test_loader), batch_time=batch_time))

    return np.vstack(logit), np.vstack(pred), np.hstack(im_ids)


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


if __name__ == '__main__':
    main()
