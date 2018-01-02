import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable

from unet import *
from util import *
from global_hint import *
from data_process import *


# Hyper Parameters


# arguments parsed when initiating
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar', choices=['cifar', 'imagenet', 'celeba', 'mscoco'])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model', type=str, default='unet100.pkl')
    parser.add_argument('--image_save', type=str, default='./images')
    parser.add_argument('--learning_rate', type=int, default=0.0002)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--resume', type=bool, default=False,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--islocal', type=bool, default=False)

    return parser.parse_args()


def main(args):
    dataset = args.data
    gpu = args.gpu
    batch_size = args.batch_size
    model_path = args.model_path
    log_path = args.log_path
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    start_epoch = args.start_epoch
    islocal = args.islocal

    # make directory for models saved when there is not.
    make_folder(model_path, dataset) # for sampling model
    make_folder(log_path, dataset) # for logpoint model
    make_folder(log_path, dataset +'/ckpt') # for checkpoint model

    # see if gpu is on
    print("Running on gpu : ", gpu)
    cuda.set_device(gpu)

    # set the data-loaders
    train_dataset, train_loader, val_loader, imsize = Color_Dataloader(dataset, batch_size)

    # declare unet class
    unet = UNet(imsize, islocal)

    # make the class run on gpu
    unet.cuda()

    # Loss and Optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    criterion = torch.nn.SmoothL1Loss()

    # optionally resume from a checkpoint
    if args.resume:
        ckpt_path = os.path.join(log_path, dataset, 'ckpt/local/model.ckpt')
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint")
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            unet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            print("=> Meaning that start training from (epoch {})".format(checkpoint['epoch']+1))
        else:
            print("=> Sorry, no checkpoint found at '{}'".format(args.resume))

    # record time
    tell_time = Timer()
    iter = 0
    # Train the Model
    for epoch in range(start_epoch, num_epochs):

        unet.train()
        for i, (images, _) in enumerate(train_loader):

            batch = images.size(0)
            '''
            additional variables for later use.
            change the picture type from rgb to CIE Lab.
            def process_data, def process_global in util file
            '''
            if islocal:
                input, labels, _ = process_data(images, batch, imsize, islocal)
                local_ab, local_mask = process_local(labels, batch, imsize)
                side_input = torch.cat([local_ab, local_mask], 1) # concat([batch x 2 x imsize x imsize , batch x 1 x imsize x imsize], 1) = batch x 3 x imsize x imsize
                random_expose = random.randrange(1, 101)
                if random_expose == 100:
                    print("Jackpot! expose the whole!")
                    local_mask = torch.ones(batch_size, 1, imsize, imsize)
                    side_input = torch.cat([labels, local_mask], 1)
            else: # if is local
                input, labels, ab_for_global = process_data(images, batch, imsize, islocal)
                side_input = process_global(images, ab_for_global, batch, imsize, hist_mean=0.03, hist_std=0.13)


            # make them all variable + gpu avialable

            input = Variable(input).cuda()
            labels = Variable(labels).cuda()
            side_input = Variable(side_input).cuda()

            # initialize gradients
            optimizer.zero_grad()
            outputs = unet(input, side_input)

            # make outputs and labels as a matrix for loss calculation
            outputs = outputs.view(batch, -1)  # 100 x 32*32*3(2048)
            labels = labels.contiguous().view(batch, -1)  # 100 x 32*32*3

            loss_train = criterion(outputs, labels)
            loss_train.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.10f, iter_time: %2.2f, aggregate_time: %6.2f'
                      % (epoch + 1, num_epochs, i + 1, (len(train_dataset) // batch_size), loss_train.data[0],
                         (tell_time.toc() - iter), tell_time.toc()))
                iter = tell_time.toc()

        torch.save(unet.state_dict(), os.path.join(model_path, dataset, 'unet%d.pkl' % (epoch + 1)))

        # start evaluation
        print("-------------evaluation start------------")

        unet.eval()
        loss_val_all = Variable(torch.zeros(100), volatile=True).cuda()
        for i, (images, _) in enumerate(val_loader):

            # change the picture type from rgb to CIE Lab
            batch = images.size(0)

            if islocal:
                input, labels, _ = process_data(images, batch, imsize, islocal)
                local_ab, local_mask = process_local(labels, batch, imsize)
                side_input = torch.cat([local_ab, local_mask], 1)
                random_expose = random.randrange(1, 101)
                if random_expose == 100:
                    print("Jackpot! expose the whole!")
                    local_mask = torch.ones(batch_size, 1, imsize, imsize)
                    side_input = torch.cat([labels, local_mask], 1)
            else: # if is local
                input, labels, ab_for_global = process_data(images, batch, imsize, islocal)
                side_input = process_global(images, ab_for_global, batch, imsize, hist_mean=0.03, hist_std=0.13)

                # make them all variable + gpu avialable

            input = Variable(input).cuda()
            labels = Variable(labels).cuda()
            side_input = Variable(side_input).cuda()

            # initialize gradients
            optimizer.zero_grad()
            outputs = unet(input, side_input)

            # make outputs and labels as a matrix for loss calculation
            outputs = outputs.view(batch, -1)  # 100 x 32*32*3(2048)
            labels = labels.contiguous().view(batch, -1)  # 100 x 32*32*3

            loss_val = criterion(outputs, labels)

            logpoint = {
                'epoch': epoch + 1,
                'args': args,
            }
            checkpoint = {
                'epoch': epoch + 1,
                'args': args,
                'state_dict': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            loss_val_all[i] = loss_val

            if i == 30:
                print('Epoch [%d/%d], Validation Loss: %.10f'
                      % (epoch + 1, num_epochs, torch.mean(loss_val_all).data[0]))
                torch.save(logpoint, os.path.join(log_path, dataset, 'Model_e%d_train_%.4f_val_%.4f.pt' %
                                                  (epoch + 1, torch.mean(loss_train).data[0],
                                                   torch.mean(loss_val_all).data[0])))
                torch.save(checkpoint, os.path.join(log_path, dataset, 'ckpt/model.ckpt'))
                break


if __name__ == '__main__':
    args = parse_args()
    main(args)