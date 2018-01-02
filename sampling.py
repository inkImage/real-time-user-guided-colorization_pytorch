import os
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from skimage.color import rgb2lab, lab2rgb, rgb2gray

from unet import *
from util import *
from global_hint import *
from data_process import *



# Hyper Parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar', choices=['cifar', 'imagenet', 'celeba', 'mscoco'])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--model', type=str, default='unet100.pkl')
    parser.add_argument('--image_save', type=str, default='./images')
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--global_hist', type=bool, default=False)
    parser.add_argument('--global_sat', type=bool, default=False)
    parser.add_argument('--hist_ref_idx', type=int, default=1)
    parser.add_argument('--sat_ref_idx', type=int, default=1)
    parser.add_argument('--islocal', type=bool, default=False)
    parser.add_argument('--nohint', type=bool, default=False)


    return parser.parse_args()



def main(args):
    dataset     = args.data
    gpu         = args.gpu
    batch_size  = args.batch_size
    model_path  = args.model_path
    image_save  = args.image_save
    model       = args.model
    idx         = args.idx
    global_hist = args.global_hist
    global_sat  = args.global_sat
    hist_ref_idx = args.hist_ref_idx
    sat_ref_idx  = args.hist_ref_idx
    islocal     = args.islocal
    nohint      = args.nohint

    make_folder(image_save, dataset)

    print("Running on gpu : ", gpu)
    cuda.set_device(gpu)

    _, _, test_loader, imsize = Color_Dataloader(dataset, batch_size)

    unet = UNet(imsize, islocal)

    unet.cuda()

    unet.eval()
    unet.load_state_dict(torch.load(os.path.join(model_path, dataset, model)))


    for i, (images, _) in enumerate(test_loader):

        batch = images.size(0)
        '''
        additional variables for later use.
        change the picture type from rgb to CIE Lab.
        def process_data, def process_global in util file
        '''
        if islocal:
            input, labels, _ = process_data(images, batch, imsize, islocal)
            local_ab, local_mask = process_local_sampling(batch_size, imsize, p=1)
            if nohint:
                local_ab = torch.zeros(batch_size, 2, imsize, imsize)
                local_mask = torch.zeros(batch_size, 1, imsize, imsize)

            side_input = torch.cat([local_ab, local_mask], 1)


        else:
            input, labels, ab_for_global = process_data(images, batch, imsize, islocal)

            print('global hint for histogram : ', global_hist)
            print('global hint for saturation : ', global_sat)

            side_input = process_global_sampling(batch, imsize, 0.03, 0.13,
                                                 global_hist, global_sat, hist_ref_idx, sat_ref_idx)

        # make them all variable + gpu avialable

        input = Variable(input).cuda()
        labels = Variable(labels).cuda()
        side_input = Variable(side_input).cuda()

        outputs = unet(input, side_input)

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(outputs, labels)
        print('loss for test data: %2.4f'%(loss.cpu().data[0]))


        colored_images = torch.cat([input,outputs],1).data # 100 x 3 x 32 x 32
        gray_images = torch.zeros(batch_size, 3, imsize, imsize)
        img_gray =np.zeros((imsize, imsize,3))

        colored_images_np = colored_images.cpu().numpy().transpose((0,2,3,1))

        j = 0
        # make sample images back to rgb
        for img in colored_images_np:

            img[:,:,0] = img[:,:,0]*100
            img[:, :, 1:3] = img[:, :, 1:3] * 200 - 100
            img = img.astype(np.float64)
            img_RGB = lab2rgb(img)
            img_gray[:,:,0] = img[:,:,0]
            img_gray_RGB = lab2rgb(img_gray)

            colored_images[j] = torch.from_numpy(img_RGB.transpose((2,0,1)))
            gray_images[j] = torch.from_numpy(img_gray_RGB.transpose((2,0,1)))
            j+=1

        #
        torchvision.utils.save_image(images,
                                 os.path.join(image_save, dataset, '{}_real_samples.png'.format(idx)))
        torchvision.utils.save_image(colored_images,
                                     os.path.join(image_save, dataset, '{}_colored_samples.png'.format(idx)))
        torchvision.utils.save_image(gray_images,
                                     os.path.join(image_save, dataset, '{}_input_samples.png'.format(idx)))


        print('-----------images sampled!------------')
        break


if __name__ == '__main__':
    args = parse_args()
    main(args)