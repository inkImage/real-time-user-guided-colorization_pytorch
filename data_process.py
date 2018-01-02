import torch
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from skimage.color import rgb2lab

from global_hint import *


def Color_Dataloader(dataset, batch_size):
    if dataset == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = dsets.CIFAR10(root='./data/',
                                      train=True,
                                      transform=transform,
                                      download=True)

        val_dataset = dsets.CIFAR10(root='./data/',
                                     train=False,
                                     transform=transform)
        # Data Loader-> it will hand in dataset by size batch
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        imsize = 32

    elif dataset == 'imagenet':

        traindir = './data/tiny-imagenet-200/train/'
        valdir = './data/tiny-imagenet-200/val/'
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = dsets.ImageFolder(traindir, transform)
        val_dataset = dsets.ImageFolder(valdir, transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=2)
        imsize = 64


    elif dataset == 'celeba':

        traindir = './data/CelebA/trainimages/images'
        valdir= './data/CelebA/valimages'
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = dsets.ImageFolder(traindir, transform=transform)
        val_dataset = dsets.ImageFolder(valdir, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=2)
        imsize = 128

    elif dataset == 'mscoco':

        traindir = './data/mscoco/trainimages_resized'
        valdir = './data/mscoco/valimages_resized'
        # Load mscoco data
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = dsets.ImageFolder(traindir, transform=transform)
        val_dataset = dsets.ImageFolder(valdir, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=2)
        imsize = 32

    return train_dataset, train_loader, val_loader, imsize


def process_data(image_data, batch_size, imsize, islocal):
    input = torch.zeros(batch_size, 1, imsize, imsize)
    labels = torch.zeros(batch_size, 2, imsize, imsize)
    images_np = image_data.numpy().transpose((0, 2, 3, 1))

    if islocal == False:
        ab_for_global = torch.zeros(batch_size, 2, imsize, imsize)

        for k in range(batch_size):
            img_lab = rgb2lab(images_np[k])

            img_l = img_lab[:, :, 0] / 100
            input[k] = torch.from_numpy(np.expand_dims(img_l, 0))

            img_ab_scale = (img_lab[:, :, 1:3] + 100) / 200
            labels[k] = torch.from_numpy(img_ab_scale.transpose((2, 0, 1)))

            img_ab_unscale = img_lab[:, :, 1:3]
            ab_for_global[k] = torch.from_numpy(img_ab_unscale.transpose((2, 0, 1)))

    if islocal == True:
        for k in range(batch_size):
            img_lab = rgb2lab(images_np[k])

            img_l = img_lab[:, :, 0] / 100
            input[k] = torch.from_numpy(np.expand_dims(img_l, 0))

            img_ab_scale = (img_lab[:, :, 1:3] + 100) / 200
            labels[k] = torch.from_numpy(img_ab_scale.transpose((2, 0, 1)))

            ab_for_global = 0 # just to make the room. don't need it in local net

    return input, labels, ab_for_global


def process_global(images, input_ab, batch_size, imsize, hist_mean, hist_std):
    glob_quant = Global_Quant(batch_size, imsize)
    X_hist = glob_quant.global_histogram(input_ab)  # batch x 313 x imsize x imsize
    X_sat = glob_quant.global_saturation(images).unsqueeze(1)  # batch x 1
    B_hist, B_sat = glob_quant.global_masks(batch_size)  # if masks are 0, put uniform random(0~1) value in it

    for l in range(batch_size):
        if B_sat[l].numpy() == 0:
            X_sat[l] = torch.normal(torch.FloatTensor([hist_mean]), std=torch.FloatTensor([hist_std]))
        if B_hist[l].numpy() == 0:
            tmp = torch.rand(313)
            X_hist[l] = torch.div(tmp, torch.sum(tmp))
    global_input = torch.cat([X_hist, B_hist, X_sat, B_sat], 1).unsqueeze(2).unsqueeze(2)
    # batch x (q+1) = batch x 316 x 1 x 1

    return global_input

def process_local(input_ab, batch_size, imsize):
    num_points = torch.zeros(batch_size).geometric_(0.125).long() # number of points to give as hints
    block_size = torch.zeros(batch_size, 1).uniform_(-0.5, 2.49).round().clamp(0, 2).long() # size of blocks to average
    local_ab = torch.zeros(batch_size, 2, imsize, imsize) # output local hint (ab channel)
    local_mask = torch.zeros(batch_size, 1, imsize, imsize).long() # output local hint (mask)

    for i in range(batch_size): # for all batches and
        for j in range(num_points[i]):
            gaussian_points = torch.zeros(2).normal_(mean=imsize/2, std=imsize/4).round().clamp(0, imsize-1).long()
            local_ab[i], local_mask[i] = \
                local_get_average_value(local_ab[i], input_ab[i], local_mask[i], gaussian_points, block_size[i], imsize)

    return local_ab, local_mask.float()

# get average value in local_ab for random sized box at certain points.
def local_get_average_value(local_ab, input_ab, local_mask, loc, p, imsize):  # width 0~4

    low_v = loc[0]-p[0] #lower bound 0
    if low_v<0:
        low_v=0
    high_v = loc[0]+p[0]+1 #higher bound imsize-1
    if high_v>=imsize:
        high_v=imsize
    low_h = loc[1]-p[0] #lower bound 0
    if low_h<0:
        low_h=0
    high_h = loc[1]+p[0]+1 #higher bound imsize-1
    if high_h>=imsize:
        high_h=imsize


    local_mask[:, low_v:high_v, low_h:high_h] = 1
    local_ab = torch.mul(local_mask.repeat(2, 1, 1).float(), input_ab)
    local_mean_a = torch.sum(local_ab[0,:,:]) / len(torch.nonzero(local_ab[0,:,:]))
    local_mean_b = torch.sum(local_ab[1,:,:]) / len(torch.nonzero(local_ab[1,:,:]))
    local_a = local_mask.float() * local_mean_a # 1 x 32 x 32
    local_b = local_mask.float() * local_mean_b
    local_ab = torch.cat([local_a, local_b], dim=0)
    return local_ab, local_mask



def process_global_sampling(batch_size, imsize, hist_mean, hist_std,
                            HIST=False, SAT=False, hist_ref_idx=1, sat_ref_idx=1):
    glob_quant = Global_Quant(batch_size, imsize)

    if HIST==True:
        input_ab_for_hist = hist_ref(batch_size, imsize, hist_ref_idx)
        X_hist = glob_quant.global_histogram(input_ab_for_hist)  # batch x 313 x imsize x imsize
        B_hist = torch.ones(batch_size, 1)

    else:
        tmp = torch.rand(batch_size, 313)
        X_hist = torch.div(tmp, torch.sum(tmp, dim=1).unsqueeze(1).repeat(1, 313))
        B_hist = torch.zeros(batch_size, 1)

    if SAT==True:
        image_for_sat = (batch_size, imsize, sat_ref_idx)
        X_sat = glob_quant.global_saturation(image_for_sat).unsqueeze(1)  # batch x 1
        B_sat = torch.ones(batch_size, 1)  # if masks are 0, put uniform random(0~1) value in it

    else:
        X_sat = torch.randn(batch_size, 1)
        for l in range(batch_size):
            X_sat[l] = torch.normal(torch.FloatTensor([hist_mean]), std=torch.FloatTensor([hist_std]))
        B_sat = torch.zeros(batch_size, 1)

    global_input = torch.cat([X_hist, B_hist, X_sat, B_sat], 1).unsqueeze(2).unsqueeze(2)
    # batch x (q+1) = batch x 316 x 1 x 1

    return global_input

def process_local_sampling(batch_size, imsize, p):

    ab_input = torch.FloatTensor([0,0]).unsqueeze(0)
    xy_input = torch.LongTensor([0,0]).unsqueeze(0)
    q=0
    while q is not -1:
        ab_list = []
        xy_list = []
        x = int(input("Enter a number for x: "))
        y = int(input("Enter a number for y: "))
        a = int(input("For <a channel> which color you want to apply?: (between -100 and 100)"))
        b = int(input("For <b channel> which color you want to apply?: (between -100 and 100)"))
        a = ((a+100)/200)
        b = ((b+100)/200)
        xy_list.append(x)
        xy_list.append(y)
        ab_list.append(a)
        ab_list.append(b)
        xy_list = torch.LongTensor([xy_list])
        ab_list = torch.FloatTensor([ab_list])
        xy_input = torch.cat([xy_input, xy_list], dim=0)  # n x 2 with 1 x 2 all zeros
        ab_input = torch.cat([ab_input, ab_list], dim=0) # n x 2 with 1 x 2 all zeros
        q = int(input("Enter -1 to finish: "))

    local_ab = torch.zeros(batch_size, 2, imsize, imsize) # output local hint (ab channel)
    local_mask = torch.zeros(batch_size, 1, imsize, imsize).long() # output local hint (mask)
    # print(torch.sum(local_ab))
    # print(torch.sum(local_mask))
    for i in range(batch_size): # for all batches and
        for j in range(ab_input.size(0)-1):
            # print(ab_input.size(0)-1)
            # print(ab_input[j+1])

            low_v = xy_input[j+1][0] - p  # lower bound 0
            if low_v < 0:
                low_v = 0
            high_v = xy_input[j+1][0] + p + 1  # higher bound imsize-1
            if high_v >= imsize:
                high_v = imsize
            low_h = xy_input[j+1][1] - p  # lower bound 0
            if low_h < 0:
                low_h = 0
            high_h = xy_input[j+1][1] + p + 1  # higher bound imsize-1
            if high_h >= imsize:
                high_h = imsize

            local_ab[i,0, low_v:high_v, low_h:high_h] = ab_input[j + 1][0]
            local_ab[i,1, low_v:high_v, low_h:high_h] = ab_input[j + 1][1]
            local_mask[i,:,low_v:high_v, low_h:high_h] = 1
            print(len(torch.nonzero(local_ab[i])), len(torch.nonzero(local_mask[i])))

    return local_ab, local_mask.float()

def hist_ref(batch, imsize, idx=1):
    valdir = './data/sample/hist'
    transform = transforms.Compose([
        transforms.Scale((imsize,imsize)),
        transforms.ToTensor(),

        ])

    val_dataset = dsets.ImageFolder(valdir, transform)


    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=2)

    for i, (image, _) in enumerate(val_loader):
        if i==(idx-1):
            ref_image = image
            print('%dth image chosen as reference for histogram'%(idx))
            break

    ref_image = ref_image.numpy().transpose((0, 2, 3, 1))
    img_lab = rgb2lab(ref_image)
    img_ab = img_lab[:, :, :, 1:3]

    pick_ref = torch.from_numpy(img_ab.transpose((0, 3, 1, 2))).repeat(batch,1,1,1).float()

    return pick_ref

def sat_ref(batch, imsize, idx=1):
    valdir = './data/sample/sat'
    transform = transforms.Compose([
        transforms.Scale((imsize,imsize)),
        transforms.ToTensor(),

        ])

    val_dataset = dsets.ImageFolder(valdir, transform)


    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=2)

    for i, (image, _) in enumerate(val_loader):
        if i==(idx-1):
            ref_image = image
            print('%dth image chosen as reference for saturation'%(idx))
            break

    print(ref_image.size())
    pick_ref = ref_image.repeat(batch, 1, 1, 1).float()

    return pick_ref