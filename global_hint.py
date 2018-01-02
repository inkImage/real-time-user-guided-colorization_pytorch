import torch
import numpy as np
import sklearn.neighbors as neigh
from skimage.color import rgb2hsv

from unet import *
import util


class Global_Quant():
    ''' Layer which encodes ab map into Q colors
    '''
    def __init__(self, batch, imsize):
        self.quantization = Quantization(batch, imsize, km_filepath='./data/pts_in_hull.npy')

    def global_histogram(self, input):
        out = self.quantization.encode_nn(input) # batch x 313 x imsize x imsize
        out = out.type(torch.FloatTensor) # change it to tensor
        X_onehotsum = torch.sum(torch.sum(out, dim=3), dim=2) # sum it up to batch x 313
        X_hist = torch.div(X_onehotsum, util.expand(torch.sum(X_onehotsum, dim=1).unsqueeze(1), X_onehotsum)) # make 313 probability
        return X_hist

    def global_saturation(self, images): # input: tensor images batch x 3 x imsize x imsize (rgb)
        images_np = images.numpy().transpose((0, 2, 3, 1)) # numpy: batch x imsize x imsize x 3
        images_h = torch.zeros(images.size(0), 1, images.size(2),images.size(2))
        for k in range(images.size(0)):
            img_hsv = rgb2hsv(images_np[k])
            img_h = img_hsv[:, :, 1]
            images_h[k] = torch.from_numpy(img_h).unsqueeze(0) #  batch x 1 x imsize x imsize
        avgs = torch.mean(images_h.view(images.size(0), -1),dim=1) # batch x 1
        return avgs

    def global_masks(self, batch_size):  # both for histogram and saturation
        B_hist = torch.round(torch.rand(batch_size, 1))
        B_sat = torch.round(torch.rand(batch_size, 1))
        return B_hist, B_sat

class Quantization():
    # Encode points as a linear combination of unordered points
	# using NN search and RBF kernel
    def __init__(self,batch, imsize, km_filepath='./data/pts_in_hull.npy' ):

        self.cc = torch.from_numpy(np.load(km_filepath)).type(torch.FloatTensor) # 313 x 2
        self.K = self.cc.shape[0]
        self.batch = batch
        self.imsize = imsize

    def encode_nn(self,images): # batch x imsize x imsize x 2

        images = images.permute(0,2,3,1) # batch x 2 x imsize x imsize -> batch x imsize x imsize x 2
        images_flt = images.contiguous().view(-1, 2)
        P = images_flt.shape[0]
        inds = self.nearest_inds(images_flt, self.cc).unsqueeze(1) # P x 1
        images_encoded = torch.zeros(P,self.K)
        images_encoded.scatter_(1, inds, 1)
        images_encoded = images_encoded.view(self.batch, self.imsize, self.imsize, 313)
        images_encoded = images_encoded.permute(0,3,1,2)
        return images_encoded

    def nearest_inds(self, x, y):  # x= n x 2, y= 313 x 2  n x 2, 2 x 313 = n x 313
        inner = torch.matmul(x, y.t())
        normX = torch.sum(torch.mul(x, x), 1).unsqueeze(1).expand_as(inner)
        normY = torch.sum(torch.mul(y, y), 1).unsqueeze(1).t().expand_as(inner)  # n x 313
        P = normX - 2 * inner + normY
        nearest_idx = torch.min(P, dim=1)[1]
        return nearest_idx



	# def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
	# 	pts_enc_flt = util.flatten_nd_array(pts_enc_nd,axis=axis)
	# 	pts_dec_flt = np.dot(pts_enc_flt,self.cc)
	# 	pts_dec_nd = util.unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
	# 	return pts_dec_nd
    #
	# def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
	# 	pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
	# 	pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
	# 	if(returnEncode):
	# 		return (pts_dec_nd,pts_1hot_nd)
	# 	else:
	# 		return pts_dec_nd