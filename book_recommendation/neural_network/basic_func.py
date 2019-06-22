# -*- coding: utf-8 -*-
"""

姓名：周恒成
文件描述：定义基本函数

"""
from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets,transforms
import os
import scipy.io
from model import ft_net
import numpy as np

batchsize = 32
use_gpu = torch.cuda.is_available()

def load_network(network):#加载神经网络
    network.load_state_dict(torch.load('./net.pth'))
    return network


def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
# 提取图片特征
def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)#可视化过程，测试用
        ff = torch.FloatTensor(n,512).zero_()
        for i in range(2):
            if i==1:
                img = fliplr(img)
            input_img = Variable(img)
            if use_gpu:
                input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu().float()
            ff = ff+f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features

# 按相似度排序
def sort_img(qf, gf):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()#与库中所有图片进行对比
    index = np.argsort(score)
    index = index[::-1]
    return index

