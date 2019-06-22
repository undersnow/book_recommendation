# -*- coding: utf-8 -*-
"""

姓名：周恒成
文件描述：输入查询图片，返回图片名称列表

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
from basic_func import *

batchsize = 32
use_gpu = torch.cuda.is_available()

def getSimliarPhotos(path1,path2,num):
    list=[]
    #data_dir = path1
    if use_gpu:
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    #图片预处理
    data_transforms = transforms.Compose([
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #加载图片
    image_datasets = {'gallery': datasets.ImageFolder( path1 ,data_transforms),'query':datasets.ImageFolder( path2 ,data_transforms)}
    print(image_datasets['query'].classes)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,shuffle=False, num_workers=16) for x in ['query']}


    # 加载模型
    model_structure = ft_net(751, stride = 2)
    model = load_network(model_structure)

    #去掉全连接层
    model.classifier.classifier = nn.Sequential()

    # test
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # 提取特征
    result1 = scipy.io.loadmat('gallery.mat')
    gallery_feature = torch.FloatTensor(result1['gallery_f'])
    with torch.no_grad():
        query_feature = extract_feature(model, dataloaders['query'])
    if use_gpu:
        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

    index = sort_img(query_feature[0],gallery_feature)
    query_path, _ = image_datasets['query'].imgs[0]
    result = {'gallery_f': gallery_feature.cpu().numpy()}
    for i in range(num):
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        list.append(img_path)
    return list

#测试用例
if __name__ == '__main__':
    listg=getSimliarPhotos('./pytorch/gallery','./pytorch/query',10)
    print(listg)
