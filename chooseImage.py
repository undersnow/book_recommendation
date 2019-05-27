# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets,transforms
import os
from model import ft_net
import numpy as np

batchsize = 32
use_gpu = torch.cuda.is_available()
# 加载训练好的模型
def load_network(network):
    network.load_state_dict(torch.load('./net.pth'))
    return network

# 提取图片特征
def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)#可视化过程，可省略
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
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features

# 按相似度排序
def sort_img(qf, gf):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    return index



def getSimliarPhotos(path,num):
    list=[]
    data_dir = path
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
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,shuffle=False, num_workers=16) for x in ['gallery','query']}
    class_names = image_datasets['query'].classes
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs


    # 加载模型
    model_structure = ft_net(751, stride = 2)
    model = load_network(model_structure)

    #去掉全连接层
    model.classifier.classifier = nn.Sequential()

    # test
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    #print(use_gpu)
    # 提取特征
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'])
        query_feature = extract_feature(model, dataloaders['query'])
    if use_gpu:
        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()


    index = sort_img(query_feature[0],gallery_feature)
    query_path, _ = image_datasets['query'].imgs[0]
    for i in range(num):
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        list.append(img_path)
    return list

if __name__ == '__main__':
    list=getSimliarPhotos('./pytorch',10)
    print(list)