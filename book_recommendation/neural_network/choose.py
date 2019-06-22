# -*- coding: utf-8 -*-
"""

姓名：周恒成
文件描述：计算爬取图片特征值

"""
from basic_func import *
if use_gpu:
    torch.cuda.set_device(0)
    cudnn.benchmark = True

#图片预处理，改变尺寸为符合要求的大小
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#加载爬取的图片,路径需要根据实际修改
image_datasets = {'gallery': datasets.ImageFolder( './pytorch/gallery' ,data_transforms)}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,shuffle=False, num_workers=16) for x in ['gallery']}


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
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
if use_gpu:
    gallery_feature = gallery_feature.cuda()

result = {'gallery_f': gallery_feature.cpu().numpy()}
scipy.io.savemat('gallery.mat', result)#将库中所有图片的特征值保存下来

