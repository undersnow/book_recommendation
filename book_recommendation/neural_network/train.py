"""

姓名：周恒成
文件描述：使用训练集对神经网络进行训练

"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time
import os
from shutil import copyfile
from model import ft_net
version =  torch.__version__


download_path = '/home/user/reID/Market'#训练集初始路径，根据实际修改
data_dir = '/home/user/reID/Market/pytorch'#训练集路径，根据实际修改

#修改训练集为我们所需的格式
if not os.path.isdir(download_path):
    print('wrong')
save_path = download_path + '/pytorch'#pytorch里面为我们所需格式的训练集，保存路径根据实际修改
if not os.path.isdir(save_path):
    os.mkdir(save_path)
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'#查询的图片
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
query_path = download_path + '/gt_bbox'
if os.path.isdir(query_path):
    query_save_path = download_path + '/pytorch/multi-query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
#gallery中的图片
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)
for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)
for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

use_gpu=0#判断是否使用gpu
batchsize=16
stride=2
lr=0.05
droprate=0.5
# 对gpu的支持
if torch.cuda.is_available():
    use_gpu=1
    torch.cuda.set_device(0)
    cudnn.benchmark = True
transform_train_list = [
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all'),data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),data_transforms['val'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,shuffle=True, num_workers=8, pin_memory=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)#测试用

# 训练

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('---------------')

        #训练加验证  train+val
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0.0
            for data in dataloaders[phase]:
                # 获得输入
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<batchsize:
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())

                optimizer.zero_grad()
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                if int(version[0])>0 or int(version[2]) > 3: #对不同版本的支持
                    running_loss += loss.item() * now_batch_size
                else :
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
            epoch_loss = running_loss / dataset_sizes[phase]#损失函数
            epoch_acc = running_corrects / dataset_sizes[phase]
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            if phase == 'val':
                last_model_wts = model.state_dict()
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(last_model_wts)
    save_network(model)
    return model


# 保存pth文件
def save_network(network):
    save_path = './net.pth' #生成pth文件
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(0)


#加载网络，丢弃全连接层
model = ft_net(len(class_names), droprate, stride)
ignored_params = list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
         {'params': base_params, 'lr': 0.1*lr},
         {'params': model.classifier.parameters(), 'lr': lr}
     ], weight_decay=5e-4, momentum=0.9, nesterov=True)

#学习率参数
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
if use_gpu:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=60)#训练的epoch数量设置

