import os
import random
import pandas as pd
from torch._inductor import scheduler
from torch.optim.lr_scheduler import *  # PyTorch学习率调度器
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Lenet5 import Lenet5
from MyNet import MYCNN;
from dataset import CatDogDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt  # 可视化库
########################################
batch_size =32
EPOCHS = 10  # 迭代次数
cPath = os.getcwd()
cat_train_dir = cPath+'/dataset/A/cat'
dog_train_dir = cPath+'/dataset/A/dog'
cat_train_file = os.listdir(cat_train_dir)
dog_train_file = os.listdir(dog_train_dir)
test_dir = cPath+'/dataset/B'
test_file = os.listdir(test_dir)

# 训练集转换
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

cat_files = [tf for tf in cat_train_file if 'cat' in tf]
dog_files = [tf for tf in dog_train_file if 'dog' in tf]

cats = CatDogDataset(cat_files,cat_train_dir,transform=train_transform)
dogs = CatDogDataset(dog_files,dog_train_dir,transform=train_transform)

train_set = ConcatDataset([cats,dogs])
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)

# 测试集转换

test_set = CatDogDataset(test_file, test_dir, mode='test')
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=0)

def get_val_dataset():
    cat_files = [tf for tf in cat_train_file if 'cat' in tf]
    dog_files = [tf for tf in dog_train_file if 'dog' in tf]
    val_cat_files=[]
    val_dog_file=[]
    for _ in range(1500):
        r = random.randint(0,len(cat_files)-1)
        val_dog_file.append(dog_files[r])
        val_cat_files.append(cat_files[r])
        cat_files.remove(cat_files[r])
        dog_files.remove(dog_files[r])

    train_loader = _extracted_from_get_val_dataset_15(
        cat_files, train_transform, dog_files
    )
    val_loader = _extracted_from_get_val_dataset_15(
        val_cat_files,train_transform,val_dog_file
    )
    return train_loader, val_loader

def _extracted_from_get_val_dataset_15(cat_files, transform, dog_files):
    cats = CatDogDataset(cat_files, cat_train_dir, transform=train_transform)
    dogs = CatDogDataset(dog_files, dog_train_dir, transform=train_transform)

    train_set = ConcatDataset([cats, dogs])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader

# train
def train(model,train_loader, optimizer, epoch,criterion):
    model.train() # 设置为训练模式
    train_loss = 0.
    train_acc =0.
    percent =100

    for batch_idx,(x,label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output,label)

        loss.backward()
        optimizer.step()
        loss =loss.item()
        train_loss +=loss
        pred =torch.argmax(output,dim=1)
        train_acc += torch.eq(pred,label).float().sum().item()

        if (batch_idx+1)%percent ==0:
            processed_samples =(batch_idx+1)*len(x)
            total_samples=len(train_loader.dataset)

            progress = 100.0*batch_idx/len(train_loader)
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\t'.format(epoch, processed_samples, total_samples,
                                                                             progress, loss))
    train_loss *=batch_size
    train_loss/=len(train_loader.dataset)
    print('\ntrain epoch: {}\tloss: {:.6f}\taccuracy:{:.4f}% '.format(epoch, train_loss, 100. * train_acc))

    # 新增根据训练准确率调整训练策略
    scheduler.step()

    return train_loss,train_acc
#validate
def val(model,val_loader,optimizer,epoch,criterion):
    model.eval()  #一定要调模式
    val_loss =0.0
    correct =0

    for (x,label) in val_loader:
        # 验证集要禁止自动求导
        with torch.no_grad():
            out = model(x)
            val_loss +=criterion(out,label).item()

            pred = out.max(1,keepdim=True)[1]
            correct +=pred.eq(label.view_as(pred)).sum().item()


    val_loss *=batch_size
    val_loss /=len(val_loader.dataset)
    val_acc = correct/len(val_loader.dataset)
    print("\nval set: epoch{} average loss: {:.4f}, accuracy: {}/{} ({:.4f}%) \n"
          .format(epoch, val_loss, correct, len(val_loader.dataset), 100. * val_acc))
    return val_loss,100.*val_acc

# test
def test(model, test_loader,epoch):
    model.eval()  # test也是eval模式
    filename_list=[]
    pred_list=[]

    for x,filename in test_loader:


        with torch.no_grad():
            output =model(x)
            pred =torch.argmax(output,dim=1)
            # 将文件名和预测结果添加到列表中
            filename_list += [n[:-4] for n in filename]
            pred_list += [p.item() for p in pred]

    print(f"\ntest epoch: {epoch}\n")

    # 创建提交文件
    submission = pd.DataFrame({"id": filename_list, "label": pred_list})
    submission.to_csv(f'preds_{str(epoch)}.csv', index=False)

if __name__ == '__main__':

    # 模型
    model =MYCNN()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=5e-4)

    scheduler = StepLR(optimizer,step_size =5)
    # 统计答案
    criterion = nn.CrossEntropyLoss()
    train_counter = []  # 训练集数量
    train_losses = []  # 训练集损失
    train_acces = []  # 训练集准确率
    val_counter = []  # 验证集数量
    val_losses = []  # 验证集损失
    val_acces = []  # 验证集准确率

    for epoch in range(EPOCHS):
        # 刷新读取数据集
        train_loader, val_loader = get_val_dataset()
        tr_loss ,tr_acc =train(model,train_loader,optimizer,epoch+1,criterion)
        train_counter.append((epoch) * len(train_loader.dataset))
        train_losses.append(tr_loss)
        train_acces.append(tr_acc)

        val_loss,val_acc = val(model,val_loader,optimizer,epoch+1,criterion)
        val_counter.append((epoch - 1) * len(train_loader.dataset))
        val_losses.append(val_loss)
        val_acces.append(val_acc)
    test(model, test_loader, 1)
    fig = plt.figure()  # 创建图像
    plt.plot(train_counter, train_losses, color='blue')  # 画出训练损失曲线
    plt.scatter(val_counter, val_losses, color='red')  # 画出测试损失散点图
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # 图例标识
    plt.xlabel('number of training examples seen')  # x轴标签
    plt.ylabel('negative log likelihood loss')  # y轴标签






