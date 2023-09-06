import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision

from matplotlib import pyplot as plt
from gradient import  plot_image,plot_curve ,one_hot

batch_size =512
# step1 加载工具包
train_loader = torch.utils.data.DataLoader( # 将已经下好的数据集拖到 最外层文件夹
    torchvision.datasets.MNIST('mnist_data',train =True,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081))

                                    # 正则化

                               ])),
    batch_size=batch_size,shuffle= True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',train= False,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081))



                               ])),
    batch_size=batch_size,shuffle =False)

x,y =next(iter(train_loader))
# plot_image(x,y,'image sample')

# step2 创建网络 三层

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1=nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10) # 最后10 由数据集决定

    def forward(self,x):
        # x :[b,1,28,28]
        # h1 = wx+b
        x= F.relu(self.fc1(x))
        # h2 =relu(h1w+b)
        x =F.relu(self.fc2(x))

        x =self.fc3(x)

        return x

# step 训练
train_loss=[]
net =Net()
# net.parameters = [w1,b1,w2,b2,w3,b3]
optimizer = optim.SGD(net.parameters(),lr =0.01,momentum =0.9)
for epoch in range(3):
    for batch_idx, (x,y) in enumerate(train_loader):   #对batch迭代一次

        # x [b,1,28,28]
        # [b,featrue] 只能接受2维
        x =x.view(x.size(0),28*28)
        out =net(x)
        # =>[b,10] b=batch
        # y [b,10]
        y_onehot =one_hot(y)
        # loss =mse(out,y_onshot) 均方差

        loss = F.mse_loss(out,y_onehot)

        optimizer.zero_grad() # 清空梯度
        loss.backward()  # 计算梯度

        # w' =w-lr*grdient
        optimizer.step()
        train_loss.append(loss.item())


plot_curve(train_loss)
# get [w1,b1,w2,b2,w3,b3]

# step 准确度测试

total_correct=0
for x,y in test_loader:
    x = x.view(x.size(0),28*28)
    out =net(x)
    # out [b,10]
    pred =out.argmax(dim=1)  #范围的是索引

    correct =pred.eq(y).sum().float().item()
    total_correct +=correct;
total_num = len(test_loader.dataset)

acc =total_correct/total_num

print('test.acc',acc)

x,y =next(iter(test_loader))

out = net(x.view(x.size(0),28*28))
pred =out.argmax(dim=1)
plot_image(x,pred,'test')





