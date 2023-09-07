import torch
from torch import nn
from torch.nn import functional as F

class MYCNN(nn.Module):
    def __init__(self, num_classes=2):  # 初始化
        super().__init__()  # 调用父类的初始化函数
        # 定义卷积层和池化层的序列
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # (224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2+1=27
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # (27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2+1=13
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (13-3)/2+1=6
        )  # 6*6*128=9126
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 定义平均池化层
        # 定义全连接层序列
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 丢弃部分数据
            nn.Linear(128 * 6 * 6, 2048),  # 全连接层，输出维度为2048
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout(),  # 丢弃部分数据
            nn.Linear(2048, 512),  # 全连接层，输出维度为512
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(512, num_classes),  # 全连接层，输出维度为num_classes
        )
        # softmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):  # 定义前向传播函数
        x = self.features(x)  # 输入经过卷积和池化层序列
        x = self.avgpool(x)  # 输入经过平均池化层
        x = x.view(x.size(0), -1)  # 将多维张量展平成一维
        x = self.classifier(x)  # 输入经过全连接层序列
        x = self.logsoftmax(x)  # 输入经过LogSoftmax层
        return x  # 返回计算结果


    # def forward(self ,x):

