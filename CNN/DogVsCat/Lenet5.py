import torch
from torch import nn
from torch.nn import functional as F
class Lenet5(nn.Module):
    """
    for cifar test
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x[b,3,32,32]=》[b,6,]
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0), # channel-in 是相片的通道RGB3
            #pooling 求滑窗的max
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),

            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
        )
        # flatten 打平
        #fc unit
        self.fc_unit=nn.Sequential(
            nn.Linear(16*5*5,120), # 一维全连接
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,2)
        )
       # self.criteon  = nn.CrossEntropyLoss()
    def forward(self,x): #前向路径 只用写前项:
        batch_sz =x.size(0)
        x  =self.conv_unit(x)

        # flatten
        x  = x.view(batch_sz,16*5*5)

        logtis = self.fc_unit(x) #[b,2]
        # loss = self.criteon(logtis,y)
        return logtis

#
# def main():
#     net =Lenet5()
#     tmp =torch.randn(2,3,32,32)
#     out =net(tmp)
#     print(out.shape)
#
# if __name__ == '__main__':
#     main()