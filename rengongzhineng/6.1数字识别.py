# 使用pytorch完成数字的识别
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam
import os
Batch_SIZE = 2


# 1.准备数据集
# 如果传入未true就是训练集，如果不是就是测试集

def get_dataloader(train=True):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))  # mean和std的通道数相同
    ])
    dataset = MNIST(root='data', train=train, transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=Batch_SIZE, shuffle=True)
    return data_loader


# 2.构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        """
        :param input:[batch_size,1,28,28]
        :return:
        """
        # 修改形状
        x = input.view([input.size(0), 1 * 28 * 28])
        # 2.全连接操作
        x = self.fc1(x)
        # 3.进行激活函数处理,形状不会变化
        x = F.relu(x)
        # 4.输出层处理
        out = self.fc2(x)
        return F.log_softmax(out,dim=-1)

model = MnistModel()
optimizer = Adam(model.parameters(),lr=0.001)
if(os.path.exists('model/model1.pkl')):
    model.load_state_dict(torch.load('model/model1.pkl'))
    optimizer.load_state_dict(torch.load('model/opti1.pkl'))


def train(epoch):
    data_loder = get_dataloader()
    for idx,(input,target) in enumerate(data_loder):
        optimizer.zero_grad()
        output = model(input)#调用模型获得预测值
        loss = F.nll_loss(output,target)#得到损失
        loss.backward()#反向传播
        optimizer.step()#梯度更新
        if idx%100==0:
            torch.save(model.state_dict(),'model/model1.pkl')
            torch.save(optimizer.state_dict(),'model/opti1.pkl')
            print(loss.item())
if __name__=="__main__":
    for i in range(3):
        train(i)

