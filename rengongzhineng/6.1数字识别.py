# 使用pytorch完成数字的识别
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam
import os

Batch_SIZE = 128
TEST_BATCH_SIZE = 1000


# 1.准备数据集
# 如果传入未true就是训练集，如果不是就是测试集
# 数据处理部分
def get_dataloader(train=True,batch_size=Batch_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))  # mean和std的通道数相同
    ])
    dataset = MNIST(root='data', train=train, transform=transform_fn)
    # shuffer代表True代表打乱顺序
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
        return F.log_softmax(out, dim=-1)


model = MnistModel()
optimizer = Adam(model.parameters(), lr=0.001)
if os.path.exists('model/model1.pkl'):
    model.load_state_dict(torch.load('model/model1.pkl'))
    optimizer.load_state_dict(torch.load('model/opti1.pkl'))


def train(epoch):
    data_loder = get_dataloader()
    for idx, (input, target) in enumerate(data_loder):
        optimizer.zero_grad()
        output = model(input)  # 调用模型获得预测值
        loss = F.nll_loss(output, target)  # 得到损失
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
        if idx % 100 == 0:
            torch.save(model.state_dict(), 'model/model1.pkl')
            torch.save(optimizer.state_dict(), 'model/opti1.pkl')


def atest():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False,batch_size=TEST_BATCH_SIZE)
    for idx, (input, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss)
            pred = output.max(dim=-1)[-1]
            # eq是进行比较，浮点型，计算均值
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失",np.mean(acc_list), np.mean(loss_list))


if __name__ == "__main__":
    atest()
