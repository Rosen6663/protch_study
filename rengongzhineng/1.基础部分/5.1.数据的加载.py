import torch
from torch.utils.data import Dataset,DataLoader

data_path = '../data/SMSSpamCollection'


# 完成数据集类
class MyDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path,encoding='utf-8').readlines()

    # 获取索引对应位置的数据
    def __getitem__(self, index):
        cur_line = self.lines[index].strip()
        lable = cur_line[:4].strip()
        content = cur_line[4:].strip()
        return lable,content

    # 返回数据总量
    def __len__(self):
        return len(self.lines)


my_dataset = MyDataset()
data_loader=DataLoader(dataset=my_dataset,batch_size=2,shuffle=True)
if __name__ == '__main__':
    # my_dataset = MyDataset()

    # print(my_dataset[1000])
    # print(len(my_dataset))
    for index,(lable,content) in enumerate(data_loader):
        print(index,lable,content)

