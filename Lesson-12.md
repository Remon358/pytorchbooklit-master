Lesson-12

逻辑斯特回归问题：用于处理二分类问题-->二分类表示输出或者是0， 或者是1， 输出只有两种选择。

损失函数采用交叉熵

```python
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
x_data = torch.tensor([[1],[2],[3]],dtype=torch.float32)
y_data = torch.tensor([[0],[0],[1]],dtype=torch.float32)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return torch.sigmoid(y_pred)

model = Model()
criterion = torch.nn.BCELoss(size_average=False)
optim = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())
    # time.sleep(0.1)

    optim.zero_grad()
    loss.backward()
    optim.step()

x = np.linspace(0,10,200)
x_t = torch.tensor(x,dtype=torch.float32).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.show()
```

网络的层数越多，学习能力越强。那么，我们不就可以一直增加网络的层数了吗？答案是否定的，一方面，由于网络层数越多，那么学习能力越强，但是也会把系统中的噪声也学习出来。在真是的环境中，系统的噪声在不同的环境下可能会有差异的。这就要求网络要有一定的泛化能力。

分类预测问题： 保险行业预测问题

数据类型一般选择为float32, 这是因为1， 在中低端显卡中，多支持的是32位的浮点运算。在高端显卡中才支持64为的浮点运算。 2 在神经网络的计算过程中，32位浮点数足够使用

下面的例子显示一个糖尿病例子的预测问题：

```python
import numpy as np
from matplotlib import pyplot as plt
import torch
data_xy = np.loadtxt('/home/chasing/Documents/pytorchbooklit/diabetes.csv.gz', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(data_xy[:,:-1])
y_data = torch.from_numpy(data_xy[:,-1]).reshape(-1,1)

class LinearExample(torch.nn.Module):
    def __init__(self):
        super(LinearExample, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        # self.linear4 = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        # x = self.linear4(x)
        return self.relu(x)
        # return self.sigmoid(x)


model = LinearExample()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2)

loss_list = list()

if __name__ == '__main__':
    for epoch in range(300):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    plt.plot(loss_list)
    plt.show()
```

采用一个batch会提升并行计算能力，提升计算的速度。采用一组数据会避免陷入鞍点（局部最优）, 本历程中，我们采用dataset构造数据集。

```python
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
from matplotlib import pyplot as plt

class LinearExample(torch.nn.Module):
    def __init__(self):
        super(LinearExample, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        # self.linear4 = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        # x = self.linear4(x)
        return self.relu(x)
        # return self.sigmoid(x)

class DiabetesDatset(Dataset):
    def __init__(self):
        data_xy = np.loadtxt('/home/chasing/Documents/pytorchbooklit/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = data_xy.shape[0]
        self.data_x = torch.from_numpy(data_xy[:,:-1])
        self.data_y = torch.from_numpy(data_xy[:,-1]).reshape(-1,1)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len

model = LinearExample()
dataset = DiabetesDatset()
train_loader =  DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

loss_list = list()

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss_list.append(loss)

    plt.plot(loss_list)
    plt.show()
```
