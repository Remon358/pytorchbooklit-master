Lesson-12plus

逻辑斯特回归问题：用于处理二分类问题-->二分类表示输出或者是0， 或者是1， 输出只有两种选择。

逻辑回归（Logistic regression 或logit regression），即逻辑模型（英语：Logit model，也译作“评定模型”、“分类评定模型”）是离散选择法模型之一，属于多重变量分析范畴，是社会学、生物统计学、临床、数量心理学、计量经济学、市场营销等统计实证分析的常用方法。虽然叫做回归其实是个分类模型。逻辑回归实际上是用sigmoid函数将线性回归进行了归一化，把输出值压缩到0-1之间，这个值代表的是发生的概率。
![输入图片说明](https://images.gitee.com/uploads/images/2021/1010/225133_66fefbda_9672955.png "image-20211010223718410.png")
对这张图一定不陌生，它在神经网络中作为常用的激活函数，将输出值控制在0-1之间。他就是Sigmoid函数！当然它也被称为Logistic函数，这就是逻辑斯蒂回归算法的由来。还有一点要注意的是，虽然名字叫做逻辑斯蒂回归，但他解决的其实是分类问题。其实原理很简单，对于二分类任务来说，大于0就是正类，小于0就是负类。

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

在深度学习中，由于问题的复杂性，我们往往会使用较深层数的网络进行训练，相信很多炼丹的朋友都对调参的困难有所体会，尤其是对深层神经网络的训练调参更是困难且复杂。在这个过程中，我们需要去尝试不同的学习率、初始化参数方法（例如Xavier初始化）等方式来帮助我们的模型加速收敛。深度神经网络之所以如此难训练，其中一个重要原因就是网络中层与层之间存在高度的关联性与耦合性。下图是一个多层的神经网络，层与层之间采用全连接的方式进行连接。

![输入图片说明](https://images.gitee.com/uploads/images/2021/1010/225200_7fce9204_9672955.png "image-20211010224228682.png")

![输入图片说明](https://images.gitee.com/uploads/images/2021/1010/225212_e67c5678_9672955.png "image-20211010224248956.png")

我们规定左侧为神经网络的底层，右侧为神经网络的上层。那么网络中层与层之间的关联性会导致如下的状况：随着训练的进行，网络中的参数也随着梯度下降在不停更新。一方面，当底层网络中参数发生微弱变化时，由于每一层中的线性变换与非线性激活映射，这些微弱变化随着网络层数的加深而被放大（类似蝴蝶效应）；另一方面，参数的变化导致每一层的输入分布会发生改变，进而上层的网络需要不停地去适应这些分布变化，使得我们的模型训练变得困难。上述这一现象叫做Internal Covariate Shift。

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
