Lesson-13 多分类问题

采用softmax函数进行

$$P(y=i) = \frac{e^{z_i}}{\sum_{j=0}^{k}{e^{z_j}}} \quad j=0,1,2,3...i...k$$

保证了求和后为1， 且每个值都是正数， $e^x$是一个单调增函数

同时计算损失的时候，之前的BCE已经不再适用（该问题是针对的二分类问题）， 所以引入一种新的损失计算方法 CrossEntropyLoss()， 计算表达式为

$-y log\hat{y}$

```python
import torch
criterion = torch.nn.CrossEntropyLoss()
y = torch.LongTensor([2, 0, 1])

y1 = torch.tensor([[0.1,0.2,0.9],
                   [1.1,0.1,0.2],
                   [0.2,2.1,0.1]])

y2 = torch.tensor([[0.8,0.2,0.3],
                   [0.2,0.3,0.5],
                   [0.2,0.2,0.5]])

loss1 = criterion(y1, y)
loss2 = criterion(y2, y)
print(f'loss1= {loss1}, loss2={loss2}')
```

手写数字识别例题:

```python
import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np

batch_size = 64
batch_size_test = 100
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

minist_tainloader = datasets.MNIST(root='./', train=True, download=True, transform=data_transform)
minist_testloader = datasets.MNIST(root='./', train=False, download=True, transform=data_transform)

trainloader = DataLoader(minist_tainloader, batch_size=batch_size, shuffle=True)
testloader = DataLoader(minist_testloader, batch_size=batch_size_test, shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        return self.linear5(x)

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
loss_list = list()

def test_accuracy():
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            pred = model(images)
            total_num = 0
            correct = 0
            for i in range(batch_size_test):
                labels_np = labels.numpy().tolist()
                pred_np = pred.numpy().tolist()
                total_num += 1
                if labels_np[i] == pred_np[i].index(max(pred_np[i])):
                    correct += 1
            print(f'Accuracy = {correct/total_num}, i = {i}')


if __name__ == '__main__':
    for epoch in range(10):
        for i, data in enumerate(trainloader, 0):
            inputs, label = data
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss_list.append(loss)
            loss.backward()

            optimizer.step()
        print(f'[{epoch}]: loss = {loss}')

    plt.plot(loss_list)
    plt.show()

    test_accuracy()
```

通过PIL识别图像

```python
import numpy as np
from PIL import Image

a = Image.open('test.jpg')
c = a.convert('L')
c.show()
# print(c)
im = np.array(a)
im_gray = np.array(c)
print(im_gray.shape)
print(im_gray)
print(im.shape)
# print(im)
b = np.array([[[1,2,3],[2,3,3],[3,4,5]],[[2,1,2],[3,4,5],[4,5,6]]])
# print(b.shape)
# a.show()
# print(a)
```

