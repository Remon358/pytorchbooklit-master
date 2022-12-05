Lesson-14

卷积神经网络

首先需要完成卷积网络的维度的推断

```python
import torch
width, height = 28, 28
in_channle = 1
batch_size = 1
inputs = torch.randn(batch_size, in_channle,
                     width, height)
print(inputs.shape)
conv_lay1 = torch.nn.Conv2d(in_channels=1,
                            out_channels=10,
                            kernel_size=5)
output1 = conv_lay1(inputs)
print(output1.shape)
maxpool_lay = torch.nn.MaxPool2d(kernel_size=2)
output2 = maxpool_lay(output1)
print(output2.shape)
conv_lay2 = torch.nn.Conv2d(in_channels=10,
                            out_channels=20,
                            kernel_size=5)
output3 = conv_lay2(output2)
print(output3.shape)
output4 = maxpool_lay(output3)
print(output4.shape)
output5 = output4.view(1, -1)
linear_lay = torch.nn.Linear(320, 10)
output6 = linear_lay(output5)
print(output6.shape)
```

下面将手写数字识别的程序修改成带有卷积操作的深度神经网络结构

```python
import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np

batch_size = 64
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

minist_tainloader = datasets.MNIST(root='./', train=True, download=True, transform=data_transform)
minist_testloader = datasets.MNIST(root='./', train=False, download=True, transform=data_transform)

trainloader = DataLoader(minist_tainloader, batch_size=batch_size, shuffle=True)
testloader = DataLoader(minist_testloader, batch_size=batch_size, shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.linear = torch.nn.Linear(320, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.pooling(self.conv1(x)))
        x = self.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x

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
            for i in range(batch_size):
                labels_np = labels.numpy().tolist()
                pred_np = pred.numpy().tolist()
                total_num += 1
                if labels_np[i] == pred_np[i].index(max(pred_np[i])):
                    correct += 1
            print(f'Accuracy = {correct / total_num}')


if __name__ == '__main__':
    for epoch in range(3):
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

后面, 在卷积神经网络的后续学习中, 提供对应的googlenet框架模型, $1\times1$卷积核的作用, residual net网络框架. 通过手写数字集在进行复现.