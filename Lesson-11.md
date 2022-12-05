Lesson-11

利用pytorch实现线性回归

本节课我们将采用pytorch标准模块来实现上面的例子。

[优化器参考文档https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html)

计算流程如下所示：

```python
for input, target in dataset:
    output = model(input)
    loss = loss_fn(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(）
```
同时之前的例子可以改写为：
```python
import torch
from matplotlib import pyplot as plt

x_data = torch.tensor([[1], [2], [3]], dtype=torch.float)
y_data = torch.tensor([[2], [4], [6]], dtype=torch.float)


class LinearExample(torch.nn.Module):
    def __init__(self):
        super(LinearExample, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearExample()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epoch_list = list()
a_list = list()

if __name__ == '__main__':
    for epoch in range(100):
        y_hat = model(x_data)
        loss = criterion(y_hat, y_data)
        a_list.append(model.linear.weight.item())
        epoch_list.append(epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(epoch_list, a_list)
    plt.show()
```
