### Lesson-10 

##### 反向传播

 **知识点回顾：链式传播法则**

$y=f(g(x))$ 则有 $\frac{dy}{dx}=\frac{\partial{f}}{\partial{g}} \frac{\partial{g}}{\partial{x}}$

在pytorch中，tensor的数据类型中，一方面包含有数据的数值，还有一个数据包含有数据的梯度。 如下例子

```python
import torch

a = torch.tensor([2, 3], requires_grad=True, dtype=torch.float)
b = torch.tensor([6, 4], requires_grad=True, dtype=torch.float)

Q = 3 * a ** 3 - b ** 2

extern_gradient = torch.tensor([1, 1])
Q.backward(gradient=extern_gradient)
print(a.grad)
print(b.grad)
```

我们定义$Q=3a^3 - b^2$, 则可以计算出 $\frac{\partial{Q}}{\partial{a}} = 9a^2$ 且 $\frac{\partial{Q}}{\partial{b}}=-2b$, 然后我们将$a，b$的值带入就可以计算出对应的梯度值。

```python
import torch

x = torch.tensor(3, dtype=torch.float32, requires_grad=True)
y = torch.tensor(4, dtype=torch.float32, requires_grad=True)
b = torch.tensor(5, dtype=torch.float32, requires_grad=True)
z = x*y + b
"Z = xy"
print(z)
z.backward()
print(z.requires_grad, x.grad, y.grad, b.grad)
```

在pytorch中，可以通过backward()自动的反向计算梯度，对应的数据类型应该是torch中的tenser.  在上一个例子中，我们需要计算的是对应变量$a$的梯度，因此，我们对上一个例子进行修改。首先，引入pytorch的包， import torch. 

定义数据类型为 a= torch.Tensor([7.0]) 且声明 a. requires_grad=True

```python
from matplotlib import pyplot as plt
import torch

data_x = [1, 2, 3]
data_y = [2, 4, 6]
loss_list = list()
a_list = list()

alpha = 0.01


def forward(x):
    return a * x


def lossFunction(x, y):
    y_pred = forward(x)
    loss = (y_pred - y) ** 2
    return loss

if __name__ == '__main__':
    a = torch.Tensor([7.0])
    a.requires_grad = True
    for epoch in range(1000):
    # for a in np.arange(0, 4, 0.1):
        sum_loss = 0
        for i in range(3):
            sum_loss += lossFunction(data_x[i], data_y[i])
            l = lossFunction(data_x[i],data_y[i])
            l.backward()
            a.data = a.data - alpha*a.grad
            a.grad = None
            a_list.append(a.data)
            # a = gradient(a, data_x[i], data_y[i])
        loss_list.append(sum_loss / 3)

    print(a_list)
    plt.subplot(211)
    plt.plot(a_list)
    plt.subplot(212)
    plt.plot(loss_list)
    plt.show()
```
