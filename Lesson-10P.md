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

 ```python
#调用线性包
from matplotlib import pyplot as plt
import torch

data_x = [1, 2, 3]#创建数据
data_y = [2, 4, 6]
loss_list = list()#损失的列表  创建list函数
a_list = list()

alpha = 0.01  #调线性图像系数

#预测值运算
def forward(x):
    return a * x


def lossFunction(x, y):
    y_pred = forward(x)#预测值
    loss = (y_pred - y) ** 2 #损失值，将预测值减去真实值得平方
    return loss

if __name__ == '__main__':
    a = torch.Tensor([7.0])# 数据节点
    a.requires_grad = True #requires_grad必须为True，才能求偏导
    for epoch in range(1000):#梯度
    # for a in np.arange(0, 4, 0.1):
        sum_loss = 0
        for i in range(3):
            sum_loss += lossFunction(data_x[i], data_y[i])#获取data_x，data_y列表数据
            l = lossFunction(data_x[i],data_y[i])
            l.backward()
            a.data = a.data - alpha*a.grad #梯度运算
            a.grad = None
            a_list.append(a.data)
            # a = gradient(a, data_x[i], data_y[i])
        loss_list.append(sum_loss / 3)#取平均数
    #输出数据，图像显示
    print(a_list)
    plt.subplot(211)
    plt.plot(a_list)
    plt.subplot(212)
    plt.plot(loss_list)
    plt.show()
 ```
