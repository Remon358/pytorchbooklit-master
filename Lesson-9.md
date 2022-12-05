### Lesson-9

线性模型

$f(x) = ax + b$

深度学习的框架 数据集 模型, 训练 , 推理

| x    | y    |
| ---- | ---- |
| 1    | 2    |
| 2    | 4    |
| 3    | 6    |
| 4    | ?    |

**前三行可以成为训练集, 第四行为测试集**

很容易的,我们可以猜到最后的$y$值应该是$4$, 但是我们在怎么让机器去预测最后的$y$值呢? 换句话说,如果我们能够找到一个直线方程,能够最大限度的穿过给定的x, y点也可以. (这里为了简化计算, 假设已经知道直线方程过原点,也就是说$b$ 等于0) 

下面给出一个方法:

定义损失函数 $loss = (\hat{y} -y)^2 = (a\cdot x+b-y)^2$, 可以看出,只要找到了正确的a值,就可以得到一个较小的loss函数值.

首先,我们采用穷举的方式查找最小的损失函数

本例题的目标就是找到一个组合适的$x, b$使得上式中的$loss$最小

```python
import numpy as np
from matplotlib import pyplot as plt

data_x = [1, 2, 3]
data_y = [2, 4, 6]
loss_list = list()
a_list = list()


def forward(x):
    return a * x


def lossFunction(x, y):
    y_pred = forward(x)
    loss = (y_pred - y) ** 2
    return loss

def predict(x,a_):
    return a_*x


if __name__ == '__main__':
    for a in np.arange(0, 4, 0.1):
        sum_loss = 0
        for i in range(3):
            sum_loss += lossFunction(data_x[i], data_y[i])
        loss_list.append(sum_loss / 3)
        a_list.append(a)
    plt.figure()
    plt.plot(a_list, loss_list)
    # plt.title("")
    plt.xlabel('a')
    plt.ylabel('loss')
    # plt.show()
    min_value = min(loss_list)
    index_lossMin = loss_list.index(min_value)
    print(index_lossMin)
    proper_a = a_list[index_lossMin]
    print(proper_a)
    print("Please input the desired x:")
    desired_x = input()
    print(f"The predict output for the linear model is {predict(float(desired_x),proper_a)}")
```

```python
下面给出采用sklearn的方法来实现线性回归的效果

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import pandas as pd
lrm = linear_model.LinearRegression()
x_data = np.array([1,2,3])
y_data = np.array([2,4,6])
z_data = np.zeros([3,2])
m_data = np.zeros([3,2])
z_data[:,0] = x_data
z_data[:,1] = y_data
m_data[:,0] = x_data
m_data[:,1] = x_data
lrm.fit(m_data,z_data)
print(lrm.predict([[4,4]]))


练习题：
'''
给定训练集为     x=1, y=6.8
          x=2, y=9.8
          x=3, y=13.2
          x=4, y=16.2
测试集       x=5, y=? '''

import numpy as np
from matplotlib import pyplot as plt
x_data = [1,2,3,4]
y_data = [6.8,9.8,13.2,16.2]

loss_list = list()
def forward(a,x,b):
    return a*x+b

def lossFunction(a,x,y,b):
    y_pred = forward(a,x,b)
    loss = (y_pred - y)**2
    return loss

a_list = list()
b_list = list()
if __name__ == '__main__':
    for a in np.arange(0,6,0.1):
        for b in np.arange(0,6,0.1):
            sum_loss = 0
            for i in range(4):
                sum_loss += lossFunction(a, x_data[i], y_data[i],b)
            loss_list.append(sum_loss/4)
            a_list.append(a)
            b_list.append(b)

    plt.plot(a_list,loss_list)
    plt.xlabel('a')
    plt.ylabel('loss')
    print(min(loss_list))
    loss_min_index = loss_list.index(min(loss_list))
    print(loss_min_index)
    a_wanted = a_list[loss_min_index]
    b_wanted = b_list[loss_min_index]
    print(f'a_wanted = {a_wanted}, b_wanted ={b_wanted}')
    # plt.show()
    # a_wanted = a_list[loss_list.index(min(loss_list))]
    # print(forward(a_wanted, 4))
    print(forward(a_wanted, 5, b_wanted))

```

通过下面的代码可以直观的看出，当取得不同$b$的时候，得到的直线的样子。

```python
def LinearFunction(x,a=3.2,b=3.4):
    return a*x+b
def LinearFunction2(x,a=3.2,b=3.5):
    return a*x+b
x_data = [1,2,3,4]
y_data = [6.8, 9.8, 13.2, 16.2]
z_data = [6,12,18,24]
n_data = np.arange(5)
m_data = np.zeros([5,1])
l_data = np.zeros([5,1])
for i in range(5):
    m_data[i] = LinearFunction(n_data[i])
    l_data[i] = LinearFunction2(n_data[i])
plt.scatter(x_data,y_data)
plt.plot(n_data,m_data,'r')
plt.plot(n_data,l_data,'g')
plt.show()
```

上面的实例中存在一个问题,那就是我们是采用穷举的方式进行a值的预测,这样就要求a的最佳值最好我们所选取的范围内,否则只能得到一个局部最优值,但是这个值是不是全局最优值就不知道了. 也就是说,上面的例子中, 如果我们选取的(0,4)的范围不对,是(3,6), 那么我们得到的预测结果就不是最优的了. 

那么,有没有什么方法可以避免这个问题,可以让程序自己找到一个最小的Loss呢?  

**局部最小值与全局最小值**, 我们可以认为上面的方法很有可能找到一个局部最小值.   画图说明局部最小值的问题

![1](C:\Users\Administrator\Desktop\课件\pytorchbooklit\1.png)

接下来,我们采用梯度的方法进行网络的改写

```python
import numpy as np
from matplotlib import pyplot as plt

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


def predict(x, a_):
    return a_ * x


def gradient(a, x, y):
    a = a - alpha * 2 * (a * x - y) * x
    return a


if __name__ == '__main__':
    a = 0
    for epoch in range(1000):
    # for a in np.arange(0, 4, 0.1):
        sum_loss = 0
        for i in range(3):
            sum_loss += lossFunction(data_x[i], data_y[i])
            a = gradient(a, data_x[i], data_y[i])
        loss_list.append(sum_loss / 3)
        a_list.append(a)
    plt.subplot(211)
    plt.plot(a_list)
    plt.subplot(212)
    plt.plot(loss_list)
    plt.show()

    plt.figure()
    plt.plot(a_list, loss_list)
    plt.xlabel('a')
    plt.ylabel('loss')
    plt.show()
    min_value = min(loss_list)
    index_lossMin = loss_list.index(min_value)
    print(index_lossMin)
    proper_a = a_list[index_lossMin]
    print(proper_a)
    print("Please input the desired x:")
    desired_x = input()
    print(f"The predict output for the linear model is {predict(float(desired_x), proper_a)}")
```

```python
import time

import numpy as np
from matplotlib import pyplot as plt
import random

x_data = [1,2,3]
y_data = [2,4,6]

loss_list = list()
a_b_list = list()

def forward(a, x):
    return a*x

def lossFunction(a,x,y):
    y_pred = forward(a,x)
    loss = (y - y_pred)**2
    return loss

alpha = 0.1
def gradient(x, a, y):
    a = a - alpha*2*x*(x*a -y)
    return a


a_list = list()
b_list = list()
if __name__ == '__main__':
    a = random.randint(0, 10)
    for epoch in range(100):
        sum_loss = 0
        for i in range(3):
            sum_loss += lossFunction(a, x_data[i], y_data[i])
            a = gradient(x_data[i], a, y_data[i])

        loss_list.append(sum_loss/3)
        a_list.append(a)
        b_list.append(epoch)
        print(f'epoch = {epoch}, a = {a}, loss = {sum_loss/3}')
        time.sleep(0.5)
        # time.sleep(0.5)
    # plt.plot(a_list, loss_list)
    plt.plot(b_list, loss_list)
    plt.show()
```

