

### Lesson-9plus

### 线性模型

$f(x) = ax + b$

深度学习的框架 数据集 模型, 训练 , 推理

|      x      |      y      |
|     ---     |     ---     |
|      1      |      2      |
|      2      |      4      |
|      3      |      6      |
|      4      |      ?      |


#### **前三行可以成为训练集, 第四行为测试集**

很容易的,我们可以猜到最后的$y$值应该是$4$, 但是我们在怎么让机器去预测最后的$y$值呢? 换句话说,如果我们能够找到一个直线方程,能够最大限度的穿过给定的x, y点也可以. (这里为了简化计算, 假设已经知道直线方程过原点,也就是说$b$ 等于0) 

**“斜率”参数（w，也叫作权重或系数）被保存在coef_属性中，而偏移或截距（b）被保存在intercept_属性中：**

#### 线性模型利用输入特征的线性函数（linear function）进行预测。

普通最小二乘法（ordinary least squares，OLS），是回归问题最简单也最经典的线性方法。线性回归寻找参数 w 和 b，使得对训练集的预测值与真实的回归目标值 y之间的均方误差最小。均方误差（mean squared error）是预测值与真实值之差的平方和除
以样本数。线性回归没有参数，这是一个优点，但也因此无法控制模型的复杂度。

 岭回归的预测公式与普通最小二乘法相同。但在岭回归中，对系数（w）的选择不仅要在训练数据上得到好的预测结果，而且还要拟合附加约束。我们还希望系数尽量小。、w 的所有元素都应接近于 0。直观上来看，这意味着每个特征对输出的影响应尽可能小（即斜率很小），同时仍给出很好的预测结果。这种约束是正则化（regularization）。正则化是指对模型做显式约束以避免过拟合。岭回归使用的是L2正则化。岭回归在 linear_model.Ridge 中实现。对扩展的波士顿房价数据集进行岭回归预测（图1）

![不同alpha值的岭回归与线性回归的系数比较](https://images.gitee.com/uploads/images/2021/1010/225532_1bace6e3_9672955.png "image-20211010221429206.png")

​                                                                            **图一：不同alpha值的岭回归与线性回归的系数比较**

![输入图片说明](https://images.gitee.com/uploads/images/2021/1010/225607_aa1434b2_9672955.png "image-20211010221910663.png")

​                                                                            **图二：不同 alpha 值的 lasso 回归与岭回归的系数比较**

除了 Ridge，还有一种正则化的线性回归是 Lasso。与岭回归相同，使用 lasso 也是约束系数使其接近于 0，但用到的方法不同，叫作 L1 正则化。L1 正则化的结果是，使用 lasso 时某些系数刚好为 0。这说明某些特征被模型完全忽略。这可以看作是一种自动化的特征选择。某些系数刚好为 0，这样模型更容易解释，也可以呈现模型最重要的特征。

常见的两种线性分类算法是 Logistic 回归（logistic regression）和线性支持向量机（linear support vector machine，线性 SVM），前者在 linear_model.LogisticRegression 中实现，后者在 svm.LinearSVC（SVC 代表支持向量分类器）中实现。

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

![输入图片说明](https://images.gitee.com/uploads/images/2021/1010/225801_edc73c66_9672955.png "1.png")

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

**优点、缺点和参数**
线性模型的主要参数是正则化参数，在回归模型中叫作alpha，在LinearSVC和LogisticRegression中叫作C。alpha值较大或C值较小，说明模型比较简单。特别是对于回归模型而言，调节这些参数非常重要。通常在对数尺度上对C和alpha进行搜索。此外还需要确定的是用L1正则化还是L2正则化。如果假定只有几个特征是真正重要的，那么应该用L1正则化，否则应默认使用L2正则化。如果模型的可解释性很重要的话，使用L1也会有帮助。由于L1只用到几个特征，所以更容易解释哪些特征对模型是重要的，以及这些特征的作用。
线性模型的训练速度非常快，预测速度也很快。这种模型可以推广到非常大的数据集，对稀疏数据也很有效。如果数据包含数十万甚至上百万个样本，可能需要研究如何使用LogisticRegression和Ridge 模型的 solver='sag' 选项，在处理大型数据时，这一选项比默认值要更快。其他选项还有SGDClassifier类和SGDRegressor类，它们对以上的线性模型实现了可扩展性更强的版本。
线性模型的另一个优点在于，利用我们之前见过的用于回归和分类的公式，理解如何进行预测是相对比较容易的。不幸的是，往往并不完全清楚系数为什么是这样的。如果数据集中包含高度相关的特征，这一问题尤为突出。在这种情况下，可能很难对系数做出解释。
如果特征数量大于样本数量，线性模型的表现通常都很好。它也常用于非常大的数据集，只是因为训练其他模型并不可行。但在更低维的空间中，其他模型的泛化性能可能更好。
