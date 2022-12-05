#### Lesson-6 数据可视化工具包 matplotlib



matplotlib是最初由John D.hunter创建，首次发布与2007年。其代码风格与matlab中的plot相似。其中mat表示矩阵，plot表示显示，lib表示工具包。Matplotlib是python中最为基础也是使用最为广泛的可视化工具包。



##### matplotlib 绘制图形例子

```python
from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0, np.pi * 2, 50)
plt.plot(x, np.sin(x))
plt.show()
```

一些常用的API

| 函数名  | 作用                   | 示例                           |
| ------- | ---------------------- | ------------------------------ |
| title   | 图片标题               | plt.title('sin function plot') |
| xlabel  | x轴标题                | plt.xlabel('x')                |
| ylabel  | y轴标题                | plt.ylabel('sinx')             |
| grid    | 显示网格               | plt.grid()                     |
| xlim    | 设置x轴的坐标范围      | plt.xlim([0,8])                |
| ylim    | 设置y轴的坐标范围      | plt.ylim([-2,2])               |
| legend  | 显示图注               | plt.legend()                   |
| plot    | 绘制图形               | plt.plot(x, y)                 |
| subplot | 绘制子图               | plt.subplot(x,y)               |
| show    | 显示图像               | plt.show()                     |
| figure  | 创建一副图像，返回句柄 | plt.figure()                   |

**注：plot(x,y , 'r*',label=r'$\alpha$') 表示绘制图形的样式为红色，对应的图注为希腊字母 $\alpha$**

下面的代码给一个示例程序:

```python
from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0, np.pi * 2, 50)
plt.figure(figsize=[50, 50])
plt.subplot(211)
plt.plot(x, np.sin(x), 'r*', label=r'$\beta$')
plt.title('Sinx Figure')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.xlim([0, 10])
plt.ylim([-2, 2])
plt.grid()
plt.legend()
# plt.show()
plt.subplot(212)
plt.plot(x, np.cos(x), 'b', label='cosx')
plt.show()
```

练习题：绘制y=|x|,y=log(x),y=eˣ,以及y=sin(x)的图像，并将其放在一个图形中，分2*2显示。

```python
from matplotlib import pyplot as plt
import numpy as np
x = np.linspace(-5,5,50)
# print(x)
plt.subplot(221)
plt.xlabel('x')
plt.ylabel('sinx')
plt.plot(x,np.sin(x),'g')
plt.subplot(222)
plt.xlabel('x')
plt.ylabel('|x|')
plt.plot(x,np.abs(x),'g')
plt.subplot(223)
plt.xlabel('x')
plt.ylabel('$e^x$')
plt.plot(x,np.e**x,'g')
plt.subplot(224)
plt.xlabel('x')
plt.ylabel('logx')
x1 = np.linspace(0.1,10,100)
plt.plot(x1,np.log(x1),'g')
plt.show()

```

