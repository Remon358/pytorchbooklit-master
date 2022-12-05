#### Lesson 5

##### list介绍

list 介绍： 列表 a = list()，我们通常采用list来记录中间变量，采用如下方式完成数据的添加和弹出，以及删除

list常用的API函数， max(), min(), len(), list()

| 函数名 | 功能                     |      |
| :----- | ------------------------ | ---- |
| max()  | 返回list中的最大值       |      |
| min()  | 返回list中的最小值       |      |
| len()  | 返回list的长度           |      |
| list() | 将元祖强制转换成一个list |      |

注：下式中的两句等同

```python
a  = list('hello')
b  = list("hello")
```

list中的常用方法(Method), append, count, index, insert, pop, remove, reverse, sort

| 方法名  | 功能                                   | 示例                 |
| ------- | -------------------------------------- | -------------------- |
| append  | 从列表末尾追加新元素                   | a_list.append(1)     |
| count   | 统计某元素在列表中出现的次数           | a_list.count(2)      |
| index   | 返回从列表中查找第一个匹配项的索引值   | a_list.index(2)      |
| insert  | 在列表中插入某值                       | a_list.insert(3,'m') |
| pop     | 移除列表最后一个值，返回移除的值       | a_list.pop()         |
| reverse | 列表中的所有元素反向排列               | a_list.reverse()     |
| sort    | 对列表中的元素进行排序（相同数据类型） | a_list.sort()        |

思考题：

	  c = list([1,2,"h","hello",1.23])， 此时是否存在max(c), min(c), 以及len(c)

```Python
a = list()
a.append(1)
a.append(2.3)
a.append("hello")
a.pop()
a.remove(2.3)
```

list中数据的访问,采用[]的形式对列表的数据进行访问，支持切片访问

```python
a = list()
for i in range(10):
	a.append(i)
print(a[1])
print(a[3:6])
```

#### Numpy介绍

NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

NumPy 的前身 Numeric 最早是由 Jim Hugunin 与其它协作者共同开发，2005 年，Travis Oliphant 在 Numeric 中结合了另一个同性质的程序库 Numarray 的特色，并加入了其它扩展而开发了 NumPy。NumPy 为开放源代码并且由许多协作者共同维护开发。

NumPy 是一个运行速度非常快的数学库，主要用于数组计算，包含：

- 一个强大的N维数组对象 ndarray

- 广播功能函数

- 整合 C/C++/Fortran 代码的工具

- 线性代数、傅里叶变换、随机数生成等功能

##### Numpy的导入

```python
import numpy as np
```

np.array([2,3,4],[1,2,3])可以理解为生成一个两行三列的矩阵

np.zeros((3,2))表示生成一个3行2列的矩阵，

numpy中数组创建函数

| 函数         | 说明                                                     |
| ------------ | -------------------------------------------------------- |
| array        | 创建一个数组，可以直接指定数据类型                       |
| arange       | 类似于内置的range, 但是返回的是一个ndarray而不是一个列表 |
| ones         | 全1数组                                                  |
| zeros        | 全0数组                                                  |
| empty        | 创建一个新的数组，只分配内存空间但不填充任何值           |
| eye/identity | 创建一个N维的单位矩阵                                    |

numpy中的随机数产生

```python
np.random.rand(2,3)   #均匀分布
np.random.randint(2,8,size=(3,2))
np.random.randn(3,3)   #正态分布
```



numpy中的数据类型，dtype进行， 例如

```python
a = np.array([2,1,3],dtype=np.int16 )
```



| 名称       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| bool_      | 布尔型数据类型（True 或者 False）                            |
| int_       | 默认的整数类型（类似于 C 语言中的 long，int32 或 int64）     |
| intc       | 与 C 的 int 类型一样，一般是 int32 或 int 64                 |
| intp       | 用于索引的整数类型（类似于 C 的 ssize_t，一般情况下仍然是 int32 或 int64） |
| int8       | 字节（-128 to 127）                                          |
| int16      | 整数（-32768 to 32767）                                      |
| int32      | 整数（-2147483648 to 2147483647）                            |
| int64      | 整数（-9223372036854775808 to 9223372036854775807）          |
| uint8      | 无符号整数（0 to 255）                                       |
| uint16     | 无符号整数（0 to 65535）                                     |
| uint32     | 无符号整数（0 to 4294967295）                                |
| uint64     | 无符号整数（0 to 18446744073709551615）                      |
| float_     | float64 类型的简写                                           |
| float16    | 半精度浮点数，包括：1 个符号位，5 个指数位，10 个尾数位      |
| float32    | 单精度浮点数，包括：1 个符号位，8 个指数位，23 个尾数位      |
| float64    | 双精度浮点数，包括：1 个符号位，11 个指数位，52 个尾数位     |
| complex_   | complex128 类型的简写，即 128 位复数                         |
| complex64  | 复数，表示双 32 位浮点数（实数部分和虚数部分）               |
| complex128 | 复数，表示双 64 位浮点数（实数部分和虚数部分）               |

数组属性

| 属性             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| ndarray.ndim     | 秩，即轴的数量或维度的数量                                   |
| ndarray.shape    | 数组的维度，对于矩阵，n 行 m 列                              |
| ndarray.size     | 数组元素的总个数，相当于 .shape 中 n*m 的值                  |
| ndarray.dtype    | ndarray 对象的元素类型                                       |
| ndarray.itemsize | ndarray 对象中每个元素的大小，以字节为单位                   |
| ndarray.flags    | ndarray 对象的内存信息                                       |
| ndarray.real     | ndarray元素的实部                                            |
| ndarray.imag     | ndarray 元素的虚部                                           |
| ndarray.data     | 包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。 |

重点函数

| 函数名              | 说明                       | 示例                           |
| ------------------- | -------------------------- | ------------------------------ |
| reshape             | 不改变数据的条件下修改形状 | a = np.arange(10).reshape(2,5) |
| transpose           | 对换数组的维度             | a.transpose()                  |
| squeeze             | 从数组的形状中删除一维条目 | a.squeeze()                    |
| sin()、cos()、tan() | np中的三角函数             |                                |
|                     |                            |                                |
|                     |                            |                                |
|                     |                            |                                |
|                     |                            |                                |

游戏设计

```
class Game():

    def __init__(self, name, sex, hp, exp):
        self.name = name
        self.sex = sex
        self.hp = hp
        self.exp = exp

    def showhp(self):
        self.hp -= 20
        if self.hp <= 0:
            print('your hero is dead...')
        else:
            pass
        return self.hp

    def showexp(self):
        self.exp += 50
        if self.exp == 50:
            print('你升级了')
        else:
            pass
        return self.exp


if __name__ == '__main__':
    Button = str(input('是否开始游戏 y/n '))
    if Button == 'y':
        Player1 = Game(name='Little Boy', sex='male', hp=100, exp=0)
        for i in range(3):
            Player1.showhp()
        for i in range(5):
            Player1.showexp()
    else:
        pass
    print('你剩余的生命值为')
    print(Player1.hp)
```
