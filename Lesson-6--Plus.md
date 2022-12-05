#### Lesson-6-Plus

##### Pandas简介

pandas是强大的结构化数据分析工具，可以用于数据挖掘与数据分析，也可以用于数据清洗。

pandas的数据结构有Serials形式，以及dateframe形式。

```python
import pandas as pd
c = pd.Series([1,2,3,4,6,6,7])
print(c)
```

```python
import pandas as pd
import numpy as np
c = pd.Series([1,2,3,4,np.nan,6,7])#创建一个索引函数Series的一维数组，nan不是一个数，是一个float类型的数据，所以相关计算都无法得到数字
print(c)
```

```python
d = pd.DataFrame(np.arange(12).reshape((3,4)))#定义一个数据框，分别是以3行4列，数据是0~12排列组成的数据表
print(d)
```

```python
df = pd.Series(['a',1,3,5.0])#创建一个索引函数Series的一维数组，包括字符、整型、float类型
print(df)
print(df.index#查看索引，输出打印RangeIndex(start=0, stop=4, step=1)，start为起点、stop为终止、step为歩长
print(df.values#返回视图对象，输出打印['a' 1 3 5.0]
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e']#创建一个索引函数Series的一维数组，然后index作为标量序号
print(s)
s3 = {'h':1,'b':2, 'm':3}#定义一个以h、b、m为1、2、3数值的序号数组
s4 = pd.Series(s3) #将创建好的数组s3利用索引函数Series进行整理赋值给s4
print(s4)
```

```python
d = pd.read_csv('ab.csv',delimiter=',') #pandas提供了pd.read_csv()方法可以读取其中的数据，返回DataFrame文件，利用逗号(,)隔开
print(d)
d.loc[0,'chinese'] = 60 #索引从第0行开始查chinese成绩等于60分
print(d)
# print(d.loc[0,'chinese']) #输出打印索引从第0行开始查到chinese成绩等于60分
# print(d.iloc[1,:])#通过整数索引第1行号获取行数据，不能是字符
```

ab.csv中数据如下:

name,chinese,math
lihua,98,99
xiaozhang,95,60
xiaoli,100,59
xiaowang,60,86

