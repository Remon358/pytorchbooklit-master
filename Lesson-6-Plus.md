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
c = pd.Series([1,2,3,4,np.nan,6,7])
print(c)
```

```python
d = pd.DataFrame(np.arange(12).reshape((3,4)))
print(d)
```

```python
df = pd.Series(['a',1,3,5.0])
print(df)
print(df.index)
print(df.values)
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print(s)
s3 = {'h':1,'b':2, 'm':3}
s4 = pd.Series(s3)
print(s4)
```

```python
d = pd.read_csv('ab.csv',delimiter=',')
print(d)
d.loc[0,'chinese'] = 60
print(d)
# print(d.loc[0,'chinese'])
# print(d.iloc[1,:])
```

ab.csv中数据如下:

name,chinese,math
lihua,98,99
xiaozhang,95,60
xiaoli,100,59
xiaowang,60,86

