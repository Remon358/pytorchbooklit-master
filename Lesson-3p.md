### Lesson-3

###### python函数的定义

1. 格式 def + 函数名（传递参数）: 函数体， return 

   ```python
   def fool(a,b):
   	return a+b
   
   print(fool(1,2))
   ```

2. 返回类型可以是多个变量，返回值为一个列表，可以通过列表的访问形式进行访问; 也可以采用多个返回值来表示。

   ```python
   def fool(a,b):
   	return a+b, a
   c,d = fool(1,2)	
   ```

3. 传递参数可以有默认值；传递参数可以是对象，且对象不必添加类名字

4. 函数名建议采用驼峰命名法
##### python中的类

1. 讲不同的类型放在一起，组成一个整体，形成类。类的出现方便的数据的管理，利于程序的编写，是面向对象变成的体现。类包含有自己的属性，对外提供的方法，以及类内的函数。C++中对类的描述分为public, protect, private三种属性。一般采用public表示对外提供的接口以及方法，采用protect和private表示保护以及私有属性，保护以及私有属性，一般在类外不能访问。在python的编程中，建议采用相同的编程逻辑。

2. 格式 class 类名（父类）： 类体 

   ```python
   class Person():
   
   ​         def __init(self,name="liming", age = 18, classname="classthree")__:
   
   ​			self. name = name
   
   ​			self. age = age
   
   ​			self. classname = classname
   
   ​        def ShowName(self):
   
   ​            return self.name
   
   ​        ......
   ```

3. 对象的建立 person = Person()  表示建立默认的对象，由于没有指定传递参数，采用默认的传递参数。也可以在建立对象的时候指定传递参数。比如 person1 = Person("xiaohau", 20, "classone")

4. 一般采用类中提供的方法实现功能（方法即为类中对外提供的函数）

5. 类的封装性：只对外提供想要的实现，对类内封闭的属性进行封装。python中允许直接方针对象的属性

   练习题目：创建一个汽车类，包含有汽车的名字，车长，车重， 并创建实现方法显示其中的属性。

   练习题目： 在游戏中为了增加互动性，通常会创建一个游戏角色，该角色具有一定的属性，比如性别，名字，初始生命值。 我们假设，当受到一次攻击时，生命值会减少10， 当生命值小于等于0时，角色死亡；当击打别人一次，经验值增加10， 当经验值大于50的时候，升级。请编程模拟该过程。初始人物为 {” Little Boy“， male,  100}, 受到2次攻击，同时击打别人5次的过程。
   
   #### 类中的继承  待补充
##### python中采用import导入外部pack或者外部程序

1. 具体写法为import Person 表示引入Person.py的文件，同时可以使用Person.py文件中的内容

2. import numpy as np表示引入numpy的同时给numpy起一个别名np，后面在使用的时候，可以采用np来代替numpy.

3. 本节课中常用的外部包有numpy, time, math, matplot, tensor

   练习从另一个.py文件中，引入变量，函数，类。

   例：从键盘输入两个整数，调用四个函数，分别显示输出+-*/（加减乘除）

   ```python
   def functionAdd(num1,num2): #加法
       return num1+num2
   def functionSub(num1,num2): #减法
       return num1-num2
   def functionMul(num1,num2): #乘法
       return num1*num2
   def functionDiv(num1,num2): #除法
       if not num2:
           return 99999
       return num1/num2
   
   if __name__=='__main__':
       #请输入数字
       num1= int(input('Please input a  number:'))  
       num2= int(input('Please input a  number:'))
       
       #打印加减乘除的结果
       print(functionAdd(num1,num2))
       print(functionSub(num1,num2))
       print(functionMul(num1,num2))
       
       #除法判断是否有误
       divResult =functionDiv(num1,num2)
       if divResult ==99999:
           print('the second is zero...wrong input')
       else:
           print(functionDiv(num1,num2))
   ```
   
   汽车类型代码
   
   ```python
   #这是一个Car类
   class Car():
       '''
       kind : 车的品牌
       length ：车的长度
       weight ： 车的重量
       '''
       def __init__(self,kind='BMW',length=4.5,weight=1.6):
           self.kind=kind
           self.length=length
           self.__weight=weight #__weight 表示私有属性 不能直接调用
           
       #显示车的重量
       def showWeight(self):
           return self.__weight
       
   #初始化Car类
   BMW=Car()
   
   #打印车的每个属性
   print(BMW.kind)
   print(BMW.showWeight())
   print(BMW.length)
   ```
   
   

