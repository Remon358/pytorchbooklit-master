##### python语言简介

python英文翻译成“蟒蛇”， 并不是由于其具有爬虫的特点。

python是一门解释型语言，具有如下特点：

1. 与C++/C 相比，不需要编译操作。

   由于C/C++需要进行编译操作，将源代码最终翻译成机器代码，所以从执行效率上是最高的。

   但是程序的修改相对比较麻烦，需要维护CMakeLists.txt文件。与之相对应的Python不需要进行相应的编译操作，可以直接运行，方便了程序的修改和调试。当时执行效率相对较低。

2. 伪多线程的存在

3. Python和C++一样是面向对象的编程。

   3.1 ”面向对象的编程“与”面向过程的编程“。

   面向对象的编程：将具有相同属性的变量或者函数进行封装，较为符合人们的理解习惯，同时在大型工程中，利于分工与合作，代码的管理等。

   面向过程的编程：注重代码的实现过程，目前在底层驱动方面用得较多，以C语言为主。编程语法较少，适合初学者入门。

4. Python具有丰富的包。通过这几年的发展，python在各个行业都已经有各种较为成熟的代码包可供使用，这大大降低了工程的开发难度，同时，python这种解释性语言，也很方便的将自己写的代码做成代码包，方便管理与复用。

5. 应用领域较广：机器视觉，机器学习，科学计算，网页爬虫，文本处理，UI界面，无人机控制等领域

##### python 语言的编程环境

1. 在Windows下和Ubuntu下均有较好的编程体验。建议采用Pytharm进行编程操作。本次授课采用的Jupyter进行编程授课。jupyter notebook的好处在于将文本的说明和代码运行较好的集成在了一起，一方面文字的说明给了大家教学的内容提示，以及编程的说明。另一方面，代码可以立刻出结果，则方便了对大家对编程的理解。

2. 将上课地点选在实验室，方便大家随时跟着进行练习。Pycharm的基本使用，将不再进行详细的讲述。在课件以及教案中，或者下课的辅助公众号中，给出常用的使用方法，希望同学们能够多多练习，多百度。

   2.1 需要安排2名同学进行公众号的维护，一般选择学习委员和班长，其他有兴趣的同学可以积极踊跃报名。该同学主要负责： 公众号的维护，问题的反馈，课程教案的维护文字处理工作。同时，会经常和我本人进行交流，提升一定的专业知识。

##### 整个教学安排

1. 课程前1/3部分，我们会对python以及其常用库进行介绍。包括且不限于
   1. python基础语法：能够完成简单的Python基础编程。python的基本数据类型，array, list, directory等，循环，分支，顺序。函数，类，import导入。
   
   2. 本门课程用的python包教学：Numpy, matplotlib, torch等
   
   3. 基于Pytorch框架的深度学习课程教学：梯度 --> CNN , RNN 
   
##### 初识python

1. Win+R  然后 输入cmd. 得到

![image-20210902234435012](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210902234435012.png)

在里面输入python，即可进入python环境

![image-20210902234512792](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210902234512792.png)

其中1显示了当前的Python版本号

我们可以直接在python环境中输入

```python
print('hello,world')
```

或者在python环境中直接键入

```python
2**10
```
注： python中支持的运算表达为

| 表达式     | 运算结果 | 说明                   |
| ---------- | -------- | ---------------------- |
| 9 + 4      | 13       | 加法                   |
| 9 - 4      | 5        | 减法                   |
| 9 * 4      | 36       | 乘法                   |
| 9 / 4      | 2.25     | 数学除法               |
| 9 // 4     | 2        | 取整除法               |
| 9 % 4      | 1        | 取余（模运算）         |
| 2**10      | 1024     | 乘方，2的10次方        |
| 36**0.5    | 6        | 乘方运算的特例：平方根 |
| 7+9**0.5   | 10.0     | 乘方的优先级高         |
| (7+9)**0.5 | 4.0      | 括号改变优先级         |
