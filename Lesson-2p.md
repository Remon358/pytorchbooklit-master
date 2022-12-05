### Lesson-2

python基础知识+ 运用

1. 使用缩进完成语法对齐功能

   ```python
   def Hello():
   	print("hello")
   ```

   ```python
   for i in range(10):
   	print("i = %d" %i)
   ```

2. 程序的注释

   采用 # 进行单行注释，采用‘’‘ This is comment''' 来完成多行注释。（Remark: 在pycharm中可以采用 ***ctrl + /*** 完成对程序的注释，同时对于已经注释的程序，则再次 ***ctrl+/*** 可以取消注释）''' 多行注释'''

3. 命名法则

   python对程序中的小写敏感。

   3.1 变量名：一般采用驼峰命名法。首字母小写，后面的单词首字母大写，例如 "printFunction"

   3.2 类名建议首字母大写

   3.3 Python中33个保留字，不能重名：def , class, if , for , while, not , and , pass 等

   

5. python中的变量属性
   
   5.1. python中变量可以直接进行赋值  (str int float bool  type(valueName)) True False
   
   (C++ true false)
   
      ```python
      a_float = 1.2
      print(type(a_float))
      b_str = "hello"
      print(type(b_str))
      print(type(person))
      a_int = 1
      print(type(a_int))
      ```
   
   5.2. 不同数据类型可以默认转换 强制类型转换 str(valueName) int(a_str) float(a_float)
   
      ```python
      print(a_int + a_float)
      print(str(a_int + a_float) + a_str)
      ```
   
   5.3. python中含有bool型数据类型，即 True, False 判断语句中，True 非零值， False 
   
   if (a)
   
6. python中的标准输入输出

   1. print("hello") 默认含有换行回车操作，若要取消换行，需要print("hello", end="")

   2. print("a = %f \t b=%f" %(a,b))

   3. print(f"a = {a}, b={b}")

   4. ```python
      name = input()
      print("hello"+"\t"+name)
      ```

      注：输入的一定是字符串
      
      ###### python程序编程
      
      顺序，分支，循环三种程序结构
      
      1. 在C语音的程序编写中，程序的入口为
      
         ```c++
         int main(int argc, char** argv)
         {
         	/*
         	main program to be programmed
         	*/
             return 1;
         }
         ```
      
      python采用脚本式编程，程序自上而下顺序执行。遇到函数以及类时，跳过不执行。一般我们采用如下语句申明main函数
      
      2. 程序的执行方式：顺序，分支，循环，下面分别予以介绍
      
      3. 分支程序， 包含有 if else,  注：逻辑运算符 not  and or (if elif)
      
         ```python
         if Ture:
         	## program
         else:
             ## program
             
         if A and B(A or B):  ## if not A:
             ## program
         else:
             ## program
         ```
      
         注：添加编程训练，从键盘输入两个数，判断大小，并输出显示较大的数。
      
         输入第一个数：
      
         输入第二个数：
      
         如果 第一个数大于第二个数:
      
         ​			输出第一个数
      
         再如果 第一个数等于第二个数：
      
         ​			输入有误
      
         否则 
      
         ​			输出第二个数
      
         
      
      4. 循环结构，包含有whie , for。若想停止循环，可以选择break;  进行下一次分支语句，continue; pass语句表示什么都不执行，进行占位
      
         C/C++语言中 
         
         for( int i=0 ; i< 6; i++)
         
         {
         
         ​			// program...
         
         }
         
         int i=0
         
         while(i<10)
         
         {
         
         ​	i++;
         
         ​		//program;
         
         }
         
         ```python
         for i in range(10):
         	print(i)
         ```
         
         以及对应的while循环
         
         ```python
         while True:
         	print("hello")
         	time.sleep(1)
             
         ```
         python中的占位符。
         
         
         
         
         
         常用的range循环结构，重点掌握其中的前四个
   
   | 序列                    | Python                                         | C/C++/Java              |
   | ----------------------- | ---------------------------------------------- | ----------------------- |
   | **[0,1,2,…,9]**         | range(10)                                      | for (i=0;  i<=9; i++)   |
   | **[0,1,2,…,n-1]**       | range(n)                                       | for (i=0;  i<n ; i++)   |
   | **[n-1, …, 1, 0]**      | range(n-1, -1, -1)                             | for (i=n-1; i>=0; i--)  |
   | **[1,2,…,n]**           | range(1, n+1)                                  | for (i=1;  i<=n; i++)   |
   | **[1,2,3,4 ……**         | import itertoolsfor i in itertools.count(1):   | for (i=1;    ; i++)     |
   | **小于****n****的奇数** | range(1, n, 2)                                 | for (i=1;  i<n ; i=i+2) |
   | 所有奇数                | import itertoolsfor i in itertools.count(1,2): | for (i=1;    ; i=i+2)   |
   
   猜数字：
   
   随机产生一个1～100的正数， 从键盘输入一个正整数，程序判断告诉：输入数据大了，还是小了。最终找到这个数字。
   
   ```python
   import random
   #生成1~100任意的一个随机数
   randNum=random.randint(1,100)
   while True:
       number=float(input("请输入一个100以内的数:")) #键盘输入一个100以内的数
       if(number>randNum): #如果大于输出 "输入值偏大"
           print("输入值偏大")
       elif(number<randNum):#如果小于输出 "输入值偏小"
           print("输入值偏小")
       else:          		 #相等输出 “数值正确”
           print("数值正确")
           break
   ```

​       

输出1--100之间能够被3整除的整数：

```
for num in range(100):
    if num%3==0:
        print(num,end='\t')
```

