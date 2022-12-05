#### Lesson-4

#####   python中的异常处理

1. 采用关键字 try, except, finally. 其中try 表示有可能出错的代码； except表示如果出错了怎么办，finally表示不管错没错都需要执行的代码。

   ```python
   a = 1.0
   b = float(input())  #请从键盘输入一个数，强制转换为浮点型
   try:
       c = a/b
       print(f"c={c}")
   except:
       print("b is zero...")
   finally:
       print("program end...")
   ```

   在上述示例代码中，如果输入的b值为0，则进入except中，输出“b is zero...", 如果输入的b值为非零值，则计算输出c的值。但是无论输入的b值是多少，都会执行最终的打印输出值"program end..."

