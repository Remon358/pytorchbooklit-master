Lesson-7 多线程与多进程

电脑中时会有很多单独运行的程序，每个程序有一个独立的进程，而进程之间是相互独立存在的. 

进程想要执行任务就需要依赖线程。换句话说，就是进程中的最小执行单位就是线程，并且一个进程中至少有一个线程。

实现多线程是采用一种并发执行机制.

并发执行机制原理：简单地说就是把一个处理器划分为若干个短的时间片，每个时间片依次轮流地执行处理各个应用程序，由于一个时间片很短，相对于一个应用程序来说，就好像是处理器在为自己单独服务一样，从而达到多个应用程序在同时进行的效果 [3] 。

多线程就是把操作系统中的这种并发执行机制原理运用在一个程序中，把一个程序划分为若干个子任务，多个子任务并发执行，每一个任务就是一个线程。这就是多线程程序 [3] 。

*多线程的优势*:

​	1、使用线程可以把占据时间长的[程序](https://baike.baidu.com/item/程序)中的[任务](https://baike.baidu.com/item/任务)放到[后台](https://baike.baidu.com/item/后台)去处理 [4] 。

​	2、用户界面可以更加吸引人，这样比如用户点击了一个按钮去触发某些事件的处理，可以弹出一个进度条来显示处理的进度 [4] 。

​	3、程序的运行速度可能加快 [4] 。

​	4、在一些等待的[任务](https://baike.baidu.com/item/任务)实现上如用户输入、文件读写和网络收发数据等，线程就比较有用了。在这种[情况](https://baike.baidu.com/item/情况)下可以释放一些珍贵的资源如[内存](https://baike.baidu.com/item/内存)占用等 [4] 。

​	5、[多线程技术](https://baike.baidu.com/item/多线程技术/5764231)在IOS软件开发中也有举足轻重的作用 [4] 

*多线程的缺点:*

​	1、如果有大量的线程，会影响性能，因为[操作系统](https://baike.baidu.com/item/操作系统)需要在它们之间切换 [4] 。

​	2、更多的线程需要更多的[内存](https://baike.baidu.com/item/内存)空间 [4] 。

​	3、线程可能会给[程序](https://baike.baidu.com/item/程序)带来更多“bug”，因此要小心使用 [4] 。

​	4、线程的中止需要考虑其对[程序](https://baike.baidu.com/item/程序)运行的影响 [4] 。

​	5、通常块模型数据是在多个线程间共享的，需要防止线程死锁[情况](https://baike.baidu.com/item/情况)的发生 [4] 



**注: 本部分知识在本节课中并不做重点介绍**

##### 多线程主要API

| 函数名           | 作用                            | 示例                                          |
| ---------------- | ------------------------------- | --------------------------------------------- |
| threading.Thread | 创建一个线程,并绑定一个回调函数 | thread1=threading.Thread(target=functionName) |
| setDaemon        | 设置是否开启后台运行            | thread1.setDaemon(True)                       |
| start            | 开启线程运行                    | thread1.start()                               |



```python
def printHello():
    while True:
        print("This is the hello threading...")
        time.sleep(1)

def printNihao():
    while True:
        print("This is the Nihao threading...")
        time.sleep(2)

if __name__ == '__main__':
    t1 = threading.Thread(target=printHello)
    t2 = threading.Thread(target=printNihao)
    t1.setDaemon(False)
    t2.setDaemon(False)
    t1.start()
    t2.start()
    print("main threading ended...")
```

倒计时游戏， 从60开始进行循环倒计时，每一秒自减一，按下按键‘b'，开始倒计时并显示， 按下按键’t'停止倒计时。

```python
import time
import threading
begin = False
def numCounter():
    num = 60
    global begin
    while True:
        if begin:
            num = num -1
            print(f'num={num}, begin={begin}')
            time.sleep(1)
            if num<=0:
                num = 60
        else:
            pass

def keyDectect():
    # TODO:need to be revised by using the threading lock
    global begin
    while True:
        key = input()
        if key == 'b':
            begin = True
        elif key == 't':
            begin = False
        else:
            print('wrong input..')


if __name__ == '__main__':
    t1 = threading.Thread(target=numCounter)
    t2 = threading.Thread(target=keyDectect)
    t1.setDaemon(False)
    t2.setDaemon(False)
    t1.start()
    t2.start()

```

