## Lesson-7 多线程与多进程

电脑中时会有很多单独运行的程序，每个程序有一个独立的进程，而进程之间是相互独立存在的. 

进程想要执行任务就需要依赖线程。换句话说，就是进程中的最小执行单位就是线程，并且一个进程中至少有一个线程。

实现多线程是采用一种并发执行机制.

并发执行机制原理：简单地说就是把一个处理器划分为若干个短的时间片，每个时间片依次轮流地执行处理各个应用程序，由于一个时间片很短，相对于一个应用程序来说，就好像是处理器在为自己单独服务一样，从而达到多个应用程序在同时进行的效果 [3] 。

多线程就是把操作系统中的这种并发执行机制原理运用在一个程序中，把一个程序划分为若干个子任务，多个子任务并发执行，每一个任务就是一个线程。这就是多线程程序 [3] 。0

### *多线程的优势*:

​	1、使用线程可以把占据时间长的[程序](https://baike.baidu.com/item/程序)中的[任务](https://baike.baidu.com/item/任务)放到[后台](https://baike.baidu.com/item/后台)去处理 [4] 。

​	2、用户界面可以更加吸引人，这样比如用户点击了一个按钮去触发某些事件的处理，可以弹出一个进度条来显示处理的进度 [4] 。

​	3、程序的运行速度可能加快 [4] 。

​	4、在一些等待的[任务](https://baike.baidu.com/item/任务)实现上如用户输入、文件读写和网络收发数据等，线程就比较有用了。在这种[情况](https://baike.baidu.com/item/情况)下可以释放一些珍贵的资源如[内存](https://baike.baidu.com/item/内存)占用等 [4] 。

​	5、[多线程技术](https://baike.baidu.com/item/多线程技术/5764231)在IOS软件开发中也有举足轻重的作用 [4] 

### *多线程的缺点:*

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

python中提供两个标准库thread和threading用于对线程的支持，python3中已放弃对前者的支持，后者是一种更高层次封装的线程库，接下来均以后者为例。

#### 创建线程

python中有两种方式实现线程：

1. 实例化一个threading.Thread的对象，并传入一个初始化函数对象（initial function )作为线程执行的入口；
2. 继承threading.Thread，并重写run函数；

##### 方式1：创建threading.Thread对象

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

##### 方式2：继承threading.Thread，并重写run

```python
import threading
import time

class CustomThread(threading.Thread):
    def __init__(self, thread_name):
        # step 1: call base __init__ function
        super(CustomThread, self).__init__(name=thread_name)
        self._tname = thread_name

    def run(self):
        # step 2: overide run function
        time.sleep(0.5)
        print("This is %s running...." % self._tname)

if __name__ == "__main__":
    t1 = CustomThread("thread 1")
    t2 = CustomThread("thread 2")
    t1.setDaemon(False)
    t2.setDaemon(False)
    t1.start()
    t2.start()
    print("This is main function")
```

上面两种方法本质上都是直接或者间接使用threading.Thread类

下面是threading.Thread提供的线程对象方法和属性：

> - start()：创建线程后通过start启动线程，等待CPU调度，为run函数执行做准备；
> - run()：线程开始执行的入口函数，函数体中会调用用户编写的target函数，或者执行被重载的run函数；
> - join([timeout])：阻塞挂起调用该函数的线程，直到被调用线程执行完成或超时。通常会在主线程中调用该方法，等待其他线程执行完成。
> - name、getName()&setName()：线程名称相关的操作；
> - ident：整数类型的线程标识符，线程开始执行前（调用start之前）为None；
> - isAlive()、is_alive()：start函数执行之后到run函数执行完之前都为True；
> - daemon、isDaemon()&setDaemon()：守护线程相关；

#### **GIL锁**

​        提起python多线程就不得不提一下GIL(Global Interpreter Lock 全局解释器锁)，这是目前占统治地位的python解释器CPython中为了保证数据安全所实现的一种锁。不管进程中有多少线程，只有拿到了GIL锁的线程才可以在CPU上运行，即时是多核处理器。对一个进程而言，不管有多少线程，任一时刻，只会有一个线程在执行。对于CPU密集型的线程，其效率不仅仅不高，反而有可能比较低。python多线程比较适用于IO密集型的程序。对于的确需要并行运行的程序，可以考虑多进程。多线程对锁的争夺，CPU对线程的调度，线程之间的切换等均会有时间开销。

#### 线程与进程区别

下面简单的比较一下线程与进程

- 进程是资源分配的基本单位，线程是CPU执行和调度的基本单位；

- 通信/同步方式：

- 进程：

  - 通信方式：管道，FIFO，消息队列，信号，共享内存，socket，stream流；

  - 同步方式：PV信号量，管程

    线程：

  - 同步方式：互斥锁，递归锁，条件变量，信号量
  - 通信方式：位于同一进程的线程共享进程资源，因此线程间没有类似于进程间用于数据传递的通信方式，线程间的通信主要是用于线程同步。

- CPU上真正执行的是线程，线程比进程轻量，其切换和调度代价比进程要小；

- 线程间对于共享的进程数据需要考虑线程安全问题，由于进程之间是隔离的，拥有独立的内存空间资源，相对比较安全，只能通过上面列出的IPC(Inter-Process Communication)进行数据传输；

- 系统有一个个进程组成，每个进程包含代码段、数据段、堆空间和栈空间，以及操作系统共享部分 ，有等待，就绪和运行三种状态；

- 一个进程可以包含多个线程，线程之间共享进程的资源（文件描述符、全局变量、堆空间等），寄存器变量和栈空间等是线程私有的；

- 操作系统中一个进程挂掉不会影响其他进程，如果一个进程中的某个线程挂掉而且OS对线程的支持是多对一模型，那么会导致当前进程挂掉；

- 如果CPU和系统支持多线程与多进程，多个进程并行执行的同时，每个进程中的线程也可以并行执行，这样才能最大限度的榨取硬件的性能；



实例：倒计时游戏， 从60开始进行循环倒计时，每一秒自减一，按下按键‘b'，开始倒计时并显示， 按下按键’t'停止倒计时。

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

