作为机器学习的基础进行快速入门
## 格式化字符串
>python使用""默认都是str数据类型，如需网络传输或当作文件存储则需要转换为字节

b'ABC'中的b为byte的意思，其类型为utf-8
### format()
```python
'Hello, {0}, 成绩提升了 {1:.1f}%'.format('小明', 17.125)
'Hello, 小明, 成绩提升了 17.1%'
```
**or使用c语言同样的占位符**
```
%5d    %5.2f     %.3f
```
### f-string风格(字符串快速格式化)
```python
r = 2.5
s = 3.14 * r ** 2
print(f'The area of a circle with radius {r} is {s:.2f}')
The area of a circle with radius 2.5 is 19.62
```
## list与tuple
### list列表
元素数据类型可不一致
```python
classmates = ['Michael', 'Bob', 'Tracy']  #基本定义


classmates[1]                             #访问Bob
classmates.append('Adam')
classmates.extend(AA)                     #将AA列表放入最后
classmates.insert(1,'Jack')               #插入
A=classmates.pop(i)                       #删除末尾或i索引位置,返回至A
```
### tuple元组
基本与列表相同，可以理解为常列表。
```python
classmates = ('A','B','C') 
classmates.count("A")                     #记录A的数量
```
### set集合
集合是**无序**的**可修改**的，且内容**不可重复**
```python
#定义集合
{}
#空集合
A=set()
```
### dict字典
key与value可以是任意类型的（key不能为字典）
```python
dict1={"111":111,"222":222}
```
必须通过key查找value
- 新增元素
```python
dict["AAA"]=11                            #如果存在AAA就修改，不存在新增
```
- 删除元素
```python
score=dict.pop("AAA")                     #删除元素AAA
```
- 清空元素
```python
dict.clear()
```
- 获取所有key
```python
A=dict.keys()
```
## 模式匹配
### match == switch
复杂模式匹配
```python
age = 15

match age:
    case x if x < 10:
        print(f'< 10 years old: {x}')
    case 10:
        print('10 years old.')
    case 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18:
        print('11~18 years old.')
    case 19:
        print('19 years old.')
    case _:
        print('not sure.')
```
```python
args = ['gcc', 'hello.c', 'world.c']
# args = ['clean']
# args = ['gcc']
match args:
    # 如果仅出现gcc，报错:
    case ['gcc']:
        print('gcc: missing source file(s).')
    # 出现gcc，且至少指定了一个文件:
    case ['gcc', file1, *files]:
        print('gcc compile: ' + file1 + ', ' + ', '.join(files))
    # 仅出现clean:
    case ['clean']:
        print('clean')
    case _:
        print('invalid command.')
```
>第一个`case ['gcc']`表示列表仅有`'gcc'`一个字符串，没有指定文件名，报错；
>第二个`case ['gcc', file1, *files]`表示列表第一个字符串是`'gcc'`，第二个字符串绑定到变量`file1`，后面的任意个字符串绑定到`*files`
>第三个`case ['clean']`表示列表仅有`'clean'`一个字符串；
>最后一个`case _`表示其他所有情况。

## 函数
### 函数的定义
```python
def func():
```
### 默认参数

```python
def power(x,n=2):    #默认参数必须放在最后
```
==注意！==：默认参数是一个变量，指向一个内存地址，注意其定义域！必须指向不变对象！
```python
def add_end(L=[]):
    L.append('END')
    return L
```
其多次运行结果如下
```python
>>> add_end()
['END', 'END']
>>> add_end()
['END', 'END', 'END']
```
### 可变参数
基本定义
```python
def func(*var):       #最好使用args作为可变参数
```
此时变量var传入后自动组装为一个tuple（元组）。那么我们如何将一个list或tuple作为参数传入呢？看下面的例子
```python
nums=[1,2,3]
func(*nums)

func(nums[0],nums[1],nums[2])    #比较复杂
```
### 关键字参数(传入字典类型)
允许任意个含参数名的参数并在函数内部组装为dict
```python
def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
```
调用方法与可变参数类似
```python
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, city=extra['city'], job=extra['job'])
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}

>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, **extra)
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}

```
#### 关键字参数约束
对传入关键字合法性检测的方法
``` python
def person(name, age, **kw):
    if 'city' in kw:
        # 有city参数
        pass
    if 'job' in kw:
        # 有job参数
        pass
    print('name:', name, 'age:', age, 'other:', kw)
```
那么如何命名关键字参数呢？
``` python
def person(name, age, *, city, job):         #只接受 city与job值
    print(name, age, city, job)


>>> person('Jack', 24, city='Beijing', job='Engineer')
Jack 24 Beijing Engineer
```

### 切片
在python中字符串，list，tuple都可以被切片
```python
 L[A:B:2]
```
可以在一条语句中进行多次切片
```python
L[::-1][2:4:]
```
其意义为 \[A,B\)区间内每2个取一个
### 迭代
任何可迭代类型皆可使用for循环迭代
```python
>>> d = {'a': 1, 'b': 2, 'c': 3}
>>> for key in d:
...     print(key)


>>> for ch in 'ABC':
...     print(ch)


```
### 列表生成式
这是对列表类型的快速筛选，可以简短代码
``` python
>>> [x for x in range(1, 11) if x % 2 == 0]
[2, 4, 6, 8, 10]
```
### 生成器
主要是保存一种生成模式，在需要时调用即可生成目标参数。
``` python
>>> g = (x * x for x in range(10))
>>> g
<generator object <genexpr> at 0x1022ef630>
```
如何将一个函数也改造为generator？
``` python
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b    #这里是关键所在
        a, b = b, a + b
        n = n + 1
    return 'done'
```
调用结果如下
```python
>>> for n in fib(6):
...     print(n)
1
1
2
3
5
8
```

### Iterable与Iterator的区别
我们可以使用`isinstance()`方法来检测变量的数据类型
``` python
>>> from collections.abc import Iterable
>>> isinstance([], Iterable)
True
>>> isinstance({}, Iterable)
True
>>> isinstance('abc', Iterable)
True
>>> isinstance((x for x in range(10)), Iterable)
True
>>> isinstance(100, Iterable)
False
```
Iterable代表已知的数据集合，而Iterator代表的是一种数据运算结构，甚至可以无限生成！能for循环的都是==Iterable==而能`next()`的都是==Iterator==

### 模块的导入
```python
from time import sleep             #导入sloop函数
import time                        #导入整个time
from time import *                 #导入time中所有的函数
```
#### \_\_main__变量的作用
在下面的代码中，只有当直接运行文件时__name__的值才会等于__main\_\_这样就可以防止运行main函数时其他模块的测试函数运行
```python
def test(x,y):
    return x+y
if __name__=='__main__':
    print(test(1,2))         #测试本模块函数的运行情况
```
### python包
>**从物理上看**包就是一个文件夹
>**从逻辑上看**包依然是一个模块

文件夹中含有init的文件夹才是包
![[Pasted image 20240515132305.png]]
### 几个常用函数
#### map
将第二个参数==（必须是iterable）==统统带第一个参数的函数中
```python
>>> def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```
### reduce
叠加的map，其必须有两个参数
``` python
>>> from functools import reduce
>>> def fn(x, y):
...     return x * 10 + y
...
>>> reduce(fn, [1, 3, 5, 7, 9])
13579
```
### filter
根据第一参数函数返回bool值判断是否保留
``` python
def is_odd(n):
    return n % 2 == 1

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]
```
注意到`filter()`函数返回的是一个`Iterator`，也就是一个惰性序列，所以要强迫`filter()`完成计算结果，需要用`list()`函数获得所有结果并返回list。
### sorted
基础用法
``` python
>>> sorted([36, 5, -12, 9, -21])
[-21, -12, 5, 9, 36]
```
可以传入参数
``` python
>>> sorted([36, 5, -12, 9, -21], key=abs)
[5, 9, -12, -21, 36]
```

### 匿名函数
可以简化函数编写，避免命名冲突
``` python
lambda x:x*x
```
:前为传入参数，后仅能跟一个表达式并直接返回结果

## 基础文件操作
### 打开文件并读取
```python
open("路径","r",encoding="utf-8")
```
### 文件写入
```python
f=open("","w",encoding="utf-8")
f.write("hello world")
f.flush()                        #从内存中写入文件
f.close()                        #内置flush功能
```
### 文件的追加
```python
f=open("路径","a",encoding="utf-8")  #a表示append，追加模式
#其他内容与写入相同
```

### 异常
```python
try:
	#语句
except:
	#异常处理
else:
	#无异常情况
finally:
	#不管有无都会运行
```

### json的格式

>本质是列表（列表中内嵌字典）or字典，方便不同语言间进行数据交互

```python
import json
data= [{"name":"大山","age":11},{"name":"大锤","age":11},{"name":"赵四","age":11}]
json_tr = json.dumps(data,ensure_ascii=false)           #转换为json
```

#### json.dumps()
转换为json数据
#### json.loads()
json转换为字典or列表

## oop内容
### 类中的方法定义
```python
class Student:
	name=none
	def say_hi(self):
		print(f"hello{name}")           #self是每个方法必须传入的
```
如果需要使用属性，则方法必须包含self参数

### 辅助方法
辅助方法只是一种规范，以下划线开头，常用于重构类内部复杂方法。即将逻辑拆分至多个函数
```python
    def run_game(self):
        """开始游戏主循环"""
        while True:
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         sys.exit()
            self._check_events()
            self.ship.update()
            # self.screen.fill(self.settings.bg_color)
            # self.ship.blitme()
            # pygame.display.flip()   #刷新屏幕
            self._update_screen()  # 重构
            self.clock.tick(60)  # 设置帧率

    def _check_events(self):
        """相应按键与鼠标事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_RIGHT:
                #     self.ship.moving_right = True
                # elif event.key == pygame.K_LEFT:
                #     self.ship.movint_left = True

                self._check_keydown_events()  # 重构

            elif event.type == pygame.KEYUP:
                # if event.key == pygame.K_RIGHT:
                #     self.ship.moving_right = False
                # elif event.key == pygame.K_LEFT:
                #     self.ship.movint_left = False

                self._check_keyup_events()

    def _check_keydown_events(self, event):
        """响应按下操作"""
        if event.key == pygame.K_RIGHT:
            self.ship.moving_right = True
        elif event.key == pygame.K_LEFT:
            self.ship.movint_left = True

    def _check_keyup_events(self, event):
        """相应释放"""
        if event.key == pygame.K_RIGHT:
            self.ship.moving_right = False
        elif event.key == pygame.K_LEFT:
            self.ship.movint_left = False
```
### 构造方法 
```python
class Student:
	name =none
	age=none
	tel=none
	
	def __init__(self,name,age,tel):     #构造方法固定名称
		self.name=name
		self.age=age
		self.tel=tel
```

### 魔术方法
魔术方法是内置的类方法，可以实现很多功能。
#### __str__
```python
class Student:
    def __init__(self,name,age,tel) -> None:
        self.name=name
        self.age=age
        self.tel=tel
    def __str__(self) -> str:
        return f"你好{self.name}"
stu=Student("stone",11,123456)
print(stu.name)
print(stu)            #输出为 你好stone
```

还包括 \_\_lt\_\_ \_\_le\_\_ \_\_eq\_\_ 三个魔术方法，具体功能查一下就行

### 封装
无非是对现实世界的一个抽象过程
![[Pasted image 20240517085436.png]]
#### 私有成员（java中的private）
**为什么需要私有成员？**
![[Pasted image 20240517085547.png]]
在python中私有成员或方法以__开头
```python
 class Phone:
    __current_voltage = None
    def __keep_single_core(self):
        print("only one core running")
phone=Phone()
phone.__keep_single_core()     #运行将报错
```

### 继承
为什么需要继承？继承在现实中的意义是什么？
![[Pasted image 20240517091609.png]]
#### 基本语法
```python
class Phone2020(Phone):        #继承了Phone类
```
#### 多继承
![[Pasted image 20240517092030.png]]
```python
class Phone2020(father1,father2,...):
```

#### 复写
```python
class Phone:
    IMEI="stone"
    def call_by_5g(self):
        print("old")    
class Phone2020(Phone):

    IMEI="xt"
    def call_by_5g():
        super().call_by_5g()    #直接使用方法
        Phone.call_by_5g(self)  #self必须传入
        print("plus new func")
```
直接重新定义同名属性与方法，那么如何调用父类的方法与属性呢？
* **使用super().父类方法/属性**
```python

```
* 使用父类.成员变量访问

### 类型注解
* 为啥需要类型注解呢？
> 说明数据类型，帮助IDE做类型判断

支持类型注解的几种类型
1. 基础容器类型
2. 类对象
3. 基础数据类型
```python
var_1: int = 10               #或者使用 type: int
var_2: str = "xt"             #type: str

class Student:
	pass
stu: Student = Student()

my_list:list = [1,2,3]
my_tuple:tuple = (1,2,3)

my_list: list[int] = [1]
my_tuple: tuple(int,str,float) = (1,"asd",11.1) 
```
**类型注解非强制性，只是起到提示作用**

#### 函数类型注解
```python
def 函数方法名(形参:类型,...) -> int:             #int为返回值的类型注解
```

#### union类型
联合注解，为所给数据确定了一个类型范围
```python
from typing import union

my_list: list[union[str,int]] = [1,2,"xt","stone"]

def func(data:union[int,str])->union[int,str]:
```

### 多态
>多种状态，完成某个行为时，使用不同的对象会得到不同的状态

![[Pasted image 20240517110024.png]]
#### 抽象类
作为顶层设计标准，以便子类做具体实现。同时是对子类的**约束**，要求子类必须实现父类的抽象方法
![[Pasted image 20240517110132.png]]
![[Pasted image 20240517110344.png]]
---
### 闭包
>是一种双层的嵌套函数，内层函数可以访问外层函数变量并返回内层函数对象
```python
def outer(logo):
    def inner(msg):
        print(f"{logo},{msg},{logo}")
    return inner

fn1=outer("heima")
fn1("111")
```
这种写法可以有效地避免logo参数被其他代码修改，如果需要修改logo值则需要在inner的logo前加入nonlocal关键字

### 装饰器
>也是一种闭包，在不破坏原有函数代码下**增加功能**

```python
def outer(func):
    def inner():
        print("start sleep")
        func()
        print("end sleep")
    return inner

@outer        
def sleep():
    import random
    import time
    print("sleeping")
    time.sleep(random.randint(1,5))

fn=outer(sleep)
fn()
#其运行的基本原理如上，时基于闭包实现的
sleep()
#在使用@outer后直接可以省略fn=outer(sleep)
```

---
## 设计模式

### 单例模式
* defnation :保证工具类仅有一个实例，并全局访问
* destination:节省内存，节省创建对象的开销
```python
#str_tools.py
class StrTools:
	pass
str_tool = StrTools()

#main.py
from str_tools import str_tool
str_t1 = str_tool
str_t2 = str_tool
#二者对象相同
```

### 工厂模式
* defination: 创建一个专门创建对象的类
* 易于维护
* 发生修改仅更改工厂类
```python
class Person:
    pass
class Worker(Person):
    pass
class Student(Person):
    pass
class Teacher(Person):
    pass
class PersonFactory:
    def get_person(self,p_type:str):
        if p_type == 'w':
            return Worker()
        elif p_type == 's':
            return Student()
        else:
            return Teacher()
            
pf=PersonFactory()
worker=pf.get_person("w")
student=pf.get_person('s')
teacher=pf.get_person('t')
```