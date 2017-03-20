
Phone Screen - someone from infrastructure team 
1, resume 
2, Given an application that a class with big data, 

so creating an instance of this class is slow, how to solve it? 

3, Someone implement hash table and it is slow, why? 

4, use hash table to store data, but there is much more data than the machine's RAM, how to deal with that? 

add one more machine, rehash and reconstruct the hash table 

5, A application involves with multiple machines and it is slow, 

figure out why can use matrix to measure the latency and throughput of each machine 

6, process & thread 7, throughput & latency 

On site: （悲剧地没只撑到中午饭 = =） 

1, write a function to determine if a given long num is power of 4 

Using a random number iterator to crate a iterator that only return multiply of 5

2, given a 2 dimension matrix, set all cubes to 1 or 0 according to the number of neighbors' value of 

the cube at the same time (game of life) 

If we have multiple processor, 

how do u improve that If we have multiple machines, how do we apply 

3, Java find bug, multimap implements a map interface and inherits other abstract map


/*
I have a one hour phone screen. 

Interviewer spent 3 minute measures himself and then told me that there will be 6 knowledge questions 

plus one algorithm implementation. 

1. Describe a design pattern (eg factory pattern, MVC)  

2. Describe a hashtable, How does it work? 

3. How to deal with collisions in a hashtable? 

4. Difference between Throughput and Latency   

5. Difference between a process and a thread and difference between their communication  

6. Given a stream of numbers, how would you calculate the median (used 2 heaps, question is on Leetcode) 

*/
Coding question: You are given an iterator, 
use it to write an iterator that returns multiples of 5. 
(write the next () and hasNext () functions).   
My recruiter sent me An email a couple of hours 
after my interview to tell me that the feedback is positive. They move really fast.  


/*
http://www.themianjing.com/?s=two+sigma

2. iterator.. 那个题..他給你一个可以产生随机数的iterator，
然后你自己写一个iterator…调用他的iterator….
然后你的iterator..要把非5的倍数的数过滤掉。 
完成你自己的hasnext（）、next（） 和remove（）。
hasnext（）里也得调用他給的iterator.. 
必须找着下一个5的倍数了才能返回true…
所以next和hasnext里面都掉用了他給的iterator的next（）了…所以这里得用个boolean 之类的..
记住hasnext里面已经找过了.
下次再调用hasnext（）的时候不要又调用他的iterator的next（）了。
*/

int next()
{
	return val;
}

bool hasNext()
{
	while(iter.hasNext())
	{
		int num = iter.next();
		if(num%5 == 0)
		{
			val  = num;
			return true;
		}
	}
	
	return false;
}

coding
判断罗马数字是不是valid的，如果valid 就转化成普通数字。
lz蛋疼了….不动罗马数字的规则，只知道leetcode上有转化成数字的，但是不会判断valid 否。


http://codereview.stackexchange.com/questions/88644/checking-for-roman-numeral-validity


1.
a. reverse polish notation, 然后引申到怎么做比较generatic, 用户要添加任意的计算字符
提示： design pattern
b. 给你一段写好的代码，就说一个树的结构，每一个node都指向了它的parent,给你一个index，要你删除所有它的子树
2.
a. 两个 independent queue,
每个 queue 都存着 timestamp,只能有 getNext()来取 queue
里面的 timestamp,每个 timestamp 只能被取一次,比较这两个 queue 里的 timestamp,如
果差值<1,print 这两个 timestamp。
例如: Q1 0.2, 1.4, 3.0 Q2 1.0 1.1, 3.5 
output: (0.2, 1.0), (1.4, 1.0), (0.2, 1.1), (1.4, 1.1), (3.0, 3.5) 
提示： 多线程， 一个线程负责queue1, 一个负责queue2 写伪代码就行 
b. 一个网站用户访问特别慢，怎么解决，分析各种可能原因 
3. wildcard matching 先写junit test case, 
再写代码 三轮都是中国人，LZ水平太烂木有珍惜。
感觉two sigma非常高大上，选人标准也很strict，pkg很好~也不是很累。
题目基本都是面经题，会解没用，还会问的很深入。
上午过了下午见了VP过得可能性比较大。 
然后面试条件很好，送很多东西，
给订了三晚上酒店，每天吃的报100刀，LZ旅游的很爽~大家可以去试试。 



Observer pattern :     

Factory method : 

A factory method is a design pattern that allows objects to be created in a polymorphic way, 
so the client doesn’t need to know the exact type of the returned object, 
only the base class that provides the desired interface.

It also helps to hide a complex set of creation steps to instantiate particular classes.

In the factory method design pattern, 
the objective is to hide the complexity and introduce indirection 
when creating an instance of a particular class. 
Instead of asking clients to perform the initialization steps, 
factory methods provide a simple interface that can be called to create the object and return a reference.


Singleton : 

The singleton pattern is used to model situations
in which you know that only one instance of a particular class can validly exist. 
This is a situation that occurs in several applications, and in finance, 
I present the example of a clearing house for options trading.

A singleton is a class that can have at most one active instance. 
The singleton design pattern is used to control access to this single object 
and avoid creating copies of this unique instance.

Observer : 
The Observer pattern facilitates communication 
between a parent class and any dependent child classes, 
allowing changes to the state of the parent class 
to be sent to the dependent child classes. 
We can use this pattern to allow the state changes 
in a class to be sent to all its dependent classes. 
The class relationship is one-to-many between the class and all its dependents.

Another common application of design patterns is in processing financial events
such as trades.
The observer design patterns allows you to decouple the classes that receive trading
transactions from the classes that process the results, which are the observers.

Through the observer design pattern, it is possible to simplify the logic and the amount of code necessary 
to support these common operations, such as the development of a trading ledger, for example.

The observer pattern allows objects to receive notifications for important events 
occurring in the system. This pattern also reduces the coupling between objects in the system, 
since the generator of notification events doesn’t need to know the details of the observers.

Visitor : 
The visitor patter allows a member function of an object to be called 
in response to another dynamic invocation implemented in a separate class. 
The Visitor pattern allows changes or additions to a class’s structure 
without changing the actual class.
The visitor pattern therefore provides the mechanism for dispatching messages 
based on a combination of two objects, instead of the single object-based dispatch that is common with OO languages.

The visitor pattern is popular in financial software designs for two of its features: 
1. It allows us to extend the functionality of classes without modifying their structure or interface.
2. It provides a viable solution to the lack of double dispatch in common object-oriented languages 
like Java and C++, which provide only single dispatch through virtual functions. 
In single dispatch, the compiler chooses the right function at run-time using the dynamic type of calling object.

In order to give us the flexibility of adding new operations to classes, 
the visitor pattern lets us separate their structure from their algorithms.


//Go Design Patterns
//Principles of Quantitative Development

Roman numeral validation

http://stackoverflow.com/questions/267399/how-do-you-match-only-valid-roman-numerals-with-a-regular-expression


Composite Pattern:
1. Composite pattern is used where we need to  treat a group of objects in similar way as a single object.
2. Composite pattern composes objects in term of a tree structure to represent part as well as whole hierarchy. 
3. This type of design pattern comes under structural pattern as this pattern creates a tree structure of group of objects.
4. This pattern creates a class that contains group of its own objects.
	4.1. This class provides ways to modify its group of same objects.
	
The Composite pattern provides a way to 
represent a part-whole relationship 
where the client can ignore the difference 
between individual objects and compositions of objects.  
The key elements of the Composite pattern are the leaf and the composite:

The leaf defines the behavior for primitive objects in the composition.
The composite defines the behavior for components consisting of leaves.

DFS and BFS Graph Clone
http://blog.csdn.net/cinderella_niu/article/details/43232887

先是project介绍，我大概说了20分钟。我挑的是去年暑假实习的project, 
结果说了18分钟左右。面试官说，你这个project好复杂啊，你3个月能做完么。（面试官之前在我实习公司干了2年）。
我说我实习被延长了几周当苦力，
有的时候一周干70小时，还有100小时的时候。 
面试官表示惊讶，说你这个project做的很不错啊，还投入使用了。
面试表示，我的project应该是1个full-time大概半年的量。. 
解释一下lz这个project, 垮了3个大的framework，
前后端+databse+security的full stack，
并且code test coverage 80%+ field testing。 
主要backend 是java和perl, 也写了很多前端， 写了5周spring, 写了2周jqeury, 1周ajax，1周html, css， 1周android等等等。

然后就是开始那些问答题了，顺序我记得不太多。 面前知道几个人答的还不错的，
都挂了电面。有的告诉我follow up问的很细节，答的不好会挂。我就准备的很深。以至于，
面试官问我 ipc, itc是怎么相互通信的。lz说了，我分别知道3到4种通信，然后先开始定义概念，
然后比较每种通信方式的发展历史，和什么情况用哪种，哪写operating system, framework在用哪些。 
然后面试官说，你懂的真多。回答足够了，就打断了lz。
然后还问了design pattern， 我从实践里面分析了各种design pattern。
还问了多线程， deadlock, startvation等，就是简单的回答概念和例子。
问了latency和throughput， 我用例子回答的，我说我做过latency的testing。
还问了procee 和thread的区别，我先范讲概念和区别，然后深入解释，说为什么operating system这么设计，面试官打断，说足够了。
coding是moving average, median 和还有一个忘了，很简单，我回答的hashmap， 说如果信息量大怎么办，我回答的多大机器同时处理。
然后和面试官聊了大概15分钟。
3天左右收到onsite。 
http://www.1point3acres.com/bbs/thread-226426-2-1.html

https://en.wikipedia.org/wiki/Starvation_(computer_science)
https://en.wikipedia.org/wiki/Thread_(computing)#Multithreading
https://en.wikipedia.org/wiki/Inter-process_communication

Starvation

	Starvation describes a situation where a thread is unable to gain regular access to shared resources 
	and is unable to make progress. This happens when shared resources are made unavailable 
	for long periods by "greedy" threads. 
	For example, suppose an object provides a synchronized method that often takes a long time to return. 
	If one thread invokes this method frequently, other threads that also need frequent synchronized 
	access to the same object will often be blocked.

Livelock

	A thread often acts in response to the action of another thread. 
	If the other thread's action is also a response to the action of another thread, 
	then livelock may result. 
	As with deadlock, livelocked threads are unable to make further progress. 
	However, the threads are not blocked — they are simply too busy responding to each other to resume work. 
	This is comparable to two people attempting to pass each other in a corridor: 
	Alphonse moves to his left to let Gaston pass, while Gaston moves to his right to let Alphonse pass. 
	Seeing that they are still blocking each other, Alphone moves to his right, 
	while Gaston moves to his left. They're still blocking each other, so...