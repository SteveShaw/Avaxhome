/*
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
*/

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

