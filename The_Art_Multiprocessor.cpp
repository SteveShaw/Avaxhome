/*
Basics:
1. threads, cpus.
	1.1 A software thread is a stream of instructions that the processor executes
	1.2 A hardware thread is the hardware resources that execute a single software thread. A multicore processor has multiple hardware threads—these are the virtual CPUs. Other sources might refer to hardware threads as strands. Each hardware thread can support a software thread.
	1.3 context switching: When there are more active software threads than there are hardware threads to run them, the operating system will share the "virtual CPUs" between the software threads. Each thread will run for a short period of time, and then the operating system will swap that thread for another thread that is ready to work. The act of moving a thread onto or off the virtual CPU is called a <context switch>.
	1.4 Affinity: used to mean keeps threads local to where they were previously executing. When a thread wakes up, it is common that this thread is unable to be scheduled to exactly the same virtual CPU that was running it earlier. However, if he thread can be scheduled to a virtual CPU that shares the same locality group, the disturbance will be less than running it on a virtual processor that shares nothing with the original virtual processor.
	1.5 Branch Predict: To keep fetching instructions, the processor needs to guess the next instruction that will be executed. Most of the time this will be the instruction at the following address in memory. However, a branch instruction might change the address where the instruction is to be fetched from—but the processor will know this only once all the conditions that the branch depends on have been resolved and once the actual branch instruction has been executed.
		1.5.1 The usual approach to dealing with this is to predict whether branches are taken and then to start fetching instructions from the predicted address. 
			*If the processor predicts correctly, then there is no interruption to the instruction steam—and no cost to the branch.
			*If the processor predicts incorrectly, all the instructions executed after the branch need to be flushed, and the correct instruction stream needs to be fetched from memory. These are called branch mispredictions, and their cost is proportional to the length of the pipeline. The longer the pipeline, the longer it takes to get the correct instructions through the pipeline in the event of a mispredicted branch.
	1.6 Pipeline: More recent processors have four or more pipelines. Each pipeline is specialized to handle a particular type of instruction. It is typical to have a memory pipeline that handles loads and stores, an integer pipeline that handles integer computations (integer addition, shifts, comparison, and so on), a floating-point pipeline (to handle floating-point computation), and a branch pipeline (for branch or call instructions). 
	
2. cache line: The number of lines in a cache can be calculated by dividing one by the other. For example, a 4KB cache that has a cache line size of 64 bytes will hold 64 lines.
	1.1 In a simple cache line system, one cache line only maps to a location in the memory. So, a cache line of a 4KB cache will map each line at every 4KB interval in memory.
	2.2 a program that accessed memory in 4KB strides would end up just using a single entry in the cache and could suffer from poor performance if it needed to simultaneously use multiple cache lines.
	2.3 The way around this problem is to increase the associativity of the cache—that is, make it possible for a single cache line to map into more positions in the cache and therefore reduce the possibility of there being a conflict in the cache. In a two-way associative cache, each cache line can map into one of two locations. Doubling the number of potential locations for each cache line means that the interval between lines in memory that map onto the same cache line is halved, but overall this change will result in more effective utilization of the cache and a reduction in the number of cache misses. (Because the cache line could be used to get data from two locations not one location in the memory)
	2.4 If multiple threads share a level of cache, cache need to have high associativity. For example, if two threads share a cache that map to one location in memory only, and they access the same virtual memory address, then they will be attempting to use the same line in the cache, and only one will succeed. But this success is short-lived since the other copy will immediately replace this line of data with the line of the data they need.
3. Virtual memory: 
	3.1 When an address was accessed that was not in physical memory, the operating system would write a page containing data that had not been used in a while to disk and then fetch the data that was needed into the physical memory that had just been freed.
	3.2 Allows to run multiple applications, because the same address can be reused.
	
4. TLB: The critical step in using virtual memory is the translation of a virtual address, as used by an application, into a physical address, as used by the processor, to fetch the data from memory. This step is achieved using a part of the processor called the translation look-aside buffer (TLB)
	4.1  there will be one TLB for translating the address of instructions (the instruction TLB or ITLB) and a second TLB for translating the address of data (the data TLB, or DTLB).
	4.2 Each TLB is a list of the virtual address range and corresponding physical address range of each page in memory. 
	4.3 Translaton:
		* first splits the address into a virtual page (the high-order bits) and an offset from the start of that page (the low-order bits). 
		* looks up the address of this virtual page in the list of translations held in the TLB.
		* gets the physical address of the page and adds the offset to this to get the address of the data in physical memory
		* if the translation is not in TLB (TLB miss), it will be fetched from page table. 
	4.4 Capacity miss: the amount of memory being mapped by the application is greater than the amount of memory that can be mapped by the TLB.
	4.5 Conflict miss: multiple pages in memory map into the same TLB entry; adding a new mapping causes the old mapping to be evicted from the TLB.
	4.6 To reduce the miss rate, one possible way is to enlarge the size of the page. Larger page sizes means that fewer TLB entries are needed to map the virtual address space, and then means less chance of them being knocked out of the TLB when a new entry is loaded.
		4.6.1 For example, mapping a 1GB address space with 4MB pages requires 256 entries, whereas mapping the same memory with 8KB pages would require 131,072. It might be possible for 256 entries to fit into a TLB, but 131,072 would not.
		
5. cache coherence: In a system with multiprocessors, not only can data be held in memory, but it can also be held in the caches of one of the other processors. For code to execute correctly, there should be only a single up-to-date version of each item of data. This feature is called <cache coherence>.
	5.1 The common approach to providing cache coherence is called <snooping>. Each processor broadcasts the address that it wants to either read or write. The other processors watch for these broadcasts. When they see that the address of data they hold can take one of two actions, they can return the data if the other processor wants to read the data and they have the most recent copy. If the other processor wants to store a new value for the data, they can invalidate their copy.
	
6. Memory layout and latency:
	6.1 A multiprocessor system could be configured with all the memory attached to one processor or the memory evenly shared between the processors.
	6.2 For systems where memory is attached to multiple processors, there are two options for reducing the performance impact. 
		6.2.1 One approach is to interleave memory, often at a cache line boundary, so that for most applications, half the memory accesses will see the short memory access, and half will see the long memory access; so, on average, applications will record memory latency that is the average of the two extremes. This approach typifies what is known as a uniform memory architecture (UMA), where all the processors see the same memory latency.
		6.2.2 The other approach is to accept that different regions of memory will have different access costs for the processors in a system and then to make the operating system aware of this hardware characteristic. With operating system support, this can lead to applications usually seeing the lower memory cost of accessing local memory. A system with this architecture is referred to as having cache coherent nonuniform memory architecture (ccNUMA). For the operating system to manage ccNUMA memory characteristics effectively, it has to do a number of things. 
			* First, it needs to be aware of the locality structure of the system so that for each processor it is able to allocate memory with low access latencies. T
			* The second challenge is that once a process has been run on a particular processor, the operating system needs to keep scheduling that process to that processor. If the operating system fails to achieve this second requirement, then all the locally allocated memory will become remote memory when the process gets migrated.
			
7. Assembly code
	7.1  x86 in 32-bit mode has a stack-based calling convention. This means that all the parameters that are passed into a function are stored onto the stack, and then the first thing that the function does is to retrieve these stored parameters. Hence, the first thing that the code does is to load the value of the pointer from the stack
	7.2 SPARC is a reduced instruction set computer (RISC), meaning it has a small number of simple instructions, and all operations must be made up from these simple building blocks. x86 is a complex instruction set computer (CISC), so it has instructions that perform more complex operations. 
	7.3 The x86 code used [esp] as the stack pointer, which points to the region of memory where the parameters to the function call are held. In contrast, the SPARC code passed the parameters to functions in registers. The method of passing parameters is called the calling convention, and it is part of the application binary interface (ABI) for the platform.
	7.4 SPARC actually has 32 general-purpose registers, whereas the x86 processor has eight general-purpose registers. Some of these general-purpose registers have special functions. The SPARC processor ends up with about 24 registers available for an application to use, while in 32-bit mode the x86 processor has only six. However, because of its CISC instruction set, the x86 processor does not need to use registers to hold values that are only transiently needed—in the example, the current value of the variable in memory was not even loaded into a register. So although the x86 processor has many fewer registers, it is possible to write code so that this does not cause an issue.
	7.5 If there are insufficient registers available, a register has to be freed by storing its contents to memory and then reloading them later. This is called <register spilling and filling>, and it takes both additional instructions and uses space in the caches.

8. 32-bit vs 64-bit: 64-bit instruction improved performance by eliminating or reducing two problems.
	8.1 stack-based calling convention: This convention leads to the code using lots of store and load instructions to pass parameters into functions. In 32-bit code when a function is called, all the parameters to that function needed to be stored onto the stack. The first action that the function takes is to load those parameters back off the stack and into registers. In 64-bit code, the parameters are kept in registers, avoiding all the load and store operations.
	8.2 64-bit transition increases the number of general-purpose registers from about 6 in 32-bit code to about 14 in 64-bit code. Increasing the number of registers reduces the number of register spills and fills.
	8.3 perform loss is caused by 64-bit address space.
	
9. Memory ordering: Memory ordering is the order in which memory operations are visible to the other processors in the system. 
	9.1 The memory ordering instructions are given the name memory barriers (membar) on SPARC and memory fences (mfence) on x86. These instructions stop memory operations from becoming visible outside the thread in the wrong order. 
		Example: Suppose a variable <count> protected by a lock and will be incremented. Assume the lock is acquired with value 1 contained. <count> is incremented and then lock is released by storing value 0. 
		
			LOAD [&count], %A
			INC %A
			STORE %A, [&count]
			STORE 0, [&lock]
			
		As soon as the value 0 is stored into the variable <lock>, then another thread can acquire the lock and modify the variable <count>. For performance reasons, some processors implement a weak ordering of memory operations, meaning that stores can be moved past other stores or loads can be moved past other loads. If the previous code is run on a machine with a weaker store ordering, then the code at execution time could look like the code shown as below:
		
			LOAD [&count], %A
			INC %A
			STORE 0, [&lock]
			STORE %A, [&count]
			
		At runtime, the processor makes the store to the lock to become visible to the rest of the system before the store to the variable <count>. Hence, the lock is released before the new value of count is visible. Another processor could see that the lock was free and load up the old value of count rather than the new value.
		
		The solution is to place a memory barrier between the two stores to tell the processor not to reorder them. In the following code (SPARC related), the membar instruction ensures that all previous store operations have completed before the next store instruction is executed.
		
				LOAD [&count], %A
				INC %A
				STORE %A, [&count]
				MEMBAR #store, #store
				STORE 0, [&lock]
				
		A similar issue could occur when the lock is acquired. The load that fetches the value of <count> might be executed before the store that sets the lock to be acquired. In such a situation, it would be possible for another processor to modify the value of <count> between the time that the value was retrieved from memory and the point at which the lock was acquired.
		
10. Differences Between Processes and Threads: Threads and processes are ways of getting multiple streams of instructions to coordinate.

	10.1 The advantage of processes is that each process is isolated—if one process dies, then it can have no impact on other running processes. The disadvantages of multiple processes is that each process requires its own TLB entries, which increases the TLB and cache miss rates. The other disadvantage of using multiple processes is that sharing data between processes requires explicit control, which can be a costly operation.
	
	10.2 Multiple threads have advantages in low costs of sharing data between threads—one thread can store an item of data to memory, and that data becomes immediately visible to all the other threads in that process. The other advantage to sharing is that all threads share the same TLB and cache entries, so multithreaded applications can end up with lower cache miss rates. The disadvantage is that one thread failing will probably cause the entire application to fail.
	
	10.3 


*/

Chapter 007
1. Test-And-Set Locks: The testAndSet() operation, with consensus number two, was the principal synchronization instruction provided by many early multiprocessor architectures. This instruction operates on a single memory word (or byte). That word holds a binary value, true or false. The testAndSet() instruction atomically stores true in the word, and returns that word’s previous value, swapping the value true for the word’s current value. At first glance, this instruction seems ideal for implementing a spin lock. The lock is free when the word’s value is false, and busy when it is true. The lock() method repeatedly applies testAndSet() to the location until that instruction returns false (i.e., until the lock is free). The unlock() method simply writes the value false to it.
	1.1 The first method: directly uses std::atomic_flag's test_and_set.
		1.1.2 std::atomic_flag: only in one of two states: "set" or "clear". Must be initialized with ATOMIC_FLAG_INIT. objects of this type always starts "clear". only three things can be applied
			* destroy : destructor
			* clear: clear() member function, which is a <store> operation, cannot have memory_order_acquire or memory_order_acq_rel.
			* set and query the previous value: test_and_set() member function, which is a <read-modify-write> operation.
		1.1.3 std::atomic_flag cannot be copy-constructed or assign one to another. Reason:
			* A single operation on two distinct objects cannot be atomic. Copy-contruct or copy assignment involves two separate operations on two distinct objects: read the value from one object and then write to the other. The combination cannot be atomic.
		
							class TASLock
							{
								public:
									TASLock()
										:state(ATOMIC_FLAG_INIT)
									{
										
									}
									
									void lock()
									{
										//loop on test_and_set() until the old value is false,
										//indicating that current thread will quit the loop. 
										while( state.test_and_set( memory_order_acquire ) )
										{
											
										}
									}
									
									void unlock()
									{
										state.clear(std::memory_order_release);
									}
									
								private:
									std::atomic_flag state;
							}
		1.1.4 Analysis: How does the lock work?
			Assume there are two threads: t1 and t2. They share one TASLock object: tas_lock. Both use this lock to protect a critical section as follows:
			
			t1: 													t2:
			
			tas_lock.lock()												tas_lock.lock()
			{															{
				//t1's critical section										//t2's critical section
			}															}
			tas_lock.unlock()											tas_lock.unlock()
			
			Without lossing generalsity, assume t1 is the first thread to enter the lock() function. In this case, t1 will change tas_lock's state from "clear" to "set", and the loop in the function lock() is stopped. Thus, t1 goes to its own critical section. For t2, state now is set to set, thus state.test_and_set returns a true value, t2 will continue to loop until t1 set the state to clear by calling unlock() function. Simply put, if any thread loop inside the lock() function, it cannot go to its critical section.
	1.2 The alternative way is to use std::atomic<bool> since std::atomic_flag does not have a member function to get state only
		1.2.1 std::atomic<bool> supports:
			* writes : store()
			* replace the stored value with a new one and returns the original value: exchange() (read-modify-write)
			* reads : load()
		1.2.2 TASLock using std::atomic<bool>
		
				class TASLock
				{
					std::atomic<bool> state;
					
					public:
						TASLock() : state(false) {}
						
						void lock()
						{
							while( true )
							{
								while( state.load( memory_order_acquire ) )
								{
									
								}
								
								if( !state.exchange( true, memory_order_acq_rel ) )
								{
									return;
								}
							}
						}
						
						void unlock()
						{
							state.store(false, memory_order_release);
						}
				}
		1.2.3 How does this lock work?
			Assume there are two threads: t1 and t2. They share one TASLock object: tas_lock. Both use this lock to protect a critical section as follows:
			
			t1: 													t2:
			
			tas_lock.lock()												tas_lock.lock()
			{															{
				//t1's critical section										//t2's critical section
			}															}
			tas_lock.unlock()											tas_lock.unlock()
			
			Assume t1 is the first thread to enter the lock() function. In this case, state.load() return false and then state.exchange() also return false. Thus, t1 goes to its own critical section. For t2, state now is set to set, thus state.load returns a true value, t2 will continue to loop until the state is read as false. Then t2 will call state.exchange() to try to acquire the lock (set to state to true). If t2 successfully acquire the lock, state.exchange returns false which means between t2's calling state.load() and state.exchange(), t1 does not acquire the lock or t1 cannot acquire the lock successfully. Thus t2 go into its own critical section. 
	1.3 The performance of these two implementations: The second one is better than the first one.
		1.3.1 Multiprocessor uses a shared bus as communication channel between processors. At one time, only one processor can broadcast on the bus.
		1.3.2 Each processor has a cache. 
		1.3.3 cache is first read when a processor reads from an address in memory.
			* if the contents are present in its cache, it is a cache hit and the load is immediately.
			* if not, it is a cache miss and must find the data either in the memory or in another processor's cache.
				-->Then the processor broadcast the address on the bus. The other processors receive the broadcast and check if they have the data. If one processor has that address in its cache, it broadcast the address and value to the bus.
				--> if no processor has that addresss, the memory itself will respnd with the vaue at that address.
		1.3.4 For the first implementation:
			1) Each test_and_set() call is a broadcast on the bus. All threads are delayed by their test_and_set() calls even those not waiting for the lock.
			2) One processor's test_and_set() call forces other processors to discard their own cached copies of the <state>, so every thread spinning on the loop encounters a cache miss almost every time. These threads must use the bus to fetch the new, but unchanged value. 
			3) When the threading holding the lock tries to release it, it may also be delayed because the bus is occupied by the spinning threads. 
		1.3.5 For the second implementation:
			1) Assume thread A and B are contenting to acquire the lock. If the lock is held by thread A, the first time thread B reads the state, it takes a cache miss. B is forced to block and the value is loaded into B's cache. B repeatedly reads the state's value, but hits in the cache every time while A is holding the lock. Thus thread B produces no bus traffic and does not slow down other threads' memory access. 
			2) If A release the lock, there will be a problem. A releases the lock by writing false to the state, which immediately invalidates B (the spinning thread)'s cached copies. Each one take a cache miss, rereads the new value, and they all call exchange() to acquire the lock. The first thread acquires the lock will invalidate the others, who must then reread the value, causing a storm of bus traffic. 
		1.3.6 Local spinning: threads repeatedly reread cached values instead of repeatedly using the bus, is an important principle in the design of spin locks.

