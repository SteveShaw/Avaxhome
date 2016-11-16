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

