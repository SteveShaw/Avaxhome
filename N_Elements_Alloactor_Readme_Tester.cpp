/*stack_alloc

Howard Hinnant

Introduction

Update 2015-12-13:

Thanks to this stackoverflow question I have revisited this handy little allocator and made several improvements.

Update:

I've updated this article with a new allocator that is fully C++11 conforming. The allocator it replaces was not fully C++03 nor C++11 conforming because copies were not equal.

Sometimes you need a container that is almost always going to hold just a few elements, but it must be prepared for the "large" use case as well. It is often advantageous to have the container allocate off of the local stack up to a given size, in order to avoid a dynamic allocation for the common case.

Programmers often believe they need to write custom containers to get this optimization instead of using the standard containers such as std::vector<T> and std::list<T>. This is certainly doable. However I do not believe this is the best way to go.

Instead I prefer to write a custom allocator which can be used with any of the standard containers. This custom allocator can be written in several different variants and this brief paper outlines only one. But in general, such an allocator refers to a buffer either on the stack, or with static or thread storage duration. So the end result is the same as writing your own container, but you get to reuse the standard container code.

Allocator requirements for C++11 are backwards compatible in that C++98/03 allocators will work in C++11 containers. However C++11 allocators will not work in C++98/03 containers. The allocator demonstrated here is purposefully a C++11 allocator and thus will not work with C++98/03 containers. I chose this presentation because C++11 allocators need not have a lot of distracting boiler-plate (it can be defaulted). However it is a relatively trivial task to add the boiler-plate C++98/03 allocator requirements to this allocator if desired.

This allocator will dole out memory in units of alignment which defaults to alignof(std::max_align_t). This is the same alignment which malloc and new hand out memory. If there is room on the stack supplied buffer (you specify the buffer size), the memory will be allocated on the stack, else it will ask new for the memory.

If memory is tight (when is it not!), you can reduce the alignment requirements via the defaulted third template argument of short_alloc. If you do this, and then the container attempts to allocate memory with alignment requirements greater than you have specified, a compile-time error will result. Thus you can safely experiment with reducing the alignment requirement, without having to know implementation details of the container you're using short_alloc with.

A small vector*/

#include "short_alloc.h"
#include <iostream>
#include <new>
#include <vector>

// Replace new and delete just for the purpose of demonstrating that
//  they are not called.

std::size_t memory = 0;
std::size_t alloc = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    memory += s;
    ++alloc;
    return malloc(s);
}

void  operator delete(void* p) throw()
{
    --alloc;
    free(p);
}

void memuse()
{
    std::cout << "memory = " << memory << '\n';
    std::cout << "alloc = " << alloc << '\n';
}

// Create a vector<T> template with a small buffer of 200 bytes.
//   Note for vector it is possible to reduce the alignment requirements
//   down to alignof(T) because vector doesn't allocate anything but T's.
//   And if we're wrong about that guess, it is a comple-time error, not
//   a run time error.
template <class T, std::size_t BufSize = 200>
using SmallVector = std::vector<T, short_alloc<T, BufSize, alignof(T)>>;

int main()
{
    // Create the stack-based arena from which to allocate
    SmallVector<int>::allocator_type::arena_type a;
    // Create the vector which uses that arena.
    SmallVector<int> v{a};
    // Exercise the vector and note that new/delete are not getting called.
    v.push_back(1);
    memuse();
    v.push_back(2);
    memuse();
    v.push_back(3);
    memuse();
    v.push_back(4);
    memuse();
    // Yes, the correct values are actually in the vector
    for (auto i : v)
        std::cout << i << ' ';
    std::cout << '\n';
}

/*In the above example I've created a custom new/delete just for the purpose of monitoring heap allocations. A vector<int> is created that will allocate up to 200 bytes's before going to the heap.

A small list

This works with list too. Note though that here I have defaulted the alignment requirements because std::list<T> allocates internal nodes that may have larger alignment requirements than T. If desired, I could experiment with reduced alignment requirements here and get a per-instantiation compile-time check for each experiment. Those results won't be portable, but on porting the experiment will either run correctly or fail to compile.*/

template <class T, std::size_t BufSize = 200>
using SmallList = std::list<T, short_alloc<T, BufSize>>;

int main()
{
    SmallList<int>::allocator_type::arena_type a;
    SmallList<int> v{a};
    v.push_back(1);
    memuse();
    v.push_back(2);
    memuse();
    v.push_back(3);
    memuse();
    v.push_back(4);
    memuse();
    for (auto i : v)
        std::cout << i << ' ';
    std::cout << '\n';
}

memory = 0
alloc = 0
memory = 0
alloc = 0
memory = 0
alloc = 0
memory = 0
alloc = 0
1 2 3 4 
/*A small unordered_set

Yes, you can make a small unordered_set too. Here I've experimentally determined on my system that I can reduce the alignment requirement down as low as 8 for many types T. The compiler may tell you that this number is too small on your system.*/

template <class T, std::size_t BufSize = 200>
using SmallSet = std::unordered_set<T, std::hash<T>, std::equal_to<T>,
                      short_alloc<T, BufSize, alignof(T) < 8 ? 8 : alignof(T)>>;

int main()
{
    SmallSet<int>::allocator_type::arena_type a;
    SmallSet<int> v{a};
    ...
/*Next time you're tempted to write your own container, take a moment to explore the possibility of reusing the standard containers. That's what they are there for. You may be surprised at how they can be customized to meet your needs.*/