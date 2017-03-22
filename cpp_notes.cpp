// 2017-01-30
/////////////////////////////////////////


/*
Variables:

It simply means that once the variable has been initialized, it remains in memory until the end of the program.

static variables exist for the "lifetime" of the translation unit that it's defined in, and:

If it's in a namespace scope (i.e. outside of functions and classes), then it can't be accessed from any other translation unit. 
This is known as "internal linkage". (Dont' do this in headers, it's just a terrible idea)

If it's a variable in a function, it can't be accessed from outside of the function, just like any other local variable. (this is the local they mentioned)

class members have no restricted scope due to static, but can be addressed from the class as well as an instance (like std::string::npos). 
[Note: you can declare static members in a class, but they should usually still be defined in 
a translation unit (cpp file), and as such, there's only one per class]

Before any function in a translation unit is executed (possibly after main began execution), 
the variables with static storage duration in that translation unit will be "constant initialized" 
(to constexpr where possible, or zero otherwise), 
and then non-locals are "dynamically initialized" properly in the order they are defined in the translation unit 
(for things like std::string="HI"; that aren't constexpr). 
Finally, function-local statics are initialized the first time execution "reaches" the line where they are declared. 
They are all destroyed in the reverse order of initialization.

The easiest way to get all this right is to make all static variables that are not constexpr initialized into function static locals, which makes sure all of your statics/globals are initialized properly when you try to use them no matter what, thus preventing the static initialization order fiasco.

T& get_global() {
    static T global = initial_value();
    return global;
}
Be careful, because when the spec says namespace-scope variables have "static storage duration" by default, they mean the "lifetime of the translation unit" bit, but that does not mean it can't be accessed outside of the file.

Functions

Significantly more straightforward, static is often used as a class member function, and only very rarely used for a free-standing function.

A static member function differs from a regular member function in that it can be called without an instance of a class, and since it has no instance, it cannot access non-static members of the class. Static variables are useful when you want to have a function for a class that definitely absolutely does not refer to any instance members, or for managing static member variables.

struct A {
    A() {++A_count;}
    A(const A&) {++A_count;}
    A(A&&) {++A_count;}
    ~A() {--A_count;}

    static int get_count() {return A_count;}
private:
    static int A_count;
}

int main() {
    A var;

    int c0 = var.get_count(); //some compilers give a warning, but it's ok.
    int c1 = A::get_count(); //normal way
}
A static free-function means that the function will not be referred to by any other translation unit, and thus the linker can ignore it entirely. This has a small number of purposes:

Can be used in a cpp file to guarantee that the function is never used from any other file.
Can be put in a header and every file will have it's own copy of the function. Not useful, since inline does pretty much the same thing.
Speeds up link time by reducing work
Can put a function with the same name in each TU, and they can all do different things. For instance, you could put a static void log(const char*) {} in each cpp file, and they could each all log in a different way.
*/


/*
static_cast is the first cast you should attempt to use. 
It does things like implicit conversions between types (such as int to float, or pointer to void*), 
and it can also call explicit conversion functions (or implicit ones). In many cases, explicitly stating static_cast isn't necessary, 
but it's important to note that the T(something) syntax is equivalent to (T)something and should be avoided (more on that later). 
A T(something, something_else) is safe, however, and guaranteed to call the constructor.

static_cast can also cast through inheritance hierarchies. It is unnecessary when casting upwards (towards a base class), 
but when casting downwards it can be used as long as it doesn't cast through  virtual inheritance. 
It does not do checking, however, and it is undefined behavior to static_cast down a hierarchy to a type that isn't actually the type of the object.
*/

/*
dynamic_cast is almost exclusively used for handling polymorphism. 
You can cast a pointer or reference to any polymorphic type to any other class type (a polymorphic type has at least one virtual function, declared or inherited). You can use it for more than just casting downwards -- 
you can cast sideways or even up another chain. 
The dynamic_cast will seek out the desired object and return it if possible. If it can't, it will return nullptr in the case of a pointer, or throw std::bad_cast in the case of a reference.

dynamic_cast has some limitations, though. 
It doesn't work if there are multiple objects of the same type in the inheritance hierarchy (the so-called 'dreaded diamond') 
and you aren't using virtual inheritance. 
It also can only go through public inheritance - it will always fail to travel through protected or private inheritance.
This is rarely an issue, however, as such forms of inheritance are rare.

reinterpret_cast is the most dangerous cast, and should be used very sparingly. 
It turns one type directly into another - such as casting the value from one pointer to another, 
or storing a pointer in an int, or all sorts of other nasty things. 
Largely, the only guarantee you get with reinterpret_cast is that normally if you cast the result back to the original type, 
you will get the exact same value (but not if the intermediate type is smaller than the original type). 
There are a number of conversions that reinterpret_cast cannot do, too. 
It's used primarily for particularly weird conversions and bit manipulations, like turning a raw data stream into actual data, or storing data in the low bits of an aligned pointer.
*/

// 2017-02-21
/////////////////////////////////////////
//https://github.com/ryancoleman/lotsofcoresbook2code/tree/master/Pearls2_Chapter07

//Basically, rvalue references allow you to detect when objects are temporaries 
//and you don't have to preserve their internal state. 
//This allows for much more efficient code where C++03 used to have to copy all the time, 
//in C++0x you can keep re-using the same resources. In addition, rvalue references enable perfect forwarding.
/*
In C++03, you can't distinguish between a copy of a non-mutable lvalue and an rvalue.

std::string s;
std::string another(s);           // calls std::string(const std::string&);
std::string more(std::string(s)); // calls std::string(const std::string&);
In C++0x, this is not the case.

std::string s;
std::string another(s);           // calls std::string(const std::string&);
std::string more(std::string(s)); // calls std::string(std::string&&);
Consider the implementation behind these constructors. 
In the first case, the string has to perform a copy to retain value semantics, 
which involves a new heap allocation. 
However, in the second case, 
we know in advance that the object which was passed in to our constructor 
is immediately due for destruction, and it doesn't have to remain untouched. 
We can effectively just swap the internal pointers 
and not perform any copying at all in this scenario, 
which is substantially more efficient. 
Move semantics benefit any class 
which has expensive or prohibited copying of internally referenced resources.

Consider the case of std::unique_ptr- now that our class can distinguish between temporaries and non-temporaries, 
we can make the move semantics work correctly so that the unique_ptr cannot be copied but can be moved, 
which means that std::unique_ptr can be legally stored in Standard containers, sorted, etc, 
whereas C++03's std::auto_ptr cannot.

Now we consider the other use of rvalue references- perfect forwarding. 
Consider the question of binding a reference to a reference.

std::string s;
std::string& ref = s;
(std::string&)& anotherref = ref; // usually expressed via template
Can't recall what C++03 says about this, 
but in C++0x, the resultant type when dealing with rvalue references is critical. 
An rvalue reference to a type T, where T is a reference type, becomes a reference of type T.

(std::string&)&& ref // ref is std::string&
(const std::string&)&& ref // ref is const std::string&
(std::string&&)&& ref // ref is std::string&&
(const std::string&&)&& ref // ref is const std::string&&
Consider the simplest template function- min and max. In C++03 you have to overload for all four combinations of const and non-const manually. In C++0x it's just one overload. Combined with variadic templates, this enables perfect forwarding.

template<typename A, typename B> auto min(A&& aref, B&& bref) {
    // for example, if you pass a const std::string& as first argument,
    // then A becomes const std::string& and by extension, aref becomes
    // const std::string&, completely maintaining it's type information.
    if (std::forward<A>(aref) < std::forward<B>(bref))
        return std::forward<A>(aref);
    else
        return std::forward<B>(bref);
}
I left off the return type deduction, because I can't recall how it's done offhand, 
but that min can accept any combination of lvalues, rvalues, const lvalues.
*/

/*
vector:

Contiguous memory.
Pre-allocates space for future elements, so extra space required beyond what's necessary for the elements themselves.
Each element only requires the space for the element type itself (no extra pointers).
Can re-allocate memory for the entire vector any time that you add an element.
Insertions at the end are constant, amortized time, but insertions elsewhere are a costly O(n).
Erasures at the end of the vector are constant time, but for the rest it's O(n).
You can randomly access its elements.
Iterators are invalidated if you add or remove elements to or from the vector.
You can easily get at the underlying array if you need an array of the elements.
list:

Non-contiguous memory.
No pre-allocated memory. The memory overhead for the list itself is constant.
Each element requires extra space for the node which holds the element, including pointers to the next and previous elements in the list.
Never has to re-allocate memory for the whole list just because you add an element.
Insertions and erasures are cheap no matter where in the list they occur.
It's cheap to combine lists with splicing.
You cannot randomly access elements, so getting at a particular element in the list can be expensive.
Iterators remain valid even when you add or remove elements from the list.
If you need an array of the elements, you'll have to create a new one and add them all to it, since there is no underlying array.
In general, use vector when you don't care what type of sequential container that you're using, but if you're doing many insertions or erasures to and from anywhere in the container other than the end, you're going to want to use list. Or if you need random access, then you're going to want vector, not list. Other than that, there are naturally instances where you're going to need one or the other based on your application, but in general, those are good guidelines.
*/



Consider the below problems statement.

There 100 different types of caps each having a unique id from 1 to 100. Also, 
there ‘n’ persons each having a collection of variable number of caps. 
One day all of these persons decide to go in a party wearing a cap 
but to look unique they decided that none them will wear the same type of cap. 
So, count the total number of arrangements or ways such that none of them is wearing same type of cap.

Constraints: 1 <= n <= 10 Example:

First line contains value of n, next n lines contain collections 
of all the n persons.
Input: 
3
5 100 1     // Collection of first person.
2           // Collection of second person.
5 100       // Collection of Third person.

Output:
4
Explanation: All valid possible ways are (5, 2, 100),  (100, 2, 5),
            (1, 2, 5) and  (1, 2, 100)
			
What is Bitmasking?
Suppose we have a collection of elements which are numbered from 1 to N. 
If we want to represent a subset of this set than it can be encoded by a sequence of N bits 
(we usually call this sequence a “mask”). 
In our chosen subset the i-th element 
belongs to it if and only if the i-th bit of the mask I set i.e.,
 it equals to 1. For example, 
 the mask 10000101 means that the subset of the set [1… 8] 
 consists of elements 1, 3 and 8. 
 We know that for a set of N elements there are total 2N subsets thus 2N masks are possible, 
 one representing each subset. Each mask is in fact an integer number written in binary notation.

Our main methodology is to assign a value to each mask (and, therefore, to each subset) 
and thus calculate the values for new masks using values of the already computed masks. Usually our main target is to calculate value/solution for the complete set i.e., for mask 11111111. Normally, to find the value for a subset X we remove an element in every possible way and use values for obtained subsets X’1, X’2… ,X’k to compute the value/solution for X. This means that the values for X’i must have been computed already, so we need to establish an ordering in which masks will be considered. It’s easy to see that the natural ordering will do: go over masks in increasing order of corresponding numbers. Also, We sometimes, start with the empty subset X and we add elements in every possible way and use the values of obtained subsets X’1, X’2… ,X’k to compute the value/solution for X.

We mostly use the following notations/operations on masks:
bit(i,mask) – the i-th bit of mask
count(mask) – the number of non-zero bits in mask
first(mask) – the number of the lowest non-zero bit in mask
set(i, mask) – set the ith bit in mask
check(i, mask) – check the ith bit in mask

The idea is to use the fact that there are upto 10 persons. 
So we can use a integer variable as a bitmask to store which person is wearing cap and which is not.

Let i be the current cap number (caps from 1 to i-1 are already 
processed). Let integer variable mask indicates the the persons w
earing and not wearing caps.  If i'th bit is set in mask, then 
i'th person is wearing a cap, else not.

		     // consider the case when ith cap is not included 
                     // in the arrangement
countWays(mask, i) = countWays(mask, i+1) +             
                    
                    // when ith cap is included in the arrangement
                    // so, assign this cap to all possible persons 
                    // one by one and recur for remaining persons.
                    ∑ countWays(mask | (1 << j), i+1)
                       for every person j that can wear cap i 
 
Note that the expression "mask | (1 << j)" sets j'th bit in mask.
And a person can wear cap i if it is there in the person's cap list
provided as input.


1. Can destructor throw exceptions
2. When static_cast will fail
3. forbid object create in stack, but can permit new in heap.



What are the operators that cannot be overloaded in C++?
In C++, following operators can not be overloaded:

. (Member Access or Dot operator)
?: (Ternary or Conditional Operator )
:: (Scope Resolution Operator)
.* (Pointer-to-member Operator )
sizeof (Object size Operator)
typeid (Object type Operator)