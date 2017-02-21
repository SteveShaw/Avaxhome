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

