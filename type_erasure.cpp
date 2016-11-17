
/*
My understanding: type-erasure can be thought as a type that represent a series of types. By this way, we can erase those represented types and replaced with the new type. For example: std::function type can erase all function-like types.
*/

/*From https://akrzemi1.wordpress.com/ */


/*OO-based type erasure

We could attempt to solve some of these problems by introducing an OO-style type hierarchy. We can enforce every user of our code/library to derive their types from our super base class. This super base class we would probably call Object (as it is usually the case for OO-style designs) and require that all derived classes implement a member function lessThan(Object*) and probably a couple of other functions. Our sorting function could then be implemented as something like this:
*/
class MyInteger : public Object
{
  // ...
  public: int lessThan (const Object* rhs) const override
  {
    auto irhs = dynamic_cast<MyInteger const*>(rhs);
 
    if (irhs == nullptr) {
      throw SomeException{};
    }
 
    return irhs->getInt() - this->getInt();
  }
};
 
Object* OO_search (Object* val, Object** base, size_t size);

/*
Now we have solved one problem. In case a compared object has an incompatible type, an exception will be thrown at run-time.

This is another “basic” type of type erasure known probably for a long time to most C++ programmers: inheriting from a common base class and passing around a pointer/reference to the interface. Unlike with void* we are now guaranteed that whatever our type really is (MyInt or MyString, etc.) certain operations on our pointer (like lessThan) will always work; that is, never render a UB. Our type has been erased, but our type’s interface has not! We no longer have to worry how we retrieve the original type of our object, because nearly anything we will ever need of it can be achieved via the interface.

But now our type is tied to the sorting function. We can fix that by passing sorting function as a separate argument:
*/
int i_bigger (const Object* a, const Object* b)
{
  auto ia = dynamic_cast<MyInteger const*>(a);
  auto ib = dynamic_cast<MyInteger const*>(b);
 
  if (ia == nullptr || ib == nullptr) {
    throw SomeException{};
  }
 
  return ib->getInt() - ia->getInt();
}
 
Object* OO_search (Object** base, 
                   size_t size,
                   int (*comp)(const Object*, const Object*));
/*
and if we are really keen on OO, instead of a function pointer we could introduce a yet another interface:
*/

Object* OO_search (Object* val, Object** base, size_t size, Ordering & compar);

/*
There are a couple of problems with this approach though. First, we can no longer sort arrays of ints: ints can never inherit from Object. As you can see we had to wrap them in a class that would have otherwise been unnecessary. Although it only stores an int, its size is now greater than than of an int, because wee need a room for a pointer to vtable. In fact, we now impose a broader additional requirement: all user’s types must now inherit from Object or else they won’t work with our component. This puts a burden on users. What if the user wanted to use a comparison function from a third party vendor? Such comparator couldn’t possibly derive from an interface we invented. And we may not be able to change third party code. Also, imagine that this is a library that you are writing and you require that all users’ types derive from MyLib::Object; but the user also wants to use another library with a similar philosophy, which in turn forces him to also derive all his types from OtherLib::Object. Also, there are a lot of function objects out there, taking two ints and returning a bool (e.g., std::greater<int>) which we now will not be able to use due to our OO constraints. If we choose to use OO interface Ordering we will not even be able to pass closures as predicate.

Second, note that I changed the type of the first argument to Object** (a pointer to pointer). The outer pointer serves as the iterator; the inner pointer is necessary because now we no longer can store our objects directly as values in containers, because we do not know the size of the object under the interface. We would not know by how much to increase the pointer. On the other hand, if we store pointers, any pointer has always the same size. To fix that, we could go back to passing nmemb argument to our function, or add member function size to the “minimum” interface or Object.

Third, we only partially solved the problem of applying a comparator functor to a mismatching type. While we are now avoiding UB, we turned it into a run-time exception. While this is an improvement, it is still poor a solution, compared with the original example with std::equal_range, where such mismatch is detected by the compiler. Just because we throw an exception, it doesn’t mean we handled the situation properly. What will the user of a GUI application see when he tries to pick a widget? “Bad comparison function passed to function OO_search“? “Internal error”? Bugs should be detected at compilation time rather than at program execution time. This may not be always doable; but in our case it was — we missed the opportunity. This problem could be fixed, though, by turning our searching function back into a template and making the interface Ordering also a template, as explained in detail down below.

Fourth, note that if we choose to use a custom comparator (not tie it to the element type) and figure out the size of the structures by other means, Object does not contain any useful (for us) member. We inherit only for the sake of being able to pass a pointer to an empty base class and apply the inheritance test with dynamic_cast. Not a very useful interface: almost like void*.

Finally, our function only works with pointers as iterators. We cannot search in a map or anything but a raw array. We could introduce a yet another OO interface: Iterator (and it would also know by how much bytes to advance to the next element). But STL containers do not provide iterators derived from MyLib::Iterator. You will have to add a wrapper for each iterator you use. Then, you will have to answer next questions: Do you want to pass these Iterators by reference? You want to avoid slicing, don’t you? By reference to const or non-const object? (If non-const, where will you create them before passing them to function. If const, you will not be able to increment them easily.) How will you return the iterator to the found object? By value (risking slicing)? By reference (risking a dangling reference)? Also, we would now have created another iterator, incompatible with STL — because STL algorithms pass iterators by value and expect no slicing.

OO-style interfaces do not play well with value semantics (https://akrzemi1.wordpress.com/2012/02/03/value-semantics/) . It somehow forces us to pass references/pointers to our objects. This is especially difficult when returning a type-erased object from a function: you cannot return by value, because you are risking slicing. You often cannot return by reference, because, you would be returning a reference to an automatic object that would have died before you try to access it. You could return by a smart pointer, but this only opens a new set of problems: which smart pointer? unique_ptr? shared_ptr? — But neither will work if you need to (deeply) copy the returned object.

Value-semantic type erasure

I will disappoint you. I do not have a perfect solution for the problem I was complaining about above. In order to enforce at compile-time that the comparison function matches the element type, I think you need to mention the type explicitly — I cannot see any other way. In order to provide a reasonable solution we will have to go back to using a template. You can consider it cheating. But this will be a different template; it will require much less instantiations.

Using templates (apart from all disadvantages) offers two advantages: faster programs and more type safety. By abandoning templates we abandoned both the advantages; whereas our original trade-off was to sacrifice faster program for faster build — but not sacrifice type-safe program. So, let me put the template back, but make is a bit more “constrained”. Our original function template is parametrized over three things. Well, technically it is two things, but in practise you can think of it as three:

1. Type of the comparison function.
2. Type of the element in the collection.
3. Type of the iterator.

You cannot see the dependence on element type directly, because you obtain the T from the type of the iterator. Nonetheless it is there, and this is the only parameter that matters to us (in order to guarantee type safety), so we will extract it as a template parameter, and erase the type of the iterator and the type of the comparison function. How? Let’s start with the comparison. You probably know the tool for this already: std::function. It is well documented Boost.Function documentation. In short, it is a holder for any kind of function, function object or closure of an appropriate type. The “type” of function appropriate for us is something taking two ints and returning a bool:
*/

std::function<bool(int, int)> predicate;

/*
We can assign to it any function pointer or function object with a matching function signature:
*/
bool bigger (int a, int b) { return a > b; }
 
predicate = &bigger;
predicate = [](int a, int b) { return a > b; };
predicate = std::greater<int>{};

/*
Not only is predicate able to “hold” any function-like entity (of an appropriate function signature), but is is also a value: it can be copied or assigned to. It can be passed by value with no risk of slicing, because it guarantees to make a deep copy of the underlying concrete object. And we use it just as any other function:
*/
bool ans = predicate (2, 1);

/*
our predicate works well with STL algorithms: it is still a function object. We give it anything that has operator()(int, int) and we get something that has operator()(int, int), but with erased type. Note that std::function’s requirements on the types it can be assigned/initialized with are non-intrusive (or duck-typing-like (http://wiki.c2.com/?DuckTyping) ): if it has operator() it is a good candidate, nothing else is required: no inheritance.

And note one more thing: we did not use any particular language feature for this; std::function is a library component. Isn’t that amazing? If you are wondering how it is even possible to implement such a thing, you can have a look at this publication.(https://akrzemi1.files.wordpress.com/2013/10/mpool2007-marcus.pdf)

You might be puzzled about one thing, though: if std::function is used for type erasure, why is it a template itself? Well, it is a template for generating type-erased interfaces. It is like an “interface template”. Above, we were only interested in one interface: one instantiation of the template, capable of erasing a lot of types. But to solve our main problem from this post, we will need more than a sorting predicate for ints, we will need a predicate for any T. We can express this with an alias template (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2003/n1489.pdf):
*/
template <typename T>
using AnyBinaryPredicate = std::function<bool(T const&, T const&)>;
/*
This defines a new “typedef” or “alias” that can be used with one template parameter, e.g., AnyBinaryPredicate<int>, which means “any binary predicate capable of comparing ints.” Note that it is only a “typedef”: it does not introduce new types or functions.

With std::function we (1) erase the type of the underlying function/function-like object, (2) preserve the interface (operator()), (3) we are able to pass it by value, (4) we require of the erased types no declaration of conformance to an interface (no inheritance).

But this is a special case for functions, because they are popular in C++. How do we erase the type of the iterator? Iterators are also popular in C++: we will use another library already waiting for us. We could use Thomas Becker’s any_iterator.(http://thbecker.net/free_software_utilities/type_erasure_for_cpp_iterators/any_iterator.html)

It comes with a class template IteratorTypeErasure::any_iterator, another “interface template” similar to std::function. It is parametrized by two parameters:

1. Value type of the underlying sequence (container).
2. Iterator category (forward iterator, random access iterator, etc.).
We will fix the second parameter to “forward iterator” tag. This is the minimum that we require for searching functions. Iterator category is something we do not want to erase:
*/

template <typename T>
using AnyForwardIter = IteratorTypeErasure::any_iterator<
  T,                         // variable parameter
  std::forward_iterator_tag  // fixed parameter
>;

/*
Now we have an another value-semantic interface capable of managing anything that is a forward iterator:
*/
std::vector<int> vec {1, 2, 3};
std::list<int> list {2, 4, 6};
 
AnyForwardIter<int> it { vec.begin() }; // initialize
it = list.begin();                      // rebind
AnyForwardIter<int> it2 = it;           // copy (deep)

/*And it is an iterator itself:*/
++it;
int i = *it;
it == it2;

/*
But let’s go one step further, rather than using two iterators, let’s use a range — a type-erased range: boost::any_range. Again, like std::function and IteratorTypeErasure::any_iterator, it is an “interface template”, so we will only pick some specializations. Template boost::any_range requires at least 4 parameters:

1. Value type of the container.
2. Iterator category.
3. Type of the reference returned by dereferencing an iterator.
4. Type of the iterator difference.

We will fix the three latter parameters and only leave the value type as the parameter. We will use an alias template again:
*/
template <typename T>
using AnyForwardRange = boost::any_range<
  T,                            // real param
  boost::forward_traversal_tag, // fixed
  T&,                           // repeated param
  std::ptrdiff_t                // fixed
>;

/*
This means “any forward range capable of iterating over ints.” This is how we can use it:
*/
std::vector<int> vec {9, 8, 5, 4, 2, 1, 1, 0};
std::set<int> set {1, 2, 3, 5, 7, 9};
 
AnyForwardRange<int> rng = vec; // initialize interface
std::distance (boost::begin(rng), boost::end(rng));
 
rng = set;                      // rebind interface
std::distance (boost::begin(rng), boost::end(rng));

/*
Thus, we have two value-semantic type-erased interfaces: AnyForwardRange<T> and AnyBinaryPredicate<T>. Using them we can define our (partially) type-erased searching function:
*/
template <typename T>
AnyForwardRange<T> Search (AnyForwardRange<T> rng, T const& v, AnyBinaryPredicate<T> pred) 
{
  auto ans = std::equal_range (rng.begin(), rng.end(), v, pred);
  return {ans.first, ans.second};
}
/*
We still use std::equal_range inside: it is a good algorithm. But because it is wrapped in our new interface, we will control its instantiations: only one per element type:
*/
std::equal_range <typename boost::range_iterator<AnyForwardRange<T>>::type,T>

/*
Our Search is still a template, but it will only generate new instantiations when we want to sort different types of elements. It will not generate different instantiations for different iterator types or different predicates: their type will be erased. This is how we can use our Search:
*/

std::vector<int> vec {9, 8, 5, 4, 2, 1, 1, 0};
auto greater = [](int a, int b) { return a > b; };
AnyForwardRange<int> ans = Search<int> (vec, 1, greater);
assert (distance(ans) == 2);
     
std::set<int> set {1, 2, 3, 5, 7, 9};
ans = Search<int> (set, 4, std::less<int>{});
assert (distance(ans) == 0);

/*
It comes with one inconvenience: you have to specify which instantiation you choose. I must admit, I do not know how to overcome this without introducing more templates. This type erasure works at the expense of reduced run-time performance. The implementations of boost::any_range and std::function internally use indirect function call techniques, like OO interfaces.

The implementation of std::function and boost::any_range uses templates and template instantiations too, so you might conclude that in order to avoid templates we introduced even more templates. This is true, but only to certain extent. We have now new template instantiations, indeed; but they are localized to places where you bind objects to interfaces. Once you do that other functions/algorithms that use the type-erased interfaces do not have to be templates, or as was the case for our Search, they do not have to generate this many instantiations.

If we erased the types completely, you could hide the function’s implementation to a “cpp” file and compile separately: you do not need to expose it to the users: you still reduce compilation times (although less than with void*). This is not an ideal solution but an alternative when making important trade-offs in your projects: an attractive alternative.

To be continued…

And that’s it for today. You may still feel that I have cheated you. std::function and boost::any_range may be a solution if anything you ever wanted to do in your program was to invoke functions and advance iterators. But what if you want to use a custom interface with custom member functions? I will try to cover it in the next post (but I already confess, I do not have a perfect solution for this). I will also try to explain why all the names of the interfaces in the examples above start with “Any”. I will also try to cover some practical examples of type-safe type erasure.
*/
///////////////////////////////////////////////////////////////////////
/* PART 2*/

/*
While there are many ways to erase a type, I will use name type erasure to denote the usage of std::function-like value-semantic interface. This convention appears to be widely accepted in C++ community.

In this post, we will see how to create a custom type-erased interface. This is not easy, as there is no language feature for that. 

Using std::function is easy, but this is because someone has made an effort to implement it for us. So let’s try to see how much effort it is to write our own type-erased interface. We will be using countdown counters: something that (1) can be decremented and (2) tested if the count reached zero. A simple model for such counter is type int: you can assign it an initial value:
*/
int counter = 10;
//decrement it:
--counter;
//and test for reaching zero:
if (counter) {}   // still counting
if (!counter) {}  // done counting
 
if (--counter) {} // decrement and test

/*
I realize that this may not be a convincing real-life example of a concept, but this is just to illustrate the techniques. I needed a concept that is (a) small and (b) can be ‘modeled’/’implemented’ by built-in types that have no member functions. Apart from an int, we can imagine other implementations of counters: a counter that logs each decrement, a counter that contains a list of other counters and decrements all of them when it itself is decremented.

If we had Concepts Lite (for description see (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3580.pdf), for technical specification draft see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3651.pdf ), especially with support for variable templates (as has been indicated in Concepts SG mailing list (https://groups.google.com/a/isocpp.org/forum/?fromgroups#!forum/concepts) ) we could specify the above requirements formally:
*/

// maybe in C++17
template <typename C>
concept bool CountdownCounter = requires (C c) {
  --c;
  (bool(c));      // contextual conversion to bool
  { !c } -> bool; // !c must be convertible to bool
  (bool(--c));    // decrement and test
 
  requires std::is_copy_constructible<C>::value;
  requires std::is_copy_assignable<C>::value;
};
/*
and even test if a type satisfies them:
*/

// maybe in C++17
static_assert (CountdownCounter<int>, "not a counter");

/*
This is a different (than OO) way of specifying interfaces: we say what expressions (including function calls) are valid on our type — not member functions that it has. This way built in types are also able to comply to interfaces and we can also use free functions in addition to member functions. If type erasure was supported by the language, we could probably bind a model to the interface with a single instruction:
*/

// not in C++
ERASED<CountdownCounter> counter = int{3};
/*
So, how do we do the same without language support?

Boost Concept Check Library

We may (but don’t have to) specify a concept. This is not strictly necessary, but it can improve our diagnostic messages when someone tries to bind an incorrect type to our interface. For specifying concepts, we could use Boost Concept Check Library.
*/

#include <boost/concept_check.hpp>
 
template <typename C>
struct CountdownCounter : boost::CopyConstructible<C>
                        , boost::Assignable<C>
{
  BOOST_CONCEPT_USAGE(CountdownCounter)
  {
    --c;
    (bool(c));   // contextual conversion to bool
    bool b = !c; // !c must be convertible to bool
    (bool(--c)); // decrement and test
  }
     
private:
    C c;
};
/*
The library also allows us to test if a given type models (satisfies the requirements of) a given concept:
*/
BOOST_CONCEPT_ASSERT((CountdownCounter<int>));         // ok
BOOST_CONCEPT_ASSERT((CountdownCounter<std::string>)); // error
/*
Double parentheses are the price to be paid for using a library-based solution to concepts rather than concepts built into the language. There is a lot of macro and template meta-programming magic involved to make this library work.

Adobe.Poly

Now, to type erasure. We will use Adobe.Poly (http://stlab.adobe.com/group__poly__related.html) library first. We will need to declare two classes and one class template. First an internal interface. The users will never see it:
*/

#include <adobe/poly.hpp>
 
struct CounterIface : adobe::poly_copyable_interface
{
  virtual void decrement() = 0;  // for operator--
  virtual bool done() const = 0; // for operator bool
};

/*
This interface corresponds to our concept, but does not have to be identical, and in fact, it cannot be identical in case we are using non-member functions. Instead of names decrement and done, we could have used names operator-- and explicit operator bool, and this would have even been more sane, but my goal was to stress that the names in the internal interface need not be the same as these in the external interface — the one that is really going to be used. Class adobe::poly_copyable_interface is a base interface that we extend. It provides definitions of member functions common to all value-semantic interfaces: assign, clone, exchange for copying and swapping, plus some other functions for querying the exact type bound to the outer interface — but this is done for us: we only need to worry about specifying the operations custom to our interface.

Next, we have to provide a class template for creating implementations of our internal interface:
*/

template <typename T>
struct CounterImpl : adobe::optimized_storage_type<T, CounterIface>::type
{
  BOOST_CONCEPT_ASSERT((CountdownCounter<T>)); //line 5
 
  using base_t = typename adobe::optimized_storage_type<T, CounterIface>::type; //line 7
 
  CounterImpl(T x) : base_t(x) {}
  CounterImpl(adobe::move_from<CounterImpl> x)
    : base_t(adobe::move_from<base_t>(x.source)) {}
 
  void decrement() override { --this->get(); }
  bool done() const override { return bool(this->get()); }
};
/*
Here, T is the to-be-erased type that users want to bind to our external value-semantic interface. T has to model our concept CountdownCounter. We check that in line 5. This is a compile-time test, and it makes template error messages more readable. But this line is not strictly necessary. If we omit it, we will simply get less readable error messages if we bind a type to the incompatible interface. Line 7 is an alias declaration, it is the new substitute for typedefs. adobe::optimized_storage_type is a utility for picking the most efficient storage for type T. In short, if T is small enough, rather than allocating it on a heap, we will use a potentially stack-based <aligned_storage>(http://www.cplusplus.com/reference/type_traits/aligned_storage/). This trick is often called a <small buffer optimization>.

We also need to define two constructors: one allows initialization from T, the other is Adobe’s way of implementing move constructor. Finally, we implement the internal interface’s member functions with T’s interface. Expression this->get() returns T& (or T const& respectively). This is where we map T’s interface onto our internal interface. Note that we lost the type returned by T’s operator--. That’s fine here. We will restore it in the next class we define. Other member functions that need to be overridden, which deal with copying and swapping, are already defined in base_t, the class we derive from: the framework does it for us.

Finally, we define a class that represents the external interface:
*/

struct Counter : adobe::poly_base<CounterIface, CounterImpl>
{
  using base_t = adobe::poly_base<CounterIface, CounterImpl>;
  using base_t::base_t; // inherit constructors
 
  Counter(adobe::move_from<Counter> x) 
    : base_t(adobe::move_from<base_t>(x.source)) {}
 
  Counter& operator--() 
  { 
    interface_ref().decrement(); 
    return *this; 
  }
 
  explicit operator bool() const
  { 
    return interface_ref().done();
  }
};
/*
Now, the outer interface has the same interface as our original concept. operator-- returns a reference to self. We map from the inner interface to the outer interface. But we will not use naked Counter. Our users will have to use a derived type:
*/
using AnyCounter = adobe::poly<Counter>;

/*
Now we can statically test that our interface is at the same time the model of concept CountdownCounter:
*/
BOOST_CONCEPT_ASSERT((CountdownCounter<AnyCounter>));
/*
And we can also test our counters at runtime. Let’s invent some other model of CountdownCounter:
*/
struct LoggingCounter
{
  int c = 2; // by default start from 2
 
  explicit operator bool () const { return c; }
 
  LoggingCounter& operator--() 
  { 
    --c;
    std::cout << "decremented\n"; 
    return *this;
  }
};
/*
And the test:
*/

AnyCounter counter1 {2};  // bind to int (initially 2)
assert (counter1);        // still counting
assert (--counter1);      // still counting (1)
AnyCounter counter2 = counter1; 
                          // counter2 (int) counts from 1
--counter1;
assert (!counter1);       // done
assert (counter2);        // counter2 still 1
assert (!--counter2);     // counter2 also done
   
counter1 = AnyCounter{LoggingCounter{}};
                          // reset with a different type 
assert (counter1);        // 2
--counter1;
assert (counter1);        // 1
--counter1;
assert (!counter1);       // 0

/*
Well, performing mutating operations in assertions is a bad idea. I just tried to make the example short. Don’t do it at home. You can see that adobe::poly offers no implicit conversion from T to the interface object. I had to use explicit initialization. Also, you can see tat there is a lot of boiler-plate code involved in creating an interface. One would expect some macro-based automation for this process. For a complete, working program code, see following.
*/

//Adobe.Poly — Counter

#include <cassert>
#include <iostream>
#include <adobe/poly.hpp>
#include <boost/concept_check.hpp>
 
// (1) Concept for compile-time tests
 
template <typename C>
struct CountdownCounter : boost::Assignable<C>
                        , boost::CopyConstructible<C>
{
  BOOST_CONCEPT_USAGE(CountdownCounter)
  {
    --c;
    (bool(c));
    bool b = !c;
    (bool(--c));
  }
   
private:
    C c;
};
 
// (2) The inner interface
 
struct CounterIface : adobe::poly_copyable_interface
{
  virtual void decrement() = 0;
  virtual bool done() const = 0;
};
 
// (3) The inner interface implementation
 
template <typename T>
struct CounterImpl 
  : adobe::optimized_storage_type<T, CounterIface>::type
{
    using base_t = typename
    adobe::optimized_storage_type<T, CounterIface>::type;
 
  BOOST_CONCEPT_ASSERT((CountdownCounter<T>));
   
  CounterImpl(T x) : base_t(x) {}
  CounterImpl(adobe::move_from<CounterImpl> x) 
    : base_t(adobe::move_from<base_t>(x.source)) {}
   
  void decrement() override { --this->get(); }
  bool done() const override { return bool(this->get()); }
};
 
// (4) The outer interface specification
 
struct Counter : adobe::poly_base<CounterIface, CounterImpl>
{
  using base_t = adobe::poly_base<CounterIface, CounterImpl>;
  using base_t::base_t; // Inherit constructors
 
  Counter(adobe::move_from<Counter> x) 
    : base_t(adobe::move_from<base_t>(x.source)) {}
   
  Counter& operator--() 
  { 
    interface_ref().decrement(); 
    return *this; 
  }
   
  explicit operator bool() const
  { 
    return interface_ref().done(); 
  }
};
 
// (5) The interface
 
typedef adobe::poly<Counter> AnyCounter;
 
// (6) Another counter for testing
 
struct LoggingCounter
{
  int c = 2;
  explicit operator bool () const { return c; }
   
  LoggingCounter& operator--() 
  { 
    std::cout << "decremented\n";
    --c;
    return *this;
  }
};
  
// (7) Compile-time test
 
BOOST_CONCEPT_ASSERT((CountdownCounter<int>)); 
BOOST_CONCEPT_ASSERT((CountdownCounter<LoggingCounter>)); 
BOOST_CONCEPT_ASSERT((CountdownCounter<AnyCounter>));
 
// (8) Run-time test
 
void  test_counter()
{
  AnyCounter counter1 {2};  // bind to int (initially 2)
  assert (counter1);        // still counting
  assert (--counter1);      // still counting (1)
  AnyCounter counter2 = counter1;
                            // counter2 (int) counts from 1
  --counter1;
  assert (!counter1);       // done
  assert (counter2);        // counter2 still 1
  assert (!--counter2);     // counter2 also done
      
  counter1 = AnyCounter{LoggingCounter{}};
                            // reset with a different type
  assert (counter1);        // 2
  --counter1;
  assert (counter1);        // 1
  --counter1;
  assert (!counter1);       // 0
}
 
int main()
{
  test_counter();
}

/*
Boost.TypeErasure

Another library serving a similar purpose is Steven Watanabe’s Boost.TypeErasure. It follows a slightly different philosophy. Rather than creating one concept composed of a number of requirements, we create a separate concept per each operation. We can later combine them together into bigger concepts, as needed. We have two operations in our concept: counter decrement and the test for reaching zero. We will start with the first. The library already offers concepts for nearly every C++ operator. The concept that we need is boost::type_erasure::decrementable:
*/

#include <boost/type_erasure/operators.hpp>
using boost::type_erasure::decrementable;

/*
Regarding the other operation, we are very unlucky: the library offers convenience tools for specifying operators, named member functions and named free functions, but it does not offer any convenience for conversion functions. Hopefully this will be addressed soon (see https://svn.boost.org/trac/boost/ticket/9436 ), but for now, we have to do it the long way: customize the framework with a template specialization. First we define our mini-concept:
*/

template <class T>
struct testable
{
  static bool apply(const T& arg) { return bool(arg); }
};

/*
Its name is testable. It checks for the valid expression inside the body, and it offers the external interface (function apply) recognized by the framework. Now, the framework needs to be taught about our new concept:
*/

#include <boost/type_erasure/any.hpp>
 
namespace boost { namespace type_erasure {
 
  template <class T, class Base>
  struct concept_interface<testable<T>, Base, T> : Base
  {
    explicit operator bool () const
    { return call(testable<T>(), *this); }
  };
 
}}

/*
We can see that specializations of concept_interface for our concept testable expose the explicit conversion to bool themselves. Function call in the implementation internally calls our function testable<T>::apply.

That’s quite a lot of boiler palate, but this is an exceptional situation (a conversion function), and likely to be fixed. Also, note that we had to define our mini-concept only once. Now it can be used to build many other composed concepts: not only our counter. Now we have to compose our mini-concepts together.
*/
#include <boost/mpl/vector.hpp>
namespace te = boost::type_erasure;
 
using Counter = boost::mpl::vector<
  te::copy_constructible<>, 
  decrementable<>, 
  testable<te::_self>, 
  te::relaxed
>;

/*
This is four requirements rather than two. You already know why we need copy_constructible, but what about this relaxed (http://www.boost.org/doc/libs/1_55_0/doc/html/boost/type_erasure/relaxed.html)? According to the documentation, it allows the interface object to provide a couple of other operations that we would often want to have: rebind the interface to another implementation of different type, create a null interface, equality comparison, and a few more. Now, with thus defined concept, we can produce the interface type:
*/
using AnyCounter = te::any<Counter>;

/*
And that’s it; you can apply the same compile-time and run-time test as with adobe::poly examples. One difference with Boost.TypeErasure is that objects bind to interfaces implicitly. So, you can write:
*/

AnyCounter counter1 = 2; 
AnyCounter counter2 = counter1; 
counter1 = LoggingCounter{};
/*
For a complete, working program code, see following.
*
Boost.TypeErasure — Counter
*/

#include <cassert>
#include <iostream>
#include <boost/type_erasure/any.hpp>
#include <boost/type_erasure/operators.hpp> // decrementable
#include <boost/concept_check.hpp>
#include <boost/mpl/vector.hpp>
 
// (1) Concept, for compile-time tests
 
template <typename C>
struct CountdownCounter : boost::Assignable<C>
                        , boost::CopyConstructible<C>
{
  BOOST_CONCEPT_USAGE(CountdownCounter)
  {
    --c;
    (bool(c));
    bool b = !c;
    (bool(--c));
  }
   
private:
    C c;
};
 
// (2) Mini-concept for contextual conversion to bool
 
template <class T>
struct testable
{
    static bool apply(const T& arg) { return bool(arg); }
};
 
// (3) Teaching framework the conversion to bool
 
namespace boost { namespace type_erasure {
 
  template<class T, class Base>
  struct concept_interface <testable<T>, Base, T> : Base
  {
    explicit operator bool () const
    { return call(testable<T>(), *this); }
  };
}}
 
using boost::type_erasure::decrementable;
namespace te = boost::type_erasure;
 
// (4) Composing the interface functions
 
using Counter = boost::mpl::vector<
  te::copy_constructible<>, 
  decrementable<>, 
  testable<te::_self>, 
  te::relaxed
>;
 
// (5) The type-erased interface
 
using AnyCounter = te::any<Counter>;
 
// (6) Custom Counter for tests
 
struct LoggingCounter
{
  int c = 2;
  explicit operator bool () const { return c; }
 
  LoggingCounter& operator--() 
  { 
    std::cout << "decremented\n";
    --c;
    return *this;
  }
};
 
// (7) Compile-time test
 
BOOST_CONCEPT_ASSERT((CountdownCounter<int>)); 
BOOST_CONCEPT_ASSERT((CountdownCounter<LoggingCounter>)); 
BOOST_CONCEPT_ASSERT((CountdownCounter<AnyCounter>)); 
 
// (8) Run-time test
 
void test_counter()
{
  AnyCounter counter1 = 2;     // bind to int (initially 2)
  assert (counter1);           // still counting
  assert (--counter1);         // still counting (1)
  AnyCounter counter2 = counter1;
                               // counter2 (int) counts from 1
  --counter1;
  assert (!counter1);          // done
  assert (counter2);           // counter2 still 1
  assert (!--counter2);        // counter2 also done
      
  counter1 = LoggingCounter{};
                               // reset with a different type
  assert (counter1);           // 2
  --counter1;
  assert (counter1);           // 1
  --counter1;
  assert (!counter1);          // 0
}
 
int main ()
{
  test_counter();
}

/*
For another example…

Since this post is supposed to be an introduction to type erasure libraries, let me show you briefly an another example, so that you can see how these libraries work with types having member functions and free functions in their interface. Let’s define the following, somewhat silly, concept:
*/
template <typename H>
struct HolderConcept : boost::Assignable<H>, boost::CopyConstructible<H>
{
  BOOST_CONCEPT_USAGE(HolderConcept)
  {
    h.store(i);  // member
    i = load(h); // non-member (free)
  }
   
private:
    H h;
  int i;
};

/*
And here is a type that models our concept:
*/

struct SomeHolder
{
  int val = 0;
  void store(int i) { val = i; }
};
int load(SomeHolder& h) { return h.val; }
 
BOOST_CONCEPT_ASSERT((HolderConcept<SomeHolder>));

/*
First, using Adobe.Poly:
*/

struct HolderIface : adobe::poly_copyable_interface
{
  virtual void store(int) = 0;
  virtual int free_load() = 0;
};
 
template <typename T>
struct HolderImpl : adobe::optimized_storage_type<T, HolderIface>::type
{
  using base_t = typename adobe::optimized_storage_type<T, HolderIface>::type;
 
  BOOST_CONCEPT_ASSERT((HolderConcept<T>));
  HolderImpl(T x) : base_t(x) {}
  HolderImpl(adobe::move_from<HolderImpl> x) 
    : base_t(adobe::move_from<base_t>(x.source)) {}
   
  void store(int i) override { this->get().store(i); }
  int free_load() override { return load(this->get()); }
};
 
struct Holder : adobe::poly_base<HolderIface, HolderImpl>
{
  using base_t = adobe::poly_base<HolderIface, HolderImpl>;
  using base_t::base_t;
 
  Holder(adobe::move_from<Holder> x) 
    : base_t(adobe::move_from<base_t>(x.source)) {}
   
  void store(int i) { interface_ref().store(i); }
 
  friend int load(Holder & h) // free function
  { return h.interface_ref().free_load(); }
};
 
using AnyHolder = adobe::poly<Holder>;

/*
It is boringly similar to the previous example. There are two things worth noting. First, look how member function free_load is implemented in terms of free function load. Second, note how I used friend declaration to declare a free function inside class, visible in a namespace enclosing the class.

Now, with Boost.TypeErasure:
*/

BOOST_TYPE_ERASURE_MEMBER((has_member_store), store, 1)
BOOST_TYPE_ERASURE_FREE((has_free_load), load, 1) 
 
namespace te = boost::type_erasure;
 
using Holder = boost::mpl::vector<
  te::copy_constructible<>,
  has_member_store<void(int)>,
  has_free_load<int(te::_self&)>,
  te::relaxed
>;
 
using AnyHolder = te::any<Holder>;

/*
This may require some explanation. First line declares a mini-concept has_member_store. It requires that the model has member function store taking 1 argument: we do not specify yet what the type of this argument is or if the member function is const. similarly, second line defines a mini-concept has_free_load, requiring that there exists a free function load taking one argument (the model). Next, we compose the requirements. Now, we do specify the types missed previously. The strange _self& means that we want to pass argument to function load by non-const reference. Because we didn’t specify otherwise, our member function store is non-const.
*/