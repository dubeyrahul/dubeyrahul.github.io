---
layout: post
title: "Writing Pythonic code: Part 2"
date: 2019-08-04
---

This is the second post ([here's the first]({% post_url 2019-08-03-writing-pythonic-code %})) on writing Pythonic code. In this post, we will first go over some aspects of writing effective Python code by brushing over topics such as: slicing, list comprehension, generator expressions.

+ [Writing effective Python code](#writing-effective-python-code)
  - [Slicing](#slicing)
  - [List comprehensions](#list-comprehensions)
  - [Looping effectively](#looping-effectively)
  - [Understanding try, catch, else, except, finally](#understanding-try--catch--else--except--finally)
  - [Knowing the difference between str, unicode, and bytes](#knowing-the-difference-between-str--unicode--and-bytes)
* [Conclusion](#conclusion)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

### Writing effective Python code
#### Slicing
* **What is slicing:**
	* Slicing allows you to easily access a subset of a sequence in Python
	* Syntax: `array[start:end:stride]` : This results into a subset of this array starting from index `start`, till index `end` (exclusive), and picking up every n<sup>th</sup> item if `stride=n`.
	* Negative integers can be used as offset from the end of the list, so `arr[:-2]` gets all elements of the list till 2<sup>nd</sup> last element (excluding) of the list. Similarly, `a[-2:]` gets all elements from 2<sup>nd</sup> last element till the end of the list. Don't use `-0` to slice the list, it results into weird results.
* **Things to keep in mind while slicing:**
	*  If you're starting the slice from the start of the list, then don't specify `start=0`, just leave it blank. For example, do this: `arr[:10]` instead of `arr[0:10]`. Similarly, if you're slicing till the end of the list, do this: `arr[5:]` instead of `arr[5:len(arr)]`.
	*  Slicing results into a new list. So references to the objects in old list are maintained. If you modify elements of the new list, then the elements of old list are not modified.
	*  When creating a copy of a list, remember that `b = a[:]` creates a new list `b`. Modifying elements of `b` does not affect `a`. On the other hand, if you do `b = a`, then the references are shared and modifying one changes the other.
* **Things to avoid while slicing:**
	* A common usage of slicing is to reverse a string (Byte string and ASCII characters) by slicing it with `stride=-1`.  

{% highlight python %}
x = b'apple'
y = x[::-1]
print(x) # apple
print(y) # elppa
{% endhighlight %}
But this doesn't work if the string is UTF-8 encoded.


How readable is a code with a bunch of statements like this:

{% highlight python %}
x = a[2:10:-2]
y = b[1:20:3]
z = c[10:-2:-1]
{% endhighlight %}


To me, it seems sort of readable because **I** wrote it and it's just a couple of statements. Now imagine a legacy codebase that you're trying to understand that is full of these expressions. Moreover, imagine that these indices are variables instead of integer values. Now that may become a nightmare for you to parse. Hence, it is best to avoid using start, end, and slice all in the same expression. If you really have to use stride, then skip start/end, and if you can't then break it into two expressions. So instead of the above listed expressions, we can split them like:
{% highlight python %}

temp = a[::-2]
x = temp[2:10]

temp = b[::3]
y = temp[1:20]

temp = c[10:-2]
z = temp[::-1]
{% endhighlight %}


#### List comprehensions
* **What are list comprehensions:**
	* Python provides a compact and beautiful way of deriving one list from the other, especially if the same function is applied to most of the elements in the original list. Say, you want to square every number in a list. These are some ways of doing it.
{% highlight python %}

a = [1, 2, 3, 4, 5, 6, 7, 8]

# Method 1: Using for loop
even_squares = []
for elem in a:
    even_squares.add(elem*elem)
# Method 2: Using map()
even_squares = map(lambda x: x*x, a)

# Method 3: Using list comprehension
even_squares = [x*x for x in a]
{% endhighlight %}
As one can see, both method 2 and 3 are compact and readable. I strongly prefer list comprehension over map(), and this example from the book illustrates why.
Say you want to compute squares of all even numbers in the list, then the above methods look like:
{% highlight python %}
# Method 1: Using for loop
even_squares = []
for elem in a:
	if elem % 2 == 0:
   		even_squares.add(elem*elem)	        
# Method 2: Using map()
even_squares = map(lambda x: x*x, filter(lambda x: x % 2 == 0, a)
# Method 3: Using list comprehension
even_squares = [x*x for x in a if x % 2 == 0]
{% endhighlight %}

In my opinion, list comprehensions become much easier to read once `filter` gets involved with `map`.

Note: list comprehension syntax can be applied to `set` and `dict` as well.

* **When to not use list comprehensions**
	* Avoid list comprehensions the moment they become unreadable, or when they start taking as much space as using for loops would. I think after that point one is just being a pretentious Python programmer :p
	* One common scenario is multiple level of comprehension. Something like: `flat = [x for row in matrix for x in row] # flattens a 2D-matrix` is a good use of list comprehension, but the moment this becomes more than 2 level, it becomes difficult to read. Say you had a 3D array to be flatted, the code would look like:
{% highlight python %}
# Method 1: Using list comprehension:
flat = [x for level_1 in matrix
		 for level_2 in level_1
		 for x in level_3]

# Method 2: Using for loops:
flat = []
for level_1 in matrix:
	for level_2 in matrix:
		flat.extend(level_2
{% endhighlight %}

It seems pretty clear that for-loop is more readable here and less confusing too.

* One major issue with list comprehension is that it is not scalable, in the sense it consumes a lot of memory in creating and holding the new list in memory. This is not ideal in dealing with big data, or large streams of infinite data, or while reading a big file. One common application in Machine Learning is reading large files, parsing it line by line, selecting some parts of the line as features and storing it in lists.
{% highlight python %}

# let parse_and_extract_features(line) parse a line and extract features
# let dataset.txt contain 1M rows. Then,
file_name = 'dataset.csv'
output_value = [parse_and_extract_features(x) for x in open(file_name)]
output_file = open('output.csv', w)
output_file.write(output_value)
# output_value will be a list of 1M items and we want to write it to a new file
{% endhighlight %}


In this case, a list of 1M items will be generated only to be written to the file and never used again. A better approach in this case is to use **generator expressions**.
{% highlight python %}

def genexp_extract_features(file_name):
	# generator to parse feature for a given file
	for row in open(file_name):
	    yield parse_and_extract_features(row)
file_name = 'dataset.csv'
output_generator = genexp_extract_features(file_name)
output_file = open('output.csv', w)
for output_value in output_generator:
	output_file.write(output_value)
{% endhighlight %}


By using generator, we do not create a list of 1M items in memory and instead generate feature for one row at a time by creating output_generator object and then iterating over it. Note, we can iterate over genexp like this because essentially they are like iterators. 		

* In conclusion, generator expresssions are a more general form of list comprehension. They are also more efficient because they do not create the entire list in memory. Instead genexp (in short) evaluates to an iterator, that can be used to generate the result-list, one item at a time by calling next() on the evaluated iterator. In the above example, we could use genexp instead of list comprehension to evaluate and write one line at a time to the transformed_dataset file.
* One thing to keep in mind is that genexp evaluates to iterators and once you call next() on it, that result is evaluated and returned and that value cannot be obtained again.

#### Looping effectively
* **Avoiding range / Using enumerate**
	* In Python 2, always use `xrange` instead of `range`. Difference being `xrange` generates an iterator to iterate over whereas `range` produces actual list which can be memory inefficient when iterating over a large range. In Python 3, there's only `range` function and it's a generator.

	* For people coming from languages like Java and C++, Python looping may seem a little strange initially because the most common way of looping in Python does not give you access to indices, rather it gives you the elements.
So instead of doing this:
{% highlight python %}

states = ['California', 'North Carolina', 'Texas', 'Nevada']
n = len(states)
for i in range(n):
    print (states[i])
{% endhighlight %}

do this:
{% highlight python %}

states = ['California', 'North Carolina', 'Texas', 'Nevada']
for state in states:
    print (state)
{% endhighlight %}

It seems pretty clear that the second option is more readable, and does the job effectively.

Now there may arise a case when you absolutely do need the index of the element in the list. In that case, one might think range is the way to go. But there's a better option: *enumerate*

Instead of doing this:
{% highlight python %}

states = ['California', 'North Carolina', 'Texas', 'Nevada']
n = len(states)
for i in range(n):
    print (i, states[i])
{% endhighlight %}

do this:
{% highlight python %}

states = ['California', 'North Carolina', 'Texas', 'Nevada']
for i, state in enumerate(states):
    print (i, state)
{% endhighlight %}


enumerate function takes anything that's iterable and creates a lazy generator. This generator spits out two things: index and the next element from the iterator. It is essentially made for this task.  

Note: You can also provide enumerate an additional argument, a start index, which the function uses to start its indexing (by default it begins with index=0)

* **Using zip / izip:**
	* Sometimes you want to loop over multiple lists of same length which often happens at least in data science tasks, when you're looping over multiple feature list for your dataset, where each list is of size n, where n = number of examples in dataset. In that case, one might write such loop as:
{% highlight python %}

for i in range(len(dataset)):
    feat_1 = feature_1[i]
    feat_2 = feature_2[i]
    print (feat_1, feat_2)
{% endhighlight %}


In this case, zip provides a useful generator that takes in multiple iterators and yields tuples containing the next value from each iterator (in the same order as they are given to zip)

{% highlight python %}
for feat_1, feat_2 in zip(feature_1, feature_2):
    print (feat_1, feat_2)
{% endhighlight %}

Result: More readable, like a pseudocode.

* Things to keep in mind with zip:
	* In Python 2, zip is not a generator, it actually evaluates both the lists and thus may cause memory issues if the lists are massive. **Tip:** Use izip in Python from itertools module.
	* If the length of the iterables provided to zip are not same, then zip will stop iterating as soon as any of the iterable is exhausted, so keep that in mind. 	

#### Understanding try, catch, else, except, finally
* **Using try and finally:**
	* Use this when you want exceptions to propagate up to the caller, but also want to do cleanup.
{% highlight python %}
handle = open('data.txt')
try:
    data = handle.read()
finally:
    handle.close()
{% endhighlight %}


This ensures that if the exception occurs at open() then we won't attempt at closing the file handler, which is correct. We will only call close() if try block is attempted, which means open() succeeded.  

PS: I always used to write open() inside the try block with a similar finally clause, and never thought about it. So I am glad to learn this new bit.

* **Using try, except, and else:**
	* By using this trio we can make it clear which exceptions will be handled by our code and which caller is suppose to handle. When try runs successfully, else block runs. It kind of helps minimize the code in try block so you can keep the try block only for code that can raise exceptions and do other computations in else block.
{% highlight python %}
def load_json_key(json_data, key):
    try:
        data_dict = json.loads(json_data)
    except ValueError as e:
        raise KeyError as e
    else:
        return data_dict[key]
{% endhighlight %}

Now if json loading fails, that will be caught by our try except block. If it goes through, then else will try to extract the value. If it can't find the key, it'll raise that error to the caller. This is done to make error propagation clear.

* **Using try, except, else, and finally:** In this case,  consider an example where we read something from a file, perform some computation, and return the result of that computation. In this case, we can have:
	 - try block attempting to read the file
	 - except catches an exception that try block is expected to throw
	 - else block runs the computation if try block runs successfully
	 - finally block does the clean up like closing the file handler.

#### Knowing the difference between str, unicode, and bytes
* This is something I wasn't fully aware of. I had come across both unicode and str, but I never needed to delve into details about their differences, and especially how they behave in Python 2 v/s 3. As someone who works on projects in both Python 2 and Python 3 regularly, this is definitely something I should keep in mind.

	| Contains  | Python 2 data-type | Python 3 data-type |
	|-----------|--------------------|--------------------|
	| raw 8-bit | str                | bytes              |
	| unicode   | unicode            | str                |


* Also, in Python 3, the encoding argument for open() defaults to UTF-8, which makes read and write operations expect the str data type containing unicode characters instead of bytes data type containing binary data. One may encounter this issue while writing binary data to a .bin file. In Python 2, you can open with 'w' argument for writing, whereas in Python 3, you've to use 'wb' argument for writing to a binary file.

## Conclusion
In this post, we saw several techniques to make our code more efficient and effective. Note that most of these skills build up over time. But it's helpful to know about them so that we you see them in other people's code, you can appreciate and understand its intent better.
