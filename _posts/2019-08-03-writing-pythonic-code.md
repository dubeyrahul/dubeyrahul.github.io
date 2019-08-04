---
layout: post
title: "Python tips: Part 1 - Writing Pythonic code"
date: 2019-08-03
---

## Part 1: Writing Pythonic code
### Why should we learn to write Pythonic code
Before starting any task, it is very important to understand the *why*, hence I am going to describe my *why*. I am a Machine Learning Engineer at Yelp and I write Python every day, so one may wonder why am I learning to write Pythonic code *now*.  

Like many machine learning engineers, Python was not my first programming language. In fact, I never learned Python from a proper textbook like I learned Java or C++ (ages ago). The sole reason I started to code in Python is because I wanted to learn Machine Learning, and wanted to learn it quickly. And Python is ideal for that because of [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) and the [PyData](https://pydata.org/) stack comprising of popular libraries such as NumPy, SciPy, Pandas and scikit-learn to name a few. These packages along with the ability to quickly prototype models in Python gives Python an edge in Machine Learning community.

So I started learning Python on a case-to-case basis, meaning, I'd only learn certain topics that I need to perform a certain Machine Learning task. It ranged from data scraping from web, calling REST APIs, doing a lot of file handling, and obviously writing machine learning code (either from scratch, or using scikit-learn). Thus, I never learned how to write Pythonic OOP, tests, closures, or treating functions as first-class object. Even though I used some of these features incidentally, I never learned it thoroughly or the reasoning behind it.

But all of this changed when I started my graduate studies and started programming extensively in Python (again, the reason being that I was specializing in machine learning). I especially started using NumPy and Scikit-learn a lot, learned a bit about reading and processing large files using generators (and was amazed by it). I wrote some simple closures, wrote some functions that I could pass as parameter to other scikit-learn functions (thereby treating functions as first class objects). But I reached a state where I was doing things very ineffectively and inconsistently. I'd repeat myself a lot, write really complicated list comprehensions, totally forget that generators exist, wouldn't use closures as effectively as they can be used. Also, I started reading more Python code which made me realize that I need to **relearn** Python and let go of old bad habits.

After I joined Yelp as Machine Learning Engineer, my Python knowledge and skills improved a lot because of the highest quality of code reviews that I got from my mentor and peers. I learnt many best practices, learnt how to use pytest properly. I am aware that many non-traditional Python programmers who take up Python because it serves their purpose (like Data science/ Machine Learning) do not know the true power and beauty of Python. Hence, I am starting this new blog series to document some important Python principles that I learned from the book [Effective Python](https://effectivepython.com/), and I hope that this could help someone overcome the obstacles that I faced during my long and awkward journey to learning to write effective Python.  

I'll try to write two kinds of blog posts: one will be a summary of what I learned from each chapter of [Effective Python](https://effectivepython.com/) and the other will be a deep dive in some topic that I found important or interesting.

This post lists and explains my learnings from Chapter 1 : Pythonic Thinking, [Effective Python](https://effectivepython.com/). Most examples are borrowed from the book. I **strongly** encourange folks to get this Python book!

We'll first go over some aspects of writing clean and readable Python code, then we'll dive into some cool Python constructs like slicing, list comprehension, generator expressions. We'll take a look at how to loop effectively, how to catch exceptions correctly, differences in string data type in Python 2 v/s 3, and we'll end the post with a brief overview of different Python runtimes.

### Writing clean Python code:  

* **What is Pythonic thinking?**   
Pythonic thinking can be described as a style of coding that has become the "standard" way of doing things in Python. These are not strict compiler enforced rules that will throw errors if not followed. These are more subtle ways of making Python a practical and beautiful language to code in. Hence, it is important to know how to do the most common things correctly in Python, i.e. in a **Pythonic** way. In this section, we'll briefly go over some core concepts behind Pythonic thinking and coding.    

	**Be explicit:** Your intent in the code should be explicit to the readers of your code   
	For example: When you want to invoke a function import its module explicitly  
	So, instead of:  
	{% highlight python %}
	from os import *  
	print(getcwd())
	# do:  
	import os  
	print(os.getcwd())
	{% endhighlight %}
	    

	**Prefer simple over complex:**    
	
	For example:
	2.1. use the boolean variable in the condition instead of comparing the boolean variable with  `True` or `False`  

	{% highlight python %}
	# instead of this:
	my_bool_val = 10 > 5
	if my_bool_val == True:
	    print ('my_bool_val is True')  	    
	# do this: 
	my_bool_val = 10 > 5
	if my_bool_val:
	    print ('my_bool_val is True')
	{% endhighlight %}

	2.2. Break complicated single-line expressions into small helper functions especially if they are used often. Let's say you have a dictionary where the values are in list of size 1, the element in the list is string representation of integer, and the values could be a list with empty string or just None. You want to use default value of 0 in that case. One way of handling it could be:
	{% highlight python %}
	val = my_dict.get(my_key,  [''])[0] or 0			    
	{% endhighlight %}
	Now, this above complicated expression looks fine if it is used just once, but you might need to use this several times for reading numerous keys from your dict. Also, what if you decide to change the default value? Then you'll need to change it for all the `get` statements. To tackle this, you could make a small helper function like this: 
	
				def get_first_val(my_dict, key, default = 0):
				    val_found = my_dict.get(key, [''])
				    if val_found[0]:
				        val_found = int(val_found[0])
				    else:
				        val_found = default
				    return val_found
			This allows you to separate out a complex dictionary reading, write a testable piece of code and gives you flexibility to change the default value.
	
	* **Maximize readability:** 
	Your code may be processed by a machine but it will be read and consumed repeatedly by humans, so be nice to your fellow species. As the author of Python, Guido van Rossum said:
>  **Code is read much more often than it is written.** 

		There's a lot that goes into maximizing readibility, ranging from organizing imports, to using blank lines and spaces at the right places, to writing helper functions to remove complicated code blocks, and finally to write helpful meaningful comments and docstrings. Best suggestion is to go over PEP-8 style guide document.
* **What is PEP8?**
	* [PEP8](https://www.python.org/dev/peps/pep-0008/) is basically the official Python coding convention guide; to improve the readability and estabilish consistency of Python code
	* It primarily describes how to neatly lay out the code, use whitespaces and commenting. It also dives into naming conventions and some general programming recommendations to keep in mind, some of which will be highlighted here.
	* I strongly recommend people to at least give PEP8 a light read, and try to remember the important conventions. But as the authors themselves say, use your best judgement in deciding when to be inconsistent with the PEP8 guide. I like to keep this quote in mind from the PEP8 guide:  
 
>  **Consistency with this style guide is important. Consistency within a project is more important. Consistency within one module or function is the most important.**
	
* **Why does whitespace matter in Python?**
	*	Whitespace is a crucial part of Python syntax; and this may seem a bit odd to developers moving from languages like C++ or Java where braces `{` , `}` and `;` are important part of the syntax. Check out this [link](https://www.python.org/dev/peps/pep-0008/#whitespace-in-expressions-and-statements) for PEP8 conventions on whitespaces.
	* Use spaces instead of tabs: yes, tabs are definitely easier to type than typing 4 spaces, so perhaps configure your IDE to convert tab to 4 spaces. Most IDEs provide this functionality, and it helps you write awesome Pythonic code. For example, follow [this](https://stackoverflow.com/questions/11816147/pycharm-convert-tabs-to-spaces-automatically) to do configure PyCharm
	* Blank lines are important too in long files to separate functions and classes: 2 blank lines between different functions, 1 blank line between class methods, use 1 blank line sparingly to indicate logical sections in a function
	
* **Naming stuff in Python?**
	* Again, a major switch for Java developers is to not go for camelCase all the time. In Python, `lowercase_underscore` format is most common for all variables and functions.
	* Protected and private attributes are denoted by `_leading_underscore` and `__double_leading_underscore`
	* Use `CamelCase` for declaring classes
	* For constants, use `ALL_CAPS`
* **Organizing imports, statements and code blocks**
	* Use `if somelist` instead of `if len(somelist) != 0`, because empty values evaluate to `False` in python. Use that for clarity.
	* Split long if/for/while statements to multiple lines to improve readability
	* Always put import statements at top and use absolute import names. For example:
	use `from foo import bar` instead of  
	`import bar`
	* Organize imports: standard library modules, then 3rd party modules, then own modulels. All of them in alphabetical order in their own group.
	* Now that's a lot to keep in mind, do I focus on coding or do I practice the so-called Pythonic principles. Well, fear not, because Pylint is here.  
	
* **What is Pylint tool?**  
	* Well, first of all we should know what is `lint/linter`. It is basically a piece of software or a tool that performs static analysis on your code for programming bugs, stylistic errors, and symantic discrepancies.
	* [Pylint](https://www.pylint.org/) is one such popular tool for Python. It helps you follow PEP-8 conventions without remembering them all, especially for length of a line of code, variable naming convention, import organization, unused imports/variables, some basic error detection. One can also customize Pylint based on what conventions they do/do not want to follow by modifying your `pylintrc` file
	* Simplest way of using Pylint is to run  `pylint your_python_file.py`
	* Pylint can be configured in your [IDE](https://kirankoduru.github.io/python/pylint-with-pycharm.html), which is awesome so that while developing itself you can take care of writing clean readable Python code
	* Another way of using Pylint especially if you are working on OSS, or at a company where you definitely have a git repo and you want to maintain the sanctity of Python code, is to add pylint in your pre-commit hook like [this] (https://github.com/pre-commit/mirrors-pylint). This ensures that anything that you commit is PEP-8 compatible
	* I'd strongly recommend reading the [Pylint doc](https://pylint.readthedocs.io/en/latest/intro.html) before actually using it so that you understand what it is, and what it is not.
	* There are other similar linters as well: flake8 and pycodestyle are particularly popular.
	* Refer [this](https://jeffknupp.com/blog/2016/12/09/how-python-linters-will-save-your-large-python-project/) helpful article on incorporating linters in your build process if you are creating a new service or a large project in Python.

### Writing effective Python code:
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

			x = b'apple'
			y = x[::-1]
			print(x) # apple
			print(y) # elppa
	But this doesn't work if the string is UTF-8 encoded.
	* How readable is a code with a bunch of statements like this: 
	
			x = a[2:10:-2]
			y = b[1:20:3] 
			z = c[10:-2:-1]. 
	To me, it seems sort of readable because **I** wrote it and it's just a couple of statements. Now imagine a legacy codebase that you're trying to understand that is full of these expressions. Moreover, imagine that these indices are variables instead of integer values. Now that may become a nightmare for you to parse. Hence, it is best to avoid using start, end, and slice all in the same expression. If you really have to use stride, then skip start/end, and if you can't then break it into two expressions. So instead of the above listed expressions, we can split them like: 
			
			temp = a[::-2]
			x = temp[2:10]
			
			temp = b[::3]
			y = temp[1:20]
			
			temp = c[10:-2]
			z = temp[::-1]

#### List comprehensions
* **What are list comprehensions:**
	* Python provides a compact and beautiful way of deriving one list from the other, especially if the same function is applied to most of the elements in the original list. Say, you want to square every number in a list. These are some ways of doing it.
	
			a = [1, 2, 3, 4, 5, 6, 7, 8]
			
			# Method 1: Using for loop
			even_squares = []
			for elem in a:
			    even_squares.add(elem*elem)
			        
			# Method 2: Using map()
			even_squares = map(lambda x: x*x, a)
			
			# Method 3: Using list comprehension
			even_squares = [x*x for x in a]
		As one can see, both method 2 and 3 are compact and readable. I strongly prefer list comprehension over map(), and this example from the book illustrates why. Say you want to compute squares of all even numbers in the list, then the above methods look like:
		
			# Method 1: Using for loop
			even_squares = []
			for elem in a:
				if elem % 2 == 0:
			   		even_squares.add(elem*elem)
			        
			# Method 2: Using map()
			even_squares = map(lambda x: x*x, filter(lambda x: x % 2 == 0, a)
			
			# Method 3: Using list comprehension
			even_squares = [x*x for x in a if x % 2 == 0]
		In my opinion, list comprehensions become much easier to read once `filter` gets involved with `map`.
	* Note: list comprehension syntax can be applied to `set` and `dict` as well.
				 
* **When to not use list comprehensions**
	* Avoid list comprehensions the moment they become unreadable, or when they start taking as much space as using for loops would. I think after that point one is just being a pretentious Python programmer :p
	* One common scenario is multiple level of comprehension. Something like: `flat = [x for row in matrix for x in row] # flattens a 2D-matrix` is a good use of list comprehension, but the moment this becomes more than 2 level, it becomes difficult to read. Say you had a 3D array to be flatted, the code would look like:
	
			# Method 1: Using list comprehension:
			flat = [x for level_1 in matrix 
					 for level_2 in level_1 
					 for x in level_3]
					 
			# Method 2: Using for loops:
			flat = []
			for level_1 in matrix:
				for level_2 in matrix:
					flat.extend(level_2)
	It seems pretty clear that for-loop is more readable here and less confusing too.
	* One major issue with list comprehension is that it is not scalable, in the sense it consumes a lot of memory in creating and holding the new list in memory. This is not ideal in dealing with big data, or large streams of infinite data, or while reading a big file. One common application in Machine Learning is reading large files, parsing it line by line, selecting some parts of the line as features and storing it in lists. 

			# let parse_and_extract_features(line) parse a line and extract features
			# let dataset.txt contain 1M rows. Then, 
			file_name = 'dataset.csv'
			output_value = [parse_and_extract_features(x) for x in open(file_name)]
			output_file = open('output.csv', w)
			output_file.write(output_value)
			# output_value will be a list of 1M items and we want to write it to a new file
			

		In this case, a list of 1M items will be generated only to be written to the file and never used again. A better approach in this case is to use **generator expressions**.
		
			def genexp_extract_features(file_name):
				# generator to parse feature for a given file
				for row in open(file_name):
				    yield parse_and_extract_features(row)
			file_name = 'dataset.csv'
			output_generator = genexp_extract_features(file_name)
			output_file = open('output.csv', w)
			for output_value in output_generator:
				output_file.write(output_value)
			
		By using generator, we do not create a list of 1M items in memory and instead generate feature for one row at a time by creating output_generator object and then iterating over it. Note, we can iterate over genexp like this because essentially they are like iterators. 		
		
	* In conclusion, generator expresssions are a more general form of list comprehension. They are also more efficient because they do not create the entire list in memory. Instead genexp (in short) evaluates to an iterator, that can be used to generate the result-list, one item at a time by calling next() on the evaluated iterator. In the above example, we could use genexp instead of list comprehension to evaluate and write one line at a time to the transformed_dataset file.
	* One thing to keep in mind is that genexp evaluates to iterators and once you call next() on it, that result is evaluated and returned and that value cannot be obtained again.

#### Looping effectively
* **Avoiding range / Using enumerate** 
	* Note: In Python, always use `xrange` instead of `range`. Difference being `xrange` generates an iterator to iterate over whereas `range` produces actual list which can be memory inefficient when iterating over a large range. In Python 3, there's only `range` function and it's a generator.

	* For people coming from languages like Java and C++, Python looping may seem a little strange initially because the most common way of looping in Python does not give you access to indices, rather it gives you the elements.
So instead of doing this:

			states = ['California', 'North Carolina', 'Texas', 'Nevada']
			n = len(states)
			for i in range(n):
			    print (states[i])
		    
		do this:

			states = ['California', 'North Carolina', 'Texas', 'Nevada']
			for state in states:
			    print (state)
It seems pretty clear that the second option is more readable, and does the job effectively.

	* Now there may arise a case when you absolutely do need the index of the element in the list. In that case, one might think range is the way to go. But there's a better option: *enumerate* 

		Instead of doing this:

			states = ['California', 'North Carolina', 'Texas', 'Nevada']
			n = len(states)
			for i in range(n):
			    print (i, states[i])
			    
		do this:

			states = ['California', 'North Carolina', 'Texas', 'Nevada']
			for i, state in enumerate(states):
			    print (i, state)
			
	* **Enumerate** function takes anything that's iterable and creates a lazy generator. This generator spits out two things: index and the next element from the iterator. It is essentially made for this task.  
Note: You can also provide enumerate an additional argument, a start index, which the function uses to start its indexing (by default it begins with index=0) 

* **Using zip / izip:**
	* Sometimes you want to loop over multiple lists of same length which often happens at least in data science tasks, when you're looping over multiple feature list for your dataset, where each list is of size n, where n = number of examples in dataset. In that case, one might write such loop as:

			for i in range(len(dataset)):
			    feat_1 = feature_1[i]
			    feat_2 = feature_2[i]
			    print (feat_1, feat_2)
			    
		whereas, zip provides a useful generator that takes in multiple iterators and yields tuples containing the next value from each iterator (in the same order as they are given to zip)

			for feat_1, feat_2 in zip(feature_1, feature_2):
			    print (feat_1, feat_2)
Result: More readable, like a pseudocode.
	* Things to keep in mind with zip:
		* In Python 2, zip is not a generator, it actually evaluates both the lists and thus may cause memory issues if the lists are massive. Tip: Use izip in Python from itertools module.
		* If the length of the iterables provided to zip are not same, then zip will stop iterating as soon as any of the iterable is exhausted, so keep that in mind. 	

#### Understanding try, catch, else, except, finally
* Using try and finally:
	* Use this when you want exceptions to propagate up to the caller, but also want to do cleanup.
	
			handle = open('data.txt')
			try:
			    data = handle.read()
			finally:
			    handle.close()
	This ensures that if the exception occurs at **open** then we won't attempt at closing the file handler, which is correct. We will only call close() if try block is attempted, which means open() succeeded.  
	PS: I always used to write open() inside the try block with a similar finally clause, and never thought about it. So I am glad to learn this new bit. 
	
* Using try, except, and else:
	* By using this trio we can make it clear which exceptions will be handled by our code and which caller is suppose to handle. When try runs successfully, else block runs. It kind of helps minimize the code in try block so you can keep the try block only for code that can raise exceptions and do other computations in else block.
	
			def load_json_key(json_data, key):
			    try:
			        data_dict = json.loads(json_data)
			    except ValueError as e:
			        raise KeyError as e
			    else:
			        return data_dict[key]
	So if json loading fails, that will be caught by our try except block. If it goes through, then else will try to extract the value. If it can't find the key, it'll raise that error to the caller. This is done to make error propagation clear.
	
* Using try, except, else, and finally:
	* In this case,  consider an example where we read something from a file, perform some computation, and return the result of that computation. In this case, we can have try block attempting to read the file; except catches an exception that try block is expected to throw; else block runs the computation if try block runs successfully; and finally block does the clean up like closing the file handler.
			    
#### Knowing the difference between str, unicode, and bytes
* This is something I wasn't fully aware of. I had come across both unicode and str, but I never needed to delve into details about their differences, and especially how they behave in Python 2 v/s 3. As someone who works on projects in both Python 2 and Python 3 regularly, this is definitely something I should keep in mind.
	
	| Contains  | Python 2 data-type | Python 3 data-type |
	|-----------|--------------------|--------------------|
	| raw 8-bit | str                | bytes              |
	| unicode   | unicode            | str                |


* Also, in Python 3, the encoding argument for open() defaults to UTF-8, which makes read and write operations expect the str data type containing unicode characters instead of bytes data type containing binary data. One may encounter this issue while writing binary data to a .bin file. In Python 2, you can open with 'w' argument for writing, whereas in Python 3, you've to use 'wb' argument for writing to a binary file. 

## Conclusion
In this post, we saw several techniques to make your code more Pythonic, more readable, clear, concise, and efficient. Note that most of these skills build up over time. But it's helpful to kno about them so that we you see them in other people's code, you can appreciate and understand its intent better.
