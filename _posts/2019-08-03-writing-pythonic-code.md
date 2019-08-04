---
layout: post
title: "Python tips: Part 1 - Writing clean Python code"
date: 2019-08-03
---

In this post, we will go over some basic techniques of making your Python code more readable by following some simple Pythonic principles laid out in PEP-8.


+ [Why should we learn to write Pythonic code](#why-should-we-learn-to-write-pythonic-code)
+ [Writing clean Python code:](#writing-clean-python-code)
  - [Pythonic thinking?](#what-is-pythonic-thinking)
  - [Be explicit](#be-explicit)
  - [Prefer simple over complex](#prefer-simple-over-complex)
  - [Maximize readability](#maximize-readability)
* [Conclusion](#conclusion)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

### Why should we learn to write Pythonic code
Before starting any task, it is very important to understand the *why*, hence I am going to describe my *why*. I am a Machine Learning Engineer at Yelp and I write Python every day, so one may wonder why am I learning to write Pythonic code *now*.  

Like many machine learning engineers, Python was not my first programming language. In fact, I never learned Python from a proper textbook like I learned Java or C++ (ages ago). The sole reason I started to code in Python is because I wanted to learn Machine Learning, and wanted to learn it quickly. And Python is ideal for that because of [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) and the [PyData](https://pydata.org/) stack comprising of popular libraries such as NumPy, SciPy, Pandas and scikit-learn to name a few. These packages along with the ability to quickly prototype models in Python gives Python an edge in Machine Learning community.

So I started learning Python on a case-to-case basis, meaning, I'd only learn certain topics that I need to perform a certain Machine Learning task. It ranged from data scraping from web, calling REST APIs, doing a lot of file handling, and obviously writing machine learning code (either from scratch, or using scikit-learn). Thus, I never learned how to write Pythonic OOP, tests, closures, or treating functions as first-class object. Even though I used some of these features incidentally, I never learned it thoroughly or the reasoning behind it.

But all of this changed when I started my graduate studies and started programming extensively in Python (again, the reason being that I was specializing in machine learning). I especially started using NumPy and Scikit-learn a lot, learned a bit about reading and processing large files using generators (and was amazed by it). I wrote some simple closures, wrote some functions that I could pass as parameter to other scikit-learn functions (thereby treating functions as first class objects). But I reached a state where I was doing things very ineffectively and inconsistently. I'd repeat myself a lot, write really complicated list comprehensions, totally forget that generators exist, wouldn't use closures as effectively as they can be used. Also, I started reading more Python code which made me realize that I need to **relearn** Python and let go of old bad habits.

After I joined Yelp as Machine Learning Engineer, my Python knowledge and skills improved a lot because of the highest quality of code reviews that I got from my mentor and peers. I learnt many best practices, learnt how to use pytest properly. I am aware that many non-traditional Python programmers who take up Python because it serves their purpose (like Data science/ Machine Learning) do not know the true power and beauty of Python. Hence, I am starting this new blog series to document some important Python principles that I learned from the book [Effective Python](https://effectivepython.com/), and I hope that this could help someone overcome the obstacles that I faced during my long and awkward journey to learning to write effective Python.  

I'll try to write two kinds of blog posts: one will be a summary of what I learned from each chapter of [Effective Python](https://effectivepython.com/) and the other will be a deep dive in some topic that I found important or interesting.

This post lists and explains my learnings from Chapter 1 : Pythonic Thinking, [Effective Python](https://effectivepython.com/). Most examples are borrowed from the book. I **strongly** encourange folks to get this Python book!

In this post, we'll only cover aspects of writing clean and readable Python code and in the subsequent Python posts, we'll dive deeper into writing effective and efficient Python code.

### Writing clean Python code

#### What is Pythonic thinking?
Pythonic thinking can be described as a style of coding that has become the "standard" way of doing things in Python. These are not strict compiler enforced rules that will throw errors if not followed. These are more subtle ways of making Python a practical and beautiful language to code in. Hence, it is important to know how to do the most common things correctly in Python, i.e. in a **Pythonic** way. In this section, we'll briefly go over some core concepts behind Pythonic thinking and coding.    

#### Be explicit
Your intent in the code should be explicit to the readers of your code   
For example: When you want to invoke a function import its module explicitly  
So, instead of:  
{% highlight python %}
from os import *  
print(getcwd())
# do:  
import os  
print(os.getcwd())
{% endhighlight %}
    

#### Prefer simple over complex

- use the boolean variable in the condition instead of comparing the boolean variable with  `True` or `False`  

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


- Break complicated single-line expressions into small helper functions especially if they are used often. Let's say you have a dictionary where the values are in list of size 1, the element in the list is string representation of integer, and the values could be a list with empty string or just None. You want to use default value of 0 in that case. One way of handling it could be:

{% highlight python %}
val = my_dict.get(my_key,  [''])[0] or 0			    
{% endhighlight %}
Now, this above complicated expression looks fine if it is used just once, but you might need to use this several times for reading numerous keys from your dict. Also, what if you decide to change the default value? Then you'll need to change it for all the `get` statements. To tackle this, you could make a small helper function like this: 

{% highlight python %}
def get_first_val(my_dict, key, default = 0):
    val_found = my_dict.get(key, [''])
    if val_found[0]:
        val_found = int(val_found[0])
    else:
        val_found = default
    return val_found
{% endhighlight %}

This allows you to separate out a complex dictionary reading, write a testable piece of code and gives you flexibility to change the default value.
	
#### Maximize readability
Your code may be processed by a machine but it will be read and consumed repeatedly by humans, so be nice to your fellow species. As the author of Python, Guido van Rossum said:
>  **Code is read much more often than it is written.** 

There's a lot that goes into maximizing readibility: organizing imports, using blank lines and spaces at the right places, writing helper functions to remove complicated code blocks, write helpful meaningful comments and docstrings, etc. Best suggestion is to go over PEP-8 style guide document.
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
	* Another way of using Pylint especially if you are working on OSS, or at a company where you definitely have a git repo and you want to maintain the sanctity of Python code, is to add pylint in your pre-commit hook like [this](https://github.com/pre-commit/mirrors-pylint). This ensures that anything that you commit is PEP-8 compatible
	* I'd strongly recommend reading the [Pylint doc](https://pylint.readthedocs.io/en/latest/intro.html) before actually using it so that you understand what it is, and what it is not.
	* There are other similar linters as well: flake8 and pycodestyle are particularly popular.
	* Refer [this](https://jeffknupp.com/blog/2016/12/09/how-python-linters-will-save-your-large-python-project/) helpful article on incorporating linters in your build process if you are creating a new service or a large project in Python.

## Conclusion
In this post, we saw several techniques to make your code more Pythonic, more readable, clear, and concise. We learned that we can apply most of these techniques by using automated tools and make our code more Pythonic. In the [next post](/blog/2019/08/04/writing-effective-python-code) we'll go over writing effective Python code.
