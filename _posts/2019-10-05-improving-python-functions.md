---
layout: post
title: "Writing better Python functions"
date: 2019-10-05
---
This is the 3rd post on writing Effective Python code. In this post, we will go over writing Python functions in a better, more Pythonic way. We will explore concepts such as: functions as first-class objects, closures, decorators, variable scoping, generators. In the end we will go over some techniques of passing arguments to a function to make it less error prone and enforce clarity.

+ [Functions as first-class object](#functions-as-first-class-object)
  - [what does it mean to be a first-class object](#first-class-objects)
  - [Examples of functions treated as first-class-object](#examples-of-functions-as-first-class-object)
+ [Closures][#closures]
  - [what is a closure](#what-is-a-closure)
  - [when-to-use-closures](#when-to-use-closures)
  - [variable scoping in closures](#variable-scoping-in-closures)
  - [decorators](#how-decorators-use-closures)
+ [Generators][#generators]
  - [what is a generator](#what-is-a-generator)
  - [passing generator as argument](#passing-generator-as-argument)
  - [iterators](#passing-iterator-as-argument)
+ [Effective argument passing][#effective-argument-passing]
  - [different types of arguments](#different-types-of-arguments)
  - [when to use what](#when-to-use-what)
* [Conclusion](#conclusion)
