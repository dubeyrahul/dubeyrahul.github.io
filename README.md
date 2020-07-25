# Blog
This repository contains code for my Personal blog, where I mainly talk about Python, ML, Data, and Distributed Systems.

## Note to self:
- bundle exec jekyll build  
- bundle exec jekyll serve  
- `/sync` before notebook push

## Notebook push:
Based on: https://github.com/fastai/fastpages

Create a Jupyter Notebook with the contents of your blog post.

Create a markdown cell at the beginning of your notebook with the following contents:

# Title
> Awesome summary
- toc: False
- metadata_key1: metadata_value1
- metadata_key2: metadata_value2
Replace Title, with your desired title, and Awesome summary with your desired summary.
fast_template will automatically generate a table of contents for you based on markdown headers! You can toggle this feature on or off by setting toc: to either True or False.
Additional metadata is optional and allows you to set custom front matter.
Save your notebook with the naming convention YYYY-MM-DD-*.ipynb into the /_notebooks folder of this repo. For example 2020-01-28-My-First-Post.ipynb. This naming convention is required by Jekyll to render your blog post.

Be careful to name your file correctly! It is easy to forget the last dash in YYYY-MM-DD-. Furthermore, the character immediately following the dash should only be an alphabetical letter. Examples of valid filenames are:

2020-01-28-My-First-Post.ipynb
2012-09-12-how-to-write-a-blog.ipynb
If you fail to name your file correctly, fast_template will automatically attempt to fix the problem by prepending the last modified date of your notebook to your generated blog post in /_posts, however, it is recommended that you name your files properly yourself for more transparency.

Commit and push your notebook to GitHub. Important: At least one of your commit messages prior to pushing your notebook(s) must contain the word /sync in order to trigger automatic notebook conversion. Furthermore, the automatic conversion only occurs when a push is made to the master branch.

The requirement of the /sync keyword is designed to alleviate instances of unwanted local conflicts that would otherwise require you to pull a fresh copy of your repo after each commit. When a notebook is converted to a blog post, a new file is committed to your repo automatically. See How Does it Work? for more details and with instructions on customizing this behavior.
