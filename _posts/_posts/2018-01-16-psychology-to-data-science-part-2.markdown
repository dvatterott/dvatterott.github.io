---
layout: post
title: "Psychology to Data Science: Part 2"
date: 2018-01-16 20:18:34 -0600
comments: true
categories: [data science, psychology, academia, industry]
---

This is the second post in a series of posts about moving from a PhD in Psychology/Cognitive Psychology/Cognitive Neuroscience to data science. The [first post](https://danvatterott.com/blog/2018/01/10/psychology-to-data-science-part-1/) answers many of the best and most common questions I've heard about my transition. This post focuses on the technical skills that are often necessary for landing a data science job.

Each header in this post represents a different technical area. Following the header I describe what I would know before walking into an interview.

### SQL
SQL is not often used in academia, but it's probably the most important skill in data science (how do you think you'll get your data??). It's used every day by data scientists at every company, and while it's 100% necessary to know, it's stupidly boring to learn. But, once you get the hang of it, it's a fun language because it requires a lot of creativity. To learn SQL, I would start by doing the [mode analytics tutorials](https://community.modeanalytics.com/sql/tutorial/introduction-to-sql/), then the [sql zoo](http://sqlzoo.net/) problems. [Installing postgres on your personal computer](https://www.fullstackpython.com/blog/postgresql-python-3-psycopg2-ubuntu-1604.html) and fetching data in Python with psycopg2 or sql-alchemy is a good idea. After, completing all this, move onto query optimization (where the creativity comes into play) - check out the [explain function](https://www.postgresql.org/docs/9.3/static/sql-explain.html) and [order of execution](https://stackoverflow.com/questions/2617661/whats-the-execute-order-of-the-different-parts-of-a-sql-select-statement). Shameless self promotion: I made a [SQL presentation](https://danvatterott.com/presentations/sql_presentation/index.html#_blank) on what SQL problems to know for job interviews.

### Python/R
Some places use R. Some places use Python. It sucks, but these languages are not interchangeable (an R team will not hire someone who only knows Python). Whatever language you choose, you should know it well because this is a tool you will use every day. I use Python, so what follows is specific to Python.

I learned Python with [codeacademy](https://www.codecademy.com/) and liked it. If you’re already familiar with Python I would practice "white board" style questions. Feeling comfortable with the beginner questions on a site like [leetcode](https://leetcode.com/) or [hackerrank](https://www.hackerrank.com/) would be a good idea. Writing answers while thinking about code optimization is a plus.

[Jeff Knupp's blog](https://jeffknupp.com/) has great tid-bits about developing in python; it's pure gold.

Another good way to learn is to work on your digital profile. If you haven't already, I would start a blog (I talk more about this is [Post 1](https://danvatterott.com/blog/2018/01/10/psychology-to-data-science-part-1/)).

### Statistics/ML
When starting here, the Andrew Ng [coursera course](https://www.coursera.org/learn/machine-learning) is a great intro. While it's impossible to learn all of it, I love to use [elements of statistical learning](https://web.stanford.edu/~hastie/ElemStatLearn/) and it’s sibling book [introduction to statistical learning](http://www-bcf.usc.edu/~gareth/ISL/) as a reference. I've heard good things about [Python Machine Learning](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130) but haven't checked it out myself.

As a psychology major, I felt relatively well prepared in this regard. Experience with linear-mixed effects, hypothesis-testing, regression, etc. serves Psychology PhDs well. This doesn't mean you can forget Stats 101 though. Once, I found myself uncomfortably surprised by a very basic probability question.

Here’s a quick list of Statistics/ML algorithms I often use: GLMs and their regularization methods are a must (L1 and L2 regularization probably come up in 75% of phone screens). Hyper-parameter search. Cross-validation! Tree-based models (e.g., random forests, boosted decision trees). I often use XGBoost and have found its [intro post](http://xgboost.readthedocs.io/en/latest/model.html) helpful.

I think you’re better off deeply (pun not intended) learning the basics (e.g., linear and logistic regression) than learning a smattering of newer, fancier methods (e.g., deep learning). This means thinking about linear regression from first principles (what are the assumptions and given these assumptions can you derive the best-fit parameters of a linear regression?). I can’t tell you how many hours I’ve spent studying Andrew Ng’s [first supervised learning lecture](http://cs229.stanford.edu/notes/cs229-notes1.pdf) for this. It’s good to freshen up on linear algebra and there isn’t a better way to do this than the [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) videos; they're amazing. This might seem too introductory/theoretical, but it’s necessary and often comes up in interviews.

Be prepared to talk about the bias-variance tradeoff. Everything in ML comes back to the bias-variance tradeoff so it's a great interview question. I know some people like to ask candidates about feature selection. I think this question is basically a rephrasing of the bias-variance tradeoff.

### Git/Code Etiquette
Make a github account if you haven't already. Get used to commits, pushing, and branching. This won't take long to get the hang of, but, again, it's something you will use every day.

As much as possible I would watch code etiquette. I know this seems anal, but it matters to some people (myself included), and having pep8 quality code can't hurt. There's a number of [python modules](https://pylint.readthedocs.io/en/latest/) that will help here. Jeff Knupp also has a [great post](https://jeffknupp.com/blog/2016/12/09/how-python-linters-will-save-your-large-python-project/) about linting/automating code etiquette.

Unit-tests are a good thing to practice/be familiar with. Like usual, Jeff Knupp has a great [post](https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/) on the topic.

I want to mention that getting a data science job is a little like getting a grant. Each time you apply, there is a low chance of getting the job/grant (luckily, there are many more jobs than grants). When creating your application/grant, it's important to find ways to get people excited about your application/grant (e.g., showing off your statistical chops). This is where code etiquette comes into play. The last thing you want is to diminish someone's excitement about you because you didn't include a doc string. Is code etiquette going to remove you from contention for a job? Probably not. But it could diminish someone's excitement.

### Final Thoughts
One set of skills that I haven't touched on is cluster computing (e.g., Hadoop, Spark). Unfortunately, I don't think there is much you can do here. I’ve heard good things about the book [Learning Spark](http://shop.oreilly.com/product/0636920028512.do), but books can only get you so far. If you apply for a job that wants Spark, I would install Spark on your local computer and play around, but it's hard to learn cluster computing when you're not on a cluster. Spark is more or less fancy SQL (aside from the ML aspects), so learning SQL is a good way to prepare for a Spark mindset. I didn’t include cluster computing above, because many teams seem okay with employees learning this on the job.

Not that there's a lack of content here, but [here](https://blog.insightdatascience.com/preparing-for-insight-ca7cc6087f91)’s a good list of must know topics that I used when transitioning from academia to data science.
