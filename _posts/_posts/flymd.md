---
layout: post
title: "Data Science Lessons Learned the Hard Way: Coding"
date: 2019-05-19 08:44:27 -0500
comments: true
categories: [data science]
---

You could summarize this post as "you will never regret good code practices" or "no project is too small for good code practices".

You might think these recommendations are not worth the time when a project seems small, but projects often grow over time. If you use good practices from the start, you will reduce the [technical debt](https://en.wikipedia.org/wiki/Technical_debt) your project accrues over time.

Here's my list of coding Data Science lessons learned the hard way.

1. You will never regret using git.

   You might think, "this project/query is just 15 minutes of code and I will never think about it again". While this might be true, it often is not. If your project/query is useful, people will ask for it again with slight tweaks. With each ask, the project grows a little. By using git, you persist the project at all change points, acknowledge that the project  will change over time, and prepare for multiple contributors.
   <br>
   Even if you never use these features, I've found that simply using git encourages other good practices.
   Also, remember [git can rescue](https://ohshitgit.com/) you when things go wrong!

2. You will never regret good documentation.

   Again, you might think, "this project is so tiny and simple, how could I ever forget how it works??". You will forget. Or another contributor will appreciate documentation.<br>
   The [numpy documentation framework](https://docs.scipy.org/doc/numpy/docs/howto_document.html) is great when working in python. Its [integration](https://numpydoc.readthedocs.io/en/latest/) with [sphinx ](http://www.sphinx-doc.org/en/stable/) can save you a lot of time when creating non-code documentation.<br>
   I recently started documenting not only *what the code is doing*, but the business rule dictating *what the code should do*. Having both lets contributors know not only know the *how* of the code but also the *why*.

3. You will never regret building unit-tests.

   Again, this might feel like over-kill in small projects, but even small projects have assumptions that should be tested. This is especially true when you add new features after stepping away from a project. By including unit-tests, you assure yourself that existing features did not break, making those pushes to production [less scary](https://dev.to/quii/why-you-should-deploy-on-friday-afternoon-285h).

4. Take the time to build infrastructure for gathering/generating sample/fake data.

   I've found myself hesitant to build unit-tests because it's hard to acquire/generate useful sample/fake data. Do not let this be a blocker to good code practices! Take the time to build infrastructure that makes good code practices easy.
   This could mean taking the time to write code for building fake data. This could mean taking the time to acquire useful sample data. Maybe it's both! Take the time to do it. You will not regret making it easy to write tests.

5. You will always find a Makefile useful.

   Once you've built infrastructure for acquiring fake or sample data, you will need a way to bring this data into your current project. I've found [Makefiles](https://en.wikipedia.org/wiki/Makefile) useful for this sort of thing. You can define a command that will download some sample data from s3 (or wherever) and save it to your repo (but don't track these files on git!).<br>
   This way all contributors will have common testing data, stored outside of git, and can acquire this data with a single, easy to remember, command.<br>
   MakeFiles are also great for installing or saving a project's dependencies.

6. Know your project's dependencies.

   Code ecosystems change over time. When you revisit a project after a break, the last thing you want is to guess what code dependencies have broken.
   It doesn't matter whether you save your project's dependencies as a anaconda environment, a requirements file, virtualenv, whatever. Just make sure to save it. Any future contributors (including yourself!!) will thank you.fLyMd-mAkEr

Most these individual points seem obvious. The overarching point is no project is too small for good code practices! Sure you might think, oh this is just a single query, but you will run that query again, or another member of your team will! While you shouldn't build a repo for each query, building a [repo for different sets of queries is not a bad idea](https://caitlinhudon.com/2018/11/28/git-sql-together/).
   