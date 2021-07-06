---
layout: post
title: "Sifting the Overflow"
date: 2017-03-04 10:11:45 -0500
comments: true
categories: [python, open source, insight, data science]
---

*DISCLAIMER: I PULLED DOWN THE EC2 INSTANCE SUPPORTING THIS PROJECT*

In January 2017, I started a fellowship at [Insight Data Science](http://insightdatascience.com/). Insight is a 7 week program for helping academics transition from academia to careers in data science. In the first 4 weeks, fellows build data science products, and fellows present these products to different companies in the last 3 weeks.  

At Insight, I built [Sifting the Overflow](http://siftingtheoverflow.com/) (this link is broken since I pulled down the ec2 instance), a chrome extension which you can install from the [google chrome store](https://chrome.google.com/webstore/detail/sifting-the-overflow/japbeffaagcpbjilckaoigpocdgncind?hl=en-US&gl=US). Sifting the Overflow identifies the most helpful parts of answers to questions about the programming language Python on [StackOverflow.com](http://stackoverflow.com/). To created Sifting the Overflow, I trained a recurrent neural net (RNN) to identify "helpful" answers, and when you use the browser extension on a stackoverflow page, this RNN rates the helpfulness of each sentence of each answer. The sentences that my model believes to be helpful are highlighted so that users can quickly find the most helpful parts of these pages.

I wrote a quick post [here](http://siftingtheoverflow.com/) about how I built Sifting the Overflow, so check it out if you're interested. The code is also available on my [github](https://github.com/dvatterott/stackex_sum).
