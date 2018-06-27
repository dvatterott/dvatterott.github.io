---
layout: post
title: "Using Cron to automate jobs on Ubuntu"
date: 2017-09-01 18:04:17 -0400
comments: true
categories: [open source, ubuntu, cron, automation, python]
---

I recently spent an entire afternoon debugging a solution for automatically launching a weekly emr job.

Hopefully, I can save someone the same pain by writing this blog post.

I decided to use Cron to launch the weekly jobs. Actually launching a weekly job on Cron was not difficult. Check out the [Ubuntu Cron manual](https://help.ubuntu.com/community/CronHowto) for a good description on how to use Cron.

What took me forever was realizing that **Cron jobs have an extremely limited path**. Because of this, specifying the complete path to executed files **and their executors** is necessary.

Below I describe how I used an ec2 instance (Ubuntu 16.04) to automatically launch this weekly job.

First, here is what my Cron job list looks like (call "crontab -e" in the terminal).

{% codeblock lang:bash %}
SHELL=/bin/bash
05 01 * * 2 $HOME/automated_jobs/production_cluster.sh
{% endcodeblock %}

The important thing to note here is that I am creating the variable SHELL, and $HOME is replaced by the actual path to my home directory.

Next, is the shell script called by Cron.

{% codeblock lang:bash %}
#!/bin/bash
source $HOME/.bash_profile

python $HOME/automated_jobs/launch_production_cluster.py
{% endcodeblock %}

Again, $HOME is replaced with the actual path to my home directory.

I had to make this shell script and the python script called within it executable (call "chmod +x" in the terminal). The reason that I used this shell script rather than directly launching the python script from Cron is I wanted access to environment variables in my bash_profile. In order to get access to them, I had to source bash_profile.

Finally, below I have the python file that executes the week job that I wanted. I didn't include the code that actually launches our emr cluster because that wasn't the hard part here, but just contact me if you would like to see it.

{% codeblock lang:python %}
#!$HOME/anaconda2/bin/python
import os
import sys
import datetime as dt
from subprocess import check_output

# setup logging
old_stdout = sys.stdout
log_file = open("production_cluster_%s.log" % dt.datetime.today().strftime('%Y_%m_%d'), "w")
sys.stdout = log_file

print 'created log file'

# organize local files and s3 files

print 'organized files'

# call emr cluster

print 'launched production job'

# close log file
sys.stdout = old_stdout
log_file.close()
{% endcodeblock %}

While the code is not included here, I use aws cli to launch my emr cluster, and I had to specify the path to aws (call "which aws" in the terminal) when making this call.

You might have noticed the logging I am doing in this script. I found logging both within this python script and piping the output of this script to additional logs helpful when debugging.

The Ubuntu Cron manual I linked above, makes it perfectly clear that my Cron path issues are common, but I wanted to post my solution in case other people needed a little guidance.
