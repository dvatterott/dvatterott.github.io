---
layout: post
title: "Integrating Apache Airflow and Databricks"
date: 2018-06-13 18:05:52 -0500
comments: true
categories: [data science, data engineering, open source]
---

Cron is great for automation, but when tasks begin to rely on each other (task C can only run after both tasks A and B finish) cron does not do the trick.

[Apache Airflow](https://airflow.apache.org/) is open source software (from airbnb) designed to handle the relationship between tasks. I recently setup an airflow server which coordinates automated jobs on [databricks](https://databricks.com/) (great software for coordinating spark clusters). Connecting databricks and airflow ended up being a little trickier than it should have been, so I am writing this blog post as a resource to anyone else who attempts to do the same in the future.

For the most part I followed [this tutorial from A-R-G-O](https://medium.com/a-r-g-o/installing-apache-airflow-on-ubuntu-aws-6ebac15db211) when setting up airflow. Databricks also has a decent [tutorial](https://docs.databricks.com/user-guide/dev-tools/data-pipelines.html) on setting up airflow. The difficulty here is that the airflow software for talking to databricks clusters (DatabricksSubmitRunOperator) was not introduced into airflow until version 1.9 and the A-R-G-O tutorial uses airflow 1.8.

Airflow 1.9 uses Celery version >= 4.0 (I ended up using Celery version 4.1.1). Airflow 1.8 requires Celery < 4.0. In fact, the A-R-G-O tutorial notes that using Celery >= 4.0 will result in the error:

{% codeblock %}
airflow worker: Received and deleted unknown message. Wrong destination?!?
{% endcodeblock %}

I can attest that this is true! If you use airflow 1.9 with Celery < 4.0, everything might appear to work, but airflow will randomly stop scheduling jobs after awhile (check the airflow-scheduler logs if you run into this). You need to use Celery >= 4.0! Preventing the Wrong destination error is easy, but the fix is hard to find (hence why I wrote this post).

After much ado, here's the fix! If you follow the A-R-G-O tutorial, install airflow 1.9, celery >=4.0 AND set broker_url in airflow.cfg as follows:

{% codeblock %}
broker_url = pyamqp://guest:guest@localhost:5672//
{% endcodeblock %}

Note that compared to the A-R-G-O tutorial, I am just adding "py" in front of amqp. Easy!
