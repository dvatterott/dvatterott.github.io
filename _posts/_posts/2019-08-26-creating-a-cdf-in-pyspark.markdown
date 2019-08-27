---
layout: post
title: "Creating a CDF in PySpark"
date: 2019-08-26 19:36:15 -0500
comments: true
categories: [python, spark, pyspark, data science]
---

[CDFs](https://en.wikipedia.org/wiki/Cumulative_distribution_function) are a useful tool for understanding your data. This tutorial will demonstrate how to create a CDF in PySpark.

I start by creating normally distributed, fake data.

{% codeblock lang:python %}
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window

sc = SparkContext("local", "Example")
spark = SparkSession(sc)

a = (sc.parallelize([(float(x),) for x in np.random.normal(0, 1, 1000)]).toDF(['X']))
a.limit(5).show() 
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>X</th>
 </tr>
 <tr>
   <td>1.3162087724709406</td>
 </tr>
 <tr>
   <td>-0.9226127327757598</td>
 </tr>
 <tr>
   <td>0.5388249247619141</td>
 </tr>
 <tr>
   <td>-0.38263792383896356</td>
 </tr>
 <tr>
   <td>0.20584675505779562</td>
 </tr>
</table>

To create the CDF I need to use a [window](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Window) function to order the data. I can then use [percent_rank](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.percent_rank) to retrieve the percentile associated with each value.

The only trick here is I round the column of interest to make sure I don't retrieve too much data onto the master node (not a concern here, but always good to think about).

After rounding, I group by the variable of interest, again, to limit the amount of data returned.

{% codeblock lang:python %}
win = Window.orderBy('X')

output = (a
          .withColumn('cumulative_probability', F.percent_rank().over(win))
          .withColumn("X", F.round(F.col("X"), 1))
          .groupBy("X")
          .agg(F.max("cumulative_probability").alias("cumulative_probability"),F.count('*').alias("my_count")))

output.limit(5).show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>X</th>
   <th>cumulative_probability</th>
   <th>my_count</th>
 </tr>
 <tr>
   <td>-3.5</td>
   <td>0.0</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-3.3</td>
   <td>0.001001001001001001</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-2.9</td>
   <td>0.002002002002002002</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-2.8</td>
   <td>0.003003003003003003</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-2.7</td>
   <td>0.004004004004004004</td>
   <td>1</td>
 </tr>
</table>

A CDF should report the percent of data less than or *equal* to the specified value. The data returned above is the percent of data less than the specified value. We need to fix this by shifting the data up.

To shift the data, I will use the function, [lead](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.lead).

{% codeblock lang:python %}
output = (a
          .withColumn('cumulative_probability', F.percent_rank().over(win))
          .withColumn("X", F.round(F.col("X"), 1))
          .groupBy("X")
          .agg(F.max("cumulative_probability").alias("cumulative_probability"),F.count('*').alias("my_count"))
          .withColumn("cumulative_probability", F.lead(F.col("cumulative_probability")).over(win))
          .fillna(1, subset=["cumulative_probability"]))

output.limit(5).show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>X</th>
   <th>cumulative_probability</th>
   <th>my_count</th>
 </tr>
 <tr>
   <td>-3.5</td>
   <td>0.001001001001001001</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-3.3</td>
   <td>0.002002002002002002</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-2.9</td>
   <td>0.003003003003003003</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-2.8</td>
   <td>0.004004004004004004</td>
   <td>1</td>
 </tr>
 <tr>
   <td>-2.7</td>
   <td>0.005005005005005005</td>
   <td>1</td>
 </tr>
</table>

There we go! A CDF of the data! I hope you find this helpful!
