---
layout: post
title: "Creating a Survival Function in PySpark"
date: 2018-12-07 21:13:48 -0600
comments: true
categories: [python, spark, pyspark, data science]
---

Traditionally, [survival functions](https://en.wikipedia.org/wiki/Survival_function) have been used in medical research to visualize the proportion of people who remain alive following a treatment. I often use them to understand the length of time between users creating and cancelling their subscription accounts.

Here, I describe how to create a survival function using PySpark. This is not a post about creating a [Kaplan-Meier estimator](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) or fitting mathematical functions to survival functions. Instead, I demonstrate how to acquire the data necessary for plotting a survival function.

I begin by creating a SparkContext.

{% codeblock lang:python %}
from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = SparkContext("local", "Example")
spark = SparkSession(sc)
{% endcodeblock %}

Next, I load fake data into a Spark Dataframe. This is the data we will use in this example. Each row is a different user and the Dataframe has columns describing start and end dates for each user. <code>start_date</code> represents when a user created their account and <code>end_date</code> represents when a user canceled their account.

{% codeblock lang:python %}
from pyspark.sql import functions as F

user_table = (sc.parallelize([[1, '2018-11-01', '2018-11-03'],
                              [2, '2018-01-01', '2018-08-17'],
                              [3, '2017-12-31', '2018-01-06'],
                              [4, '2018-11-15', '2018-11-16'],
                              [5, '2018-04-02', '2018-04-12']])
              .toDF(['id', 'start_date', 'end_date'])
             )
user_table.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>start_date</th>
   <th>end_date</th>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
 </tr>
 <tr>
   <td>2</td>
   <td>2018-01-01</td>
   <td>2018-08-17</td>
 </tr>
 <tr>
   <td>3</td>
   <td>2017-12-31</td>
   <td>2018-01-06</td>
 </tr>
 <tr>
   <td>4</td>
   <td>2018-11-15</td>
   <td>2018-11-16</td>
 </tr>
 <tr>
   <td>5</td>
   <td>2018-04-02</td>
   <td>2018-04-12</td>
 </tr>
</table>

I use <code>start_date</code> and <code>end_date</code> to determine how many days each user was active following their <code>start_date</code>.

{% codeblock lang:python %}
days_till_cancel = (user_table
                    .withColumn('days_till_cancel', F.datediff(F.col('end_date'), F.col('start_date')))
                   )

days_till_cancel.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>start_date</th>
   <th>end_date</th>
   <th>days_till_cancel</th>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
 </tr>
 <tr>
   <td>2</td>
   <td>2018-01-01</td>
   <td>2018-08-17</td>
   <td>228</td>
 </tr>
 <tr>
   <td>3</td>
   <td>2017-12-31</td>
   <td>2018-01-06</td>
   <td>6</td>
 </tr>
 <tr>
   <td>4</td>
   <td>2018-11-15</td>
   <td>2018-11-16</td>
   <td>1</td>
 </tr>
 <tr>
   <td>5</td>
   <td>2018-04-02</td>
   <td>2018-04-12</td>
   <td>10</td>
 </tr>
</table>

I use a [Python UDF](https://spark.apache.org/docs/2.3.0/api/python/pyspark.sql.html#pyspark.sql.functions.udf) to create a vector of the numbers 0 through 13 representing our *period of interest*. The start date of our *period of interest* is a user's <code>start_date</code>. The end date of our *period of interest* is 13 days following a user's <code>start_date</code>. I chose 13 days as the *period of interest* for no particular reason.

I use [explode](https://spark.apache.org/docs/2.3.0/api/python/pyspark.sql.html#pyspark.sql.functions.explode) to expand the numbers in each vector (i.e., 0->13) into different rows. Each user now has a row for each day in the *period of interest*.

I describe one user's data below.

{% codeblock lang:python %}
create_day_list = F.udf(lambda: [i for i in range(0, 14)], T.ArrayType(T.IntegerType()))

relevant_days = (days_till_cancel
                 .withColumn('day_list', create_day_list())
                 .withColumn('day', F.explode(F.col('day_list')))
                 .drop('day_list')
                )

relevant_days.filter(F.col('id') == 1).show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>start_date</th>
   <th>end_date</th>
   <th>days_till_cancel</th>
   <th>day</th>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>1</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>2</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>3</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>4</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>5</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>6</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>7</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>8</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>9</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>10</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>11</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>12</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>13</td>
 </tr>
</table>

We want the proportion of users who are active *X* days after <code>start_date</code>. I create a column <code>active</code> which represents whether users are active or not. I initially assign each user a 1 in each row (1 represents active). I then overwrite 1s with 0s after a user is no longer active. I determine that a user is no longer active by comparing the values in <code>day</code> and <code>days_till_cancel</code>. When <code>day</code> is greater than <code>days_till_cancel</code>, the user is no longer active.

I describe one user's data below.

{% codeblock lang:python %}
days_active = (relevant_days
               .withColumn('active', F.lit(1))
               .withColumn('active', F.when(F.col('day') >= F.col('days_till_cancel'), 0).otherwise(F.col('active')))
              )

days_active.filter(F.col('id') == 1).show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>start_date</th>
   <th>end_date</th>
   <th>days_till_cancel</th>
   <th>day</th>
   <th>active</th>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>0</td>
   <td>1</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>1</td>
   <td>1</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>2</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>3</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>4</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>5</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>6</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>7</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>8</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>9</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>10</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>11</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>12</td>
   <td>0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2018-11-01</td>
   <td>2018-11-03</td>
   <td>2</td>
   <td>13</td>
   <td>0</td>
 </tr>
</table>

Finally, to acquire the survival function data, I group by <code>day</code> (days following <code>start_date</code>) and average the value in <code>active</code>. This provides us with the proportion of users who are active *X* days after <code>start_date</code>.

{% codeblock lang:python %}
survival_curve = (days_active
                  .groupby('day')
                  .agg(
                      F.count('*').alias('user_count'),
                      F.avg('active').alias('percent_active'),
                  )
                  .orderBy('day')
                 )

survival_curve.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>day</th>
   <th>user_count</th>
   <th>percent_active</th>
 </tr>
 <tr>
   <td>0</td>
   <td>5</td>
   <td>1.0</td>
 </tr>
 <tr>
   <td>1</td>
   <td>5</td>
   <td>0.8</td>
 </tr>
 <tr>
   <td>2</td>
   <td>5</td>
   <td>0.6</td>
 </tr>
 <tr>
   <td>3</td>
   <td>5</td>
   <td>0.6</td>
 </tr>
 <tr>
   <td>4</td>
   <td>5</td>
   <td>0.6</td>
 </tr>
 <tr>
   <td>5</td>
   <td>5</td>
   <td>0.6</td>
 </tr>
 <tr>
   <td>6</td>
   <td>5</td>
   <td>0.4</td>
 </tr>
 <tr>
   <td>7</td>
   <td>5</td>
   <td>0.4</td>
 </tr>
 <tr>
   <td>8</td>
   <td>5</td>
   <td>0.4</td>
 </tr>
 <tr>
   <td>9</td>
   <td>5</td>
   <td>0.4</td>
 </tr>
 <tr>
   <td>10</td>
   <td>5</td>
   <td>0.2</td>
 </tr>
 <tr>
   <td>11</td>
   <td>5</td>
   <td>0.2</td>
 </tr>
 <tr>
   <td>12</td>
   <td>5</td>
   <td>0.2</td>
 </tr>
 <tr>
   <td>13</td>
   <td>5</td>
   <td>0.2</td>
 </tr>
</table>
