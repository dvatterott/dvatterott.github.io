---
layout: post
title: "Python Aggregate UDFs in Pyspark"
date: 2018-09-06 16:04:43 -0500
comments: true
categories: [python, spark, pyspark, data science, data engineering]
---

Pyspark has a great set of [aggregate](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.agg) functions (e.g., [count, countDistinct, min, max, avg, sum](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData)), but these are not enough for all cases (particularly if you're trying to avoid costly Shuffle operations).

Pyspark currently has [pandas_udfs](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf), which can create custom aggregators, but you can only "apply" one pandas_udf at a time. If you want to use more than one, you'll have to preform multiple groupBys...and there goes avoiding those shuffles.

In this post I describe a little hack which enables you to create simple python UDFs which act on aggregated data (this functionality is only supposed to exist in Scala!).

{% codeblock lang:python %}
from pyspark.sql import functions as F
from pyspark.sql import types as T

a = sc.parallelize([[1, 'a'],
                    [1, 'b'],
                    [1, 'b'],
                    [2, 'c']]).toDF(['id', 'value'])
a.show()            
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>value</th>
 </tr>
 <tr>
   <td>1</td>
   <td>'a'</td>
 </tr>
 <tr>
   <td>1</td>
   <td>'b'</td>
 </tr>
 <tr>
   <td>1</td>
   <td>'b'</td>
 </tr>
 <tr>
   <td>2</td>
   <td>'c'</td>
 </tr>
</table>

I use [collect_list](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.collect_list) to bring all data from a given group into a single row. I print the output of this operation below.

{% codeblock lang:python %}

a.groupBy('id').agg(F.collect_list('value').alias('value_list')).show()

{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>value_list</th>
 </tr>
 <tr>
   <td>1</td>
   <td>['a', 'b', 'b']</td>
 </tr>
 <tr>
   <td>2</td>
   <td>['c']</td>
 </tr>
</table>

I then create a [UDF](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.udf) which will count all the occurences of the letter 'a' in these lists (this can be easily done without a UDF but you get the point). This UDF wraps around collect_list, so it acts on the output of collect_list.

{% codeblock lang:python %}
def find_a(x):
  """Count 'a's in list."""
  output_count = 0
  for i in x:
    if i == 'a':
      output_count += 1
  return output_count

find_a_udf = F.udf(find_a, T.IntegerType())

a.groupBy('id').agg(find_a_udf(F.collect_list('value')).alias('a_count')).show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>a_count</th>
 </tr>
 <tr>
   <td>1</td>
   <td>1</td>
 </tr>
 <tr>
   <td>2</td>
   <td>0</td>
 </tr>
</table>

There we go! A UDF that acts on aggregated data! Next, I show the power of this approach when combined with [when](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.when) which let's us control which data enters F.collect_list.

First, let's create a dataframe with an extra column.

{% codeblock lang:python %}
from pyspark.sql import functions as F
from pyspark.sql import types as T

a = sc.parallelize([[1, 1, 'a'],
                    [1, 2, 'a'],
                    [1, 1, 'b'],
                    [1, 2, 'b'],
                    [2, 1, 'c']]).toDF(['id', 'value1', 'value2'])
a.show()            
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>value1</th>
   <th>value2</th>
 </tr>
 <tr>
   <td>1</td>
   <td>1</td>
   <td>'a'</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2</td>
   <td>'a'</td>
 </tr>
 <tr>
   <td>1</td>
   <td>1</td>
   <td>'b'</td>
 </tr>
 <tr>
   <td>1</td>
   <td>2</td>
   <td>'b'</td>
 </tr>
 <tr>
   <td>2</td>
   <td>1</td>
   <td>'c'</td>
 </tr>
</table>

Notice, how I included a when in the collect_list. Note that the UDF still wraps around collect_list.

{% codeblock lang:python %}

a.groupBy('id').agg(find_a_udf( F.collect_list(F.when(F.col('value1') == 1, F.col('value2')))).alias('a_count')).show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>a_count</th>
 </tr>
 <tr>
   <td>1</td>
   <td>1</td>
 </tr>
 <tr>
   <td>2</td>
   <td>0</td>
 </tr>
</table>

There we go! Hope you find this info helpful!
