---
layout: post
title: "Complex Aggregations in PySpark"
date: 2019-02-05 19:09:32 -0600
comments: true
categories: [python, spark, pyspark, data science, data engineering]
---

I've touched on this in [past posts](https://danvatterott.com/blog/2018/09/06/python-aggregate-udfs-in-pyspark/), but wanted to write a post specifically describing the power of what I call complex aggregations in PySpark.

The idea is that you have have a data request which initially seems to require multiple different queries, but using 'complex aggregations' you can create the requested data using a single query (and a single shuffle).

Let's say you have a dataset like the following. You have one column (id) which is a unique key for each user, another column (group) which expresses the group that each user belongs to, and finally (value) which expresses the value of each customer. I apologize for the contrived example.

{% codeblock lang:python %}
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
from pyspark import SparkContext

sc = SparkContext("local", "Example")
spark = SparkSession(sc)

a = sc.parallelize([[1, 'a', 5.1],
                    [2, 'b', 2.6],
                    [3, 'b', 3.4],
                    [4, 'c', 1.7]]).toDF(['id', 'group', 'value'])
a.show()            
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>id</th>
   <th>group</th>
   <th>value</th>
 </tr>
 <tr>
   <td>1</td>
   <td>'a'</td>
   <td>5.1</td>
 </tr>
 <tr>
   <td>2</td>
   <td>'b'</td>
   <td>2.6</td>
 </tr>
 <tr>
   <td>3</td>
   <td>'b'</td>
   <td>3.4</td>
 </tr>
 <tr>
   <td>4</td>
   <td>'c'</td>
   <td>1.7</td>
 </tr>
</table>

Let's say someone wants the average value of group a, b, and c, *AND* the average value of users in group a *OR* b, the average value of users in group b *OR* c AND the value of users in group a *OR* c. Adds a wrinkle, right? The 'or' clauses prevent us from using a simple groupby, and we don't want to have to write 4 different queries.

Using complex aggregations, we can access all these different conditions in a single query.

{% codeblock lang:python %}

final_data = (a
              .agg(
                F.avg(F.when(F.col('group') == 'a', F.col('value')).otherwise(None)).alias('group_a_avg'),
                F.avg(F.when(F.col('group') == 'b', F.col('value')).otherwise(None)).alias('group_b_avg'),
                F.avg(F.when(F.col('group') == 'c', F.col('value')).otherwise(None)).alias('group_c_avg'),
                F.avg((F.when(F.col('group') == 'a', F.col('value'))
                        .when(F.col('group') == 'b', F.col('value'))
                        .otherwise(None)
                      )).alias('group_ab_avg'),
                F.avg((F.when(F.col('group') == 'b', F.col('value'))
                        .when(F.col('group') == 'c', F.col('value'))
                        .otherwise(None)
                      )).alias('group_bc_avg'),
                F.avg((F.when(F.col('group') == 'a', F.col('value'))
                        .when(F.col('group') == 'c', F.col('value'))
                        .otherwise(None)
                      )).alias('group_ac_avg'),
                )
              )

final_data.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>group_a_avg</th>
   <th>group_b_avg</th>
   <th>group_c_avg</th>
   <th>group_ab_avg</th>
   <th>group_ac_avg</th>
   <th>group_bc_avg</th>
 </tr>
 <tr>
   <td>5.1</td>
   <td>3.0</td>
   <td>1.7</td>
   <td>3.7</td>
   <td>3.4</td>
   <td>2.6</td>
 </tr>
</table>

They key here is using  `when` to filter different data in and out of different aggregations.

This approach can be quite concise when used with python list comprehensions. I'll rewrite the query above, but using a list comprehension.

{% codeblock lang:python %}
from itertools import combinations

groups  = ['a', 'b', 'c']
combos = [x for x in combinations(groups,  2)]
print(combos)
#[('a', 'b'), ('a', 'c'), ('b', 'c')]

single_group = [F.avg(F.when(F.col('group') == x, F.col('value')).otherwise(None)).alias('group_%s_avg' % x) for x in groups]
double_group = [F.avg(F.when(F.col('group') == x, F.col('value')).when(F.col('group')==y, F.col('value')).otherwise(None)).alias('group_%s%s_avg' % (x, y)) for x, y in combos]
final_data = a.agg(*single_group + double_group)
final_data.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>group_a_avg</th>
   <th>group_b_avg</th>
   <th>group_c_avg</th>
   <th>group_ab_avg</th>
   <th>group_ac_avg</th>
   <th>group_bc_avg</th>
 </tr>
 <tr>
   <td>5.1</td>
   <td>3.0</td>
   <td>1.7</td>
   <td>3.7</td>
   <td>3.4</td>
   <td>2.6</td>
 </tr>
</table>

Voila! Hope you find this little trick helpful! Let me know if you have any questions or comments.
