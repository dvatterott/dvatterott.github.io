---
layout: post
title: "'Is Not In' with Pyspark"
date: 2018-02-06 21:10:32 -0600
comments: true
categories: [python, spark, pyspark, data science]
---

In SQL it's easy to find people in one list who are not in a second list (i.e., the "not in" command), but there is no similar command in pyspark. Well, at least not [a command](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.isin) that doesn't involve collecting the second list onto the master instance.

Here is a tidbit of code which replicates SQL's "not in" command, while keeping your data with the workers (it will require a shuffle).

I start by creating some small dataframes.

{% codeblock lang:python %}
import pyspark
from pyspark.sql import functions as F
a = sc.parallelize([[1, 'a'], [2, 'b'], [3, 'c']]).toDF(['id', 'valueA'])
b = sc.parallelize([[1, 'a'], [4, 'd'], [5, 'e']]).toDF(['id', 'valueB'])
{% endcodeblock %}

Take a quick look at dataframe *a*.
{% codeblock lang:python %}
a.show()
{% endcodeblock %}
<table style="width:5%">
 <tr>
   <th>id</th>
   <th>valueA</th>
 </tr>
 <tr>
   <td>1</td>
   <td>a</td>
 </tr>
 <tr>
   <td>2</td>
   <td>b</td>
 </tr>
 <tr>
   <td>3</td>
   <td>c</td>
 </tr>
</table>

And dataframe *b*.
{% codeblock lang:python %}
b.show()
{% endcodeblock %}
<table style="width:5%">
 <tr>
   <th>id</th>
   <th>valueA</th>
 </tr>
 <tr>
   <td>1</td>
   <td>a</td>
 </tr>
 <tr>
   <td>4</td>
   <td>d</td>
 </tr>
 <tr>
   <td>5</td>
   <td>e</td>
 </tr>
</table>

I create a new column in *a* that is all ones. I could have used an existing column, but this way I know the column is never null.
{% codeblock lang:python %}
a = a.withColumn('inA', F.lit(1))
a.show()
{% endcodeblock %}
<table style="width:5%">
 <tr>
   <th>id</th>
   <th>valueA</th>
   <th>inA</th>
 </tr>
 <tr>
   <td>1</td>
   <td>a</td>
   <td>1</td>
 </tr>
 <tr>
   <td>2</td>
   <td>b</td>
   <td>1</td>
 </tr>
 <tr>
   <td>3</td>
   <td>c</td>
   <td>1</td>
 </tr>
</table>

I join *a* and *b* with a left join. This way all values in *b* which are not in *a* have null values in the column "inA".
{% codeblock lang:python %}
b.join(a, 'id', 'left').show()
{% endcodeblock %}
<table style="width:5%">
 <tr>
   <th>id</th>
   <th>valueA</th>
   <th>valueB</th>
   <th>inA</th>
 </tr>
 <tr>
   <td>5</td>
   <td>e</td>
   <td>null</td>
   <td>null</td>
 </tr>
 <tr>
   <td>1</td>
   <td>a</td>
   <td>a</td>
   <td>1</td>
 </tr>
 <tr>
   <td>4</td>
   <td>d</td>
   <td>null</td>
   <td>null</td>
 </tr>
</table>

By filtering out rows in the new dataframe *c*, which are not null, I remove all values of *b*, which were also in *a*.
{% codeblock lang:python %}
c = b.join(a, 'id', 'left').filter(F.col('inA').isNull())
c.show()
{% endcodeblock %}
<table style="width:5%">
 <tr>
   <th>id</th>
   <th>valueA</th>
   <th>valueB</th>
   <th>inA</th>
 </tr>
 <tr>
   <td>5</td>
   <td>e</td>
   <td>null</td>
   <td>null</td>
 </tr>
 <tr>
   <td>4</td>
   <td>d</td>
   <td>null</td>
   <td>null</td>
 </tr>
</table>