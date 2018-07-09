---
layout: post
title: "Aggregating Sparse and Dense Vectors in Pyspark"
date: 2018-07-08 19:24:04 -0500
comments: true
categories: [python, spark, pyspark, data science, data engineering]
---

Many (if not all of) pyspark's machine learning algorithms require the input data is concatenated into a single column (using the [vector assembler](https://spark.apache.org/docs/2.3.0/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler) command). This is all well and good, but applying non-machine learning algorithms (e.g., any aggregations) to data in this format can be a real pain. Here, I describe how to aggregate (average in this case) data in sparse and dense vectors.

I start by importing the necessary libraries and creating a spark dataframe, which includes a column of sparse vectors. Note that I am using ml.linalg SparseVector and not the SparseVector from mllib. This makes a big difference!

{% codeblock lang:python %}
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.linalg import SparseVector, DenseVector
# note that using Sparse and Dense Vectors from ml.linalg. There are other Sparse/Dense vectors in spark.

df = sc.parallelize([
  (1, SparseVector(10, {1: 1.0, 2: 1.0, 3: 2.0, 4: 1.0, 5: 3.0})),
  (2, SparseVector(10, {9: 100.0})),
  (3, SparseVector(10, {1: 1.0})),
]).toDF(["row_num", "features"])

df.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>row_num</th>
   <th>features</th>
 </tr>
 <tr>
   <td>1</td>
   <td>(10,[1,2,3,4,5],[1.0, 1.0, 2.0, 1.0, 3.0])</td>
 </tr>
 <tr>
   <td>2</td>
   <td>(10,[9],[100.0])</td>
 </tr>
 <tr>
   <td>3</td>
   <td>(10,[1],[1.0])</td>
 </tr>
</table>

Next, I write a [udf](https://spark.apache.org/docs/2.3.0/api/python/pyspark.sql.html#pyspark.sql.functions.udf), which changes the sparse vector into a dense vector and then changes the dense vector into a python list. The python list is then turned into a spark array when it comes out of the udf.  

{% codeblock lang:python %}
def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array

sparse_to_array_udf = F.udf(sparse_to_array, T.ArrayType(T.FloatType()))

df = df.withColumn('features_array', sparse_to_array_udf('features'))
df.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>row_num</th>
   <th>features</th>
   <th>features_array</th>
 </tr>
 <tr>
   <td>1</td>
   <td>(10,[1,2,3,4,5],[1.0, 1.0, 2.0, 1.0, 3.0])</td>
   <td>[0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]</td>
 </tr>
 <tr>
   <td>2</td>
   <td>(10,[9],[100.0])</td>
   <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]</td>
 </tr>
 <tr>
   <td>3</td>
   <td>(10,[1],[1.0])</td>
   <td>[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]</td>
 </tr>
</table>

Now that the data is in a pyspark array, we can apply the desired pyspark aggregation to each item in the array.

{% codeblock lang:python %}
df_agg = df.agg(F.array(*[F.avg(F.col('features_array')[i]) for i in range(10)]).alias("averages"))
df_agg.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>averages</th>
 </tr>
 <tr>
   <td>[0.0, 0.66667, 0.33333, 0.66667, 0.33333, 1.0, 0.0, 0.0, 0.0, 33.33333]</td>
 </tr>
</table>

Now, let's run through the same exercise with dense vectors. We start by creating a spark dataframe with a column of dense vectors.

{% codeblock lang:python %}
df = sc.parallelize([
  (1, DenseVector([0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0])),
  (2, DenseVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])),
  (3, DenseVector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
]).toDF(["row_num", "features"])

df.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>row_num</th>
   <th>features</th>
 </tr>
 <tr>
   <td>1</td>
   <td>[0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]</td>
 </tr>
 <tr>
   <td>2</td>
   <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]</td>
 </tr>
 <tr>
   <td>3</td>
   <td>[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]</td>
 </tr>
</table>

Next, we create another pyspark udf which changes the dense vector into a pyspark array.

{% codeblock lang:python %}
def dense_to_array(v):
  new_array = list([float(x) for x in v])
  return new_array

dense_to_array_udf = F.udf(dense_to_array, T.ArrayType(T.FloatType()))

df = df.withColumn('features_array', dense_to_array_udf('features'))
df.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>row_num</th>
   <th>features</th>
   <th>features_array</th>
 </tr>
 <tr>
   <td>1</td>
   <td>[0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]</td>
   <td>[0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]</td>
 </tr>
 <tr>
   <td>2</td>
   <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]</td>
   <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]</td>
 </tr>
 <tr>
   <td>3</td>
   <td>[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]</td>
   <td>[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]</td>
 </tr>
</table>

Finally, we can use our standard pyspark aggregators to each item in the pyspark array.

{% codeblock lang:python %}
df_agg = df.agg(F.array(*[F.avg(F.col('features_array')[i]) for i in range(10)]).alias("averages"))
df_agg.show()
{% endcodeblock %}

<table style="width:100%">
 <tr>
   <th>averages</th>
 </tr>
 <tr>
   <td>[0.0, 0.66667, 0.33333, 0.66667, 0.33333, 1.0, 0.0, 0.0, 0.0, 33.33333]</td>
 </tr>
</table>

There we go! Hope you find this info helpful!
