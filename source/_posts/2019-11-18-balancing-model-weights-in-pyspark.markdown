---
layout: post
title: "Balancing Model Weights in PySpark"
date: 2019-11-18 18:57:03 -0600
comments: true
categories: [python, spark, pyspark, data science]
---

[Imbalanced classes](https://www.jeremyjordan.me/imbalanced-data/) is a common problem. Scikit-learn provides an easy fix - "balancing" class weights. This makes models more likely to predict the less common classes (e.g., [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)). 

The PySpark ML API doesn't have this same functionality, so in this blog post, I describe how to balance class weights yourself.

{% codeblock lang:python %}
import numpy as np
import pandas as pd
from itertools import chain
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

sc = SparkContext("local", "Example")
spark = SparkSession(sc)
{% endcodeblock %}

Generate some random data and put the data in a Spark DataFrame. Note that the input variables are not predictive. The model will behave randomly. This is okay, since I am not interested in model accuracy.

{% codeblock lang:python %}
X = np.random.normal(0, 1, (10000, 10))

y = np.ones(X.shape[0]).astype(int)
y[:1000] = 0
np.random.shuffle(y)

print(np.mean(y)) # 0.9

X = np.append(X, y.reshape((10000, 1)), 1)

DF = spark.createDataFrame(pd.DataFrame(X))
DF = DF.withColumnRenamed("10", "y")
{% endcodeblock %}

Here's how Scikit-learn computes class weights when "balanced" weights are requested. 

{% codeblock lang:python %}
# class weight
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# n_samples / (n_classes * np.bincount(y)).

class_weights = {i: ii for i, ii in zip(np.unique(y), len(y) / (len(np.unique(y)) * np.bincount(y)))}
print(class_weights) # {0: 5.0, 1: 0.5555555555555556}
{% endcodeblock %}

Here's how we can compute "balanced" weights with data from a PySpark DataFrame.

{% codeblock lang:python %}
y_collect = DF.select("y").groupBy("y").count().collect()
unique_y = [x["y"] for x in y_collect]
total_y = sum([x["count"] for x in y_collect])
unique_y_count = len(y_collect)
bin_count = [x["count"] for x in y_collect]

class_weights_spark = {i: ii for i, ii in zip(unique_y, total_y / (unique_y_count * np.array(bin_count)))}
print(class_weights_spark) # {0.0: 5.0, 1.0: 0.5555555555555556}
{% endcodeblock %}

PySpark needs to have a weight assigned to each instance (i.e., row) in the training set. I create a mapping to apply a weight to each training instance.

{% codeblock lang:python %}
mapping_expr = F.create_map([F.lit(x) for x in chain(*class_weights_spark.items())])

DF = DF.withColumn("weight", mapping_expr.getItem(F.col("y")))
{% endcodeblock %}

I assemble all the input features into a vector.

{% codeblock lang:python %}
assembler = VectorAssembler(inputCols=[str(x) for x in range(10)], outputCol="features")

DF = assembler.transform(DF).drop(*[str(x) for x in range(10)])
{% endcodeblock %}

And train a logistic regression. Without the instance weights, the model predicts all instances as the frequent class.

{% codeblock lang:python %}
lr = LogisticRegression(featuresCol="features", labelCol="y")
lrModel = lr.fit(DF)
lrModel.transform(DF).agg(F.mean("prediction")).show()
{% endcodeblock %}

    +---------------+
    |avg(prediction)|
    +---------------+
    |            1.0|
    +---------------+
    
With the weights, the model assigns half the instances to each class (even the less commmon one).

{% codeblock lang:python %}
lr = LogisticRegression(featuresCol="features", labelCol="y", weightCol="weight")
lrModel = lr.fit(DF)
lrModel.transform(DF).agg(F.mean("prediction")).show()
{% endcodeblock %}

    +---------------+
    |avg(prediction)|
    +---------------+
    |         0.5089|
    +---------------+
