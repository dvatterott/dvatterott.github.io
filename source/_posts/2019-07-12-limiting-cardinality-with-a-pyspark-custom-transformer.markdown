---
layout: post
title: "Limiting Cardinality with a PySpark Custom Transformer"
date: 2019-07-12 06:30:28 -0500
comments: true
categories: [python, spark, pyspark, data science, data engineering]
---

When onehot-encoding columns in pyspark, [column cardinality](https://livebook.datascienceheroes.com/data-preparation.html#high_cardinality_descriptive_stats) can become a problem. The size of the data often leads to an enourmous number of unique values. If a minority of the values are common and the majority of the values are rare, you might want to represent the rare values as a single group. Note that this might not be appropriate for your problem. [Here's](https://livebook.datascienceheroes.com/data-preparation.html#analysis-for-predictive-modeling) some nice text describing the costs and benefits of this approach. In the following blog post I describe how to implement this solution.

I begin by importing the necessary libraries and creating a spark session.

{% codeblock lang:python %}
import string
import random
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param

random.seed(1)

sc = SparkContext("local", "Example")
spark = SparkSession(sc)
{% endcodeblock %}

Next create the custom transformer. This class inherits from the `Transformer`, `HasInputCol`, and `HasOutputCol` classes. I also call an additional parameter `n` which controls the maximum cardinality allowed in the tranformed column. Because I have the additional parameter, I need some methods for calling and setting this paramter (`setN` and `getN`). Finally, there's `_tranform` which limits the cardinality of the desired column (set by `inputCol` parameter). This tranformation method simply takes the desired column and changes all values greater than `n` to `n`. It outputs a column named by the `outputCol` parameter.

{% codeblock lang:python %}
class LimitCardinality(Transformer, HasInputCol, HasOutputCol):
    """Limit Cardinality of a column."""

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, n=None):  
        """Initialize."""
        super(LimitCardinality, self).__init__()
        self.n = Param(self, "n", "Cardinality upper limit.")  
        self._setDefault(n=25)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, n=None):  
        """Get params."""
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setN(self, value):  
        """Set cardinality limit."""
        return self._set(n=value)

    def getN(self):  
        """Get cardinality limit."""
        return self.getOrDefault(self.n)

    def _transform(self, dataframe):
        """Do transformation."""
        out_col = self.getOutputCol()
        in_col = dataframe[self.getInputCol()]
        return (dataframe
                .withColumn(out_col, (F.when(in_col > self.getN(), self.getN())
                                      .otherwise(in_col))))
{% endcodeblock %}

Now that we have the tranformer, I will create some data and apply the transformer to it. I want categorical data, so I will randomly draw letters of the alphabet. The only trick is I've made some letters of the alphabet much more common than other ones.

{% codeblock lang:python %}

letter_pool = string.ascii_letters[:26]
letter_pool += ''.join([x*y for x, y in zip(letter_pool[:5], range(100,50,-10))])

a = sc.parallelize([[x, random.choice(letter_pool)] for x in range(1000)]).toDF(["id", "category"])
a.limit(5).show()
# +---+--------+                                                                  
# | id|category|
# +---+--------+
# |  0|       a|
# |  1|       c|
# |  2|       e|
# |  3|       e|
# |  4|       a|
# +---+--------+
{% endcodeblock %}

Take a look at the data.

{% codeblock lang:python %}
(a
 .groupBy("category")
 .agg(F.count("*").alias("category_count"))
 .orderBy(F.col("category_count").desc())
 .limit(20)
 .show())
# +--------+--------------+                                                       
# |category|category_count|
# +--------+--------------+
# |       b|           221|
# |       a|           217|
# |       c|           197|
# |       d|           162|
# |       e|           149|
# |       k|             5|
# |       p|             5|
# |       u|             5|
# |       f|             4|
# |       l|             3|
# |       g|             3|
# |       m|             3|
# |       o|             3|
# |       y|             3|
# |       j|             3|
# |       x|             2|
# |       n|             2|
# |       h|             2|
# |       i|             2|
# |       q|             2|
# +--------+--------------+
{% endcodeblock %}

Now to apply the new class `LimitCardinality` after `StringIndexer` which maps each category (starting with the most common category) to numbers. This means the most common letter will be 1. `LimitCardinality` then sets the max value of `StringIndexer`'s output to `n`. `OneHotEncoderEstimator` one-hot encodes `LimitCardinality`'s output. I wrap `StringIndexer`, `LimitCardinality`, and `OneHotEncoderEstimator` into a single pipeline so that I can fit/transform the dataset at one time.

Note that `LimitCardinality` needs additional code in order to be saved to disk.

{% codeblock lang:python %}
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer
from pyspark.ml import Pipeline

string_to_num = StringIndexer(inputCol="category", outputCol="category_index", stringOrderType="frequencyDesc")
censor_category = LimitCardinality(inputCol="category_index", outputCol="censored_category_index", n=10)
onehot_category = OneHotEncoderEstimator(inputCols=["category_index", "censored_category_index"],
                                     outputCols=["onehot_category", "onehot_censored_category"])
onehot_pipeline = Pipeline(stages=[string_to_num, censor_category, onehot_category])
fit_pipeline = onehot_pipeline.fit(a)

fit_pipeline.transform(a).limit(5).show()
# +---+--------+--------------+-----------------------+---------------+------------------------+
# | id|category|category_index|censored_category_index|onehot_category|onehot_censored_category|
# +---+--------+--------------+-----------------------+---------------+------------------------+
# |  0|       a|           1.0|                    1.0| (25,[1],[1.0])|          (10,[1],[1.0])|
# |  1|       c|           2.0|                    2.0| (25,[2],[1.0])|          (10,[2],[1.0])|
# |  2|       e|           4.0|                    4.0| (25,[4],[1.0])|          (10,[4],[1.0])|
# |  3|       e|           4.0|                    4.0| (25,[4],[1.0])|          (10,[4],[1.0])|
# |  4|       a|           1.0|                    1.0| (25,[1],[1.0])|          (10,[1],[1.0])|
# +---+--------+--------------+-----------------------+---------------+------------------------+

fit_pipeline.transform(a).limit(5).filter(F.col("category") == "n").show()
# +---+--------+--------------+-----------------------+---------------+------------------------+
# | id|category|category_index|censored_category_index|onehot_category|onehot_censored_category|
# +---+--------+--------------+-----------------------+---------------+------------------------+
# | 35|       n|          16.0|                   10.0|(25,[16],[1.0])|              (10,[],[])|
# |458|       n|          16.0|                   10.0|(25,[16],[1.0])|              (10,[],[])|
# +---+--------+--------------+-----------------------+---------------+------------------------+
{% endcodeblock %}

A quick improvement to `LimitCardinality` would be to set a column's cardinality so that X% of rows retain their category values and 100-X% receive the default value (rather than arbitrarily selecting a cardinality limit). I implement this below. Note that `LimitCardinalityModel` is identical to the original `LimitCardinality`. The new `LimitCardinality` has a `_fit` method rather than `_transform` and this method determines a column's cardinality.

In the `_fit` method I find the proportion of columns that are required to describe the requested amount of data. 

{% codeblock lang:python %}
from pyspark.ml.pipeline import Estimator, Model

class LimitCardinality(Estimator, HasInputCol, HasOutputCol):
    """Limit Cardinality of a column."""

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, proportion=None):
        """Initialize."""
        super(LimitCardinality, self).__init__()
        self.proportion = Param(self, "proportion", "Cardinality upper limit as a proportion of data.")
        self._setDefault(proportion=0.75)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, proportion=None):
        """Get params."""
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setProportion(self, value):
        """Set cardinality limit as proportion of data."""
        return self._set(proportion=value)

    def getProportion(self):
        """Get cardinality limit as proportion of data."""
        return self.getOrDefault(self.proportion)

    def _fit(self, dataframe):
        """Fit transformer."""
        pandas_df = dataframe.groupBy(self.getInputCol()).agg(F.count("*").alias("my_count")).toPandas()
        n = sum((pandas_df
                 .sort_values("my_count", ascending=False)
                 .cumsum()["my_count"] / sum(pandas_df["my_count"])
                ) < self.getProportion())
        return LimitCardinalityModel(inputCol=self.getInputCol(), outputCol=self.getOutputCol(), n=n)

class LimitCardinalityModel(Model, HasInputCol, HasOutputCol):
    """Limit Cardinality of a column."""

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, n=None):
        """Initialize."""
        super(LimitCardinalityModel, self).__init__()
        self.n = Param(self, "n", "Cardinality upper limit.")
        self._setDefault(n=25)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, n=None):
        """Get params."""
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setN(self, value):
        """Set cardinality limit."""
        return self._set(n=value)

    def getN(self):
        """Get cardinality limit."""
        return self.getOrDefault(self.n)

    def _transform(self, dataframe):
        """Do transformation."""
        out_col = self.getOutputCol()
        in_col = dataframe[self.getInputCol()]
        return (dataframe
                .withColumn(out_col, (F.when(in_col > self.getN(), self.getN())
                                      .otherwise(in_col))))

string_to_num = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="skip")
censor_category = LimitCardinality(inputCol="category_index", outputCol="censored_category_index", proportion=0.75)
onehot_category = OneHotEncoderEstimator(inputCols=["category_index", "censored_category_index"],
                                     outputCols=["onehot_category", "onehot_censored_category"])
onehot_pipeline = Pipeline(stages=[string_to_num, censor_category, onehot_category])
fit_pipeline = onehot_pipeline.fit(a)

fit_pipeline.transform(a).limit(5).show()
# +---+--------+--------------+-----------------------+---------------+------------------------+
# | id|category|category_index|censored_category_index|onehot_category|onehot_censored_category|
# +---+--------+--------------+-----------------------+---------------+------------------------+
# |  0|       a|           1.0|                    1.0| (25,[1],[1.0])|           (3,[1],[1.0])|
# |  1|       c|           2.0|                    2.0| (25,[2],[1.0])|           (3,[2],[1.0])|
# |  2|       e|           4.0|                    3.0| (25,[4],[1.0])|               (3,[],[])|
# |  3|       e|           4.0|                    3.0| (25,[4],[1.0])|               (3,[],[])|
# |  4|       a|           1.0|                    1.0| (25,[1],[1.0])|           (3,[1],[1.0])|
# +---+--------+--------------+-----------------------+---------------+------------------------+
{% endcodeblock %}

There are [other options](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159) for dealing with high cardinality columns such as using a clustering or a [mean encoding](https://tech.instacart.com/predicting-real-time-availability-of-200-million-grocery-items-in-us-canada-stores-61f43a16eafe) scheme.

Hope you find this useful and reach out if you have any questions.

