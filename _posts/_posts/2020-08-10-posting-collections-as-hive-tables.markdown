---
layout: post
title: "Posting Collections as Hive Tables"
date: 2020-08-10 20:03:43 -0500
comments: true
categories: [sql, spark, hive, data engineering]
---

I was recently asked to post a series of parquet collection as tables so analysts could query them in SQL. This should be straight forward, but it took me awhile to figure out. Hopefully, you find this post before spending too much time on such an easy task.

You should use the [`CREATE TABLE`](https://docs.databricks.com/spark/latest/spark-sql/language-manual/create-table.html). This is pretty straight forward. By creating a permanent table (rather than a temp table), you can use a database name. Also, by using a table (rather than  a view), you can load the data from an s3 location. 

Next, you can specify the table's schema. Again, this is pretty straight forward. Columns used to partition the data should be declared here.

Next, you can specify how the data is stored (below, I use Parquet) and how the data is partitioned (below, there are two partitioning columns).

Finally, you specify the data's location.

The part that really threw me for a loop here is that I wasn't done yet! You need one more command so that Spark can go examine the partitions - [`MSCK REPAIR TABLE`](https://spark.apache.org/docs/latest/sql-ref-syntax-ddl-repair-table.html). Also please note that this command needs to be re-run whenever a partition is added.

{% codeblock lang:python %}
spark.sql("""
CREATE TABLE my_db.my_table (
(example_key INT, example_col STRING, example_string STRING, example_date STRING)
)
USING PARQUET
PARTITIONED BY (example_string, example_date)
LOCATION 's3://my.example.bucket/my_db/my_table/'
"""
spark.sql("MSCK REPAIR TABLE my_db.my_table")
{% endcodeblock %}

Hope this post saves you some time!
