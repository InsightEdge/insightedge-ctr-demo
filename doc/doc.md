# Scalable machine learning with InsightEdge: mobile advertisement clicks prediction.

## Overview

In this blog post we will look how to use machine learning algorithms with InsightEdge. For this purpose we will predict mobile advertisement click-through rate with the [Avazu's dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

There are several compensation models in online advertising industry, probably the most notable is CPC (Cost Per Click), in which an advertiser pays a publisher when the ad is clicked.
Search engine advertising is one of the most popular forms of CPC. It allows advertisers to bid for ad placement in a search engine's sponsored links when someone searches on a keyword that is related to their business offering.

For the search engines like Google advertising became the key source of their revenue. The challenge for the advertising system is to determine what ad should be displayed for each query search engine receives.

The revenue search engine can get is essentially:

`revenue = bid * probability_of_click`

The goal is to maximize the revenue for every search engine query. Whereis the `bid` is a known value, the `probability_of_click` is not. Thus predicting the probability of click becomes the key task.

Working on machine learning problem involves a lot of experiments with feature selection, feature transformation, training different models and tuning parameters.
While there are a few excellent machine learning libraries for Python and R like scikit-learn, their capabilities are typically limited to relatively small datasets that you can fit into a single machine.

With the large datasets and/or CPU intensive workloads you may want to scale out beyond a single machine. And this is not a problem for InsightEdge at all, since it can scale the computation and data storage layers across many machines in the cluster.

## Loading the data

The [dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) consists of:
* train (5.9G) - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
* test (674M) - Test set. 1 day of ads to for testing model predictions.

The first things we want to do is to launch InsightEdge.

To get the first data insights quickly, one can [launch InsightEdge on a laptop](http://insightedge.io/docs/010/0_quick_start.html).
Though for the big datasets or compute-intensive tasks the resources of a single machine might not be enough.

For this problem we will [setup a cluster](http://insightedge.io/docs/010/13_cluster_setup.html) with four workers and place the downloaded files on HDFS.

![Alt cluster](img/0_cluster.png?raw=true "Cluster")

Let's open the interactive [Web Notebook](http://insightedge.io/docs/010/14_notebook.html) and start exploring our dataset.

The dataset is in csv format, so we will use databricks csv library to load it from hdfs into Spark dataframe:

```scala
%dep
z.load("com.databricks:spark-csv_2.10:1.3.0")
```

load the dataframe into Spark memory and cache:

```scala
val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load("hdfs://10.8.1.116/data/avazu_ctr/train")

df.cache()
```

The next thing we want to do is to copy the dataset to in-memory datagrid that is running collocated with the Spark.
This way we can restart Zeppelin session or launch another Spark applications and pick up the dataset more quicker from the memory.

With InsightEdge this can be done just with two lines of code:

```scala
import org.apache.spark.sql.insightedge._
df.write.grid.save("raw_training")
```

Any time later we can bring "raw_training" collection to Spark memory with:
```scala
val df = sqlContext.read.grid.load("raw_training")
```

As we will show later, when developing applications with InsightEdge, the datagrid can be the primary datasource for your
Spark applications, eliminating the need of ETL phase, and at the same time serve low latency transactional operations.

## Exploring the data

Now that we have the dataset in Spark memory, we can read the first rows:

```scala
df.show(10)
```

```
+--------------------+-----+--------+----+----------+--------+-----------+-------------+--------+----------+------------+---------+---------+------------+-----------+----------------+-----+---+---+----+---+---+------+---+
|                  id|click|    hour|  C1|banner_pos| site_id|site_domain|site_category|  app_id|app_domain|app_category|device_id|device_ip|device_model|device_type|device_conn_type|  C14|C15|C16| C17|C18|C19|   C20|C21|
+--------------------+-----+--------+----+----------+--------+-----------+-------------+--------+----------+------------+---------+---------+------------+-----------+----------------+-----+---+---+----+---+---+------+---+
| 1000009418151094273|    0|14102100|1005|         0|1fbe01fe|   f3845767|     28905ebd|ecad2386|  7801e8d9|    07d7df22| a99f214a| ddd2926e|    44956a24|          1|               2|15706|320| 50|1722|  0| 35|    -1| 79|
|10000169349117863715|    0|14102100|1005|         0|1fbe01fe|   f3845767|     28905ebd|ecad2386|  7801e8d9|    07d7df22| a99f214a| 96809ac8|    711ee120|          1|               0|15704|320| 50|1722|  0| 35|100084| 79|
|10000371904215119486|    0|14102100|1005|         0|1fbe01fe|   f3845767|     28905ebd|ecad2386|  7801e8d9|    07d7df22| a99f214a| b3cf8def|    8a4875bd|          1|               0|15704|320| 50|1722|  0| 35|100084| 79|
|10000640724480838376|    0|14102100|1005|         0|1fbe01fe|   f3845767|     28905ebd|ecad2386|  7801e8d9|    07d7df22| a99f214a| e8275b8f|    6332421a|          1|               0|15706|320| 50|1722|  0| 35|100084| 79|
|10000679056417042096|    0|14102100|1005|         1|fe8cc448|   9166c161|     0569f928|ecad2386|  7801e8d9|    07d7df22| a99f214a| 9644d0bf|    779d90c2|          1|               0|18993|320| 50|2161|  0| 35|    -1|157|
|10000720757801103869|    0|14102100|1005|         0|d6137915|   bb1ef334|     f028772b|ecad2386|  7801e8d9|    07d7df22| a99f214a| 05241af0|    8a4875bd|          1|               0|16920|320| 50|1899|  0|431|100077|117|
|10000724729988544911|    0|14102100|1005|         0|8fda644b|   25d4cfcd|     f028772b|ecad2386|  7801e8d9|    07d7df22| a99f214a| b264c159|    be6db1d7|          1|               0|20362|320| 50|2333|  0| 39|    -1|157|
|10000918755742328737|    0|14102100|1005|         1|e151e245|   7e091613|     f028772b|ecad2386|  7801e8d9|    07d7df22| a99f214a| e6f67278|    be74e6fe|          1|               0|20632|320| 50|2374|  3| 39|    -1| 23|
|10000949271186029916|    1|14102100|1005|         0|1fbe01fe|   f3845767|     28905ebd|ecad2386|  7801e8d9|    07d7df22| a99f214a| 37e8da74|    5db079b5|          1|               2|15707|320| 50|1722|  0| 35|    -1| 79|
|10001264480619467364|    0|14102100|1002|         0|84c7ba46|   c4e18dd6|     50e219e0|ecad2386|  7801e8d9|    07d7df22| c357dbff| f1ac7184|    373ecbe6|          0|               0|21689|320| 50|2496|  3|167|100191| 23|
+--------------------+-----+--------+----+----------+--------+-----------+-------------+--------+----------+------------+---------+---------+------------+-----------+----------------+-----+---+---+----+---+---+------+---+
only showing top 10 rows
```

The data fields are:

* id: ad identifier
* click: 0/1 for non-click/click
* hour: format is YYMMDDHH
* C1: anonymized categorical variable
* banner_pos
* site_id
* site_domain
* site_category
* app_id
* app_domain
* app_category
* device_id
* device_ip
* device_model
* device_type
* device_conn_type
* C14-C21 -- anonymized categorical variables

Let's see how many rows in the training dataset:

```scala
val totalCount = df.count()

totalCount: Long = 40428967
```

There are about 40M+ rows in the dataset.

Let's now calculate the CTR(click-through rate) of the dataset. The click-through rate is the number of times a click is made on the advertisement divided by the total impressions (the number of times an advertisement was served):

```scala
val clicks = df.filter("click = 1").count()
val ctr = clicks.toFloat / totalCount

clicks: Long = 6865066
ctr: Float = 0.16980562
```
The CTR is 0.169 (or 16.9%) which is quite high number, the common value in industry is about 0.2-0.3%. So high value is probably because non-clicks and clicks are subsampled according to different strategies as stated by Avazu.

Now, the question is which features should we use to create a predictive model? This is a difficult question that requires a deep knowledge of the problem domain. Let's try to learn it from the dataset we have.

For example, let's explore the `device_conn_type` feature. Our assumption might be that this is a categorical variable like Wi-Fi, 2G, 3G or LTE. This might be a relevant feature since clicking on an ad with a slow connection is not something common.

At first we register the dataframe as a SQL table:

```scala
df.registerTempTable("training")
```

and run the SQL query:

```sql
%sql
SELECT device_conn_type, SUM(click) as clicks_num, COUNT(click) as impression, SUM(click)/COUNT(click) as ctr
FROM training
GROUP BY device_conn_type
````

![Alt](img/6_device_conn_type.png?raw=true "device_conn_type")

![Alt](img/7_device_conn_type_2.png?raw=true "device_conn_type")

We see that there are four connection type categories. Two categories with CTR 18% and 13%, and the first one is almost 90% of the whole dataset. The other two categories have significantly lower CTR.

Another observation we may notice is that features C15 and C16 look like the ad size:

```sql
%sql
SELECT C15, C16, COUNT(click) as impression, SUM(click)/COUNT(click) as ctr
FROM training
GROUP BY C15, C16
ORDER BY ctr DESC
```

![Alt](img/11_banner_dimension.png?raw=true "banner dimension")

We can notice some correlation between the ad size and its performance. The most common one appears to be 320x50px known as "mobile leaderboard" in [Google AdSense](https://support.google.com/adsense/answer/68727?hl=en)

What about other features? All of them represent categorical values, how many unique categories for each feature?

```scala
df.columns.map(c => (c, df.select(c).distinct().count()))

res14: Array[(String, Long)] = Array((id,40428967), (click,2), (hour,240), (C1,7), (banner_pos,7), (site_id,4737), (site_domain,7745), (site_category,26), (app_id,8552), (app_domain,559), (app_category,36), (device_id,2686408), (device_ip,6729486), (device_model,8251), (device_type,5), (device_conn_type,4), (C14,2626), (C15,8), (C16,9), (C17,435), (C18,4), (C19,68), (C20,172), (C21,60))
```

We see that there are some features with a lot of unique values, for example `device_ip` has 6M+ unique values.
Usually machine learning algorithms expect to work with numbers, rather than categorical values. Converting such categorical features will result into high dimensional vector which might be very expensive.
We will need to deal with this later.

# Processing and transforming the data

Looking further at the dataset, we can see that the `hour` feature is in `YYMMDDHH` format.
To allow the predictive model effectively learn from this feature it make sense to transform it into four features: `year`, `month` and `hour`.
Let's develop the function to transform the dataframe:

```scala
import java.text.SimpleDateFormat
import java.util.{Calendar, Date}
import org.apache.spark.sql.DataFrame

object DateUtils {
  val dateFormat = new ThreadLocal[SimpleDateFormat]() {
    override def initialValue(): SimpleDateFormat = new SimpleDateFormat("yyMMddHH")
  }

  def parse(s: String, field: Int): Int = {
    val date = dateFormat.get().parse(s)
    val cal = Calendar.getInstance()
    cal.setTime(date)
    cal.get(field)
  }
}

def transformHour(df: DataFrame): DataFrame = {
  val toYear = udf[Int, String](s => DateUtils.parse(s, Calendar.YEAR))
  val toMonth = udf[Int, String](s => DateUtils.parse(s, Calendar.MONTH))
  val toDay = udf[Int, String](s => DateUtils.parse(s, Calendar.DAY_OF_MONTH))
  val toHour = udf[Int, String](s => DateUtils.parse(s, Calendar.HOUR_OF_DAY))

  df.withColumn("time_year", toYear(df("hour")))
  .withColumn("time_month", toMonth(df("hour")))
  .withColumn("time_day", toDay(df("hour")))
  .withColumn("time_hour", toHour(df("hour")))
  .drop("hour")
}
```

We can now apply this transformation to our dataframe and see the result:

```scala
val trainingHourDecoded = transformHour(df)
trainingHourDecoded.cache()
trainingHourDecoded.show()
```

![Alt](img/12_hour_transform.png?raw=true "banner dimension")

It looks like the year and month have only the single value, let's verify it:

```scala
trainingHourDecoded.select("time_month").distinct.count()
trainingHourDecoded.select("time_year").distinct.count()

res20: Long = 1
res21: Long = 1
```

We can safely drop these columns as they don't bring any knowledge to our model:

```scala
val trainingHourDecoded2 = trainingHourDecoded.drop("time_month").drop("time_year")
```

Since we now have the transformed dataset, we can remove the raw dataset from the Spark cache:
```scala
df.unpersist()
```



References
* https://en.wikipedia.org/wiki/Click-through_rate
* http://www.wordstream.com/ppc
