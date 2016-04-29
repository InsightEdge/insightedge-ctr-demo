# Scalable machine learning with InsightEdge: mobile advertisement clicks prediction.

## Overview

In this blog post we will look how to use machine learning algorithms with InsightEdge. For this purpose we will predict mobile advertisement click-through rate with the [Avazu's dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

There are several compensation models in online advertising industry, probably the most notable is CPC (Cost Per Click), in which an advertiser pays a publisher when the ad is clicked.
Search engine advertising is one of the most popular forms of CPC. It allows advertisers to bid for ad placement in a search engine's sponsored links when someone searches on a keyword that is related to their business offering.

For the search engines like Google advertising became the key source of their revenue. The challenge for the advertising system is to determine what ad should be displayed for each query search engine receives.

The revenue search engine can get is essentially:

`Revenue = ad_bid * probability_of_click`

The goal is to maximize the revenue for every search engine query. Whereis the `ad_bid` is a known value, the `probability_of_click` is not. Thus predicting the probability of click becomes the key task.

## Exploring the data

The [dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) consists of:
* train (5.9G) - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
* test (674M) - Test set. 1 day of ads to for testing model predictions.

The first things we want to do is to launch InsightEdge.

To get the first data insights quickly, one can [launch InsightEdge on a local machine](http://insightedge.io/docs/010/0_quick_start.html).
Though for the big datasets or compute-intensive tasks the resources of a single machine are not enough, so we have to scale our computation among number of machines.

For this problem we will [setup a cluster](http://insightedge.io/docs/010/13_cluster_setup.html) with four slaves and place the downloaded files on HDFS.

![Alt cluster](img/0_cluster.png?raw=true "Cluster")

Let's open the interactive [Web Notebook](http://insightedge.io/docs/010/14_notebook.html) and start exploring our dataset.

The dataset is in csv format, so we will use databricks csv library to load it from hdfs into Spark dataframe:

```scala
%dep
z.load("com.databricks:spark-csv_2.10:1.3.0")
```

load and cache the dataframe:


```scala
val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load("hdfs://10.8.1.116/data/avazu_ctr/train")

df.cache()
```

Now that's the dataset is cached in Spark memory, we can read first rows:

```scala
df.show(10)
```

![Alt](img/3_df_show.png?raw=true "Df show")

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

Let's see how many rows in the traning dataset:

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

We can notice some correlation between the ad size and its performance. The most common one appears to be 320x50px known as "mobile leaderboard" in [Google AdSense guide](https://support.google.com/adsense/answer/68727?hl=en)




# Processing and transforming the data




References
* https://en.wikipedia.org/wiki/Click-through_rate
* http://www.wordstream.com/ppc
