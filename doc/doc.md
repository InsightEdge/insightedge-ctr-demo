# Scalable machine learning with InsightEdge: mobile advertisement clicks prediction.

## Overview

In this blog post we will look how to use machine learning algorithms with InsightEdge. For this purpose we will predict mobile advertisement click-through rate with the [Avazu's dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

There are several compensation models in online advertising industry, probably the most notable is CPC (Cost Per Click), in which an advertiser pays a publisher when the ad is clicked.
Search engine advertising is one of the most popular forms of CPC. It allows advertisers to bid for ad placement in a search engine's sponsored links when someone searches on a keyword that is related to their business offering.

For the search engines like Google advertising became the key source of their revenue. The challenge for the advertising system is to determine what ad should be displayed for each query search engine receives.

The revenue search engine can get is essentially:

`Revenue = ad_bid * probability_of_click`

The goal is to maximize the revenue for every search engine query. Whereis the `ad_bid` is a known value, the `probability_of_click` is not. Thus predicting the probability of click becomes the key challenge.

## Understanding the data

The [dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) consists of:
* train (5.9G) - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
* test (674M) - Test set. 1 day of ads to for testing model predictions.

The first things we want to do is to launch InsightEdge. To get the first data insights quickly, one can [launch InsightEdge on a local machine](http://insightedge.io/docs/010/0_quick_start.html).
Though for the big datasets or compute-intensive tasks the resources of a single machine are not enough, so we have to scale our computation among number of machines.

For this problem we will [setup a cluster](http://insightedge.io/docs/010/13_cluster_setup.html) with four slaves, the downloaded files are placed on HDFS.

![Alt cluster](img/0_cluster.png?raw=true "Cluster")

Let's open the [Web Notebook](http://insightedge.io/docs/010/14_notebook.html).

We use databricks csv library to load csv files from hdfs into Spark dataframe:

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
Now we can show some data:

```scala
df.show(10)
```

![Alt](img/3_df_show.png?raw=true "Df show")

The data fields are:

* id: ad identifier
* click: 0/1 for non-click/click
* hour: format is YYMMDDHH
* C1 -- anonymized categorical variable
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
d* evice_conn_type
* C14-C21 -- anonymized categorical variables

Let's see how many rows in the traning dataset:

```scala
val totalCount = df.count()
```
![Alt](img/4_df_count.png?raw=true "Df count")

Let's calculate the CTR(click-through rate) of the dataset. The click-through rate is the number of times a click is made on the advertisement divided by the total impressions (the number of times an advertisement was served):

```scala
val clicks = df.filter("click = 1").count()
val ctr = clicks.toFloat / totalCount
```

![Alt](img/5_calc_ctr.png?raw=true "Calc CTR")

We can see the the CTR is 0.169 (or 16.9%) which is quite high number, the common numbers in industry are about 0.2-0.3%. Avazu explains this high number by specific way of data sampling, which appears to be nonuniform.



References
* https://en.wikipedia.org/wiki/Click-through_rate
* http://www.wordstream.com/ppc
