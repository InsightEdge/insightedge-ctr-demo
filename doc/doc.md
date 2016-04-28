# Scalable machine learning with InsightEdge: mobile advertisement clicks prediction.

## Overview

In this blog post we will look how to use machine learning algorithms with InsightEdge. For this purpose we will predict mobile advertisement click-through rate with the [Avazu's dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

There are several compensation models in online advertising industry, probably the most notable is CPC (Cost Per Click), in which an advertiser pays a publisher when the ad is clicked.
Search engine advertising is one of the most popular forms of CPC. It allows advertisers to bid for ad placement in a search engine's sponsored links when someone searches on a keyword that is related to their business offering.
For the search engines like Google advertising became the key source of their revenue. The challenge for the advertising system is to determine what ad should be displayed for each query search engine receives.
The revenue search engine can get is essentially:
Revenue = (ad bid) * (probability of click).
The goal is to maximize the revenue for every search engine query. Whereis the `ad bid` is a known value, the `probability of click` is not. Thus predicting the probability of click becomes the key challenge.

## del it
The probability of a click is known as click-through rate (CTR). The CTR of an advertisement is defined as the number of clicks on an ad divided by the number of times the ad is shown (impressions), expressed as a percentage. For example, if a banner ad is delivered 100 times (100 impressions) and receives one click, then the click-through rate for the advertisement would be 1%.
[CTR formula image here https://upload.wikimedia.org/math/3/c/3/3c386c95782238666a0ca05c3079c8d5.png]
## del it

## Understanding the data

The [dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) consists of:
* train (5.9G) - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
* test (674M) - Test set. 1 day of ads to for testing model predictions.

The first things we want to do is to launch InsightEdge. To get the first data insights quickly, one can [launch InsightEdge on a local machine](http://insightedge.io/docs/010/0_quick_start.html).
Though for the big datasets or compute-intensive tasks the resources of a single machine are not enough, so we have to scale our computation among number of machines. For this problem we will [setup a cluster](http://insightedge.io/docs/010/13_cluster_setup.html) with four slaves, the downloaded files are placed on HDFS.

[cluster img]

Let's open [Web Notebook](http://insightedge.io/docs/010/14_notebook.html) and load the data. We will use databricks csv library to load csv files from hdfs:

```scala
%dep
z.load("com.databricks:spark-csv_2.10:1.3.0")
```

[img #1]

```scala
val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load("hdfs://10.8.1.116/data/avazu_ctr/train")

df.cache()
```scala

[img #2]

Let's get some data:

[img #3]









References
* https://en.wikipedia.org/wiki/Click-through_rate
* http://www.wordstream.com/ppc
