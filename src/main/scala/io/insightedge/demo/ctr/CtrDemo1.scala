package io.insightedge.demo.ctr

import com.gigaspaces.spark.context.GigaSpacesConfig
import com.gigaspaces.spark.implicits._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.insightedge._
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author Oleksiy_Dyagilev
  */
object CtrDemo1 {

  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      System.err.println("Usage: CtrDemo1 <spark master url> <grid locator> <train collection>")
      System.exit(1)
    }

    val Array(master, gridLocator, trainCollection) = args

    // Configure InsightEdge settings
    val gsConfig = GigaSpacesConfig("insightedge-space", None, Some(gridLocator))
    val sc = new SparkContext(new SparkConf().setAppName("CtrDemo1").setMaster(master).setGigaSpaceConfig(gsConfig))
    val sqlContext = new SQLContext(sc)

    // load training collection from data grid
    val trainDf = sqlContext.read.grid.load(trainCollection)
    trainDf.cache()

    // use one-hot-encoder to convert 'device_conn_type' categorical feature into a vector
    val indexer = new StringIndexer()
      .setInputCol("device_conn_type")
      .setOutputCol("device_conn_type_index")
      .fit(trainDf)

    val indexed = indexer.transform(trainDf)

    val encodedDf = new OneHotEncoder()
      .setDropLast(false)
      .setInputCol("device_conn_type_index")
      .setOutputCol("device_conn_type_vector")
      .transform(indexed)

    // convert dataframe to label points RDD
    val encodedRdd = encodedDf.map { row =>
      val label = row.getAs[String]("click").toDouble
      val features = row.getAs[Vector]("device_conn_type_vector")
      LabeledPoint(label, features)
    }

    // Split data into training (60%) and test (40%)
    val Array(trainingRdd, testRdd) = encodedRdd.randomSplit(Array(0.6, 0.4), seed = 11L)
    trainingRdd.cache()

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(trainingRdd)

    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold

    // Compute raw scores on the test set
    val predictionAndLabels = testRdd.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
  }
}
