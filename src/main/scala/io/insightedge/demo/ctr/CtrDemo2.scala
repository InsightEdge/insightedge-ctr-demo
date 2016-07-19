package io.insightedge.demo.ctr

import com.gigaspaces.spark.context.GigaSpacesConfig
import com.gigaspaces.spark.implicits.all._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author Oleksiy_Dyagilev
  */
object CtrDemo2 {

  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      System.err.println("Usage: CtrDemo1 <spark master url> <grid locator> <train collection>")
      System.exit(1)
    }

    val Array(master, gridLocator, trainCollection) = args

    // Configure InsightEdge settings
    val gsConfig = GigaSpacesConfig("insightedge-space", None, Some(gridLocator))
    val sc = new SparkContext(new SparkConf().setAppName("CtrDemo2").setMaster(master).setGigaSpaceConfig(gsConfig))
    val sqlContext = new SQLContext(sc)

    // load training collection from data grid
    val trainDf = sqlContext.read.grid.load(trainCollection)
    trainDf.cache()

    // use one-hot-encoder to convert categorical features into a vector
    val encodedDf = encodeLabels(trainDf)

    // assemble multiple feature vectors into a single one
    val assembledDf = new VectorAssembler()
      .setInputCols(categoricalColumnsVectors.toArray)
      .setOutputCol("features")
      .transform(encodedDf)

    // convert dataframe to a label points RDD
    val encodedRdd = assembledDf.map { row =>
      val label = row.getAs[Double]("click")
      val features = row.getAs[Vector]("features")
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

  val categoricalColumns = Seq(
    //    "device_id",
    //    "device_ip",
    //    "device_model",
    "device_type",
    "device_conn_type",
    "time_day",
    "time_hour",
    //    "C1",
    //    "banner_pos",
    //    "site_id",
    //    "site_domain",
    //    "site_category",
    //    "app_id",
    //    "app_domain",
    //    "app_category",
    //        "C14",
            "C15",
            "C16",
            "C17",
            "C18",
            "C19",
            "C20",
            "C21"
  )

  val categoricalColumnsVectors = categoricalColumns.map(vectorCol)

  def encodeLabel(df: DataFrame, inputColumn: String): DataFrame = {
    println(s"Encoding label $inputColumn")
    val indexed = new StringIndexer()
      .setInputCol(inputColumn)
      .setOutputCol(indexCol(inputColumn))
      .fit(df)
      .transform(df)

    val encoder = new OneHotEncoder()
      .setDropLast(false)
      .setInputCol(indexCol(inputColumn))
      .setOutputCol(vectorCol(inputColumn))

    encoder.transform(indexed)
      .drop(inputColumn)
      .drop(indexCol(inputColumn))
  }

  def encodeLabels(df: DataFrame): DataFrame = {
    categoricalColumns.foldLeft(df) { case (df, col) => encodeLabel(df, col) }
  }

  def vectorCol(col: String) = col + "_vector"

  def indexCol(col: String) = col + "_index"

}
