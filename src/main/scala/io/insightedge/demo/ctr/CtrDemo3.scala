package io.insightedge.demo.ctr

import com.gigaspaces.spark.context.GigaSpacesConfig
import com.gigaspaces.spark.implicits._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.insightedge._
import org.apache.spark.sql.{SaveMode, Row, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vector

/**
  * @author Oleksiy_Dyagilev
  */
object CtrDemo3 {

  def main(args: Array[String]): Unit = {
    if (args.length < 5) {
      System.err.println("Usage: CtrDemo1 <spark master url> <grid locator> <train collection> <test collection> <prediction result path>")
      System.exit(1)
    }

    val Array(master, gridLocator, trainCollection, testCollection, predictResPath) = args

    // Configure InsightEdge settings
    val gsConfig = GigaSpacesConfig("insightedge-space", None, Some(gridLocator))
    val sc = new SparkContext(new SparkConf().setAppName("CtrDemo3").setMaster(master).setGigaSpaceConfig(gsConfig))
    val sqlContext = new SQLContext(sc)

    // load training and test collection from data grid
    val trainDf = sqlContext.read.grid.load(trainCollection)
    trainDf.cache()

    // add fictive `click` column to test dataset so that we can union test and train correctly later
    val testDf = sqlContext.read.grid.load(testCollection).withColumn("click", lit(-1))
    testDf.cache()

    // use one-hot-encoder to convert categorical features into a vector
    val (encodedTrainDf, encodedTestDf) = encodeLabels(trainDf, testDf)

    // assemble multiple feature vectors into a single one
    val assembler = new VectorAssembler()
      .setInputCols(categoricalColumnsVectors.toArray)
      .setOutputCol("features")

    // Train a model
    val lr = new LogisticRegression().setLabelCol("click")
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01 , 0.1 /*, 1.0 */))
      .addGrid(lr.elasticNetParam, Array(0.0 /*, 0.5 , 1.0 */))
      .addGrid(lr.fitIntercept, Array(false /*, true */))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("click"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(encodedTrainDf)

    // output train results
    println("Grid search results:")
    cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).foreach(println)

    println("Best set of parameters found:" + cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1)

    import sqlContext.implicits._

    // predict test dataset
    val predictionDf = cvModel.transform(encodedTestDf).select("id", "probability").map {
      case Row(id: String, probability: Vector) => (id, probability(1))
    }.toDF("id", "click")

    // save prediction to csv
    predictionDf
      .repartition(1)
      .write.mode(SaveMode.Overwrite)
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .save(predictResPath)

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

  def encodeLabel(unionDf: DataFrame, df1: DataFrame, df2: DataFrame, col: String): (DataFrame, DataFrame) = {
    println(s"Encoding label $col")
    val indexer = new StringIndexer()
      .setInputCol(col)
      .setOutputCol(indexCol(col))
      .fit(unionDf)

    def transform(df: DataFrame) = {
      val indexed = indexer.transform(df)
      val encoder = new OneHotEncoder()
        .setDropLast(false)
        .setInputCol(indexCol(col))
        .setOutputCol(vectorCol(col))

      encoder.transform(indexed)
        .drop(col)
        .drop(indexCol(col))
    }

    (transform(df1), transform(df2))
  }

  def encodeLabels(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
    // we have to encode categorical features on a union dataset,
    // since there might be labels in a test that don't exist in training
    val unionDf = trainDf.unionAll(testDf)
    unionDf.cache()
    categoricalColumns.foldLeft(trainDf -> testDf) { case ((df1, df2), col) => encodeLabel(unionDf, df1, df2, col) }
  }

  def vectorCol(col: String) = col + "_vector"

  def indexCol(col: String) = col + "_index"

}
