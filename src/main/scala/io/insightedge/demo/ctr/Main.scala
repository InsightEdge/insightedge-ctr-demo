package io.insightedge.demo.ctr

import java.util.Date

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

/**
  * @author Oleksiy_Dyagilev
  */
object Main {

  def main(args: Array[String]) = {

    val trainCsvPath = "/home/pivot/Downloads/avazu-ctr/train_1M"
    val testCsvPath = "/home/pivot/Downloads/avazu-ctr/test"

    val sc = new SparkContext(new SparkConf().setAppName("CTR").setMaster("local[2]"))

//    sc.setLogLevel("ERROR")

    val sql = new SQLContext(sc)

    val rawTrainDf = loadCsvFile(sql, trainCsvPath, clickColumn = true)
    val rawTestDf = loadCsvFile(sql, testCsvPath, clickColumn = false)

    rawTrainDf.cache()
    rawTrainDf.printSchema()

    val count = rawTrainDf.count()
    val clicks = rawTrainDf.filter("click = 1").count()

    println(s"count $count")
    println(s"clicks $clicks")
    println(s"clicks% ${clicks.toFloat / count}")

    val (encodedTrainDf, encodedTestDf) = encodeLabels(rawTrainDf, rawTestDf)

    val Array(training, validation) = assembleFeatures(encodedTrainDf)
      .select("features", "click")
      .map(rowToLabelPoint)
      .randomSplit(Array(0.8, 0.2), seed = 17)

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold

    // Compute raw scores on the test set.
    val predictionAndLabels = validation.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    calcMetrics(predictionAndLabels)

    // predict Kaggle test data
    kaggleTest(sql, model, encodedTestDf)
  }

  def rowToLabelPoint(r: Row) = LabeledPoint(r.getAs[Int]("click").toDouble, r.getAs[Vector]("features"))

  def kaggleTest(sql: SQLContext, model: LogisticRegressionModel, encodedTestDf: DataFrame) = {

    val test = assembleFeatures(encodedTestDf).select("id", "features")

    val outRdd = test.map { r =>
      val features = r.getAs[Vector]("features")
      val id = r.getAs[String]("id")
      val score = model.predict(features)
      s"$id,$score"
    }

    // TODO: to save to a single file
    outRdd.repartition(1).saveAsTextFile("/home/pivot/Downloads/avazu-ctr/test_out" + new Date())
    println("done")
  }

  def loadCsvFile(sqlContext: SQLContext, csvPath: String, clickColumn: Boolean): DataFrame = {
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(csvPath)

    val selectCols = ListBuffer(
      df("id"),
      df("device_type"),
      df("device_conn_type"),
      df("C15").cast(IntegerType),
      df("C16").cast(IntegerType),
      df("C17").cast(IntegerType),
      df("C18").cast(IntegerType),
      df("C19").cast(IntegerType),
      df("C20").cast(IntegerType),
      df("C21").cast(IntegerType)
    )

    if (clickColumn) {
      selectCols += df("click").cast(IntegerType)
    }

    df.select(selectCols: _*)
  }

  def encodeLabel(unionDf: DataFrame, df1: DataFrame, df2: DataFrame, inputColumn: String): (DataFrame, DataFrame) = {
    val indexer = new StringIndexer()
      .setInputCol(inputColumn)
      .setOutputCol(inputColumn + "_index")
      .fit(unionDf)

    def transform(df: DataFrame) = {
      val indexed = indexer.transform(df)
      val encoder = new OneHotEncoder()
        .setInputCol(inputColumn + "_index")
        .setOutputCol(inputColumn + "_vector")

      encoder.transform(indexed)
    }

    (transform(df1), transform(df2))
  }

  def encodeLabels(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
    val categoricalColumns = Seq("device_type", "device_conn_type")

    // add fictive 'click' column to testDf so we can union them
    val unionDf = trainDf.unionAll(
      testDf.withColumn("click", lit(0))
    )

    categoricalColumns.foldLeft(trainDf -> testDf) { case ((df1, df2), col) => encodeLabel(unionDf, df1, df2, col) }
  }

  def assembleFeatures(df: DataFrame): DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(Array("device_type_vector", "device_conn_type_vector", "C15", "C16", "C17", "C18", "C19", "C20", "C21"))
      .setOutputCol("features")

    assembler.transform(df)
  }

  def calcMetrics(predictionAndLabels: RDD[(Double, Double)]): Unit = {
    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
  }

}
