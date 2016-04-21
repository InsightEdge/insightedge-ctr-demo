package io.insightedge.demo.ctr

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author Oleksiy_Dyagilev
  */
object Stupid {

  def main(args: Array[String]) = {

    val csvPath = "/home/pivot/Downloads/avazu-ctr/train_100k"
    val sc = new SparkContext(new SparkConf().setAppName("CTR").setMaster("local[2]"))
    val sqlContext = new SQLContext(sc)

    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(csvPath)

    df.cache()

    df.printSchema()

    val count = df.count()
    val clicks = df.filter("click = 1").count()

    println(s"count $count")
    println(s"clicks $clicks")
    println(s"clicks% ${clicks.toFloat / count}")

    df.registerTempTable("ctr")

    val split = df.randomSplit(Array(0.8, 0.2), seed = 17)

    val trainRaw = split(0)
    val testRaw = split(1)

    val categoricalColumns = Seq("device_type", "device_conn_type")

    val encodedTrain = assembleFeatures(encodeLabels(trainRaw, categoricalColumns))
    val encodedTest = assembleFeatures(encodeLabels(testRaw, categoricalColumns))

    encodedTrain.show()

    val trainingDf = encodedTrain.select("features", "click")
    val testDf = encodedTest.select("features", "click")

    println("====")
    trainingDf.take(1).foreach(println)
    println("====")

    trainingDf.show()

    def rowToLabelPoint(r: Row) =  LabeledPoint(r.getAs[Int]("click").toDouble, r.getAs[Vector]("features"))

    val training = trainingDf.map(rowToLabelPoint)
    val test = testDf.map(rowToLabelPoint)

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    println("threshold = " + model.getThreshold)

    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    calcMetrics(predictionAndLabels)

  }


  def encodeLabel(df: DataFrame, inputColumn: String): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol(inputColumn)
      .setOutputCol(inputColumn + "_index")
      .fit(df)

    val indexed = indexer.transform(df)

    val encoder = new OneHotEncoder()
      .setInputCol(inputColumn + "_index")
      .setOutputCol(inputColumn + "_vector")

    encoder.transform(indexed)
  }

  def encodeLabels(df: DataFrame, inputColumns: Seq[String]): DataFrame = {
    inputColumns.foldLeft(df)((df, inputCol) => encodeLabel(df, inputCol))
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
