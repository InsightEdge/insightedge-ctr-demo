package io.insightedge.demo.ctr

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

/**
  * @author Oleksiy_Dyagilev
  */
object CtrDemo {

  def main(args: Array[String]) = {

    if (args.length < 3) {
      System.err.println("Usage: CtrDemo <spark master url> <trainCsvPath> <testCsvPath> <outPredictionDir>")
      System.exit(1)
    }

    val Array(master, trainCsvPath, testCsvPath, outPredictionDir) = args

    val startTime = System.currentTimeMillis()

    val sc = new SparkContext(new SparkConf().setAppName("CTR").setMaster(master))

    //    sc.setLogLevel("ERROR")

    val sql = new SQLContext(sc)

    val rawTrainDf = loadCsvFile(sql, trainCsvPath, hasClickColumn = true)
    val rawTestDf = loadCsvFile(sql, testCsvPath, hasClickColumn = false)

    val (encodedTrainDf, encodedTestDf) = encodeLabels(
      transformHour(rawTrainDf),
      transformHour(rawTestDf)
    )

    //    val hourTransformer = new HourTransformer()
    val assembler = new VectorAssembler()
      .setInputCols(categoricalColumnsVectors.toArray)
      .setOutputCol("features")

    val lr = new LogisticRegression().setLabelCol("click")

    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1 /*, 1.0 */))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5 /*, 1.0 */))
      .addGrid(lr.fitIntercept)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("click"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(encodedTrainDf)

    //    cvModel.save(outPredictionDir + "/test_out_" + new Date() + "_model")

    // predict Kaggle test data
    kaggleTest(sql, cvModel, encodedTestDf, outPredictionDir)

    val endTime = System.currentTimeMillis()
    println("time taken(s): " + (endTime - startTime) / 1000)
  }

  def kaggleTest(sql: SQLContext, cvModel: CrossValidatorModel, encodedTestDf: DataFrame, outPredictionDir: String) = {

    val predictionRdd = cvModel.transform(encodedTestDf).map { r =>
      val id = r.getAs[String]("id")
      val probVector = r.getAs[Vector]("probability")
      val clickProb = probVector(1)
      s"$id,$clickProb"
    }

    //    val header = sql.sparkContext.parallelize(Seq("id,click"))

    // TODO: to save to a single file
    val outFile = outPredictionDir + "/kaggle_prediction_" + new SimpleDateFormat("dMMMHHmm").format(new Date())
    predictionRdd.saveAsTextFile(outFile)

    println("=======")
    cvModel.avgMetrics.foreach(m => println("METRICS = " + m))

    println("best params" + cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1)
    println("=======")


    //    Utils.zipPredictionFile(outFile + "/part-00000", outFile + "/prediction.zip")
    println("done")
  }

  def loadCsvFile(sqlContext: SQLContext, csvPath: String, hasClickColumn: Boolean): DataFrame = {
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(csvPath)

    val selectCols = ListBuffer(
      df("id"),
      //      df("device_id"),
      //      df("device_ip"),
      //      df("device_model"),
      df("device_type"),
      df("device_conn_type"),
      df("hour"),
      df("C1"),
      df("banner_pos"),
      //      df("site_id"),
      //      df("site_domain"),
      df("site_category"),
      //      df("app_id"),
      //      df("app_domain"),
      df("app_category"),
      //      df("C14"),
      df("C15"),
      df("C16"),
      df("C17"),
      df("C18"),
      df("C19"),
      df("C20"),
      df("C21")
    )

    df.registerTempTable("training")

    // click col is only in train dataset, it's missing in the test dataset, but we want to keep the schema the same
    // so we can union datasets later, so for test dataset we add fictive 'click' col
    if (hasClickColumn) {
      selectCols += df("click").cast(DoubleType)
      df.select(selectCols: _*)
    } else {
      df.select(selectCols: _*).withColumn("click", lit(0))
    }
  }

  def encodeLabel(unionDf: DataFrame, df1: DataFrame, df2: DataFrame, inputColumn: String): (DataFrame, DataFrame) = {
    println(s"Encoding label $inputColumn")
    val indexer = new StringIndexer()
      .setInputCol(inputColumn)
      .setOutputCol(inputColumn + "_index")
      .fit(unionDf)

    def transform(df: DataFrame) = {
      val indexed = indexer.transform(df)
      val encoder = new OneHotEncoder()
        .setDropLast(false)
        .setInputCol(inputColumn + "_index")
        .setOutputCol(inputColumn + "_vector")

      encoder.transform(indexed)
        .drop(inputColumn)
        .drop(inputColumn + "_index")
    }

    (transform(df1), transform(df2))
  }

  val categoricalColumns = Seq(
    //    "device_id",
    //    "device_ip",
    //    "device_model",
    "device_type",
    "device_conn_type",
    "time_year",
    "time_month",
    "time_day",
    "time_hour",
    "C1",
    "banner_pos",
    //    "site_id",
    //    "site_domain",
    "site_category",
    //    "app_id",
    //    "app_domain",
    "app_category",
    //    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21")

  val categoricalColumnsVectors = categoricalColumns.map(_ + "_vector")

  def encodeLabels(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
    trainDf.cache()
    testDf.cache()

    // remove 'click' column so we can union them correctly
    val unionDf = trainDf.unionAll(testDf)

    unionDf.cache()

    categoricalColumns.foldLeft(trainDf -> testDf) { case ((df1, df2), col) => encodeLabel(unionDf, df1, df2, col) }
  }

  def transformHour(df: DataFrame): DataFrame = {
    val toYear = udf[Int, String](s => DateUtils.parse(s)._1)
    val toMonth = udf[Int, String](s => DateUtils.parse(s)._2)
    val toDay = udf[Int, String](s => DateUtils.parse(s)._3)
    val toHour = udf[Int, String](s => DateUtils.parse(s)._4)

    df.withColumn("time_year", toYear(df("hour")))
      .withColumn("time_month", toMonth(df("hour")))
      .withColumn("time_day", toDay(df("hour")))
      .withColumn("time_hour", toHour(df("hour")))
      .drop("hour")
  }

  object DateUtils {
    val dateFormat = new ThreadLocal[SimpleDateFormat]() {
      override def initialValue(): SimpleDateFormat = new SimpleDateFormat("yyMMddHH")
    }

    def parse(s: String): (Int, Int, Int, Int) = {
      val date = dateFormat.get().parse(s)
      val cal = Calendar.getInstance()
      cal.setTime(date)
      val year = cal.get(Calendar.YEAR)
      val month = cal.get(Calendar.MONTH)
      val day = cal.get(Calendar.DAY_OF_MONTH)
      val hour = cal.get(Calendar.HOUR_OF_DAY)
      (year, month, day, hour)
    }
  }

}
