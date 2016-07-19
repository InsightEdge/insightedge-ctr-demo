package io.insightedge.demo.ctr

import java.text.SimpleDateFormat
import java.util.Calendar

import com.gigaspaces.spark.context.GigaSpacesConfig
import com.gigaspaces.spark.implicits.all._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * This is a helper utility to load sample datasets to grid on local dev machine
  *
  * Not used in the blog post (in the blog we use Zeppelin to do this)
  *
  * @author Oleksiy_Dyagilev
  */
object LoadDataToGrid {

  def main(args: Array[String]) = {

    val master = "local[*]"
    val trainCsvPath = "/home/pivot/Downloads/avazu-ctr/train_100k"
    val testCsvPath = "/home/pivot/Downloads/avazu-ctr/test_100"

    val gsConfig = GigaSpacesConfig("insightedge-space", Some("insightedge"), Some("127.0.0.1:4174"))
    val sc = new SparkContext(new SparkConf().setAppName("CTR").setMaster(master).setGigaSpaceConfig(gsConfig))

    val sql = new SQLContext(sc)

    val df = sql.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(trainCsvPath)

    val testDf = sql.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(testCsvPath)

    transformHour(testDf)
      .write.mode(SaveMode.Overwrite).grid.save("test")

  }

  def castClickType(df: DataFrame): DataFrame = {
    df.withColumn("clickTmp", df("click").cast(DoubleType))
      .drop("click")
      .withColumnRenamed("clickTmp", "click")
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
      .drop("time_month")
      .drop("time_year")
  }

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


}
