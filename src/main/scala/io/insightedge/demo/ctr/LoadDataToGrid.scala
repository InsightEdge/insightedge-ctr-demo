package io.insightedge.demo.ctr

import com.gigaspaces.spark.context.GigaSpacesConfig
import com.gigaspaces.spark.implicits._
import org.apache.spark.sql.insightedge._
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode}
import org.apache.spark.{SparkConf, SparkContext}

/**
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

    HourTransformer
      .transformHour(df)
      .write.mode(SaveMode.Overwrite).grid.save("day_21")


    val testDf = sql.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(testCsvPath)

    HourTransformer
      .transformHour(testDf)
      .write.mode(SaveMode.Overwrite).grid.save("test_tiny")



  }



}
