package io.insightedge.demo.ctr

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import com.gigaspaces.spark.context.GigaSpacesConfig
import com.gigaspaces.spark.implicits._
import org.apache.spark.sql.insightedge._

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

/**
  * @author Oleksiy_Dyagilev
  */
object LoadDataToGrid {

  def main(args: Array[String]) = {

    val master = "local[*]"
    val trainCsvPath = "/home/pivot/Downloads/avazu-ctr/train_100"
    //    val testCsvPath = "/home/pivot/Downloads/avazu-ctr/test_100"

    val gsConfig = GigaSpacesConfig("insightedge-space", Some("insightedge"), Some("127.0.0.1:4174"))
    val sc = new SparkContext(new SparkConf().setAppName("CTR").setMaster(master).setGigaSpaceConfig(gsConfig))

    val sql = new SQLContext(sc)

    val rawTrainDf = sql.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "false")
      .load(trainCsvPath)


    rawTrainDf.sample(false, 0.001d)

    rawTrainDf.write.grid.save("raw_training")

    println(rawTrainDf.count())

  }


}
