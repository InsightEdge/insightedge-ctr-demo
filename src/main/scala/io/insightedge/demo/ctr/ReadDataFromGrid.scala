package io.insightedge.demo.ctr

import com.gigaspaces.spark.context.GigaSpacesConfig
import com.gigaspaces.spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.insightedge._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

/**
  * @author Oleksiy_Dyagilev
  */
object ReadDataFromGrid {

  def main(args: Array[String]) = {

    val master = "local[*]"
    val trainCsvPath = "/home/pivot/Downloads/avazu-ctr/train_100"
    //    val testCsvPath = "/home/pivot/Downloads/avazu-ctr/test_100"

    val gsConfig = GigaSpacesConfig("insightedge-space", Some("insightedge"), Some("127.0.0.1:4174"))
    val sc = new SparkContext(new SparkConf().setAppName("CTR").setMaster(master).setGigaSpaceConfig(gsConfig))

    val sql = new SQLContext(sc)

    val df = sql.read.grid.load("raw_training")

    println(df.count())

  }


}
