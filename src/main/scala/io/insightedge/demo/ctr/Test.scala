package io.insightedge.demo.ctr

import java.lang.Double
import java.text.SimpleDateFormat

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Try

/**
  * @author Oleksiy_Dyagilev
  */
object Test {

  def main(args: Array[String]): Unit = {

//    Utils.zipPredictionFile("/home/pivot/mm.cfg", "/home/pivot/mm.zip")


    val df = new SimpleDateFormat("dMMMHH:mm")
    println(df.format(new java.util.Date()))


  }

}
