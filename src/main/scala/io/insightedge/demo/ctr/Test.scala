package io.insightedge.demo.ctr

import java.lang.Double

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author Oleksiy_Dyagilev
  */
object Test {

  def main(args: Array[String]) = {

    val a = "10000174058809263569"

    val d  = a.toDouble

    d.toString
    println(String.format("%.0f", new Double(d)))


  }

}
