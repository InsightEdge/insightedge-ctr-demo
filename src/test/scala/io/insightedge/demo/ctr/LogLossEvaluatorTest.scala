package io.insightedge.demo.ctr

import org.apache.spark.sql.types.{StructField, DoubleType, StructType}
import org.apache.spark.sql.{SQLContext, Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSuite
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors, Vector}


/**
  * @author Oleksiy_Dyagilev
  */
class LogLossEvaluatorTest extends FunSuite {

  test("testEvaluate") {
    val sc = new SparkContext(new SparkConf().setAppName("LogLossEvaluatorTest").setMaster("local[2]"))
    val sqlContext = new SQLContext(sc)

    val logLoss = new LogLossEvaluator()

    val vectorDF = sqlContext.createDataFrame(Seq(
            (0.0d, Vectors.dense(0.8, 0.2)),
            (0.0d, Vectors.dense(0.6, 0.4))
    )).toDF("click", "rawPrediction")

    val expected = 0.3669845875401002

    assert(logLoss.evaluate(vectorDF) == expected)

    val vectorDF2 = sqlContext.createDataFrame(Seq(
      (1.0d, Vectors.dense(0.2, 0.8)),
      (1.0d, Vectors.dense(0.4, 0.6))
    )).toDF("click", "rawPrediction")

    assert(logLoss.evaluate(vectorDF2) == expected)



  }

}
