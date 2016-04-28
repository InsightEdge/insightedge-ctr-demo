package io.insightedge.demo.ctr

import org.apache.commons.math3.util.FastMath
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.mllib.linalg.Vector

/**
  * @author Oleksiy_Dyagilev
  */
class LogLossEvaluator(override val uid: String) extends Evaluator {

  def this() = this(Identifiable.randomUID("logLossEval"))

  override def evaluate(dataset: DataFrame): Double = {
    val labelCol = "click"
    val rawPredictionCol = "rawPrediction"

    val epsilon = 1e-15
    val minusLogLoss = dataset.select(rawPredictionCol, labelCol)
      .map { case Row(probabilities: Vector, label: Double) =>
        val probability = Math.max(epsilon, Math.min(1 - epsilon, probabilities(1)))
        label * FastMath.log(probability) + (1 - label) * FastMath.log(1 - probability)
      }
      .mean()

    -1.0 * minusLogLoss
  }

  def copy(extra: ParamMap): Evaluator = defaultCopy(extra)

  override def isLargerBetter: Boolean = false
}
