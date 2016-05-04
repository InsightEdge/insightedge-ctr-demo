package io.insightedge.demo.ctr

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  * @author Oleksiy_Dyagilev
  */
object HourTransformer {

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
