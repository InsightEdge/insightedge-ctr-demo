package ctr.demo.insightedge.io

import io.insightedge.demo.ctr.Main.DateUtils
import org.scalatest.FunSuite

/**
  * @author Oleksiy_Dyagilev
  */
class MainTest extends FunSuite {


  test("DateUtils parse") {
    val date@(year, month, day, hour) = DateUtils.parse("14091123")

    assert(date == (2014, 8, 11, 23))

  }

}
