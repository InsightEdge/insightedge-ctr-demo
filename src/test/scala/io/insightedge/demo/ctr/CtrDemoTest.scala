package io.insightedge.demo.ctr

import io.insightedge.demo.ctr.CtrDemo.DateUtils
import org.scalatest.FunSuite

/**
  * @author Oleksiy_Dyagilev
  */
class CtrDemoTest extends FunSuite {


  test("DateUtils parse") {
    val date@(year, month, day, hour) = DateUtils.parse("14091123")

    assert(date == (2014, 8, 11, 23))

  }

}
