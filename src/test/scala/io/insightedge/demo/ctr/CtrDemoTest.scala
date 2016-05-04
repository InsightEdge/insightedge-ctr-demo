package io.insightedge.demo.ctr

import java.util.Calendar

import io.insightedge.demo.ctr.CtrDemo.DateUtils
import org.scalatest.FunSuite

/**
  * @author Oleksiy_Dyagilev
  */
class CtrDemoTest extends FunSuite {


  test("DateUtils parse") {
    assert(2014 == DateUtils.parse("14091123", Calendar.YEAR) )
    assert(8 == DateUtils.parse("14091123", Calendar.MONTH) )
    assert(11 == DateUtils.parse("14091123", Calendar.DAY_OF_MONTH) )
    assert(23 == DateUtils.parse("14091123", Calendar.HOUR_OF_DAY) )
  }

}
