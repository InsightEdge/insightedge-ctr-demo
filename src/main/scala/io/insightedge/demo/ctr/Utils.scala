package io.insightedge.demo.ctr

import java.io.{File, FileInputStream, FileOutputStream}
import java.util.zip.{ZipEntry, ZipOutputStream}

/**
  * @author Oleksiy_Dyagilev
  */
object Utils {

  def zipPredictionFile(file: String, outputFile: String) = {
    val fos = new FileOutputStream(outputFile)
    val zos = new ZipOutputStream(fos)
    val ze = new ZipEntry("prediction.txt")
    zos.putNextEntry(ze)
    val in = new FileInputStream(file)

    val buffer = new Array[Byte](1024)

    Stream.continually {
      val len = in.read(buffer)
      len
    }.takeWhile(_ != -1).foreach(len => zos.write(buffer, 0, len))

    in.close()
    zos.closeEntry()
    zos.close()

  }


}
