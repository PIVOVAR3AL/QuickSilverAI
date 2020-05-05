/*
 * The MIT License (MIT)

 * Copyright (c) 2020 Pivovar Alexander

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package QuickSilverAi.data_management

import com.intel.daal.data_management.data._
import com.intel.daal.services._


object DataManagement {
  def readCSV(filename: String, delimeter: String) : Array[Array[Double]] = {
    scala.io.Source.fromFile(filename)
      .getLines()
      .map(_.split(delimeter).map(_.trim.toDouble))
      .toArray
  }

  def convertX(data: Array[Array[Double]], context: DaalContext) : HomogenNumericTable ={
    val numRows: Int = data.length
    val numCols: Int = data(0).length

    val arrData = new Array[Double](numRows * numCols)
    var i: Int = 0
    for (i <- 0 until numRows) {
      data(i).copyToArray(arrData, i * numCols)
    }
    val table = new HomogenNumericTable(context, arrData, numCols, numRows)
    table.pack()
    table
  }

  def convertY(labels: Array[Double], context: DaalContext) : HomogenNumericTable ={

    val numRows: Int = labels.length
    val tableLabels = new HomogenNumericTable(context, labels, 1, numRows)
    tableLabels.pack()
    tableLabels
  }
}
