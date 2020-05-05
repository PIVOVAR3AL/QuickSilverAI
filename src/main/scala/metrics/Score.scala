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

package QuickSilverAi.score

import scala.collection.mutable.ArrayBuffer

object Score {

  def accuracy(y_pred: Array[Double], y_true: Array[Double]): Double = {
    val correct_predictions: Double = List.range(0, y_pred.length - 1).count(idx => y_pred(idx) == y_true(idx))
    correct_predictions / y_pred.length
  }

  def f1_score(y_pred: Array[Double], y_true: Array[Double]): Double = {
    val report = classification_report(y_pred, y_true)
    report.map(_(2)).sum / report.length.toDouble
  }

  def confusion_matrix(y_pred: Array[Double], y_true: Array[Double]): Array[Array[Int]] = {
    val labels = y_true.toSet
    val co_matrix = new ArrayBuffer[Array[Int]]()
    for (label <- labels) {
      val label_indexes = List.range(0, y_pred.length).filter(y_pred(_) == label)
      co_matrix += labels.toArray.map(true_label => {
        label_indexes.count(y_true(_) == true_label)
      })
    }
    co_matrix.toArray
  }

  def classification_report(y_pred: Array[Double], y_true: Array[Double]): Array[Array[Double]] = {
    val classification_report = new ArrayBuffer[Array[Double]]()
    val co_matrix = confusion_matrix(y_pred, y_true)
    for (i <- co_matrix.indices) {

      val precisionSum = co_matrix(i).sum.toDouble
      val recallSum = co_matrix.map(_ (i)).sum.toDouble

      val precision: Double = if (precisionSum !=0 ) co_matrix(i)(i).toDouble / precisionSum else 0
      val recall: Double = if (recallSum != 0) co_matrix(i)(i).toDouble / recallSum else 0

      val precisionRecallSum = precision + recall
      val f1_score: Double = if (precisionRecallSum != 0) 2 * (precision * recall) / precisionRecallSum else 0
      classification_report += Array(precision, recall, f1_score)
    }
    classification_report.toArray
  }
}