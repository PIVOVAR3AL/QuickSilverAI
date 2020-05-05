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

package QuickSilverAi.cross_val_score

import QuickSilverAi.evaluation.TrainTestSpliter
import QuickSilverAi.data_management.DataManagement
import QuickSilverAi.pipeline.classification.components._
import QuickSilverAi.score.Score

import com.intel.daal.services.DaalContext

import scala.collection.mutable.ArrayBuffer

class CrossValScore(val model: ModelImplementation,
                    val cv: Int,
                    val splitSize: Double) {

  def calcCrossValScore(X: Array[Array[Double]], y: Array[Double]): Double = {

    val results = new ArrayBuffer[Double]()
    val context = new DaalContext()

    for (i<-0 to cv) {
      val splitter = new TrainTestSpliter(splitSize)
      val data = splitter.trainTestSplit(X, y)
      val X_train = data._1
      val X_test = data._2
      val y_train = data._3
      val y_test = data._4

      val train_data = DataManagement.convertX(X_train, context)
      val train_labels = DataManagement.convertY(y_train, context)

      model.fit(train_data, train_labels, context)
      val test_data = DataManagement.convertX(X_test, context)
      val prediction = model.predict(test_data, context)
      val score = Score.f1_score(prediction, y_test)
      results += score
    }
  val total_score = results.sum/results.size.toDouble
  total_score
  }
}
