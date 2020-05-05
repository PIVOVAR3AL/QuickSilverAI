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

package QuickSilverAi.pipeline.classification

import QuickSilverAi.pipeline.classification.components.Pipeline
import QuickSilverAi.pipeline.classification.components._
import QuickSilverAi.cross_val_score.CrossValScore

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext


class QSClassifier(val nClasses: Int, cvSize: Int, splitSize: Double) extends Serializable {

  def fit(sc: SparkContext, X: Array[Array[Double]], y: Array[Double]) : (ModelImplementation, Double)= {
    val pipeline = Pipeline.get_pipeline(nClasses)
    val broadcastX = sc.broadcast(X)
    val broadcastY = sc.broadcast(y)
    val models: RDD[(ModelImplementation, Double)] = sc.parallelize(pipeline).map(
      model => {
        val trainData = broadcastX.value
        val trainLabel = broadcastY.value

        val cv = new CrossValScore(model, cvSize, splitSize)
        val score = cv.calcCrossValScore(trainData, trainLabel)

        val result = (model, score)
        result
      })

    val best_model = models.reduce((acc,value) => {
      if(acc._2 < value._2) value else acc})
    best_model
  }
}
