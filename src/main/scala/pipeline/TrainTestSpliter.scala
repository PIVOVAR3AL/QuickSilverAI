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

package QuickSilverAi.evaluation

import scala.util.Random

class TrainTestSpliter(var splitSize: Double) {

  def trainTestSplit(X: Array[Array[Double]], y: Array[Double]) : (Array[Array[Double]],
    Array[Array[Double]],
    Array[Double],
    Array[Double]) = {
    val n = (X.length * splitSize).toInt
    val test_indices = Random.shuffle(List.range(0, X.length-1)).take(n)
    val train_indices = List.range(0, X.length-1).filter(test_indices.contains(_) == false)
    val result = (train_indices.map(X(_)).toArray,
                  test_indices.map(X(_)).toArray,
                  train_indices.map(y(_)).toArray,
                  test_indices.map(y(_)).toArray)
    result
  }
}