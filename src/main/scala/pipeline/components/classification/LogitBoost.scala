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

package QuickSilverAi.pipeline.classification.components.algorithms.logitboost

import com.intel.daal.algorithms.classifier.prediction.ModelInputId
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId
import com.intel.daal.algorithms.classifier.training.InputId
import com.intel.daal.algorithms.classifier.training.TrainingResultId
import com.intel.daal.algorithms.logitboost.Model
import com.intel.daal.algorithms.logitboost.prediction._
import com.intel.daal.algorithms.logitboost.training._
import com.intel.daal.data_management.data.HomogenNumericTable
import com.intel.daal.services.DaalContext

import QuickSilverAi.hyperparameter_search._
import QuickSilverAi.uniform_hyperparameter._
import QuickSilverAi.pipeline.classification.components.ModelImplementation


class LogitBoost(override val nClasses: Int,
                 var accuracyThreshold: Double,
                 var maxIterations: Int) extends ModelImplementation(nClasses) {

  def this(nClasses: Int) = this(nClasses, 0.01, 100)

  def setAccuracyThreshold(_accuracyThreshold: java.lang.Double): Unit = {
    accuracyThreshold = _accuracyThreshold.toDouble
  }

  def setAccuracyThreshold(_accuracyThreshold: Double): Unit = {
    accuracyThreshold = _accuracyThreshold
  }

  def setMaxIterations(_maxIterations: java.lang.Integer): Unit = {
    maxIterations = _maxIterations.toInt
  }

  def setMaxIterations(_maxIterations: Int): Unit = {
    maxIterations = _maxIterations
  }

  var model: Model = _

  override def fit(X: HomogenNumericTable, y: HomogenNumericTable, context: DaalContext): Unit = {

    val algorithm = new TrainingBatch(context, nClasses, classOf[java.lang.Double], TrainingMethod.friedman)

    algorithm.parameter.setAccuracyThreshold(accuracyThreshold)
    algorithm.parameter.setMaxIterations(maxIterations)

    algorithm.input.set(InputId.data, X)
    algorithm.input.set(InputId.labels, y)

    val trainingResult = algorithm.compute()
    model = trainingResult.get(TrainingResultId.model)
  }

  override def predict(X: HomogenNumericTable, context: DaalContext): Array[Double] = {

    val algorithm = new PredictionBatch(context, nClasses, classOf[java.lang.Double], PredictionMethod.defaultDense)

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(NumericTableInputId.data, X)
    algorithm.input.set(ModelInputId.model, model)

    /* Compute prediction results */
    val predictionResult = algorithm.compute()
    val predictionResults = predictionResult.get(PredictionResultId.prediction)
    val numericTablePointer = predictionResults.getCObject
    val newPointer = new HomogenNumericTable(context, numericTablePointer)

    val prediction = newPointer.getDoubleArray
    prediction
  }

  override def get_hyperparameter_search_space(): HyperparameterSearch = {
    val cs = new HyperparameterSearch()
    val maxIterationsParam = new UniformIntegerHyperparameter("setMaxIterations", 100, 500, 100)
    val hyperparams_list = List(maxIterationsParam)
    cs.add_hyperparameters(hyperparams_list)
    cs
  }
}