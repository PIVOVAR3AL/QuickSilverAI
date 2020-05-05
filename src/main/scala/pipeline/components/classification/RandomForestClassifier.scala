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

package QuickSilverAi.pipeline.classification.components.algorithms.decision_forest

import com.intel.daal.algorithms.classifier.prediction.ModelInputId
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId
import com.intel.daal.algorithms.classifier.training.InputId
import com.intel.daal.algorithms.classifier.training.TrainingResultId
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId
import com.intel.daal.algorithms.decision_forest.classification.Model
import com.intel.daal.algorithms.decision_forest.classification.prediction._
import com.intel.daal.algorithms.decision_forest.classification.training._
import com.intel.daal.algorithms.decision_forest._
import com.intel.daal.data_management.data.HomogenNumericTable
import com.intel.daal.services.DaalContext

import QuickSilverAi.hyperparameter_search._
import QuickSilverAi.uniform_hyperparameter._
import QuickSilverAi.pipeline.classification.components.ModelImplementation


class RandomForestClassifier(override val nClasses: Int,
                             var nTrees: Int,
                             var observationsPerTreeFraction: Double,
                             var featuresPerNode: Long,
                             var maxTreeDepth: Long,
                             var minObservationsInLeafNode: Int) extends ModelImplementation(nClasses) {

  def this(nClasses: Int) = this(nClasses, 100, 1, 0, 0, 1)

  def setNTrees(_nTrees: java.lang.Integer): Unit = {
    nTrees = _nTrees.toInt
  }

  def setNTrees(_nTrees: Int): Unit = {
    nTrees = _nTrees
  }

  def setObservationsPerTreeFraction(_observationsPerTreeFraction: java.lang.Double): Unit = {
    observationsPerTreeFraction = _observationsPerTreeFraction.toDouble
  }

  def setObservationsPerTreeFraction(_observationsPerTreeFraction: Double): Unit = {
    observationsPerTreeFraction = _observationsPerTreeFraction
  }

  def setFeaturesPerNode(_featuresPerNode: java.lang.Long): Unit = {
    featuresPerNode = _featuresPerNode.toLong
  }

  def setFeaturesPerNode(_featuresPerNode: Long): Unit = {
    featuresPerNode = _featuresPerNode
  }

  def setMaxTreeDepth(_maxTreeDepth: java.lang.Long): Unit = {
    maxTreeDepth = _maxTreeDepth.toLong
  }

  def setMaxTreeDepth(_maxTreeDepth: Long): Unit = {
    maxTreeDepth = _maxTreeDepth
  }

  def setMinObservationsInLeafNode(_minObservationsInLeafNode: java.lang.Integer): Unit = {
    minObservationsInLeafNode = _minObservationsInLeafNode.toInt
  }

  def setMinObservationsInLeafNode(_minObservationsInLeafNode: Int): Unit = {
    minObservationsInLeafNode = _minObservationsInLeafNode
  }

  var model: Model = _

  override def fit(X: HomogenNumericTable, y: HomogenNumericTable, context: DaalContext): Unit = {
    val algorithm = new TrainingBatch(context, classOf[java.lang.Double], TrainingMethod.defaultDense, nClasses)
    algorithm.parameter.setNTrees(nTrees)
    algorithm.parameter.setObservationsPerTreeFraction(observationsPerTreeFraction)
    algorithm.parameter.setFeaturesPerNode(featuresPerNode)
    algorithm.parameter.setMaxTreeDepth(maxTreeDepth)
    algorithm.parameter.setMinObservationsInLeafNode(minObservationsInLeafNode)
    algorithm.parameter.setVariableImportanceMode(VariableImportanceModeId.MDI)
    algorithm.parameter.setResultsToCompute(ResultsToComputeId.computeOutOfBagError)

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(InputId.data, X)
    algorithm.input.set(InputId.labels, y)

    /* Train the decision forest classification model */
    val trainingResult = algorithm.compute()

    model = trainingResult.get(TrainingResultId.model)
  }

  override def predict(X: HomogenNumericTable, context: DaalContext): Array[Double] = {

    val algorithm = new PredictionBatch(context, classOf[java.lang.Double], PredictionMethod.defaultDense, nClasses)

    /* Pass a testing data set and the trained model to the algorithm */
    X.unpack(context)

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
    val nTreesParam = new UniformIntegerHyperparameter("setNTrees", 100, 500, 100)
    val minObservationsInLeafNodeParam = new UniformIntegerHyperparameter("setMinObservationsInLeafNode", 1, 8, 2)
    val hyperparams_list = List(nTreesParam, minObservationsInLeafNodeParam)
    cs.add_hyperparameters(hyperparams_list)
    cs
  }
}
