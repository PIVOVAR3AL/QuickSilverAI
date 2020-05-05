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

package QuickSilverAi.pipeline.classification.components.algorithms.gradient_boosting_trees

import com.intel.daal.algorithms.classifier.prediction.ModelInputId
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId
import com.intel.daal.algorithms.classifier.training.InputId
import com.intel.daal.algorithms.classifier.training.TrainingResultId
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId
import com.intel.daal.algorithms.gbt.classification.Model
import com.intel.daal.algorithms.gbt.classification.prediction._
import com.intel.daal.algorithms.gbt.classification.training._
import com.intel.daal.algorithms.gbt.training._
import com.intel.daal.data_management.data.HomogenNumericTable
import com.intel.daal.services.DaalContext

import QuickSilverAi.hyperparameter_search._
import QuickSilverAi.uniform_hyperparameter._
import QuickSilverAi.pipeline.classification.components.ModelImplementation


class GradientBoostingTrees(override val nClasses: Int,
                            var splitMethod: String,
                            var maxIterations: Int,
                            var maxTreeDepth: Int,
                            var shrinkage: Double,
                            var minSplitLoss: Double,
                            var lambda: Double,
                            var observationsPerTreeFraction: Double,
                            var featuresPerNode: Int,
                            var minObservationsInLeafNode: Int,
                            var maxBins: Int,
                            var minBinSize: Int) extends ModelImplementation(nClasses) {

  def this(nClasses: Int) = this(nClasses, "inexact", 50, 0, 0.3, 0, 1, 1, 0, 5, 256, 5)

  def setSplitMethod(_splitMethod: java.lang.String): Unit = {
    splitMethod = _splitMethod.toString
  }

  def setMaxIterations(_maxIterations: java.lang.Integer): Unit = {
    maxIterations = _maxIterations.toInt
  }

  def setMaxIterations(_maxIterations: Int): Unit = {
    maxIterations = _maxIterations
  }

  def setMaxTreeDepth(_maxTreeDepth: java.lang.Integer): Unit = {
    maxTreeDepth = _maxTreeDepth.toInt
  }

  def setMaxTreeDepth(_maxTreeDepth: Int): Unit = {
    maxTreeDepth = _maxTreeDepth
  }

  def setShrinkage(_shrinkage: java.lang.Double): Unit = {
    shrinkage = _shrinkage.toDouble
  }

  def setShrinkage(_shrinkage: Double): Unit = {
    shrinkage = _shrinkage
  }

  def setMinSplitLoss(_minSplitLoss: java.lang.Double): Unit = {
    minSplitLoss = _minSplitLoss.toDouble
  }

  def setMinSplitLoss(_minSplitLoss: Double): Unit = {
    minSplitLoss = _minSplitLoss
  }

  def setLambda(_lambda: java.lang.Double): Unit = {
    lambda = _lambda.toDouble
  }

  def setLambda(_lambda: Double): Unit = {
    lambda = _lambda
  }

  def setObservationsPerTreeFraction(_observationsPerTreeFraction: java.lang.Double): Unit = {
    observationsPerTreeFraction = _observationsPerTreeFraction.toDouble
  }

  def setObservationsPerTreeFraction(_observationsPerTreeFraction: Double): Unit = {
    observationsPerTreeFraction = _observationsPerTreeFraction
  }

  def setFeaturesPerNode(_featuresPerNode: java.lang.Integer): Unit = {
    featuresPerNode = _featuresPerNode.toInt
  }

  def setFeaturesPerNode(_featuresPerNode: Int): Unit = {
    featuresPerNode = _featuresPerNode
  }

  def setMinObservationsInLeafNode(_minObservationsInLeafNode: java.lang.Integer): Unit = {
    minObservationsInLeafNode = _minObservationsInLeafNode.toInt
  }

  def setMinObservationsInLeafNode(_minObservationsInLeafNode: Int): Unit = {
    minObservationsInLeafNode = _minObservationsInLeafNode
  }

  def setMaxBins(_maxBins: java.lang.Integer): Unit = {
    maxBins = _maxBins.toInt
  }

  def setMaxBins(_maxBins: Int): Unit = {
    maxBins = _maxBins
  }

  def setMinBinSize(_minBinSize: java.lang.Integer): Unit = {
    minBinSize = _minBinSize.toInt
  }

  def setMinBinSize(_minBinSize: Int): Unit = {
    minBinSize = _minBinSize
  }

  var model: Model = _

  override def fit(X: HomogenNumericTable, y: HomogenNumericTable, context: DaalContext): Unit = {

    val _splitMethod = if (splitMethod == "inexact") SplitMethod.inexact else SplitMethod.exact
    val algorithm = new TrainingBatch(context, classOf[java.lang.Double], TrainingMethod.defaultDense, nClasses)

    algorithm.parameter.setSplitMethod(_splitMethod)
    algorithm.parameter.setMaxIterations(maxIterations)
    algorithm.parameter.setMaxTreeDepth(maxTreeDepth)
    algorithm.parameter.setShrinkage(shrinkage)
    algorithm.parameter.setMinSplitLoss(minSplitLoss)
    algorithm.parameter.setLambda(lambda)
    algorithm.parameter.setObservationsPerTreeFraction(observationsPerTreeFraction)
    algorithm.parameter.setFeaturesPerNode(featuresPerNode)
    algorithm.parameter.setMinObservationsInLeafNode(minObservationsInLeafNode)
    algorithm.parameter.setMaxBins(maxBins)
    algorithm.parameter.setMinBinSize(minBinSize)

    algorithm.input.set(InputId.data, X)
    algorithm.input.set(InputId.labels, y)

    val trainingResult = algorithm.compute()
    model = trainingResult.get(TrainingResultId.model)
  }

  override def predict(X: HomogenNumericTable, context: DaalContext): Array[Double] = {

    val algorithm = new PredictionBatch(context, classOf[java.lang.Double], PredictionMethod.defaultDense, nClasses)

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

    val splitMethodParam = new CategoricalHyperparameter("setSplitMethod", List("inexact", "exact"))
    val maxIterationsParam = new UniformIntegerHyperparameter("setMaxIterations", 100, 500, 100)
    val shrinkageParam = new UniformDoubleHyperparameter("setShrinkage", 0, 1, 0.1)
    val minSplitLossParam = new UniformDoubleHyperparameter("setMinSplitLoss", 0, 1, 0.5)
    val lambdaParam = new UniformDoubleHyperparameter("setLambda", 0, 1, 0.5)
    val minObservationsInLeafNodeParam = new UniformIntegerHyperparameter("setMinObservationsInLeafNode", 1, 8, 2)

    val hyperparams_list = List(splitMethodParam, maxIterationsParam, shrinkageParam, minSplitLossParam, lambdaParam,
                                minObservationsInLeafNodeParam)

    cs.add_hyperparameters(hyperparams_list)
    cs
  }
}