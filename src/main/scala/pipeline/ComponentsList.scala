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

package QuickSilverAi.pipeline.classification.components.base

import QuickSilverAi.pipeline.classification.components.algorithms.decision_forest.RandomForestClassifier
import QuickSilverAi.pipeline.classification.components.algorithms.logitboost.LogitBoost
import QuickSilverAi.pipeline.classification.components.algorithms.adaboost.AdaBoost
import QuickSilverAi.pipeline.classification.components.algorithms.svm.SVM
import QuickSilverAi.pipeline.classification.components.algorithms.gradient_boosting_trees.GradientBoostingTrees
import QuickSilverAi.pipeline.classification.components.algorithms.mn_naive_bayes.MNNaiveBayes
import QuickSilverAi.pipeline.classification.components.algorithms.log_reg.LogisticRegression

import QuickSilverAi.pipeline.classification.components.ModelImplementation

object ComponentsList {

  def get_components(nClasses: Int): List[ModelImplementation] = {
    val components = List(new RandomForestClassifier(nClasses),
      new LogitBoost(nClasses),
      new AdaBoost(nClasses),
      new SVM(nClasses),
      new GradientBoostingTrees(nClasses),
      new MNNaiveBayes(nClasses),
      new LogisticRegression(nClasses))

    components
  }
}