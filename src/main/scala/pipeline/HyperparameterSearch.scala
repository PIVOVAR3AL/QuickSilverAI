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

package QuickSilverAi.hyperparameter_search

import QuickSilverAi.pipeline.classification.components._
import QuickSilverAi.uniform_hyperparameter._

class HyperparameterSearch() {
  var hyperparameters = List[UniformHyperparameter]()

  def add_hyperparameters(uniform_hyperparameters: List[UniformHyperparameter]): Unit ={
    hyperparameters = uniform_hyperparameters
  }

  def combinationList[T](ls:List[List[T]]):List[List[T]] = ls match {
    case Nil => Nil::Nil
    case head :: tail => val rec = combinationList[T](tail)
      rec.flatMap(r => head.map(t => t::r))
  }

  def apply_hyperparameters(model: ModelImplementation): List[ModelImplementation] ={
    if (hyperparameters.isEmpty) {
      return List(model)
    }
    val param_combinations = combinationList(hyperparameters.map(_.get_params()))
    val models = param_combinations.map(param_list => {
      param_list.map(parameter => {
        val method = model.getClass.getMethod(parameter.param_name, parameter.param_value.getClass)
        method.invoke(model, parameter.param_value.asInstanceOf[AnyRef])
        model
      })
    })
    models.flatten
  }
}