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

package QuickSilverAi.uniform_hyperparameter

import scala.collection.mutable.ListBuffer

class Param(var param_name: String, var param_value: Any) {
}

class UniformHyperparameter {

  def get_params(): List[Param] ={
    val params = new ListBuffer[Param]()
    params.toList
  }
}

class UniformIntegerHyperparameter(var name: String,
                                   var lower: Int,
                                   var upper: Int,
                                   var step: Int) extends UniformHyperparameter {

  override def get_params(): List[Param] ={

    var params = new ListBuffer[Param]()

    var i: Int = lower
    while (i < upper) {
      val param = new Param(name, i)
      params += param
      i += step
    }

    params.toList
  }
}

class UniformDoubleHyperparameter(var name: String,
                                 var lower: Double,
                                 var upper: Double,
                                 var step: Double) extends UniformHyperparameter {

  override def get_params(): List[Param] ={

    var params = new ListBuffer[Param]()

    var i: Double = lower
    while (i < upper) {
      val param = new Param(name, i)
      params += param
      i += step
    }

    params.toList
  }
}

class CategoricalHyperparameter(var name: String, var choices: List[Any]) extends UniformHyperparameter {

  override def get_params(): List[Param] ={
    choices.map(elem => {
      val param = new Param(name, elem)
      param
    })
  }
}