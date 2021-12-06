# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module related to abstract explainer
"""
import warnings
from abc import ABC
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from ..commons import batch_predictions_one_hot
from ..commons import batch_predictions_one_hot_callable
from ..commons import find_layer
from ..commons import predictions_one_hot
from ..commons import predictions_one_hot_callable
from ..commons import tensor_sanitize
from ..types import Callable
from ..types import Dict
from ..types import Optional
from ..types import Tuple
from ..types import Union


def sanitize_input_output(explanation_method: Callable):
    """
    Wrap a method explanation function to ensure tf.Tensor as inputs,
    and as output

    explanation_method
        Function to wrap, should return an tf.tensor.
    """

    def sanitize(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
        targets: Optional[Union[tf.Tensor, np.array]],
        *args
    ):
        # ensure we have tf.tensor
        inputs, targets = tensor_sanitize(inputs, targets)
        # then enter the explanation function
        return explanation_method(self, inputs, targets, *args)

    return sanitize


class BlackBoxExplainer(ABC):
    """
    Base class for Black-Box explainers.

    Parameters
    ----------
    model
        Model used for computing explanations.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    """

    # in order to avoid re-tracing at each tf.function call,
    # share the reconfigured models between the methods if possible
    _cache_models: Dict[Tuple[int, int], tf.keras.Model] = {}

    def __init__(self, model: Callable, batch_size: Optional[int] = 64):
        if isinstance(model, tf.keras.Model):
            model_key = (id(model.input), id(model.output))
            if model_key not in BlackBoxExplainer._cache_models:
                BlackBoxExplainer._cache_models[model_key] = model
            self.model = BlackBoxExplainer._cache_models[model_key]
            self.inference_function = predictions_one_hot
            self.batch_inference_function = batch_predictions_one_hot
        elif isinstance(model, (tf.Module, tf.keras.layers.Layer)):
            self.model = model
            self.inference_function = predictions_one_hot
            self.batch_inference_function = batch_predictions_one_hot
        else:
            self.model = model
            self.inference_function = predictions_one_hot_callable
            self.batch_inference_function = batch_predictions_one_hot_callable

        self.batch_size = batch_size

    @abstractmethod
    def explain(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
        targets: Optional[Union[tf.Tensor, np.array]] = None,
    ) -> tf.Tensor:
        """
        Compute the explanations of the given samples.
        Accept Tensor, numpy array or tf.data.Dataset (in that case targets is None)

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            Explanation generated by the method.
        """
        raise NotImplementedError()

    def __call__(self, inputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Explain alias"""
        return self.explain(inputs, labels)


class WhiteBoxExplainer(BlackBoxExplainer, ABC):
    """
    Base class for White-Box explainers.

    Parameters
    ----------
    model
        Model used for computing explanations.
    output_layer
        Layer to target for the output (e.g logits or after softmax), if int, will be be interpreted
        as layer index, if string will look for the layer name. Default to the last layer, it is
        recommended to use the layer before Softmax.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        output_layer: Optional[Union[str, int]] = None,
        batch_size: Optional[int] = 64,
    ):

        if output_layer is not None:
            # reconfigure the model (e.g skip softmax to target logits)
            target_layer = find_layer(model, output_layer)
            model = tf.keras.Model(model.input, target_layer.output)

            # sanity check, output layer before softmax
            try:
                if (
                    target_layer.activation.__name__
                    == tf.keras.activations.softmax.__name__
                ):
                    warnings.warn(
                        "Output is after softmax, it is recommended to "
                        "use the layer before."
                    )
            except AttributeError:
                pass

        super().__init__(model, batch_size)
