"""
Module related to DeconvNet method
"""
import numpy as np
import tensorflow as tf

from ..commons import batch_gradient
from ..commons import deconv_relu_policy
from ..commons import override_relu_gradient
from ..types import Optional
from ..types import Union
from .base import sanitize_input_output
from .base import WhiteBoxExplainer


class DeconvNet(WhiteBoxExplainer):
    """
    Used to compute the DeconvNet method, which modifies the classic Saliency procedure on
    ReLU's non linearities, allowing only the positive gradients (even from negative inputs) to
    pass through.

    Ref. Zeiler & al., Visualizing and Understanding Convolutional Networks (2013).
    https://arxiv.org/abs/1311.2901

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
        output_layer: Optional[Union[str, int]] = -1,
        batch_size: Optional[int] = 32,
    ):
        super().__init__(model, output_layer, batch_size)
        self.model = override_relu_gradient(self.model, deconv_relu_policy)

    @sanitize_input_output
    def explain(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> tf.Tensor:
        """
        Compute DeconvNet for a batch of samples.
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
            Deconv maps.
        """
        gradients = batch_gradient(self.model, inputs, targets, self.batch_size)
        return gradients
