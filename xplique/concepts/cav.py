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
Module related to the Concept extraction mechanism
"""
import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from ..commons import find_layer
from ..types import Callable
from ..types import Union


class Cav:  # pylint: disable=too-few-public-methods
    """
    Used to compute the Concept Activation Vector, which is a vector in the direction of the
    activations of that concept’s set of examples.

    Ref. Kim & al., Interpretability Beyond Feature Attribution: Quantitative Testing with Concept
    Activation Vectors (TCAV) (2018).
    https://arxiv.org/abs/1711.11279

    Parameters
    ----------
    model
        Model to extract concept from.
    target_layer
        Index of the target layer or name of the layer.
    classifier : 'SGD' or 'SVC' or Sklearn model, optional
        Default implementation use SGD with hinge classifier (linear SVM), SVC use libsvm but
        the computation time is longer.
    test_fraction
        Fraction of the dataset used for test
    batch_size
        Batch size during the activations extraction
    verbose
        If true, display information while training the classifier
    """

    def __init__(
        self,
        model: tf.keras.Model,
        target_layer: Union[str, int],
        classifier: Union[str, Callable] = "SGD",
        test_fraction: float = 0.2,
        batch_size: int = 64,
        verbose: bool = False,
    ):
        self.model = model
        self.batch_size = batch_size
        self.test_fraction = test_fraction
        self.verbose = verbose

        # configure model bottleneck
        target_layer = find_layer(model, target_layer)
        self.bottleneck_model = tf.keras.Model(model.input, target_layer.output)

        # configure classifier
        if classifier == "SGD":
            # official parameters
            self.classifier = SGDClassifier(
                alpha=0.01, max_iter=1_000, tol=1e-3, verbose=self.verbose
            )
        elif classifier == "SVC":
            self.classifier = LinearSVC(verbose=self.verbose)
        elif all(hasattr(classifier, attr) for attr in ["fit", "score"]):
            self.classifier = classifier
        else:
            raise ValueError("The classifier passed is invalid.")

    def fit(
        self, positive_dataset: tf.Tensor, negative_dataset: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute and return the Concept Activation Vector (CAV) associated to the dataset and the
        layer targeted.

        Parameters
        ----------
        positive_dataset
            Dataset of positive samples : samples containing the concept.
        negative_dataset
            Dataset of negative samples : samples without the concept

        Returns
        -------
        cav
            Vector of the same shape as the layer output
        """
        positive_activations = self.bottleneck_model.predict(
            positive_dataset, batch_size=self.batch_size
        )
        negative_activations = self.bottleneck_model.predict(
            negative_dataset, batch_size=self.batch_size
        )

        activations = np.concatenate(
            [positive_activations, negative_activations], axis=0
        )
        has_concept = np.array(
            [1 for _ in positive_activations] + [0 for _ in negative_activations],
            np.float32,
        )

        original_shape = activations.shape
        activations = activations.reshape(len(activations), -1)

        x_train, x_test, y_train, y_test = train_test_split(
            activations, has_concept, test_size=self.test_fraction
        )

        self.classifier.fit(x_train, y_train)

        if self.verbose:
            val_accuracy = self.classifier.score(x_test, y_test)
            print(f"[CAV] val_accuracy : {round(float(val_accuracy * 100), 2)}")

        # weights of each feature is the vector orthogonal to the hyperplane
        # note : x + epsilon * cav should increase the concept
        cav = self.classifier.coef_[0]
        cav = cav / np.linalg.norm(cav, 1)
        cav = np.array(cav).reshape(original_shape[1:])

        return tf.cast(cav, tf.float32)

    __call__ = fit
