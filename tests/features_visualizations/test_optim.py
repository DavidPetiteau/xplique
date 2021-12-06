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

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from xplique.features_visualizations import compose_transformations
from xplique.features_visualizations import l1_reg
from xplique.features_visualizations import l2_reg
from xplique.features_visualizations import Objective
from xplique.features_visualizations import optimize
from xplique.features_visualizations import pad
from xplique.features_visualizations import random_blur
from xplique.features_visualizations import random_flip
from xplique.features_visualizations import random_jitter
from xplique.features_visualizations import random_scale
from xplique.features_visualizations import total_variation_reg


def dummy_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((16, 16, 3)),
            tf.keras.layers.Conv2D(3, (3, 3), name="early"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(3, (3, 3), name="features"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(name="pre-logits"),
            tf.keras.layers.Dense(5, name="logits"),
        ]
    )
    model.compile()
    return model


def _assert_objective(obj):
    """Assert the objective return appropriate results under multiple setting"""
    regularizers = [l1_reg(1.0), l2_reg(1.0), total_variation_reg(1.0)]
    transformations = compose_transformations(
        [
            pad(2, 0.0),
            random_jitter(6),
            random_scale(scale_range=[0.95, 1.05]),
            random_blur(),
            random_flip(),
        ]
    )
    nb_steps = 4

    for use_fourier in [True, False]:
        for custom_shape in [None, (20, 20)]:
            for warmup_steps in [False, 2]:
                for save_every in [None, 2]:

                    res, _ = optimize(
                        obj,
                        SGD(),
                        nb_steps=nb_steps,
                        use_fft=use_fourier,
                        regularizers=regularizers,
                        warmup_steps=warmup_steps,
                        custom_shape=custom_shape,
                        transformations=transformations,
                        save_every=save_every,
                    )

                    if save_every is not None:
                        assert len(res) == nb_steps // 2
                    if custom_shape is not None:
                        assert res[0].shape[1:] == (*custom_shape, 3)
                    else:
                        assert res[0].shape[1:] == (16, 16, 3)


def test_layer():
    """Ensure we can optimize on a layer"""
    model = dummy_model()

    obj = Objective.layer(model, -1)
    _assert_objective(obj)


def test_direction():
    """Ensure we can optimize on a direction"""
    model = dummy_model()

    direction_vec = np.random.random(model.get_layer("early").output.shape[1:])
    obj = Objective.direction(model, "early", direction_vec)
    _assert_objective(obj)


def test_channel():
    """Ensure we can optimize on a channel"""
    model = dummy_model()
    nb_channels = 2  # number of channel to optim (1...n)

    obj = Objective.channel(model, "early", list(range(nb_channels)))
    _assert_objective(obj)


def test_neurons():
    """Ensure we can optimize on neurons"""
    model = dummy_model()

    obj = Objective.neuron(model, "early", list(range(3)))
    _assert_objective(obj)


def test_combinations():
    """Ensure we can optimize a combinations of objectives"""
    model = dummy_model()

    layer_obj = Objective.layer(model, -1)

    direction_vec = np.random.random(model.get_layer("pre-logits").output.shape[1:])
    direction_obj = Objective.direction(model, "pre-logits", direction_vec)

    nb_channels = 2
    channels_obj = Objective.channel(model, "early", list(range(nb_channels)))

    nb_neurons = 3
    neurons_obj = Objective.neuron(model, "features", list(range(nb_neurons)))

    combined_obj = layer_obj + direction_obj + channels_obj + neurons_obj
    _assert_objective(combined_obj)
