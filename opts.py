# Copyright (C) 2020 Clarence Chen
#
# This file is a part of BTS for Tensorflow 2 with Keras.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function
from functools import partial
from six import string_types

import tensorflow as tf
import tensorflow.keras.backend as K

def per_neuron_shape(s):
	if hasattr(s, 'rank'):
		rank = s.rank
	else:
		rank = s.shape[0]
	return [1] * (rank -1) + [s[-1]]

def neuron_norm(x):
	norm = tf.norm(tf.reshape(x, [-1, tf.shape(x)[-1]]), axis=0)
	return tf.reshape(norm, per_neuron_shape(tf.shape(x)))

def neuron_constrain(x):
	def _exec_constrain():
		neuron_mean = tf.reduce_mean(tf.reshape(x, [-1, tf.shape(x)[-1]]), 0)
		return (x -tf.reshape(neuron_mean, per_neuron_shape(tf.shape(x)))) / neuron_norm(x)
	return tf.cond(tf.rank(x) > 1, _exec_constrain, lambda: x)

class PerNeuronZeros(tf.zeros_initializer, tf.keras.initializers.Initializer):

	def __call__(self, shape, dtype=None, **kwargs):
		return super(PerNeuronZeros, self).__call__(shape=per_neuron_shape(shape), dtype=K.floatx(), **kwargs)

class NeRo(tf.keras.optimizers.Optimizer):
	r"""Implements NeRo algorithm.

	The original NeRo algorithm was proposed in `Learning by Turning: Neural Architecture Aware Optimisation`_.

	Arguments:
		learning_rate (float): learning rate of optimizer 
			(default: 0.01)
		beta (float): coefficient to scale contribution of incoming gradient to moving average 
			(default: 0.999)
		epsilon (float): numerical epsilon for avoid overflow in scaling gradient by moving average
			(default: 1e-7)
		bias_scale (float): coefficient to scale bias parameter updates to maintain relative update scale
			(default: 0.01)

	.. Learning by Turning\: Neural Architecture Aware Optimisation:
		https://arxiv.org/abd/2102.07227
	"""

	def __init__(self, learning_rate=0.01, beta=0.999, epsilon=1e-7, bias_scale=1e-2, name='NeRo', **kwargs):
		super(NeRo, self).__init__(name=name, **kwargs)
		self._set_hyper('lr', learning_rate)
		self._set_hyper('beta', beta)
		self._set_hyper('epsilon', epsilon)
		self._set_hyper('bias_scale', bias_scale)

	def _create_slots(self, var_list):
		for var in var_list:
			self.add_slot(var, 'neuron_sq_norm', initializer=PerNeuronZeros())
			var = neuron_constrain(var)

	@tf.function
	def _resource_apply_dense(self, grad, var, apply_state=None):
		var_device, var_dtype = var.device, var.dtype.base_dtype

		bias_correction = 1 - self._get_hyper('beta') ** tf.cast(self.iterations, self._get_hyper('beta').dtype)
		m = self.get_slot(var, 'neuron_sq_norm')
		m.assign(self._get_hyper('beta') * m + (1 - self._get_hyper('beta')) * neuron_norm(grad) ** 2)

		grad_normed = grad / tf.sqrt((m + self._get_hyper('epsilon')) / bias_correction)
		grad_normed = tf.where(tf.math.is_finite(grad_normed), grad_normed, 0)
		update = self._get_hyper('lr') * grad_normed
		update = update * tf.cond(tf.rank(var) > 1, lambda: 1., lambda: self._get_hyper('bias_scale'))
		var.assign_sub(update)
		var.assign(neuron_constrain(var))
		return m, var

	@tf.function
	def _resource_apply_sparse(self, grad, var, indices):
		raise NotImplementedError

	def get_config(self):
		base_config = super(NeRo, self).get_config()
		config = {'decay_var_list': self.decay_var_list,
				  'learning_rate': self._get_hyper('lr'),
				  'beta': self._get_hyper('beta'),
				  'epsilon': self._get_hyper('epsilon'),
				  'bias_scale': self._get_hyper('bias_scale')
				  }
		return dict(list(base_config.items()) + list(config.items()))
