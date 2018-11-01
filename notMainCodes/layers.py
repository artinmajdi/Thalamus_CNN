# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

# for d in ['/device:GPU:2', '/device:GPU:3']:
def weight_variable(shape, stddev=0.1):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    initial = tf.truncated_normal(shape, stddev=stddev)
    A = tf.Variable(initial)
    return A

def weight_variable_devonc(shape, stddev=0.1):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    A = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    return A

def bias_variable(shape):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    initial = tf.constant(0.1, shape=shape)
    A = tf.Variable(initial)
    return A

def conv2d(x, W,keep_prob_):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    A = tf.nn.dropout(conv_2d, keep_prob_)
    return A

def deconv2d(x, W,stride):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    A = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')
    return A

def max_pool(x,n):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    A = tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')
    return A

def crop_and_concat(x1,x2):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    A = tf.concat([x1_crop, x2], 3)
    return A

def pixel_wise_softmax(output_map):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    A = tf.div(exponential_map,evidence, name="pixel_wise_softmax")
    return A

def pixel_wise_softmax_2(output_map):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    A = tf.div(exponential_map,tensor_sum_exp)
    return A



def cross_entropy(y_,output_map):
    # for d in ['/device:GPU:2']:
    #     with tf.device(d):
    A = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
    return A
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))

