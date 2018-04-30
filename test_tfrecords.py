# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 2018

@author: MRegina, CTO at OrthoPred Ltd

# TFRecords reader and interpreter code for Numpy array data (specifically for 3D arrays with an additional channel dimension).

# The source code is partially based on TensorFlow Authors' imagenet_main.py:
# https://github.com/tensorflow/models/blob/r1.4.0/official/resnet/imagenet_main.py

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import tensorflow as tf
import os

_DATA_DIR= '.\\output\\'
batch_size=4

# train data: is_training=True is_testing=False
# validation data: is_training=False is_testing=False
# test data: is_training=False is_testing=True
is_training=True
is_testing=False


def record_parser(value):
  """Parse a TFRecord file from `value`."""
 
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/depth': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/channels': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1) 
  }

  parsed = tf.parse_single_example(value, keys_to_features)
  print("parsed one example")
  
  #decode label
  label = tf.cast(tf.reshape(parsed['image/label'], shape=[-1]),dtype=tf.int32)
  
  #decode the array shape
  width = tf.cast(tf.reshape(parsed['image/width'], shape=[]),dtype=tf.int32)
  height = tf.cast(tf.reshape(parsed['image/height'], shape=[]),dtype=tf.int32)
  depth = tf.cast(tf.reshape(parsed['image/depth'], shape=[]),dtype=tf.int32)
  channels = tf.cast(tf.reshape(parsed['image/channels'], shape=[]),dtype=tf.int32)
  
  #decode 3D array data (important to check data type and byte order)
  image = tf.reshape(tf.decode_raw(parsed['image/encoded'],out_type=tf.float32,little_endian=True),shape=[height,width,depth,channels])
    
  print("image decoded")
  
  return image, label

def filenames(is_training, is_testing, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-00002' % i)
        for i in range(2)]
  if is_testing:
      return [
        os.path.join(data_dir, 'test-%05d-of-00001' % i)
        for i in range(1)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00001' % i)
        for i in range(1)]

def input_fn(data_dir, batch_size=1, num_epochs=1):
  """Input function which provides batches."""
  dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training,is_testing, data_dir))

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser(value), num_parallel_calls=5)
  dataset = dataset.prefetch(batch_size)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  print("iterator created")
  return images, labels


# start a session to plot some data slices
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    #return tensors of image and label data
    images, labels = input_fn(_DATA_DIR,batch_size)
    print("returned")
    
    #evaluate image tensors to get actual numbers to show
    imgs=images.eval(session=sess)
    
    #plot some slices from every array in the batch
    import matplotlib.pyplot as plt
    for i in range(imgs.shape[0]):
        plt.imshow(imgs[i,:,:,20,0])
        plt.show()
    

    
    
    