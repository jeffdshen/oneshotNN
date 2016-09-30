# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright 2016 jeffdshen.

"""MNIST tutorial.
This uses Pretty Tensor to define and train either a 2 layer model or a
convolutional model in the style of LeNet 5.
See: http://yann.lecun.com/exdb/lenet/
"""
import tensorflow as tf
import prettytensor as pt
from prettytensor.tutorial import data_utils

from common import Model


BATCH_SIZE = 50
EPOCH_SIZE = 60000 // BATCH_SIZE
TEST_SIZE = 10000 // BATCH_SIZE

tf.app.flags.DEFINE_string('save_path', "data/mnist/model.chkpt", 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('model', 'conv', 'Choose one of the models, either full or conv')
tf.app.flags.DEFINE_string('mode', 'train', 'Either train or infer')
FLAGS = tf.app.flags.FLAGS

class MultilayerFullyConnected(Model):
  def _make(self, input, labels):
    """
    Creates a multi layer network of fully_connected layers. Each layer is 100 neurons.
    Please change this to experiment
    with architectures.
    Args:
      input: The input images.
      labels: The labels as dense one-hot vectors.
    Returns:
      A softmax result.
    """
    # Pretty Tensor is a thin wrapper on Tensors.
    # Change this method to experiment with other architectures
    with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
      return (pt.wrap(input).flatten().fully_connected(100).fully_connected(100)
              .softmax_classifier(10, labels))

class Lenet5(Model):
  def _make(self, input, labels):
    """
    Creates a multi layer convolutional network. The architecture is similar to that defined in LeNet 5.
    Please change this to experiment with architectures.
    Args:
      input: The input images.
      labels: The labels as dense one-hot vectors.
    Returns:
      A softmax result.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
      return (pt.wrap(input).conv2d(5, 20).max_pool(2, 2).conv2d(5, 50).max_pool(2, 2)
              .flatten().fully_connected(500).softmax_classifier(10, labels))

def conv(images, labels):
  """
  Creates a multi layer convolutional network. The architecture is similar to that defined in LeNet 5.
  Please change this to experiment with architectures.
  Args:
    images: The input images.
    labels: The labels as dense one-hot vectors.
  Returns:
    A softmax result.
  """
  images = pt.wrap(images)
  with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
    model = images.conv2d(5, 20).max_pool(2, 2)
    model = model.conv2d(5, 50).max_pool(2, 2)
    return model.flatten().fully_connected(500).dropout().softmax_classifier(10, labels)


def main(_=None):
  # Since we are feeding our data as numpy arrays, we need to create
  # placeholders in the graph.
  # These must then be fed using the feed dict.
  image_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
  labels_placeholder = tf.placeholder(tf.float32, [None, 10])

  # Create our model.  The result of softmax_classifier is a namedtuple
  # that has members result.loss and result.softmax.
  if FLAGS.model == 'full':
    model = MultilayerFullyConnected(image_placeholder, labels_placeholder)
  elif FLAGS.model == 'conv':
    model = Lenet5(image_placeholder, labels_placeholder)
  else:
    raise ValueError('model must be full or conv: %s' % FLAGS.model)

  # Grab the data as numpy arrays.
  train_images, train_labels = data_utils.mnist(training=True)
  test_images, test_labels = data_utils.mnist(training=False)

  # We can set a save_path in the runner to automatically checkpoint every so
  # often.  Otherwise at the end of the session, the model will be lost.
  runner = pt.train.Runner(save_path=FLAGS.save_path)
  with tf.Session():
    for epoch in range(10):
      # Shuffle the training data.
      train_images, train_labels = data_utils.permute_data(
          (train_images, train_labels))

      runner.train_model(
          model.phase[pt.Phase.train],
          model.loss,
          EPOCH_SIZE,
          feed_vars=(image_placeholder, labels_placeholder),
          feed_data=pt.train.feed_numpy(BATCH_SIZE, train_images, train_labels),
          print_every=100)
      classification_accuracy = runner.evaluate_model(
          model.phase[pt.Phase.test],
          TEST_SIZE,
          feed_vars=(image_placeholder, labels_placeholder),
          feed_data=pt.train.feed_numpy(BATCH_SIZE, test_images, test_labels))
      print('Accuracy after %d epoch %g%%' % (
          epoch + 1, classification_accuracy * 100))


if __name__ == '__main__':
  tf.app.run()