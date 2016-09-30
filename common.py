from abc import abstractmethod
from abc import ABCMeta
import prettytensor as pt
import tensorflow as tf


class Model(metaclass=ABCMeta):
  def __init__(self, inputs, labels):
    self.model = self._make(inputs, labels)
    self.softmax = self.model.softmax
    self.loss = self.model.loss
    self.phase = self._phases(self.model, input, labels)

  def _phases(self, model, input, labels):
    return {
      pt.Phase.test: model.softmax.evaluate_classifier(labels, phase=pt.Phase.test),
      pt.Phase.infer: model.softmax,
      pt.Phase.train: pt.apply_optimizer(self._optimizer(), losses=[model.loss])
    }

  def _optimizer(self):
    return tf.train.GradientDescentOptimizer(0.01)

  @abstractmethod
  def _make(self, input, labels):
    return
