from abc import abstractmethod
from abc import ABCMeta
import prettytensor as pt
import tensorflow as tf


class Model(metaclass=ABCMeta):
  def __init__(self, inputs, labels):
    self.inputs = inputs
    self.labels = labels
    self.model = self._make(inputs, labels)
    self.softmax = self.model.softmax
    self.loss = self.model.loss
    self._phase = self._phases(self.model)

  def phase(self, p):
    if p in self._phase:
      return self._phase[p]

    if p == pt.Phase.test:
      self._phase[p] = self.model.softmax.evaluate_classifier(self.labels, phase=pt.Phase.test)
    elif p == pt.Phase.train:
      self._phase[p] = pt.apply_optimizer(self._optimizer(), losses=[self.model.loss])

    return self._phase[p]

  def _phases(self, model):
    return {
      pt.Phase.infer: model.softmax,
    }

  def _optimizer(self):
    return tf.train.GradientDescentOptimizer(0.01)

  @abstractmethod
  def _make(self, input, labels):
    return
