import functools
import math
import operator

import numpy as np


RULES = (
    ((lambda n, x: isinstance(x, str)), 'last'),
    ((lambda n, x: x.ndim == 0), 'mean'),
    ((lambda n, x: True), 'concat'),
)


class Agg:

  def __init__(self, rules=None, maxlen=1e6):
    self.rules = RULES if rules is None else rules
    self.reducers = {}
    self.maxlen = int(maxlen)

  def add(self, key_or_dict, value=None, agg=None, prefix=None):
    if value is None:
      for key, value in key_or_dict.items():
        self._add(key, value, agg, prefix)
    else:
      self._add(key_or_dict, value, agg, prefix)

  def result(self, reset=True, prefix=None):
    metrics = {}
    for key, reducer in self.reducers.items():
      metrics[key] = reducer.current()
    if prefix:
      metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
    reset and self.reset()
    return metrics

  def reset(self):
    self.reducers.clear()

  def _add(self, key, value, agg, prefix):
    if prefix:
      key = f'{prefix}/{key}'
    if not isinstance(value, str):
      value = np.asarray(value)
    if key in self.reducers:
      self.reducers[key].update(value)
    else:
      self.reducers[key] = self._initial(key, value, agg)

  def _initial(self, key, value, agg):
    if not agg:
      for applies, agg in self.rules:
        if applies(key, value):
          break
      else:
        raise ValueError(
            "No rule applied to infer aggregation strategy for metric " +
            f"with key '{key}' and value type '{type(value)}'. Please " +
            "specify add(..., agg=...) explicitly.")
    cls = {
        'mean': Mean,
        'avg': Mean,
        'sum': Sum,
        'min': Min,
        'max': Max,
        'stack': functools.partial(Stack, maxlen=self.maxlen),
        'concat': functools.partial(Concat, maxlen=self.maxlen),
        'last': Last,
    }[agg]
    return cls(value)


class Reducer:

  def __init__(self, scalar_fn, array_fn, initial):
    self.is_scalar = isinstance(initial, (int, float))
    self.fn = scalar_fn if self.is_scalar else array_fn
    self.interm = self._input(initial)
    self.count = 1

  def update(self, value):
    value = self._input(value)
    if self._isnan(value):
      return
    if self._isnan(self.interm):
      self.interm = value
      return
    self.interm = self.fn(self.interm, value)
    self.count += 1

  def current(self):
    return np.array(self.interm)

  def _input(self, value):
    if self.is_scalar:
      return value
    else:
      return np.asarray(value, np.float64)

  def _isnan(self, value):
    if self.is_scalar:
      return math.isnan(value)
    else:
      return np.isnan(value).any()


class Mean:

  def __init__(self, initial):
    self.reducer = Sum(initial)

  def update(self, value):
    self.reducer.update(value)

  def current(self):
    return self.reducer.current() / self.reducer.count


class Stack:

  def __init__(self, initial, maxlen=1e5):
    self.stack = [initial]
    self.maxlen = int(maxlen)

  def update(self, value):
    if len(self.stack) < self.maxlen:
      self.stack.append(value)

  def current(self):
    return np.stack(self.stack)


class Concat:

  def __init__(self, initial, maxlen=1e5):
    self.values = [initial]
    self.len = len(self.values[-1])
    self.maxlen = int(maxlen)

  def update(self, value):
    if self.len < self.maxlen:
      self.values.append(value[:self.maxlen - self.len])
      self.len += len(self.values[-1])

  def current(self):
    return np.concatenate(self.values)


class Last:

  def __init__(self, initial):
    self.value = initial

  def update(self, value):
    self.value = value

  def current(self):
    return self.value


Sum = functools.partial(
    Reducer, operator.add, lambda x, y: np.add(x, y, out=x, dtype=np.float64))

Min = functools.partial(
    Reducer, min, lambda x, y: np.minimum(x, y, out=x, dtype=np.float64))

Max = functools.partial(
    Reducer, max, lambda x, y: np.maximum(x, y, out=x, dtype=np.float64))