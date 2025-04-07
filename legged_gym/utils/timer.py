import contextlib
import threading
import time
from collections import defaultdict

import numpy as np


class Timer:

  def __init__(self, enabled=True):
    self.enabled = enabled
    self.stack = defaultdict(list)
    self.paths = set()
    self.mins = defaultdict(lambda: np.inf)
    self.maxs = defaultdict(lambda: 0)
    self.sums = defaultdict(lambda: 0)
    self.counts = defaultdict(lambda: 0)
    self.start = time.perf_counter_ns()
    self.writing = False
    self.extensions = []

  @contextlib.contextmanager
  def section(self, name):
    if not self.enabled:
      yield
      return
    stack = self.stack[threading.get_ident()]
    if name in stack:
      raise RuntimeError(
          f"Tried to recursively enter timer section {name} " +
          f"from {'/'.join(stack)}.")
    stack.append(name)
    path = '/'.join(stack)
    start = time.perf_counter_ns()
    try:
      if self.extensions:
        with contextlib.ExitStack() as es:
          [es.enter_context(ext(path)) for ext in self.extensions]
          yield
      else:
        yield
    finally:
      dur = time.perf_counter_ns() - start
      stack.pop()
      if not self.writing:
        self.paths.add(path)
        self.sums[path] += dur
        self.mins[path] = min(self.mins[path], dur)
        self.maxs[path] = max(self.maxs[path], dur)
        self.counts[path] += 1

  def wrap(self, name, obj, methods):
    for method in methods:
      decorator = self.section(f'{name}.{method}')
      setattr(obj, method, decorator(getattr(obj, method)))

  def stats(self, reset=True, log=False):
    if not self.enabled:
      return {}
    self.writing = True
    time.sleep(0.001)
    now = time.perf_counter_ns()
    passed = now - self.start
    self.start = now
    mets = {}
    div = lambda x, y: x and x / y
    keys = list(self.paths)
    for key in keys:
      mets.update({
          f'{key}/sum': self.sums[key] / 1e9,
          f'{key}/min': self.mins[key] / 1e9,
          f'{key}/max': self.maxs[key] / 1e9,
          f'{key}/mean': div(self.sums[key], self.counts[key]) / 1e9,
          f'{key}/frac': self.sums[key] / passed,
          f'{key}/count': self.counts[key],
      })
    self.writing = False
    lines = []
    longest = max(len(key) for key in keys)
    descending = sorted(keys, key=lambda k: -mets[f'{k}/frac'])
    for key in descending:
      count = mets[f'{key}/count']
      if not count:
        continue
      perc = '{:3.0f}'.format(100 * mets[f'{key}/frac'])
      min_ = '{:.1f}'.format(mets[f'{key}/min'])
      mean = '{:.1f}'.format(mets[f'{key}/mean'])
      max_ = '{:.1f}'.format(mets[f'{key}/max'])
      detail = f'min={min_}s, mean={mean}s, max={max_}s, n={count}'
      space = ' ' * (longest - len(key))
      lines.append(f'- {perc}% {key} {space} ({detail})')
    mets['summary'] = '\n'.join(lines)
    if log:
      print('Timer:', mets['summary'], sep='\n')
    reset and self.reset()
    return mets

  def reset(self):
    if not self.enabled:
      return
    self.writing = True
    time.sleep(0.001)
    self.sums.clear()
    self.mins.clear()
    self.maxs.clear()
    self.counts.clear()
    self.start = time.perf_counter_ns()
    self.writing = False


global_timer = Timer()
section = global_timer.section
wrap = global_timer.wrap
stats = global_timer.stats
reset = global_timer.reset
extensions = global_timer.extensions