# runner.py
from typing import Dict, Deque, Tuple, List
import re
import os
import time
import socket
import atexit
import multiprocessing as mp
import struct
import contextlib

import asyncio
import uvicorn
from fastapi import FastAPI
from tdigest import TDigest
import msgpack
import time as _time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pydantic import BaseModel

from gauss_gym import utils

_AGG_PORT_UDP = int(os.environ.get('AGG_PORT_UDP', '27183'))
_AGG_HTTP_PORT = int(os.environ.get('AGG_HTTP_PORT', '8088'))

CACHE_SEC = 0.2


def is_port_open(host: str, port: int) -> bool:
  try:
    with socket.create_connection((host, port), timeout=0.5):
      return True
  except OSError:
    return False


def wait_for_server(host: str, port: int = _AGG_HTTP_PORT, timeout_s: float = 30.0):
  t0 = time.time()
  while time.time() - t0 < timeout_s:
    if is_port_open(host, port):
      return
    utils.print(f'Waiting for server {host}:{port}...', color='yellow')
    time.sleep(1.0)
  raise TimeoutError(f'stats server {host}:{port} not reachable after {timeout_s}s')


def match_all(keys, pattern):
  regex = re.compile(pattern)
  return [k for k in keys if regex.fullmatch(k)]


def _q(td, q: float):
  # q in [0, 1]
  if hasattr(td, 'quantile'):
    return td.quantile(q)
  if hasattr(td, 'percentile'):
    return td.percentile(q * 100.0)
  if hasattr(td, 'estimate_quantile'):
    return td.estimate_quantile(q)
  return None


def run_server(
  udp_port: int = _AGG_PORT_UDP,
  http_port: int = _AGG_HTTP_PORT,
  recent_window_sec: float = float('inf'),
  recent_window_count: int = 5000,
):
  @dataclass
  class OnlineStat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    td: TDigest = field(default_factory=TDigest)

    def update(self, x: float):
      self.n += 1
      d = x - self.mean
      self.mean += d / self.n
      self.m2 += d * (x - self.mean)
      self.td.update(x)

    @property
    def var(self):
      return self.m2 / max(1, self.n - 1)

  @dataclass
  class MetricState:
    online: OnlineStat = field(default_factory=OnlineStat)
    recent: Deque[Tuple[float, float]] = field(default_factory=deque)

  class Aggregator:
    def __init__(self, recent_window_sec=float('inf'), recent_window_count=5000):
      self.metrics: Dict[str, MetricState] = defaultdict(MetricState)
      self.recent_window_sec = recent_window_sec
      self.recent_window_count = recent_window_count

      # Locks
      self.metrics_lock = asyncio.Lock()
      self.cache_lock = asyncio.Lock()

      # Cache: key -> (expiry_monotonic, version, value)
      self.cache: Dict[Tuple, Tuple[float, int, Dict]] = {}
      self.update_version = 0  # bump on every ingest

    async def ingest(self, pkt: bytes):
      # Parse outside locks
      msg = msgpack.unpackb(pkt, raw=False)
      ts = msg.get('ts', _time.time())
      metrics = msg['metrics']

      async with self.metrics_lock:
        for k, v in metrics.items():
          st = self.metrics[k]
          seq = (
            v if (hasattr(v, '__iter__') and not isinstance(v, (str, bytes))) else [v]
          )
          for item in seq:
            x = float(item)
            st.online.update(x)
            st.recent.append((ts, x))
            while st.recent and (
              len(st.recent) > self.recent_window_count
              or ts - st.recent[0][0] > self.recent_window_sec
            ):
              st.recent.popleft()
        self.update_version += 1  # invalidate cache by version

    async def _compute_snapshot_once(
      self, keys: List[str], window_count: int
    ) -> Dict[str, Dict]:
      # FAST copy under lock
      async with self.metrics_lock:
        metric_keys = list(self.metrics.keys())
        copies = {}
        for patt in keys:
          for k in match_all(metric_keys, patt):
            st = self.metrics.get(k)
            if not st:
              continue
            online = st.online
            copies[k] = (
              (online.n, online.mean, online.m2, online.td),
              list(st.recent),
            )  # deque -> list

      # SLOW compute outside lock
      out = {}
      for k, ((n, mean, m2, td), rec) in copies.items():
        rec_sorted = sorted(rec, key=lambda x: x[0])
        vals = [v for (_, v) in rec_sorted[-window_count:]]
        if vals:
          srt = sorted(vals)
          mean_w = sum(vals) / len(vals)
          p50_w = srt[len(srt) // 2]
          p95_w = srt[int(0.95 * (len(srt) - 1))]
        else:
          mean_w = p50_w = p95_w = None
        out[k] = {
          'count_total': n,
          'mean_total': mean,
          'std_total': (m2 / max(1, n - 1)) ** 0.5 if n > 1 else 0.0,
          'p50_total': _q(td, 0.5) if n else None,
          'p95_total': _q(td, 0.95) if n else None,
          'count_window': len(vals),
          'mean_window': mean_w,
          'p50_window': p50_w,
          'p95_window': p95_w,
        }
      return out

    async def snapshot_cached(
      self, keys: List[str], window_count: int
    ) -> Dict[str, Dict]:
      key = (tuple(keys), int(window_count))
      now = time.monotonic()
      ver = self.update_version  # read once

      # Fast path (no lock)
      entry = self.cache.get(key)
      if entry:
        exp, cached_ver, val = entry
        if now < exp and cached_ver == ver:
          return val

      # Compute (multiple callers may compute in parallel—OK)
      val = await self._compute_snapshot_once(keys, window_count)

      # Store atomically
      async with self.cache_lock:
        # if someone else stored newer in the meantime, keep the newer
        cur = self.cache.get(key)
        if not cur or cur[1] != self.update_version or cur[0] <= time.monotonic():
          self.cache[key] = (time.monotonic() + CACHE_SEC, self.update_version, val)
      return val

  # Aggregation logic. Data ingested from UDP.
  agg = Aggregator()

  async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
      while True:
        # 4-byte big-endian length prefix
        hdr = await reader.readexactly(4)
        (n,) = struct.unpack('!I', hdr)
        data = await reader.readexactly(n)
        await agg.ingest(data)
        # optional 1-byte ack; cheap positive feedback to client
        writer.write(b'\x01')
        await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
      pass
    finally:
      writer.close()
      with contextlib.suppress(Exception):
        await writer.wait_closed()

  async def tcp_server(stop_event: asyncio.Event, host='0.0.0.0', port=udp_port):
    server = await asyncio.start_server(handle_client, host, port)
    async with server:
      await stop_event.wait()
      server.close()
      await server.wait_closed()

  app = FastAPI()

  class SnapshotReq(BaseModel):
    keys: List[str]
    window_count: int = 10

  @app.post('/snapshot')
  async def snapshot(req: SnapshotReq):
    return await agg.snapshot_cached(req.keys, req.window_count)

  @app.get('/healthz')
  async def healthz():
    return {'ok': True}

  async def _main():
    stop = asyncio.Event()
    tcp_task = asyncio.create_task(tcp_server(stop))

    # (optionally: move to lifespan later; on_event works but is deprecated)
    @app.on_event('shutdown')
    async def _on_shutdown():
      stop.set()
      await asyncio.sleep(0)  # let last ingests run

    config = uvicorn.Config(app, host='0.0.0.0', port=http_port, log_level='warning')
    server = uvicorn.Server(config)
    try:
      await server.serve()
    finally:
      stop.set()
      with contextlib.suppress(asyncio.CancelledError):
        await tcp_task

  asyncio.run(_main())


def run_server_in_process(http_port=_AGG_HTTP_PORT, udp_port=_AGG_PORT_UDP):
  ctx = mp.get_context('spawn')
  p = ctx.Process(
    target=run_server,
    kwargs=dict(http_port=http_port, udp_port=udp_port),
    daemon=False,
  )
  p.start()

  def cleanup():
    utils.print('Shutting down server.', color='blue')
    if p.is_alive():
      p.terminate()  # SIGTERM → graceful path in child
      p.join(5.0)  # give it a few seconds to exit
    if p.is_alive():
      utils.print('Server still alive after SIGTERM. Forcing kill...', color='red')
      p.kill()  # SIGKILL as a last resort
      p.join(5.0)
    utils.print('Server process terminated.', color='green')

  atexit.register(cleanup)
  return p
