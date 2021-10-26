import socket
import msgpack
import time
import os
import threading
import queue
import requests
import struct
from gauss_gym.utils import server_utils

AGG_HOST = os.environ.get('AGG_HOST', os.environ.get('MASTER_ADDR', 'node0'))


class StatsReporter:
  def __init__(
    self,
    global_rank: int,
    max_queue: int = 10000,
    flush_every: float = 0.1,
    host: str = AGG_HOST,
    port: int = server_utils._AGG_PORT_UDP,
    ack_timeout: float = 2.0,
  ):
    self.addr = (host, port)
    self.sock = None
    self.ack_timeout = ack_timeout
    self.q = queue.Queue(maxsize=max_queue)
    self.global_rank = global_rank
    self.flush_every = flush_every
    self._stop = threading.Event()
    self.t = threading.Thread(target=self._loop, daemon=True)
    self.t.start()

  def _connect(self):
    # Close old socket if needed
    if self.sock:
      try:
        self.sock.close()
      except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect(self.addr)
    # set short timeouts for send/ack
    s.settimeout(self.ack_timeout)
    self.sock = s

  def _send_frame(self, payload: bytes):
    if not self.sock:
      self._connect()
    frame = struct.pack('!I', len(payload)) + payload
    self.sock.sendall(frame)
    # optional 1-byte ack
    ack = self.sock.recv(1)
    if not ack:
      raise ConnectionError('No ACK from aggregator')

  def report(self, step: int, **metrics):
    pkt = msgpack.packb(
      {
        'global_rank': self.global_rank,
        'step': step,
        'ts': time.time(),
        'metrics': metrics,
      },
      use_bin_type=True,
    )
    try:
      self.q.put_nowait(pkt)
    except queue.Full:
      # drop to avoid backpressure on training
      pass

  def _loop(self):
    buf = []
    last = time.time()
    while not self._stop.is_set():
      try:
        pkt = self.q.get(timeout=self.flush_every)
        buf.append(pkt)
      except queue.Empty:
        pass

      now = time.time()
      if buf and (now - last >= self.flush_every or len(buf) >= 128):
        # Try to send the batch; on failure, reconnect once and retry
        try:
          for p in buf:
            self._send_frame(p)
          buf.clear()
          last = now
        except Exception:
          # reconnect + one retry for this batch
          try:
            self._connect()
            for p in buf:
              self._send_frame(p)
            buf.clear()
            last = now
          except Exception:
            # Drop the whole batch on persistent failure
            buf.clear()
            last = now

  def close(self):
    self._stop.set()
    self.t.join(timeout=1)
    if self.sock:
      try:
        self.sock.close()
      except Exception:
        pass


# usage in training:
# reporter = StatsReporter(global_rank=...)
# reporter.report(step, reward=..., success=float(...), fps=...)


def get_snapshot(keys, window_count=10, host=None, port=server_utils._AGG_HTTP_PORT):
  host = host or os.environ.get('AGG_HOST', os.environ.get('MASTER_ADDR', 'node0'))
  r = requests.post(
    f'http://{host}:{port}/snapshot',
    json={'keys': keys, 'window_count': window_count},
    timeout=5.0,
  )
  r.raise_for_status()
  return r.json()


def get_healthz(host=None, port=server_utils._AGG_HTTP_PORT):
  host = host or os.environ.get('AGG_HOST', os.environ.get('MASTER_ADDR', 'node0'))
  r = requests.get(f'http://{host}:{port}/healthz', timeout=1.0)
  r.raise_for_status()
  return r.json()['ok']
