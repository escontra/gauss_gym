import time
import concurrent.futures as cf

from gauss_gym.utils import server_utils, client_utils
from gauss_gym.utils.server_utils import _AGG_HTTP_PORT

host = '127.0.0.1'

# 1) Wait for the HTTP server to be ready
server_utils.wait_for_server(host)

# 2) Spin up multiple reporters
NUM_REPORTERS = 6
reporters = [
  client_utils.StatsReporter(
    global_rank=i,
    host=host,
  )
  for i in range(NUM_REPORTERS)
]


# 3) Fire multiple report() calls concurrently across reporters
def send_reports(rep, rep_id):
  # a small burst per reporter
  rep.report(step=0, completion=[])
  rep.report(step=0, completion=[False, True, True])
  rep.report(step=0, reward=1.0, success=1.0, fps=10.0)
  rep.report(
    step=0,
    reward=1.0,
    success=0.0,
    fps=10.0,
    **{f'completion/hf://escontra/gauss_gym_data/bww_stairs/{rep_id:04d}.npz': 1.0},
  )
  rep.report(
    step=0,
    reward=1.0,
    success=0.0,
    fps=10.0,
    **{f'completion/hf://escontra/gauss_gym_data/bww_stairs/{rep_id:04d}.npz': 1.0},
  )
  rep.report(step=0, reward=1.0, success=1.0, fps=10.0)
  rep.report(
    step=0,
    reward=1.0,
    success=1.0,
    fps=10.0,
    **{f'completion/hf://escontra/gauss_gym_data/long_stairs/{rep_id:04d}.npz': 1.0},
  )


with cf.ThreadPoolExecutor(max_workers=NUM_REPORTERS) as pool:
  futures = [pool.submit(send_reports, reporters[i], i) for i in range(NUM_REPORTERS)]
  for fut in futures:
    fut.result()

# 4) Give background thread + server a moment to flush/ingest
time.sleep(1.0)

print(f'Healthz: {client_utils.get_healthz(host, _AGG_HTTP_PORT)}')

# 5) Fire multiple get_snapshot() calls concurrently
SNAPSHOT_CONCURRENCY = 12


def do_snapshot():
  # randomize window_count slightly to mix cached/uncached
  # wc = random.choice([5, 10, 15])
  return client_utils.get_snapshot(
    keys=[
      '^completion$',
      '^completion/.*$',
    ],  # both aggregate key and per-item completions
    window_count=5,
    host=host,
  )


with cf.ThreadPoolExecutor(max_workers=SNAPSHOT_CONCURRENCY) as pool:
  snaps = list(pool.map(lambda _: do_snapshot(), range(SNAPSHOT_CONCURRENCY)))

# 6) Show a couple results (they should all be similar)
for idx, s in enumerate(snaps[:3]):
  print(f'snapshot[{idx}]: {s}')

# 7) Clean shutdown of reporters
for r in reporters:
  r.close()
