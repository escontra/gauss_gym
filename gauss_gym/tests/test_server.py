import time
import multiprocessing as mp
from gauss_gym.utils import server_utils


def main():
  proc = server_utils.run_server_in_process(http_port=8088, udp_port=27183)
  try:
    while True:
      print('Server is running.')
      time.sleep(1)
  finally:
    if proc.is_alive():
      proc.terminate()


if __name__ == '__main__':
  # optional but nice for consistency
  mp.set_start_method('spawn', force=True)
  main()
