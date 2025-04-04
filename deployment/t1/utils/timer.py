from dataclasses import dataclass
from typing import Optional


@dataclass
class TimerConfig:
    time_step: float = 0.002


class Timer:
    def __init__(self, config: Optional[TimerConfig] = None):
        self.config = config or TimerConfig()
        self.counter = 0

    def tick_timer_if_sim(self):
        self.counter += 1

    def get_time(self):
        return self.counter * self.config.time_step
