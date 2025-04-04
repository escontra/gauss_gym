from typing import Optional
import evdev
from sshkeyboard import listen_keyboard
import threading
from dataclasses import dataclass
import time


@dataclass
class JoystickConfig:
    max_vx: float = 0.5
    max_vy: float = 0.5
    max_vyaw: float = 0.5
    control_threshold: float = 0.1
    # logitech
    custom_mode_button: evdev.ecodes = evdev.ecodes.BTN_C
    rl_gait_button: evdev.ecodes = evdev.ecodes.BTN_B
    x_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    y_axis: evdev.ecodes = evdev.ecodes.ABS_X
    yaw_axis: evdev.ecodes = evdev.ecodes.ABS_Z

    # beitong
    # custom_mode_button: evdev.ecodes = evdev.ecodes.BTN_B
    # rl_gait_button: evdev.ecodes = evdev.ecodes.BTN_A
    # x_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    # y_axis: evdev.ecodes = evdev.ecodes.ABS_X
    # yaw_axis: evdev.ecodes = evdev.ecodes.ABS_RX


class RemoteControlService:
    """Service for handling joystick remote control input without display dependencies."""

    def __init__(self, config: Optional[JoystickConfig] = None):
        """Initialize remote control service with optional configuration."""
        self.config = config or JoystickConfig()
        self._lock = threading.Lock()
        self._running = True
        try:
            self._init_joystick()
            self._start_joystick_thread()
        except Exception as e:
            print(f"{e}, downgrade to keyboard control")
            self._init_keyboard_control()
            self._start_keyboard_thread()

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

    def get_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") != None:
            return "Left axis for forward/backward/left/right, right axis for rotation left/right"
        return "Press 'w'/'s' to increase/decrease vx; Press 'a'/'d' to increase/decrease vy; Press 'q'/'e' to increase/decrease vyaw, press 'Space' to stop."

    def get_custom_mode_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") != None:
            return "Press button B to start custom mode."
        return "Press 'b' to start custom mode."

    def get_rl_gait_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") != None:
            return "Press button A to start rl Gait."
        return "Press 'r' to start rl Gait."

    def _init_keyboard_control(self):
        self.joystick = None
        self.joystick_runner = None
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False

    def _start_keyboard_thread(self):
        self.keyboard_runner = threading.Thread(target=listen_keyboard, args=(self._handle_keyboard_press,))
        self.keyboard_runner.daemon = True
        self.keyboard_runner.start()

    def _handle_keyboard_press(self, key):
        if key == "b":
            self.keyboard_start_custom_mode = True
        if key == "r":
            self.keyboard_start_rl_gait = True
        if key == "w":
            old_x = self.vx
            self.vx += 0.1
            self.vx = min(self.vx, self.config.max_vx)
            print(f"VX: {old_x:.1f} => {self.vx:.1f}")
        if key == "s":
            old_x = self.vx
            self.vx -= 0.1
            self.vx = max(self.vx, -self.config.max_vx)
            print(f"VX: {old_x:.1f} => {self.vx:.1f}")
        if key == "a":
            old_y = self.vy
            self.vy += 0.1
            self.vy = min(self.vy, self.config.max_vy)
            print(f"VY: {old_y:.1f} => {self.vy:.1f}")
        if key == "d":
            old_y = self.vy
            self.vy -= 0.1
            self.vy = max(self.vy, -self.config.max_vy)
            print(f"VY: {old_y:.1f} => {self.vy:.1f}")
        if key == "q":
            old_yaw = self.vyaw
            self.vyaw += 0.1
            self.vyaw = min(self.vyaw, self.config.max_vyaw)
            print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
        if key == "e":
            old_yaw = self.vyaw
            self.vyaw -= 0.1
            self.vyaw = max(self.vyaw, -self.config.max_vyaw)
            print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
        if key == "space":
            self.vx = 0
            self.vy = 0
            self.vyaw = 0
            print(f"FULL STOP")

    def _init_joystick(self) -> None:
        """Initialize and validate joystick connection using evdev."""
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            joystick = None

            for device in devices:
                caps = device.capabilities()
                # print(f"Device {device.name}:")
                # print(f"Capabilities: {device.capabilities(verbose=True)}")

                # Check for both absolute axes and keys
                if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                    abs_info = caps.get(evdev.ecodes.EV_ABS, [])
                    # Look for typical gamepad axes
                    axes = [code for (code, info) in abs_info]
                    if all(code in axes for code in [self.config.x_axis, self.config.y_axis, self.config.yaw_axis]):
                        absinfo = {}
                        for code, info in abs_info:
                            absinfo[code] = info
                        self.axis_ranges = {
                            self.config.x_axis: absinfo[self.config.x_axis],
                            self.config.y_axis: absinfo[self.config.y_axis],
                            self.config.yaw_axis: absinfo[self.config.yaw_axis],
                        }
                        print(f"Found suitable joystick: {device.name}")
                        joystick = device
                        break

            if not joystick:
                raise RuntimeError("No suitable joystick found")

            self.joystick = joystick
            print(f"Selected joystick: {joystick.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}")

    def _start_joystick_thread(self):
        """Start joystick polling thread."""
        self.joystick_runner = threading.Thread(target=self._run_joystick)
        self.joystick_runner.daemon = True
        self.joystick_runner.start()

    def start_custom_mode(self) -> bool:
        """Check if custom mode button is pressed."""
        if hasattr(self, "joystick") and getattr(self, "joystick") != None:
            return self.joystick.active_keys() == [self.config.custom_mode_button]
        return self.keyboard_start_custom_mode

    def start_rl_gait(self) -> bool:
        """Check if gait button is pressed."""
        if hasattr(self, "joystick") and getattr(self, "joystick") != None:
            return self.joystick.active_keys() == [self.config.rl_gait_button]
        return self.keyboard_start_rl_gait

    def _run_joystick(self):
        """Poll joystick events."""
        while self._running:
            try:
                for event in self.joystick.read_loop():
                    if event.type == evdev.ecodes.EV_ABS:
                        # Handle axis events
                        self._handle_axis(event.code, event.value)

            except BlockingIOError:
                # No events available
                time.sleep(0.01)
            except Exception:
                break

    def _handle_axis(self, code: int, value: int):
        try:
            """Handle axis events."""
            if code == self.config.x_axis:
                self.vx = self._scale(value, self.config.max_vx, self.config.control_threshold, code)
                # print("value x:", self.vx)
            elif code == self.config.y_axis:
                self.vy = self._scale(value, self.config.max_vy, self.config.control_threshold, code)
                # print("value y:", self.vy)
            elif code == self.config.yaw_axis:
                self.vyaw = self._scale(value, self.config.max_vyaw, self.config.control_threshold, code)
                # print("value yaw:", self.vyaw)
        except Exception:
            raise

    def _scale(self, value: float, max: float, threshold: float, axis_code: int) -> float:
        """Scale joystick input to velocity command using actual axis ranges."""
        absinfo = self.axis_ranges[axis_code]
        min_in = absinfo.min
        max_in = absinfo.max

        mapped_value = ((value - min_in) / (max_in - min_in) * 2 - 1) * max
        # print(f"Axis {axis_code}, value {value} min_in {min_in}, max_in {max_in}: {value} => {mapped_value}")

        if abs(mapped_value) < threshold:
            return 0.0
        return -mapped_value

    def get_vx_cmd(self) -> float:
        """Get forward velocity command."""
        with self._lock:
            return self.vx

    def get_vy_cmd(self) -> float:
        """Get lateral velocity command."""
        with self._lock:
            return self.vy

    def get_vyaw_cmd(self) -> float:
        """Get yaw velocity command."""
        with self._lock:
            return self.vyaw

    def close(self):
        """Clean up resources."""
        self._running = False
        if hasattr(self, "joystick") and getattr(self, "joystick") != None:
            self.joystick.close()
        if hasattr(self, "joystick_runner") and getattr(self, "joystick_runner") != None:
            self.joystick_runner.join(timeout=1.0)
        if hasattr(self, "keyboard_runner") and getattr(self, "keyboard_runner") != None:
            self.keyboard_runner.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
