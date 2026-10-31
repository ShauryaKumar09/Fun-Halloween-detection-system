#!/usr/bin/env python3
"""Halloween jump scare.

Watches a camera feed and plays a random scary sound when a person stays near
the candy for a moment. Settings come from (lowest to highest priority):
built-in defaults -> settings.json -> CAMERA_URL env var -> command-line flags.

Run `python halloween_scare.py --help` for options.
"""

import argparse
import json
import logging
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "settings.json"
SOUND_EXTENSIONS = {".wav", ".ogg", ".mp3", ".flac", ".aiff", ".aif"}

DEFAULTS = {
    # Camera: RTSP URL, video file path, or webcam index like "0".
    "source": None,
    "model": "models/efficientdet_lite0.tflite",
    "sounds_dir": "sounds",
    "volume": 1.0,                 # 0.0 - 1.0, applied on top of the system volume
    "dwell_seconds": 1.5,          # how long someone must be present before the scare
    "grace_seconds": 0.5,          # missed detections shorter than this don't reset the timer
    "cooldown_seconds": 25,        # minimum gap between scares
    "min_score": 0.35,             # detector confidence for "person"
    # Which people count:
    #   "size" - box is at least min_height_frac of the frame height (close to camera).
    #            Works with camera auto-tracking.
    #   "zone" - feet (bottom-center of box) inside `zone`. Needs auto-tracking OFF.
    #   "any"  - any person anywhere in frame.
    "trigger_mode": "size",
    "min_height_frac": 0.35,
    "zone": None,                  # [x1, y1, x2, y2], normalized 0-1; set with --set-zone
    "detect_width": 640,           # frames are shrunk to this width before detection
    "keepalive_seconds": 240,      # quiet tone so Bluetooth speakers don't auto-sleep; 0 = off
    "headless": False,             # no preview window (saves CPU)
    "snapshot_dir": None,          # save frames at +0s/+1s/+2s after each scare; None = off
    "rtsp_tcp": False,             # force RTSP over TCP (more reliable on bad Wi-Fi, a bit more lag)
}

log = logging.getLogger("scare")


# --------------------------------------------------------------------------- trigger logic

class ScareTrigger:
    """Decides when to fire. Pure logic, no I/O, so it is unit-testable."""

    def __init__(self, dwell_seconds, grace_seconds, cooldown_seconds):
        self.dwell = dwell_seconds
        self.grace = grace_seconds
        self.cooldown = cooldown_seconds
        self.present_since = None
        self.last_seen = None
        self.last_scare = -math.inf

    def update(self, now, present):
        """Feed one observation. Returns True when a scare should play."""
        if present:
            if self.present_since is None:
                self.present_since = now
            self.last_seen = now
        elif self.present_since is not None and now - self.last_seen > self.grace:
            self.present_since = None

        if (
            present
            and self.present_since is not None
            and now - self.present_since >= self.dwell
            and now - self.last_scare >= self.cooldown
        ):
            self.last_scare = now
            self.present_since = None  # they have to linger again after the cooldown
            return True
        return False

    def dwell_progress(self, now):
        if self.present_since is None:
            return 0.0
        return now - self.present_since

    def cooldown_left(self, now):
        return max(0.0, self.cooldown - (now - self.last_scare))


def person_qualifies(box, frame_w, frame_h, mode, min_height_frac, zone):
    """box = (x, y, w, h) in pixels."""
    x, y, w, h = box
    if mode == "any":
        return True
    if mode == "size":
        return h / frame_h >= min_height_frac
    if mode == "zone":
        if not zone:
            return False
        fx = (x + w / 2) / frame_w
        fy = min(y + h, frame_h) / frame_h
        x1, y1, x2, y2 = zone
        return x1 <= fx <= x2 and y1 <= fy <= y2
    raise ValueError(f"unknown trigger_mode {mode!r}")


# --------------------------------------------------------------------------- camera

def _parse_source(source):
    s = str(source)
    return int(s) if s.isdigit() else s


class LatestFrameReader:
    """Reads frames on a background thread and keeps only the newest one.

    Network cameras buffer frames; if the main loop is slower than the camera,
    reading inline makes you fall further and further behind. Dropping old
    frames keeps the scare in sync with reality. Reconnects automatically.
    """

    def __init__(self, source, rtsp_tcp=False):
        self.source = _parse_source(source)
        self.is_network = isinstance(self.source, str) and "://" in self.source
        self.is_file = isinstance(self.source, str) and not self.is_network
        if rtsp_tcp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self.last_frame_time = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self):
        if self.is_network:
            try:  # timeouts so a dead camera doesn't hang the reader forever (OpenCV >= 4.6)
                params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000]
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
            except (cv2.error, TypeError, AttributeError):
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not self._stopped.is_set():
            cap = self._open()
            if not cap.isOpened():
                log.warning("Could not open camera %s; retrying in 3s", _redact(self.source))
                cap.release()
                self._stopped.wait(3)
                continue
            log.info("Camera connected (%dx%d)",
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            file_delay = 0.0
            if self.is_file:
                fps = cap.get(cv2.CAP_PROP_FPS) or 15
                file_delay = 1.0 / fps
            failures = 0
            while not self._stopped.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.is_file:
                        break  # end of file: reopen and loop (handy for testing)
                    failures += 1
                    if failures > 20:
                        log.warning("Camera stream dropped; reconnecting")
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self.last_frame_time = time.monotonic()
                if file_delay:
                    time.sleep(file_delay)
            cap.release()
            if not self.is_file:
                self._stopped.wait(2)

    def latest(self):
        with self._lock:
            return self._frame_id, self._frame

    def wait_for_frame(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            fid, frame = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.1)
        return None

    def stop(self):
        self._stopped.set()


def _redact(source):
    """Hide the password in rtsp://user:pass@host URLs before logging."""
    s = str(source)
    if "://" in s and "@" in s:
        scheme, rest = s.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return s


# --------------------------------------------------------------------------- detection

class PersonDetector:
    def __init__(self, model_path, min_score, detect_width):
        options = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=min_score,
            max_results=10,
            category_allowlist=["person"],
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.detect_width = detect_width

    def detect(self, frame):
        """Returns (small_frame, [(x, y, w, h, score), ...]) in small_frame pixels."""
        h, w = frame.shape[:2]
        if self.detect_width and w > self.detect_width:
            scale = self.detect_width / w
            frame = cv2.resize(frame, (self.detect_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        people = []
        for det in result.detections or []:
            b = det.bounding_box
            people.append((b.origin_x, b.origin_y, b.width, b.height, det.categories[0].score))
        return frame, people

    def close(self):
        self.detector.close()


# --------------------------------------------------------------------------- audio

class SoundBoard:
    """Preloads sounds into memory and plays them through the default output device.

    Uses sounddevice (PortAudio), which mediapipe already depends on. pygame was dropped
    because its bundled SDL clashes with the SDL inside OpenCV on macOS.
    """

    KEEPALIVE_RATE = 44100

    def __init__(self, sounds_dir, volume, keepalive_seconds):
        self.volume = max(0.0, min(1.0, volume))
        self.sounds = []  # (name, samples float32 [frames, channels], samplerate)
        sounds_dir = Path(sounds_dir)
        for path in sorted(sounds_dir.iterdir()) if sounds_dir.is_dir() else []:
            if path.suffix.lower() not in SOUND_EXTENSIONS:
                continue
            try:
                data, rate = sf.read(str(path), dtype="float32", always_2d=True)
            except Exception as e:  # soundfile raises several types for bad/unsupported files
                log.warning("Skipping %s: %s", path.name, e)
                continue
            self.sounds.append((path.name, data * self.volume, rate))
        if not self.sounds:
            raise SystemExit(f"No playable sounds in {sounds_dir} (supported: {sorted(SOUND_EXTENSIONS)})")
        log.info("Loaded %d sound(s): %s", len(self.sounds), ", ".join(n for n, _, _ in self.sounds))
        try:
            log.info("Audio output: %s", sd.query_devices(kind="output")["name"])
        except Exception as e:
            log.warning("No audio output device found: %s", e)
        self._last_name = None
        self.keepalive_seconds = keepalive_seconds
        self._last_audio = time.monotonic()
        self._busy_until = 0.0
        self._tone = self._make_keepalive_tone() if keepalive_seconds else None

    def _play(self, data, rate):
        """Non-blocking. Never raises: a missing speaker must not kill the detector loop."""
        try:
            sd.play(data, rate)  # stops whatever was playing
        except Exception as e:
            log.error("Audio playback failed (speaker disconnected?): %s", e)
            return False
        now = time.monotonic()
        self._busy_until = now + len(data) / rate
        self._last_audio = now
        return True

    def play(self):
        choices = [s for s in self.sounds if s[0] != self._last_name] or self.sounds
        name, data, rate = random.choice(choices)
        self._last_name = name
        self._play(data, rate)
        return name

    def busy(self):
        return time.monotonic() < self._busy_until

    def tick(self):
        """Call regularly. Plays an almost-silent tone to keep Bluetooth speakers awake."""
        now = time.monotonic()
        if self.busy():
            self._last_audio = now
        elif self._tone is not None and now - self._last_audio >= self.keepalive_seconds:
            self._play(self._tone, self.KEEPALIVE_RATE)
            log.debug("keepalive tone")

    def wait(self):
        while self.busy():
            time.sleep(0.05)

    def close(self):
        try:
            sd.stop()
        except Exception:
            pass

    @classmethod
    def _make_keepalive_tone(cls):
        # About -40 dBFS at 40 Hz for 1 second: effectively inaudible, but a real signal.
        t = np.arange(cls.KEEPALIVE_RATE) / cls.KEEPALIVE_RATE
        tone = (0.01 * np.sin(2 * np.pi * 40 * t)).astype("float32")
        return np.column_stack([tone, tone])


# --------------------------------------------------------------------------- misc helpers

def keep_awake():
    """Stop the OS from idle-sleeping while this runs. Lid-close sleep must be disabled separately."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log.info("Windows sleep blocked while running")
    elif sys.platform == "darwin":
        try:
            subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
            log.info("macOS sleep blocked while running (caffeinate)")
        except OSError:
            log.warning("caffeinate not found; disable sleep manually")
    else:
        log.info("Linux: run under `systemd-inhibit` or disable suspend in power settings")


def load_settings(args, require_zone=True):
    settings = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        settings.update(json.loads(SETTINGS_FILE.read_text()))
    if os.environ.get("CAMERA_URL"):
        settings["source"] = os.environ["CAMERA_URL"]
    for key, value in vars(args).items():
        if key in settings and value is not None:
            settings[key] = value
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        log.warning("Ignoring unknown settings: %s", ", ".join(sorted(unknown)))
    if settings["trigger_mode"] not in ("any", "size", "zone"):
        raise SystemExit(f"trigger_mode must be any, size or zone (got {settings['trigger_mode']!r})")
    if require_zone and settings["trigger_mode"] == "zone" and not settings["zone"]:
        raise SystemExit("trigger_mode is 'zone' but no zone is set. Run with --set-zone first.")
    return settings


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def save_setting(key, value):
    data = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
    data[key] = value
    SETTINGS_FILE.write_text(json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------- modes

def run_set_zone(settings):
    reader = LatestFrameReader(settings["source"], settings["rtsp_tcp"])
    frame = reader.wait_for_frame(20)
    reader.stop()
    if frame is None:
        raise SystemExit("No frame from camera within 20s")
    h, w = frame.shape[:2]
    print("Drag a box over the FLOOR area where people stand at the candy, then press ENTER (c = cancel).")
    print("Note: zones only make sense with camera auto-tracking OFF.")
    x, y, bw, bh = cv2.selectROI("Set trigger zone", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if bw == 0 or bh == 0:
        raise SystemExit("Cancelled; nothing saved")
    zone = [round(x / w, 4), round(y / h, 4), round((x + bw) / w, 4), round((y + bh) / h, 4)]
    save_setting("zone", zone)
    save_setting("trigger_mode", "zone")
    print(f"Saved zone {zone} and trigger_mode=zone to {SETTINGS_FILE.name}")


def run_test_sound(settings):
    board = SoundBoard(resolve(settings["sounds_dir"]), settings["volume"], 0)
    for name, data, rate in board.sounds:
        print(f"Playing {name} ({len(data) / rate:.1f}s) ...", flush=True)
        board._play(data, rate)
        board.wait()


def draw_overlay(frame, people, qualifying, settings, trigger, now, fps):
    h, w = frame.shape[:2]
    if settings["trigger_mode"] == "zone" and settings["zone"]:
        x1, y1, x2, y2 = settings["zone"]
        cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 0, 255), 2)
    if settings["trigger_mode"] == "size":
        line_h = int(settings["min_height_frac"] * h)
        cv2.line(frame, (5, h - 5), (5, h - 5 - line_h), (255, 0, 255), 3)  # min person height
    for (x, y, bw, bh, score), ok in zip(people, qualifying):
        color = (0, 0, 255) if ok else (0, 200, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, f"{score:.2f}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cd = trigger.cooldown_left(now)
    if cd > 0:
        status, color = f"COOLDOWN {cd:.0f}s", (200, 200, 200)
    elif trigger.present_since is not None:
        status, color = f"VICTIM {trigger.dwell_progress(now):.1f}/{trigger.dwell:.1f}s", (0, 0, 255)
    else:
        status, color = "ARMED", (0, 255, 0)
    cv2.putText(frame, f"{status}  {fps:.0f}fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def run_scare(settings):
    model_path = resolve(settings["model"])
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    board = SoundBoard(resolve(settings["sounds_dir"]), settings["volume"], settings["keepalive_seconds"])
    detector = PersonDetector(model_path, settings["min_score"], settings["detect_width"])
    trigger = ScareTrigger(settings["dwell_seconds"], settings["grace_seconds"], settings["cooldown_seconds"])
    snapshot_dir = resolve(settings["snapshot_dir"]) if settings["snapshot_dir"] else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    keep_awake()

    log.info("Connecting to %s", _redact(settings["source"]))
    reader = LatestFrameReader(settings["source"], settings["rtsp_tcp"])
    headless = settings["headless"]
    window = "Halloween scare (q = quit, s = test sound)"
    if not headless:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except cv2.error:
            log.warning("No GUI available; running headless")
            headless = True

    log.info("Running: mode=%s dwell=%.1fs cooldown=%ss", settings["trigger_mode"],
             settings["dwell_seconds"], settings["cooldown_seconds"])
    last_id = 0
    pending_snapshots = []  # (due_time, label)
    scare_count = 0
    frames = 0
    fps = 0.0
    fps_t0 = last_heartbeat = time.monotonic()
    stale_warned = False

    # Ctrl+C can land on a mediapipe/ffmpeg worker thread while the main thread is
    # blocked inside detect(), so a plain KeyboardInterrupt is unreliable. Use a flag.
    stop = threading.Event()

    def request_stop(signum, _frame):
        log.info("Stopping (signal %d)...", signum)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stop.is_set():
            board.tick()
            now = time.monotonic()
            fid, frame = reader.latest()
            if frame is None or fid == last_id:
                if reader.last_frame_time and now - reader.last_frame_time > 10 and not stale_warned:
                    log.warning("No new camera frames for 10s")
                    stale_warned = True
                time.sleep(0.005)
                continue
            last_id = fid
            stale_warned = False

            small, people = detector.detect(frame)
            sh, sw = small.shape[:2]
            qualifying = [person_qualifies(p[:4], sw, sh, settings["trigger_mode"],
                                           settings["min_height_frac"], settings["zone"]) for p in people]
            now = time.monotonic()
            if trigger.update(now, any(qualifying)):
                name = board.play()
                scare_count += 1
                log.info("BOO! #%d playing %s (%d person(s) in view)", scare_count, name, len(people))
                if snapshot_dir:
                    pending_snapshots = [(now + d, f"{scare_count:03d}_{d:.0f}s") for d in (0, 1, 2)]

            if pending_snapshots and now >= pending_snapshots[0][0]:
                _, label = pending_snapshots.pop(0)
                path = snapshot_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.jpg"
                cv2.imwrite(str(path), frame)

            frames += 1
            if now - fps_t0 >= 2:
                fps = frames / (now - fps_t0)
                frames, fps_t0 = 0, now
            if now - last_heartbeat >= 60:
                log.info("alive: %.1f fps, %d scare(s) so far", fps, scare_count)
                last_heartbeat = now

            if not headless:
                draw_overlay(small, people, qualifying, settings, trigger, now, fps)
                cv2.imshow(window, small)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    log.info("Test sound: %s", board.play())
    finally:
        reader.stop()
        detector.close()
        board.close()
        if not headless:
            cv2.destroyAllWindows()
        log.info("Stopped after %d scare(s)", scare_count)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="RTSP URL, video file, or webcam index (or set CAMERA_URL)")
    p.add_argument("--sounds-dir", dest="sounds_dir")
    p.add_argument("--volume", type=float)
    p.add_argument("--dwell", dest="dwell_seconds", type=float, help="seconds a person must linger")
    p.add_argument("--cooldown", dest="cooldown_seconds", type=float)
    p.add_argument("--min-score", dest="min_score", type=float)
    p.add_argument("--mode", dest="trigger_mode", choices=["any", "size", "zone"])
    p.add_argument("--min-height", dest="min_height_frac", type=float,
                   help="size mode: min person height as fraction of frame (0-1)")
    p.add_argument("--keepalive", dest="keepalive_seconds", type=float, help="0 disables")
    p.add_argument("--snapshots", dest="snapshot_dir", help="folder to save reaction photos")
    p.add_argument("--rtsp-tcp", dest="rtsp_tcp", action="store_true", default=None)
    p.add_argument("--headless", action="store_true", default=None)
    p.add_argument("--set-zone", action="store_true", help="draw the trigger zone and save it")
    p.add_argument("--test-sound", action="store_true", help="play every sound once and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    set_zone, test_sound = args.set_zone, args.test_sound
    del args.set_zone, args.test_sound, args.verbose
    settings = load_settings(args, require_zone=not set_zone)

    if test_sound:
        return run_test_sound(settings)
    if not settings["source"]:
        raise SystemExit("No camera source. Use --source, CAMERA_URL, or \"source\" in settings.json")
    if set_zone:
        return run_set_zone(settings)
    run_scare(settings)


if __name__ == "__main__":
    main()
