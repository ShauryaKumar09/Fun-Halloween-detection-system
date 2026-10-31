# Halloween Jump Scare

Watches your security camera. When someone lingers near the candy for ~1.5 seconds,
a random scary sound blasts from the speaker. Built on the person detector from
[Human_Detection](https://github.com/ShauryaKumar09/Human_Detection) (MediaPipe EfficientDet-Lite0).

## What it does differently from the old script

- **Lingering, not single frames**: a person has to be present for `dwell_seconds`
  (default 1.5s; brief missed detections are tolerated), so passers-by and glitches don't trigger it.
- **Cooldown** between scares (default 25s) instead of playing once ever.
- **Works with camera auto-tracking**: the default `size` mode only counts people who are big
  in the frame (i.e. close). A fixed zone would break the moment the camera pans.
- **Low lag**: frames are read on a background thread and only the newest is used, so the
  scare isn't delayed by the RTSP buffer. Reconnects if the stream drops.
- **Random sound** from `sounds/`, preloaded so playback starts instantly. Never the same twice in a row.
- **Cross-platform audio** (sounddevice, already a mediapipe dependency) instead of Windows-only `winsound`. A disconnected speaker logs an error instead of crashing.
- **Keeps Bluetooth speakers awake** with an inaudible 40 Hz tone every 4 minutes.
- **Blocks idle sleep** while running (Windows, macOS).
- **Optional reaction photos** at +0s/+1s/+2s after each scare.
- Camera password lives in `settings.json` (git-ignored), not in the code.

## Laptop requirements

Any 64-bit laptop from roughly 2014 onward with 4 GB RAM. No GPU needed.
**64-bit Python 3.9+ is required**; MediaPipe does not support 32-bit Windows.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `settings.example.json` to `settings.json` and put in your camera's RTSP URL.

Add your scary sounds to `sounds/` (`.wav`, `.mp3`, `.ogg`, `.flac`, `.aiff`). Sounds with an
**instant loud start** work best; trim any silence at the beginning or the scare lands late.

## Run

```bash
python halloween_scare.py --test-sound     # check the speaker works (plays every sound)
python halloween_scare.py                  # preview window: q = quit, s = test sound
python halloween_scare.py --headless       # no window, less CPU
```

The preview shows each person's box: **red** means they count, **orange** means detected but too
far away or outside the zone. The magenta bar on the left is the minimum height for `size` mode.

## Tuning (on the night, takes 5 minutes)

1. Run with the preview window and have someone stand where trick-or-treaters grab candy.
2. **Size mode** (default): if they show orange, lower `--min-height` (e.g. `0.25`). If people on the
   sidewalk show red, raise it.
3. If it fires too late, lower `--dwell` (e.g. `1.0`). If it fires on people walking past, raise it.
4. Night vision (IR) gives grayscale images with lower confidence scores. If people are missed in
   the dark, try `--min-score 0.25`.

**Zone mode** (only if you turn auto-tracking OFF): `python halloween_scare.py --set-zone`, then drag
a box over the floor where people stand. Saved to `settings.json`.

All options: `python halloween_scare.py --help`. Every flag can also go in `settings.json`.

## Night-of checklist

- [ ] Laptop **plugged in**. Turn off sleep **and lid-close sleep** in power settings
      (the script blocks idle sleep, but not lid-close).
- [ ] Turn off Windows Update restarts for the evening ("Pause updates").
- [ ] Speaker: AUX cable is most reliable. With Bluetooth, keep the keepalive on and test after 10+ minutes idle.
- [ ] Laptop's own volume all the way up; control loudness with the speaker knob.
- [ ] Laptop and speaker sheltered from rain.
- [ ] Do a test walk-up after dark, with the camera in night mode.

**Volume**: keep the speaker a few feet from where people stand and not aimed at a small kid's ear
height. It's just as scary, and nobody leaves with ringing ears.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open camera` / `No route to host` / `Host is down` | The laptop can't reach the camera at all. Check: same Wi-Fi network; camera's current IP (it may have changed, look in the camera app or router); on macOS, System Settings > Privacy & Security > Local Network > enable Terminal. Then test the URL in VLC. |
| Choppy/grey frames, frequent reconnects | `--rtsp-tcp` |
| Scare is late | Lower `dwell_seconds`; use the camera's sub-stream (lower resolution) URL; trim silence from the sound file |
| Low fps on old laptop | `--headless`, or lower `detect_width` in `settings.json` (e.g. 480) |
| Sound plays on laptop instead of speaker | Set the speaker as the OS default output device *before* starting the script |

## Tests

```bash
pip install pytest && python -m pytest tests
```
