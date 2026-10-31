import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halloween_scare import ScareTrigger, person_qualifies, _redact  # noqa: E402


def feed(trigger, start, end, present, step=0.1):
    """Feed observations at `step` intervals; return times at which it fired."""
    fired = []
    for i in range(round((end - start) / step)):
        t = round(start + i * step, 6)
        if trigger.update(t, present):
            fired.append(round(t, 2))
    return fired


def test_fires_after_dwell():
    t = ScareTrigger(dwell_seconds=1.5, grace_seconds=0.5, cooldown_seconds=25)
    assert feed(t, 0, 3, True) == [1.5]


def test_passerby_shorter_than_dwell_does_not_fire():
    t = ScareTrigger(1.5, 0.5, 25)
    assert feed(t, 0, 1.0, True) == []
    assert feed(t, 1.0, 5.0, False) == []


def test_brief_dropout_within_grace_keeps_timer():
    t = ScareTrigger(1.5, 0.5, 25)
    assert feed(t, 0, 1.0, True) == []
    assert feed(t, 1.0, 1.3, False) == []  # 0.3s missed detections
    assert feed(t, 1.3, 2.0, True) == [1.5]


def test_long_dropout_resets_timer():
    t = ScareTrigger(1.5, 0.5, 25)
    feed(t, 0, 1.0, True)
    feed(t, 1.0, 2.0, False)  # 1s gone > grace
    assert feed(t, 2.0, 3.4, True) == []
    assert feed(t, 3.4, 4.0, True) == [3.5]


def test_cooldown_then_must_dwell_again():
    t = ScareTrigger(1.5, 0.5, 10)
    assert feed(t, 0, 20, True) == [1.5, 11.5]


def test_slow_frame_rate_still_fires():
    t = ScareTrigger(1.5, 0.5, 25)
    assert feed(t, 0, 3, True, step=0.4) == [1.6]


def test_size_mode():
    assert person_qualifies((0, 0, 50, 200), 640, 480, "size", 0.35, None)
    assert not person_qualifies((0, 0, 50, 100), 640, 480, "size", 0.35, None)


def test_zone_mode_uses_feet():
    zone = [0.25, 0.5, 0.75, 1.0]
    assert person_qualifies((300, 100, 40, 300), 640, 480, "zone", 0, zone)   # feet at y=400
    assert not person_qualifies((300, 0, 40, 100), 640, 480, "zone", 0, zone)  # feet at y=100
    assert not person_qualifies((0, 100, 40, 300), 640, 480, "zone", 0, zone)  # left of zone
    assert not person_qualifies((300, 100, 40, 300), 640, 480, "zone", 0, None)


def test_redact_password():
    assert _redact("rtsp://cam:secret@192.168.1.5:554/live") == "rtsp://cam:***@192.168.1.5:554/live"
    assert _redact("0") == "0"
