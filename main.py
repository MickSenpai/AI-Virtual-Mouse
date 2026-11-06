"""
main.py
-------
AI Virtual Mouse - control your mouse cursor with hand gestures using a
webcam, OpenCV, and MediaPipe Hands.

Features:
    * Smooth (EMA-based) cursor movement with dead-zone filtering
    * Full-monitor coordinate mapping via numpy interpolation
    * Left click, right click, double click
    * Click / double-click / screenshot cooldowns to prevent repeats
    * Drag & drop (pinch thumb + index to grab, release to drop)
    * Scroll up / down
    * Timestamped screenshots saved to a "Screenshots/" folder
    * Real-time FPS counter and on-screen status overlay
    * Defensive error handling so the app never crashes on camera
      loss or missing hand detections

Run with:
    python main.py
Press 'q' with the video window focused to quit.
"""

import json
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NamedTuple, Optional, Tuple

if sys.platform == "win32":
    # Must run before pyautogui is imported: pyautogui otherwise sets the weaker
    # "system DPI aware" mode itself, and Windows only accepts the first call.
    # Per-monitor awareness keeps pyautogui.size() and cursor coordinates in
    # real pixels when display scaling is 125%/150%.
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except (AttributeError, OSError):
        pass

import cv2
import mediapipe as mp
import pyautogui
import util
from pynput import keyboard
from pynput.mouse import Button, Controller

log = logging.getLogger("virtual_mouse")

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
SMOOTHING = 0.7                 # EMA smoothing factor (0 = none, closer to 1 = smoother/laggier)
# Camera-frame region (px inset from each edge) that the fingertip sweeps to cover the whole screen.
# The bottom margin is large on purpose: reaching the bottom of the screen with the fingertip must not
# push the wrist out of the frame, because a partly visible hand makes tracking unreliable.
FRAME_MARGIN_LEFT = 100
FRAME_MARGIN_RIGHT = 100
FRAME_MARGIN_TOP = 40
FRAME_MARGIN_BOTTOM = 200
CLICK_COOLDOWN = 0.12           # seconds between clicks; kept short so two quick pinches make a double click
CLICK_PINCH_THRESHOLD = 40      # thumb-tip to middle/ring/pinky-tip distance (0-1000 scale) below which the pinch closes
CLICK_PINCH_RELEASE = 70        # ...and above which it counts as open again (hysteresis)
CLICK_PINCH_DROP = 15           # a pinch also counts if the thumb-fingertip distance fell by at least this much...
CLICK_PINCH_LOOKBACK = 0.8      # ...within this many seconds, even if it never went past CLICK_PINCH_RELEASE
                                #   (a hand whose resting pose already keeps the fingertips close together)
CLICK_PINCH_NEAR = 28           # the drop path needs a real contact (below this), so slow finger tremor never clicks
CLICK_PINCH_CONTACT = 12        # typical distance at contact; used only to predict readiness for the overlay
CLICK_PINCH_HOLD = 55           # once closing has started, the pinch keeps counting while below this (jitter around the threshold)
CLICK_PINCH_TIE = 10            # when two fingertips are this close in distance, the earlier one in CLICK_PINCHES wins
CLICK_INDEX_MIN_ANGLE = 70      # the pointing index may curl down to this angle while a button pinch is made
GRAB_TIE = 12                   # the index tip may be this much farther from the thumb than the closest other
                                #   fingertip and still count as a grab (a deliberate index pinch with a middle finger nearby)
CLICK_PINCH_FRAMES = 2          # consecutive frames the pinch must stay closed to click
CLICK_FREEZE_SECONDS = 0.3      # the cursor stays put this long after a pinch click (or until the pinch opens)
MOVE_STAY_ANGLE = 70            # once moving, the index finger may bend down to this angle before Move is lost
MOVE_STAY_THUMB_DIST = 90       # ...and the thumb may drift this far (0-1000 scale) from the index base
STALL_LOG_SECONDS = 0.15        # a frame slower than this is logged with where the time went
HAND_LOST_GRACE = 0.3           # seconds the last state is held when tracking drops, before everything resets
HAND_LOST_GRACE_DRAG = 0.8      # ...longer while dragging, so a tracking glitch doesn't drop the window
DRAG_RELEASE_THRESHOLD = 70     # once dragging, the thumb-index pinch must open past this (0-1000 scale) to let go
DRAG_RELEASE_FRAMES = 2         # ...and stay open this many consecutive frames, so one noisy frame doesn't drop it
SCREENSHOT_COOLDOWN = 2.0       # seconds between screenshots
SCROLL_COOLDOWN = 0.15          # seconds between individual scroll "ticks"
SCROLL_PIXELS_PER_TICK = 8      # px of vertical hand movement per scroll tick (faster movement scrolls more)
SCROLL_MAX_TICKS = 5            # cap on ticks applied in a single scroll step
DEAD_ZONE = 4                   # px; ignore cursor moves smaller than this to kill jitter
SCROLL_DEAD_ZONE_PX = 8         # px; vertical movement needed to register a scroll tick
PINCH_THRESHOLD = 40            # thumb-tip/index-tip distance (0-1000 scale) below which counts as a pinch
THUMB_NEAR_DIST = 50            # thumb-tip/index-base distance (0-1000 scale) below which the thumb "rests" on the hand
FINGER_STRAIGHT_ANGLE = 90      # degrees; a finger joint angle above this counts as extended
FINGER_BENT_ANGLE = 50          # degrees; a finger joint angle below this counts as bent
HAND_REFERENCE_SIZE = 225       # wrist -> middle-knuckle distance (0-1000 scale) at which the distance thresholds were tuned
HAND_SCALE_MIN = 0.5            # bounds for the hand-size rescaling factor; stops a foreshortened
HAND_SCALE_MAX = 2.0            #   (tiny) hand from inflating distances so much that no gesture matches
HOTKEY_PAUSE = "<ctrl>+<alt>+p" # global hotkey: pause / resume cursor control
HOTKEY_QUIT = "<ctrl>+<alt>+q"  # global hotkey: quit, works even when the video window isn't focused
GESTURE_CONFIRM_FRAMES = 3      # consecutive frames a gesture must be seen before it becomes active
MAX_CONSECUTIVE_FRAME_ERRORS = 30  # frames in a row that may fail processing before the app gives up
# Pointer acceleration: fast hand movement travels further on screen, slow movement stays 1:1.
ACCEL_ENABLED = True            # toggle live with the 'a' key
ACCEL_MAX_GAIN = 1.6            # screen distance per unit of fingertip distance at full speed (1.0 = off)
ACCEL_SPEED_START = 20          # fingertip speed (screen px per frame) where acceleration starts...
ACCEL_SPEED_FULL = 80           # ...and where it reaches ACCEL_MAX_GAIN
ACCEL_RECENTER = 0.25           # per-frame pull of the cursor back onto the finger's absolute position when moving slowly
ACCEL_RESET_SECONDS = 0.15      # a gap in cursor updates longer than this restarts the acceleration baseline
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
CALIBRATION_MIN_WIDTH = 160     # camera px; smallest comfortable area accepted by the 'c' calibration
CALIBRATION_MIN_HEIGHT = 100
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
SCREENSHOT_DIR = "Screenshots"
WINDOW_NAME = "AI Virtual Mouse"

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False  # gestures already stay within the frame; avoid corner-triggered exceptions
pyautogui.PAUSE = 0         # default is 0.1s after every call, which caps cursor movement at ~10 FPS

# Landmark indices (MediaPipe Hands): wrist, thumb tip, index base/tip, middle base/tip, ring/pinky tip.
WRIST, THUMB_TIP, INDEX_BASE, INDEX_TIP, MIDDLE_BASE, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 5, 8, 9, 12, 16, 20

# Thumb-to-fingertip pinches that act as buttons (finger -> action), see update_click_pinches().
CLICK_PINCHES = (("middle", "Left Click"), ("ring", "Right Click"), ("pinky", "Screenshot"))

# Camera-frame margins as (left, right, top, bottom); the calibration ('c') overrides them.
DEFAULT_MARGINS = (FRAME_MARGIN_LEFT, FRAME_MARGIN_RIGHT, FRAME_MARGIN_TOP, FRAME_MARGIN_BOTTOM)

# On-screen label (text, BGR color) drawn for each active gesture.
GESTURE_LABELS = {
    "Dragging": ("Gesture : Dragging", (0, 200, 255)),
    "Moving": ("Gesture : Moving", (0, 255, 0)),
}
SCROLL_LABEL_COLOR = (200, 0, 200)
# Brief confirmation drawn after a pinch action (text, BGR color).
CLICK_FLASH = {
    "Left Click": ("Left Click", (0, 255, 0)),
    "Right Click": ("Right Click", (0, 0, 255)),
    "Screenshot": ("Screenshot saved", (255, 255, 0)),
}
CLICK_FLASH_SECONDS = 0.4
SCREENSHOT_FLASH_SECONDS = 1.0  # long enough to notice an accidental screenshot


# --------------------------------------------------------------------------
# STATE
# --------------------------------------------------------------------------
@dataclass
class AppState:
    """Mutable state shared across frames."""

    mouse: Any  # pynput mouse Controller (injected so tests can pass a fake)
    prev_cursor: Optional[Tuple[float, float]] = None
    dragging: bool = False
    last_click_time: float = 0.0
    last_screenshot_time: float = 0.0
    last_scroll_time: float = 0.0
    prev_scroll_y: Optional[float] = None
    gesture_label: str = "None"
    active_gesture: str = "None"
    candidate_gesture: Optional[str] = None
    candidate_frames: int = 0
    frame_w: int = CAMERA_WIDTH
    frame_h: int = CAMERA_HEIGHT
    # Click-type gestures only fire once the Move pose has been shown since the
    # hand (re)appeared, so a hand entering the frame can't click by accident.
    armed: bool = False
    paused: bool = False          # toggled from the hotkey listener thread
    quit_requested: bool = False  # set from the hotkey listener thread
    hand_lost_since: Optional[float] = None
    drag_release_frames: int = 0  # consecutive frames the drag pinch has been open
    # Click pinches (thumb to middle / ring / pinky fingertip), see update_click_pinches().
    pinch_open: dict = field(default_factory=lambda: {finger: False for finger, _ in CLICK_PINCHES})
    pinch_frames: dict = field(default_factory=lambda: {finger: 0 for finger, _ in CLICK_PINCHES})
    pinch_history: dict = field(default_factory=lambda: {finger: deque() for finger, _ in CLICK_PINCHES})  # (time, distance)
    pinch_hold: Optional[str] = None  # finger whose click just fired: cursor stays put until it opens...
    pinch_hold_until: float = 0.0     # ...or CLICK_FREEZE_SECONDS pass, whichever comes first
    cursor_status: str = "idle (no hand)"  # why the cursor is (not) following the finger; shown in the overlay
    pinch_distances: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # latest (index, middle, ring, pinky)
    click_status: str = "locked (show the Move pose first)"  # why a button pinch would (not) register; shown in the overlay
    click_flash: Tuple[str, Tuple[int, int, int]] = ("", (0, 0, 0))
    click_flash_until: float = 0.0
    show_distances: bool = False  # toggled with the 'd' key
    # Camera region the fingertip maps to the screen, and the 'c' calibration that edits it.
    margins: Tuple[int, int, int, int] = DEFAULT_MARGINS
    index_tip: Optional[Tuple[float, float]] = None  # latest normalized fingertip, used by the calibration
    calibration_step: int = 0  # 0 = off, 1 = waiting for the top-left corner, 2 = waiting for the bottom-right one
    calibration_tl: Optional[Tuple[float, float]] = None  # camera px of the first corner
    notice: str = ""  # short message drawn on the video (calibration result, toggles)
    notice_until: float = 0.0
    # Pointer acceleration, see apply_acceleration().
    accel_enabled: bool = ACCEL_ENABLED
    accel_pos: Optional[Tuple[float, float]] = None
    prev_target: Optional[Tuple[float, float]] = None
    last_move_time: float = 0.0

    def reset_hand_tracking(self):
        """Forget per-hand tracking (called when the hand has been gone longer than the grace period)."""
        self.armed = False
        self.drag_release_frames = 0
        self.pinch_open = {finger: False for finger, _ in CLICK_PINCHES}
        self.pinch_frames = {finger: 0 for finger, _ in CLICK_PINCHES}
        self.pinch_history = {finger: deque() for finger, _ in CLICK_PINCHES}
        self.pinch_hold = None
        self.accel_pos = None
        self.prev_target = None
        self.prev_cursor = None
        self.prev_scroll_y = None
        self.active_gesture = "None"
        self.candidate_gesture = None
        self.candidate_frames = 0


# --------------------------------------------------------------------------
# LANDMARK / GESTURE HELPERS
# --------------------------------------------------------------------------
class FingerAngles(NamedTuple):
    """Joint angle (degrees) of each finger: large = extended, small = bent."""

    index: float
    middle: float
    ring: float
    pinky: float


def finger_angles(landmark_list):
    """Compute every finger's joint angle once per frame."""
    return FingerAngles(
        index=util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]),
        middle=util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]),
        ring=util.get_angle(landmark_list[13], landmark_list[14], landmark_list[16]),
        pinky=util.get_angle(landmark_list[17], landmark_list[18], landmark_list[20]),
    )


def is_extended(angle):
    return angle > FINGER_STRAIGHT_ANGLE


def is_bent(angle):
    return angle < FINGER_BENT_ANGLE


def is_move_gesture(angles, thumb_index_dist, keep=False):
    """
    Index finger extended while the thumb rests near the index base. With
    `keep` (Move is already active) the pose is held with hysteresis, so a
    thumb that drifts a little or an index that dips slightly doesn't drop
    the cursor.
    """
    if keep:
        return thumb_index_dist < MOVE_STAY_THUMB_DIST and angles.index > MOVE_STAY_ANGLE
    return thumb_index_dist < THUMB_NEAR_DIST and is_extended(angles.index)


def is_pinching(pinch_dist, keep=False):
    """
    Thumb tip close to index tip (hand-size-normalized) -> used to trigger drag
    & drop. With `keep` (already dragging) the pinch is held with hysteresis:
    it only counts as open past DRAG_RELEASE_THRESHOLD, not PINCH_THRESHOLD.
    """
    return pinch_dist < (DRAG_RELEASE_THRESHOLD if keep else PINCH_THRESHOLD)


def is_scroll_pose(angles):
    """Index, middle and ring fingers extended while the pinky is curled."""
    return (
        is_extended(angles.index)
        and is_extended(angles.middle)
        and is_extended(angles.ring)
        and is_bent(angles.pinky)
    )


def classify_gesture(angles, thumb_index_dist, pinch_dist, moving=False, dragging=False):
    """Return the name of the gesture matching this single frame (no debouncing)."""
    if is_pinching(pinch_dist, keep=dragging):
        return "Dragging"
    # Scroll is checked before Move: it is the more specific pose (it needs the
    # middle and ring fingers extended too), and Move only looks at the index
    # finger and thumb, so with the thumb resting near the index base an
    # open-hand scroll pose would otherwise always be read as Move.
    if is_scroll_pose(angles):
        return "Scrolling"
    if is_move_gesture(angles, thumb_index_dist, keep=moving):
        return "Moving"
    return "None"


def confirm_gesture(state, raw_gesture):
    """
    Debounce: a gesture only becomes the active one after being seen for
    GESTURE_CONFIRM_FRAMES consecutive frames, so a single ambiguous frame
    can't trigger a click. Returns the active gesture.
    """
    if raw_gesture == state.candidate_gesture:
        state.candidate_frames += 1
    else:
        state.candidate_gesture = raw_gesture
        state.candidate_frames = 1

    if state.candidate_frames >= GESTURE_CONFIRM_FRAMES:
        state.active_gesture = raw_gesture
    return state.active_gesture


def update_click_pinches(state, angles, distances):
    """
    Edge-detect the button pinches (see CLICK_PINCHES): thumb tip touching the
    middle fingertip is a left click, the ring fingertip a right click and the
    pinky fingertip a screenshot, while the index finger stays extended and
    keeps pointing (so the action doesn't drag the cursor along, unlike
    bending the index finger). Two quick left pinches make a double click.

    `distances` maps finger name -> thumb-to-fingertip distance (0-1000 scale).
    A pinch only counts after the fingertips were seen apart (> CLICK_PINCH_RELEASE)
    since the hand appeared or the last pinch, so a hand whose resting pose already
    has the thumb touching a curled finger can't act on its own. If several
    fingers are close to the thumb at once, the closest one wins and the others
    are consumed by the same pinch. Needs the clicks to be armed. Returns the
    action name ("Left Click", "Right Click", "Screenshot") or None.
    """
    candidates = []
    index_ok = angles.index > CLICK_INDEX_MIN_ANGLE
    now = time.time()
    for finger, name in CLICK_PINCHES:
        dist = distances[finger]
        history = state.pinch_history[finger]
        history.append((now, dist))
        while history and now - history[0][0] > CLICK_PINCH_LOOKBACK:
            history.popleft()
        recent_max = max(d for _t, d in history)
        if dist > CLICK_PINCH_RELEASE:
            state.pinch_open[finger] = True
            state.pinch_frames[finger] = 0
            if state.pinch_hold == finger:
                state.pinch_hold = None
            continue
        # Closing starts below the threshold, then keeps counting a bit above it so
        # tracking jitter around the threshold doesn't restart the count every frame.
        closing = dist < CLICK_PINCH_THRESHOLD or (state.pinch_frames[finger] > 0 and dist < CLICK_PINCH_HOLD)
        # "Fresh": the fingertips were seen apart past the release distance, or they
        # just closed in by a clear amount. The second case is what lets a hand whose
        # resting pose keeps the fingertips fairly close still click, while a hand that
        # simply rests there (no closing movement) never does.
        fresh = state.pinch_open[finger] or (recent_max - dist >= CLICK_PINCH_DROP and dist < CLICK_PINCH_NEAR)
        if closing and fresh and state.armed and index_ok:
            state.pinch_frames[finger] += 1
            if state.pinch_frames[finger] >= CLICK_PINCH_FRAMES:
                candidates.append((dist, finger, name))
        else:
            state.pinch_frames[finger] = 0

    if not candidates:
        return None

    # The closest fingertip wins, but near-ties go to the earlier finger in
    # CLICK_PINCHES: a thumb landing between the middle and ring tips is a left
    # click, not a right one.
    best = min(dist for dist, _finger, _name in candidates)
    _, finger, name = next(c for c in candidates if c[0] <= best + CLICK_PINCH_TIE)
    state.pinch_hold = finger
    state.pinch_hold_until = time.time() + CLICK_FREEZE_SECONDS
    for other, _name in CLICK_PINCHES:
        if distances[other] <= CLICK_PINCH_RELEASE:
            state.pinch_open[other] = False
            state.pinch_frames[other] = 0
            state.pinch_history[other].clear()  # so the same closing can't count again as a fresh drop
    return name


def describe_click_state(state, angles, raw_gesture):
    """Why a left-click pinch (thumb to middle fingertip) would or wouldn't register right now."""
    if not state.armed:
        return "locked (show the Move pose first)"
    if raw_gesture == "Dragging" or state.dragging:
        return "blocked (grabbing with thumb+index)"
    if angles.index <= CLICK_INDEX_MIN_ANGLE:
        return "blocked (index finger curled)"
    history = state.pinch_history["middle"]
    recent_max = max((d for _t, d in history), default=0.0)
    if not state.pinch_open["middle"] and recent_max - CLICK_PINCH_CONTACT < CLICK_PINCH_DROP:
        return "blocked (move thumb and middle finger apart first)"
    return "ready"


# --------------------------------------------------------------------------
# CURSOR / ACTION HANDLERS
# --------------------------------------------------------------------------
def compute_target_cursor(index_finger_tip, frame_w, frame_h, margins=DEFAULT_MARGINS):
    """
    Map the (normalized) index-finger-tip position (x, y) to full-monitor
    screen coordinates using numpy interpolation, insetting the usable
    camera frame by `margins` (left, right, top, bottom) so the whole
    screen is reachable without requiring the finger (or the wrist) to
    reach the extreme edges of the frame.

    frame_w / frame_h are the real frame dimensions, since the camera
    may not honor the requested CAMERA_WIDTH / CAMERA_HEIGHT.
    """
    left, right, top, bottom = margins
    raw_x = index_finger_tip[0] * frame_w
    raw_y = index_finger_tip[1] * frame_h

    screen_x = util.interpolate(raw_x, [left, frame_w - right], [0, SCREEN_WIDTH])
    screen_y = util.interpolate(raw_y, [top, frame_h - bottom], [0, SCREEN_HEIGHT])

    screen_x = util.clamp(screen_x, 0, SCREEN_WIDTH - 1)
    screen_y = util.clamp(screen_y, 0, SCREEN_HEIGHT - 1)
    return screen_x, screen_y


def apply_acceleration(state, target):
    """
    Pointer acceleration on top of the absolute finger->screen mapping. The
    cursor follows the fingertip's frame-to-frame movement multiplied by a gain
    that grows with speed (1.0 when slow, up to ACCEL_MAX_GAIN when fast), so
    crossing the screen takes less hand travel while small movements stay
    precise. Because that makes the cursor run ahead of the finger, slow
    movement gently pulls it back onto the finger's absolute position
    (ACCEL_RECENTER), so the two never drift apart for good.
    Returns the (possibly accelerated) screen position to move toward.
    """
    if not state.accel_enabled:
        state.accel_pos = None
        state.prev_target = None
        return target

    now = time.time()
    stale = state.accel_pos is None or state.prev_target is None or now - state.last_move_time > ACCEL_RESET_SECONDS
    state.last_move_time = now
    if stale:
        # First update, or the cursor wasn't being driven for a while (click
        # hold, grab, hand lost): restart on the finger's absolute position and
        # let the usual smoothing glide the cursor there.
        pos = target
        dx = dy = 0.0
    else:
        pos = state.accel_pos
        dx, dy = target[0] - state.prev_target[0], target[1] - state.prev_target[1]

    speed = math.hypot(dx, dy)
    t = util.clamp((speed - ACCEL_SPEED_START) / (ACCEL_SPEED_FULL - ACCEL_SPEED_START), 0.0, 1.0)
    weight = t * t * (3 - 2 * t)  # smoothstep: no abrupt change of feel
    gain = 1.0 + (ACCEL_MAX_GAIN - 1.0) * weight
    pull = ACCEL_RECENTER * (1.0 - weight)

    x = pos[0] + gain * dx
    y = pos[1] + gain * dy
    x += pull * (target[0] - x)
    y += pull * (target[1] - y)
    pos = (util.clamp(x, 0, SCREEN_WIDTH - 1), util.clamp(y, 0, SCREEN_HEIGHT - 1))

    state.accel_pos = pos
    state.prev_target = target
    return pos


def move_mouse(state, index_finger_tip):
    """Smoothly move the OS cursor toward the mapped target position."""
    target = compute_target_cursor(index_finger_tip, state.frame_w, state.frame_h, state.margins)
    target = apply_acceleration(state, target)
    smoothed = util.smooth_cursor(state.prev_cursor, target, SMOOTHING)

    if state.prev_cursor is not None:
        # Dead zone on the raw target, not on the smoothed step: the EMA shrinks
        # every step, so testing the step would stall the cursor
        # DEAD_ZONE / (1 - SMOOTHING) px short of the fingertip.
        dx = target[0] - state.prev_cursor[0]
        dy = target[1] - state.prev_cursor[1]
        if abs(dx) < DEAD_ZONE and abs(dy) < DEAD_ZONE:
            return  # fingertip is within the dead zone of the cursor; skip to avoid jitter

    pyautogui.moveTo(int(smoothed[0]), int(smoothed[1]))
    state.prev_cursor = smoothed


def set_notice(state, text, seconds=3.0):
    """Show a short message on the video."""
    state.notice = text
    state.notice_until = time.time() + seconds


def margins_from_corners(top_left, bottom_right, frame_w, frame_h):
    """
    Turn two fingertip positions (camera px) marking the corners of a comfortable
    area into (left, right, top, bottom) margins. Returns None if the area is
    too small (or inverted) to be usable.
    """
    left, top = top_left
    right_edge, bottom_edge = bottom_right
    if right_edge - left < CALIBRATION_MIN_WIDTH or bottom_edge - top < CALIBRATION_MIN_HEIGHT:
        return None
    return (
        max(0, int(left)),
        max(0, int(frame_w - right_edge)),
        max(0, int(top)),
        max(0, int(frame_h - bottom_edge)),
    )


def load_calibration(frame_w, frame_h):
    """Saved margins, or None if there is no usable calibration for this camera resolution."""
    try:
        with open(CALIBRATION_FILE, encoding="utf-8") as f:
            data = json.load(f)
        margins = tuple(int(v) for v in data["margins"])
        if len(margins) != 4 or list(data["frame"]) != [frame_w, frame_h]:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    left, right, top, bottom = margins
    if frame_w - left - right < CALIBRATION_MIN_WIDTH or frame_h - top - bottom < CALIBRATION_MIN_HEIGHT:
        return None
    return margins


def save_calibration(margins, frame_w, frame_h):
    try:
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump({"margins": list(margins), "frame": [frame_w, frame_h]}, f)
    except OSError as exc:
        log.warning("Could not save the calibration: %s", exc)


def clear_calibration():
    try:
        os.remove(CALIBRATION_FILE)
    except OSError:
        pass


def advance_calibration(state):
    """
    'c' key: first press starts the calibration, the next two record the
    fingertip at the top-left and bottom-right corners of the area the hand can
    reach comfortably. That area then covers the whole screen.
    """
    if state.calibration_step == 0:
        state.calibration_step = 1
        state.calibration_tl = None
        return
    if state.index_tip is None:
        set_notice(state, "Show your hand first, then press c")
        return

    point = (state.index_tip[0] * state.frame_w, state.index_tip[1] * state.frame_h)
    if state.calibration_step == 1:
        state.calibration_tl = point
        state.calibration_step = 2
        return

    margins = margins_from_corners(state.calibration_tl, point, state.frame_w, state.frame_h)
    state.calibration_step = 0
    state.calibration_tl = None
    if margins is None:
        set_notice(state, f"Area too small (min {CALIBRATION_MIN_WIDTH}x{CALIBRATION_MIN_HEIGHT} px) - press c to retry")
        return
    state.margins = margins
    state.prev_cursor = None
    state.accel_pos = None
    state.prev_target = None
    save_calibration(margins, state.frame_w, state.frame_h)
    set_notice(state, "Comfortable area saved")


def handle_key(state, key):
    """Single-key commands of the video window ('q' is handled by the main loop)."""
    if key == ord("d"):
        state.show_distances = not state.show_distances
    elif key == ord("a"):
        state.accel_enabled = not state.accel_enabled
        state.accel_pos = None
        state.prev_target = None
        set_notice(state, f"Acceleration {'on' if state.accel_enabled else 'off'}")
    elif key == ord("r"):
        state.margins = DEFAULT_MARGINS
        state.calibration_step = 0
        state.calibration_tl = None
        state.prev_cursor = None
        state.accel_pos = None
        state.prev_target = None
        clear_calibration()
        set_notice(state, "Area reset to default")
    elif key == ord("c"):
        advance_calibration(state)


def exclude_window_from_capture(title):
    """
    Hide the camera window from screen captures (Windows 10 2004+), so the
    screenshot gesture doesn't photograph the preview itself. Returns True
    on success; on other platforms the window stays visible in screenshots.
    """
    if sys.platform != "win32":
        return False
    wda_excludefromcapture = 0x00000011
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    return bool(hwnd) and bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, wda_excludefromcapture))


def disable_power_throttling():
    """
    Opt this process out of Windows' background throttling (Windows 10 1709+ /
    11). When the video window is minimized or hidden behind other apps,
    Windows treats the process as background work: it runs it on slower
    efficiency cores / lower power states (about 3x less CPU in testing) and
    ignores its timer-resolution requests, which makes tracking lag badly.
    Returns True on success.
    """
    if sys.platform != "win32":
        return False
    from ctypes import wintypes

    class ProcessPowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    process_power_throttling = 4  # PROCESS_INFORMATION_CLASS value
    execution_speed, ignore_timer_resolution = 0x1, 0x4
    # StateMask 0 with a control bit set = "never throttle this".
    state = ProcessPowerThrottlingState(1, execution_speed | ignore_timer_resolution, 0)

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    return bool(
        kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), process_power_throttling, ctypes.byref(state), ctypes.sizeof(state)
        )
    )


def read_key():
    """
    Non-blocking key poll that also pumps the window's events. cv2.waitKey(1)
    sleeps a whole 15.6 ms Windows timer tick when the window is minimized and
    has nothing to repaint, which halves the loop rate; pollKey never waits.
    """
    poll = getattr(cv2, "pollKey", None)
    return (poll() if poll else cv2.waitKey(1)) & 0xFF


def _save_screenshot(filename):
    try:
        pyautogui.screenshot().save(filename)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to save screenshot: %s", exc)


def take_screenshot(frame):
    """
    Save a timestamped screenshot into the Screenshots/ directory. Grabbing and
    PNG-encoding the screen takes ~100 ms, so it runs on a background thread
    instead of freezing the tracking loop for several frames.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(SCREENSHOT_DIR, f"Screenshot_{timestamp}.png")
    threading.Thread(target=_save_screenshot, args=(filename,), daemon=True).start()
    cv2.putText(frame, "Screenshot Taken", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def start_drag(state):
    if not state.dragging:
        state.mouse.press(Button.left)
        state.dragging = True


def stop_drag(state):
    if state.dragging:
        state.mouse.release(Button.left)
        state.dragging = False


def draw_gesture_label(frame, text, color, y=60):
    cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def handle_scroll(state, landmark_list, frame):
    """
    While the scroll pose is held, compare the current vertical
    position of the middle-finger tip against the previous frame to
    decide whether to scroll up or down.
    """
    current_y = landmark_list[12][1] * state.frame_h
    now = time.time()
    direction_label = None

    if state.prev_scroll_y is not None and now - state.last_scroll_time > SCROLL_COOLDOWN:
        delta = current_y - state.prev_scroll_y
        if abs(delta) > SCROLL_DEAD_ZONE_PX:
            # Move the hand faster -> more ticks per step (at least 1, capped).
            ticks = min(SCROLL_MAX_TICKS, max(1, int(abs(delta) // SCROLL_PIXELS_PER_TICK)))
            if delta < 0:
                state.mouse.scroll(0, ticks)  # positive -> scroll up
                direction_label = "Scroll Up"
            else:
                state.mouse.scroll(0, -ticks)  # negative -> scroll down
                direction_label = "Scroll Down"
            state.last_scroll_time = now

    state.prev_scroll_y = current_y

    draw_gesture_label(frame, f"Gesture : {direction_label or 'Scroll Ready'}", SCROLL_LABEL_COLOR)


# --------------------------------------------------------------------------
# MAIN GESTURE DISPATCH
# --------------------------------------------------------------------------
def detect_gesture(frame, landmark_list, state):
    """
    Inspect the current frame's hand landmarks and route to the
    appropriate action (move / click / drag / scroll / screenshot).
    Gestures are debounced over several frames, and click-type gestures
    fire once when the gesture starts (not repeatedly while it is held),
    in addition to their cooldowns. Updates `state.gesture_label`
    for the on-screen overlay.
    """
    now = time.time()
    state.gesture_label = "None"

    if len(landmark_list) < 21:
        if state.hand_lost_since is None:
            state.hand_lost_since = now
            state.prev_scroll_y = None  # the hand may move a lot while unseen
        if now - state.hand_lost_since <= (HAND_LOST_GRACE_DRAG if state.dragging else HAND_LOST_GRACE):
            # Tracking often drops for a few frames (hand near the frame edge,
            # motion blur). Hold the cursor, drag and gesture state instead of
            # resetting, so control resumes instantly when the hand comes back.
            state.gesture_label = "Hand lost (holding)"
            state.cursor_status = "held (hand lost)"
            state.click_status = "held (hand lost)"
            return
        state.gesture_label = "No Hand Detected"
        state.cursor_status = "idle (no hand)"
        state.click_status = "locked (no hand)"
        stop_drag(state)
        state.reset_hand_tracking()
        return
    state.hand_lost_since = None

    index_finger_tip = landmark_list[INDEX_TIP]

    # Distances are rescaled so a hand of any apparent size (near/far from
    # the camera) behaves like one of HAND_REFERENCE_SIZE.
    hand_size = util.get_distance([landmark_list[WRIST], landmark_list[MIDDLE_BASE]])
    scale = util.clamp(HAND_REFERENCE_SIZE / hand_size, HAND_SCALE_MIN, HAND_SCALE_MAX) if hand_size and hand_size > 1 else 1.0
    thumb_index_dist = util.get_distance([landmark_list[THUMB_TIP], landmark_list[INDEX_BASE]]) * scale
    pinch_dist = util.get_distance([landmark_list[THUMB_TIP], landmark_list[INDEX_TIP]]) * scale
    d_middle = util.get_distance([landmark_list[THUMB_TIP], landmark_list[MIDDLE_TIP]]) * scale
    d_ring = util.get_distance([landmark_list[THUMB_TIP], landmark_list[RING_TIP]]) * scale
    d_pinky = util.get_distance([landmark_list[THUMB_TIP], landmark_list[PINKY_TIP]]) * scale
    state.pinch_distances = (pinch_dist, d_middle, d_ring, d_pinky)

    angles = finger_angles(landmark_list)
    # A thumb-index pinch only grabs when the index fingertip is (about) the one
    # the thumb is closest to. Otherwise it is a button pinch (middle / ring /
    # pinky) whose index finger merely happens to be near; reading that as a
    # grab would swallow the click and leave the button pressed.
    grab_dist = pinch_dist
    if not state.dragging and pinch_dist >= min(d_middle, d_ring, d_pinky) + GRAB_TIE:
        grab_dist = float("inf")
    raw_gesture = classify_gesture(
        angles,
        thumb_index_dist,
        grab_dist,
        moving=state.active_gesture == "Moving",
        dragging=state.dragging,
    )

    # Let go of a drag only after the pinch has been clearly open for a couple
    # of frames: one noisy frame (fast hand, jittery landmarks) must not drop
    # the window mid-drag.
    if state.dragging and raw_gesture != "Dragging":
        state.drag_release_frames += 1
        if state.drag_release_frames >= DRAG_RELEASE_FRAMES:
            stop_drag(state)
            state.drag_release_frames = 0
            if state.active_gesture == "Dragging":
                state.active_gesture = "None"
        else:
            raw_gesture = "Dragging"
    else:
        state.drag_release_frames = 0

    gesture = confirm_gesture(state, raw_gesture)
    state.gesture_label = gesture

    if gesture == "Moving":
        state.armed = True  # button pinches are ignored until the hand has shown the Move pose

    # Button pinches: thumb touches the middle (left click), ring (right click)
    # or pinky (screenshot) fingertip. Not while grabbing or dragging: with the
    # thumb-index pinch the other fingertips usually rest next to the thumb,
    # and reading that as a button would freeze the cursor or even drop the
    # window with a stray click.
    if raw_gesture == "Dragging" or state.dragging:
        state.pinch_open = {finger: False for finger, _ in CLICK_PINCHES}  # must be seen apart again after the drag
        state.pinch_frames = {finger: 0 for finger, _ in CLICK_PINCHES}
        state.pinch_history = {finger: deque() for finger, _ in CLICK_PINCHES}
        action = None
    else:
        action = update_click_pinches(
            state, angles, {"middle": d_middle, "ring": d_ring, "pinky": d_pinky}
        )
    state.click_status = describe_click_state(state, angles, raw_gesture)

    if action == "Screenshot":
        if now - state.last_screenshot_time > SCREENSHOT_COOLDOWN:
            take_screenshot(frame)
            state.last_screenshot_time = now
            state.click_flash = CLICK_FLASH[action]
            state.click_flash_until = now + SCREENSHOT_FLASH_SECONDS
    elif action is not None and now - state.last_click_time > CLICK_COOLDOWN:
        button = Button.left if action == "Left Click" else Button.right
        state.mouse.press(button)
        state.mouse.release(button)
        state.last_click_time = now
        state.click_flash = CLICK_FLASH[action]
        state.click_flash_until = now + CLICK_FLASH_SECONDS

    # While a click pinch is closing, and for a moment after it fires, the
    # cursor stays put so the pinch can't drag the click away from its target.
    # The hold is time-limited: a pinch merely relaxed (not fully opened) must
    # not leave the cursor stuck.
    cursor_frozen = (state.pinch_hold is not None and now < state.pinch_hold_until) or any(
        state.pinch_frames.values()
    )

    state.cursor_status = f"idle (pose: {gesture})"
    if gesture == "Dragging":
        start_drag(state)
        if cursor_frozen:
            state.cursor_status = "held (click pinch)"
        else:
            move_mouse(state, index_finger_tip)  # allow dragging while moving
            state.cursor_status = "following (dragging)"

    elif gesture == "Moving":
        if raw_gesture == "Dragging":
            # The thumb-index pinch is closing but the grab isn't confirmed yet.
            # The index tip travels toward the thumb meanwhile, so hold the
            # cursor to grab what was aimed at, not where the fingertip ended up.
            state.cursor_status = "held (grabbing)"
        elif cursor_frozen:
            state.cursor_status = "held (click pinch)"
        else:
            move_mouse(state, index_finger_tip)
            state.cursor_status = "following"

    elif gesture == "Scrolling":
        handle_scroll(state, landmark_list, frame)
        state.cursor_status = "scrolling"

    label = GESTURE_LABELS.get(gesture)
    if label is not None:
        draw_gesture_label(frame, *label)
    if now < state.click_flash_until:
        draw_gesture_label(frame, *state.click_flash, y=90)

    # prev_cursor is kept across clicks/other gestures (only reset when the
    # hand is lost), so the cursor glides back instead of jumping.
    if gesture != "Scrolling":
        state.prev_scroll_y = None


# --------------------------------------------------------------------------
# OVERLAY / DRAWING
# --------------------------------------------------------------------------
def draw_tracking_visuals(frame, hand_landmarks, landmark_list, mp_draw, connections):
    """Draw hand landmarks, connections, thumb/index tips, and a tracking box."""
    mp_draw.draw_landmarks(frame, hand_landmarks, connections)
    frame_h, frame_w = frame.shape[:2]

    if len(landmark_list) >= 9:
        thumb_px = (int(landmark_list[4][0] * frame_w), int(landmark_list[4][1] * frame_h))
        index_px = (int(landmark_list[8][0] * frame_w), int(landmark_list[8][1] * frame_h))
        cv2.circle(frame, thumb_px, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, index_px, 10, (0, 255, 255), cv2.FILLED)

    xs = [lm[0] * frame_w for lm in landmark_list]
    ys = [lm[1] * frame_h for lm in landmark_list]
    if xs and ys:
        pad = 20
        x1, y1 = int(min(xs) - pad), int(min(ys) - pad)
        x2, y2 = int(max(xs) + pad), int(max(ys) + pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)


def draw_active_area(frame, margins=DEFAULT_MARGINS):
    """Outline the camera region the fingertip maps to the screen, so the hand can stay inside the frame."""
    frame_h, frame_w = frame.shape[:2]
    left, right, top, bottom = margins
    cv2.rectangle(frame, (left, top), (frame_w - right, frame_h - bottom), (150, 150, 0), 1)


def draw_notices(frame, state):
    """Calibration instructions and short messages (toggles, results)."""
    if state.calibration_step == 1:
        text = "Calibration 1/2: fingertip at the TOP-LEFT of your comfortable area, then press c"
    elif state.calibration_step == 2:
        text = "Calibration 2/2: fingertip at the BOTTOM-RIGHT of it, then press c"
        px, py = int(state.calibration_tl[0]), int(state.calibration_tl[1])
        cv2.circle(frame, (px, py), 8, (0, 200, 255), 2)
    elif time.time() < state.notice_until:
        text = state.notice
    else:
        return
    cv2.putText(frame, text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)


def draw_overlay(frame, fps, state, tracking_active):
    """Render the professional status overlay: FPS, gesture, cursor/tracking status, resolution."""
    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Gesture: {state.gesture_label}",
        f"Cursor: {'PAUSED' if state.paused else state.cursor_status}",
        f"Tracking: {'Locked' if tracking_active else 'Searching...'}",
        f"Click: {state.click_status}",
        f"Area: {'default' if state.margins == DEFAULT_MARGINS else 'custom'} | Accel: {'on' if state.accel_enabled else 'off'}",
        f"Screen Res: {SCREEN_WIDTH}x{SCREEN_HEIGHT}",
    ]
    if state.show_distances:
        d_index, d_middle, d_ring, d_pinky = state.pinch_distances
        overlay_lines.append(
            f"Thumb to idx/mid/ring/pinky: {d_index:.0f} / {d_middle:.0f} / {d_ring:.0f} / {d_pinky:.0f} "
            f"(click < {CLICK_PINCH_THRESHOLD}, open > {CLICK_PINCH_RELEASE})"
        )

    y = frame.shape[0] - 15 - (len(overlay_lines) - 1) * 22
    for line in overlay_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# --------------------------------------------------------------------------
# GLOBAL HOTKEYS
# --------------------------------------------------------------------------
def start_hotkeys(state):
    """
    Register the global pause/quit hotkeys. They work regardless of which
    window has focus, which matters because the hand is driving the cursor
    (FAILSAFE is off). Returns the listener, or None if hotkeys are unavailable.
    """

    def toggle_pause():
        state.paused = not state.paused
        log.info("Cursor control %s.", "paused" if state.paused else "resumed")

    def request_quit():
        state.quit_requested = True

    try:
        listener = keyboard.GlobalHotKeys({HOTKEY_PAUSE: toggle_pause, HOTKEY_QUIT: request_quit})
        listener.start()
    except Exception as exc:
        log.warning("Global hotkeys unavailable (%s); quit with 'q' in the video window.", exc)
        return None

    log.info("Hotkeys: %s = pause/resume, %s = quit.", HOTKEY_PAUSE, HOTKEY_QUIT)
    return listener


# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if sys.platform == "win32" and not disable_power_throttling():
        log.warning("Could not opt out of Windows power throttling; tracking may lag while the window is minimized.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        max_num_hands=1,
    )

    # DirectShow opens webcams noticeably faster on Windows than the default backend.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    state = AppState(mouse=Controller())
    hotkey_listener = start_hotkeys(state)

    prev_frame_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 30  # ~1s of dropped frames at 30 FPS before giving up
    last_frame_error = None
    consecutive_frame_errors = 0
    last_stall_log = 0.0
    capture_exclusion_tried = False
    calibration_checked = False

    try:
        if not cap.isOpened():
            log.error("Could not open the camera. Check that it is connected and not in use by another app.")
            return

        while not state.quit_requested:
            loop_start = time.perf_counter()
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                log.warning("Failed to read frame from camera; retrying...")
                if consecutive_failures > max_consecutive_failures:
                    log.error("Camera appears disconnected. Exiting gracefully.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            read_done = time.perf_counter()

            try:
                frame = cv2.flip(frame, 1)
                state.frame_h, state.frame_w = frame.shape[:2]
                if not calibration_checked:
                    # Needs the real frame size, so it can only be done once a frame arrived.
                    calibration_checked = True
                    saved_margins = load_calibration(state.frame_w, state.frame_h)
                    if saved_margins is not None:
                        state.margins = saved_margins
                        log.info("Loaded your calibrated area from %s (press r to reset it).", CALIBRATION_FILE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = hands.process(frame_rgb)
                track_done = time.perf_counter()
                draw_active_area(frame, state.margins)

                landmark_list = []
                tracking_active = False

                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    draw_tracking_visuals(frame, hand_landmarks, landmark_list, mp_draw, mp_hands.HAND_CONNECTIONS)
                    tracking_active = True
                state.index_tip = landmark_list[INDEX_TIP] if landmark_list else None

                if state.paused or state.calibration_step:
                    # Keep showing the video but don't touch the mouse (paused, or
                    # the hand is busy marking the corners of the calibration).
                    stop_drag(state)
                    state.reset_hand_tracking()
                    state.gesture_label = "Calibrating" if state.calibration_step and not state.paused else "Paused"
                else:
                    # With no hand, landmark_list is empty and detect_gesture
                    # releases any drag and resets the tracking state.
                    detect_gesture(frame, landmark_list, state)

                now = time.time()
                elapsed = now - prev_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                prev_frame_time = now

                draw_overlay(frame, fps, state, tracking_active)
                draw_notices(frame, state)
                cv2.imshow(WINDOW_NAME, frame)

                if not capture_exclusion_tried:
                    # The window only exists after the first imshow; try once.
                    capture_exclusion_tried = True
                    if not exclude_window_from_capture(WINDOW_NAME):
                        log.info("Could not hide the camera window from screenshots; it may appear in them.")

                frame_done = time.perf_counter()
                if frame_done - loop_start > STALL_LOG_SECONDS and time.time() - last_stall_log > 1.0:
                    # Where did the time go? Camera wait = the webcam delivered late
                    # (often low light -> low FPS); tracking = MediaPipe; rest = our code.
                    last_stall_log = time.time()
                    log.warning(
                        "Slow frame: %.0f ms (camera wait %.0f, hand tracking %.0f, gestures+drawing %.0f) | gesture=%s cursor=%s",
                        (frame_done - loop_start) * 1000,
                        (read_done - loop_start) * 1000,
                        (track_done - read_done) * 1000,
                        (frame_done - track_done) * 1000,
                        state.gesture_label,
                        state.cursor_status,
                    )

                consecutive_frame_errors = 0

            except Exception as exc:
                # Defensive: never let a single bad frame crash the app. The
                # traceback is logged once per distinct error, not every frame.
                if repr(exc) != last_frame_error:
                    log.exception("Skipped a frame due to an error")
                    last_frame_error = repr(exc)

                # A persistent error would otherwise leave a frozen window that
                # looks alive; give up instead of failing silently forever.
                consecutive_frame_errors += 1
                if consecutive_frame_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                    log.error(
                        "%d frames in a row failed (last error: %s). Exiting.",
                        consecutive_frame_errors,
                        last_frame_error,
                    )
                    break

            key = read_key()
            if key == ord("q"):
                break
            handle_key(state, key)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        stop_drag(state)
        if hotkey_listener is not None:
            hotkey_listener.stop()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
