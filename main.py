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

import logging
import os
import sys
import time
from dataclasses import dataclass
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
FRAME_MARGIN = 100              # px inset from the camera frame edges used when mapping to the screen
CLICK_COOLDOWN = 0.5            # seconds between left/right clicks
DOUBLE_CLICK_COOLDOWN = 1.0     # seconds between double clicks
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
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
SCREENSHOT_DIR = "Screenshots"
WINDOW_NAME = "AI Virtual Mouse"

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False  # gestures already stay within the frame; avoid corner-triggered exceptions
pyautogui.PAUSE = 0         # default is 0.1s after every call, which caps cursor movement at ~10 FPS

# Landmark indices (MediaPipe Hands): wrist, thumb tip, index base/tip, middle base.
WRIST, THUMB_TIP, INDEX_BASE, INDEX_TIP, MIDDLE_BASE = 0, 4, 5, 8, 9

# On-screen label (text, BGR color) drawn for each active gesture.
GESTURE_LABELS = {
    "Dragging": ("Gesture : Dragging", (0, 200, 255)),
    "Moving": ("Gesture : Moving", (0, 255, 0)),
    "Left Click": ("Gesture : Left Click", (0, 255, 0)),
    "Right Click": ("Gesture : Right Click", (0, 0, 255)),
    "Double Click": ("Gesture : Double Click", (255, 255, 0)),
}
SCROLL_LABEL_COLOR = (200, 0, 200)


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
    last_double_click_time: float = 0.0
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

    def reset_hand_tracking(self):
        """Forget per-hand tracking (called when the hand leaves the frame)."""
        self.armed = False
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


def is_move_gesture(angles, thumb_index_dist):
    """Index finger extended while the thumb rests near the index base."""
    return thumb_index_dist < THUMB_NEAR_DIST and is_extended(angles.index)


def is_left_click(angles, thumb_index_dist):
    return is_bent(angles.index) and is_extended(angles.middle) and thumb_index_dist > THUMB_NEAR_DIST


def is_right_click(angles, thumb_index_dist):
    return is_bent(angles.middle) and is_extended(angles.index) and thumb_index_dist > THUMB_NEAR_DIST


def is_double_click(angles, thumb_index_dist):
    return is_bent(angles.index) and is_bent(angles.middle) and thumb_index_dist > THUMB_NEAR_DIST


def is_screenshot(angles, thumb_index_dist):
    return is_bent(angles.index) and is_bent(angles.middle) and thumb_index_dist < THUMB_NEAR_DIST


def is_pinching(pinch_dist):
    """Thumb tip close to index tip (hand-size-normalized) -> used to trigger drag & drop."""
    return pinch_dist < PINCH_THRESHOLD


def is_scroll_pose(angles):
    """Index, middle and ring fingers extended while the pinky is curled."""
    return (
        is_extended(angles.index)
        and is_extended(angles.middle)
        and is_extended(angles.ring)
        and is_bent(angles.pinky)
    )


def classify_gesture(angles, thumb_index_dist, pinch_dist):
    """Return the name of the gesture matching this single frame (no debouncing)."""
    if is_pinching(pinch_dist):
        return "Dragging"
    # Scroll is checked before Move: it is the more specific pose (it needs the
    # middle and ring fingers extended too), and Move only looks at the index
    # finger and thumb, so with the thumb resting near the index base an
    # open-hand scroll pose would otherwise always be read as Move.
    if is_scroll_pose(angles):
        return "Scrolling"
    if is_move_gesture(angles, thumb_index_dist):
        return "Moving"
    if is_left_click(angles, thumb_index_dist):
        return "Left Click"
    if is_right_click(angles, thumb_index_dist):
        return "Right Click"
    if is_double_click(angles, thumb_index_dist):
        return "Double Click"
    if is_screenshot(angles, thumb_index_dist):
        return "Screenshot"
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


# --------------------------------------------------------------------------
# CURSOR / ACTION HANDLERS
# --------------------------------------------------------------------------
def compute_target_cursor(index_finger_tip, frame_w, frame_h):
    """
    Map the (normalized) index-finger-tip position (x, y) to full-monitor
    screen coordinates using numpy interpolation, insetting the usable
    camera frame by FRAME_MARGIN so the whole screen is reachable
    without requiring the finger to reach the extreme edges.

    frame_w / frame_h are the real frame dimensions, since the camera
    may not honor the requested CAMERA_WIDTH / CAMERA_HEIGHT.
    """
    raw_x = index_finger_tip[0] * frame_w
    raw_y = index_finger_tip[1] * frame_h

    screen_x = util.interpolate(
        raw_x, [FRAME_MARGIN, frame_w - FRAME_MARGIN], [0, SCREEN_WIDTH]
    )
    screen_y = util.interpolate(
        raw_y, [FRAME_MARGIN, frame_h - FRAME_MARGIN], [0, SCREEN_HEIGHT]
    )

    screen_x = util.clamp(screen_x, 0, SCREEN_WIDTH - 1)
    screen_y = util.clamp(screen_y, 0, SCREEN_HEIGHT - 1)
    return screen_x, screen_y


def move_mouse(state, index_finger_tip):
    """Smoothly move the OS cursor toward the mapped target position."""
    target = compute_target_cursor(index_finger_tip, state.frame_w, state.frame_h)
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


def take_screenshot(frame):
    """Save a timestamped screenshot into the Screenshots/ directory."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(SCREENSHOT_DIR, f"Screenshot_{timestamp}.png")
    try:
        image = pyautogui.screenshot()
        image.save(filename)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to save screenshot: %s", exc)
    cv2.putText(frame, "Screenshot Taken", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def start_drag(state):
    if not state.dragging:
        state.mouse.press(Button.left)
        state.dragging = True


def stop_drag(state):
    if state.dragging:
        state.mouse.release(Button.left)
        state.dragging = False


def draw_gesture_label(frame, text, color):
    cv2.putText(frame, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


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
        state.gesture_label = "No Hand Detected"
        stop_drag(state)
        state.reset_hand_tracking()
        return

    index_finger_tip = landmark_list[INDEX_TIP]

    # Distances are rescaled so a hand of any apparent size (near/far from
    # the camera) behaves like one of HAND_REFERENCE_SIZE.
    hand_size = util.get_distance([landmark_list[WRIST], landmark_list[MIDDLE_BASE]])
    scale = util.clamp(HAND_REFERENCE_SIZE / hand_size, HAND_SCALE_MIN, HAND_SCALE_MAX) if hand_size and hand_size > 1 else 1.0
    thumb_index_dist = util.get_distance([landmark_list[THUMB_TIP], landmark_list[INDEX_BASE]]) * scale
    pinch_dist = util.get_distance([landmark_list[THUMB_TIP], landmark_list[INDEX_TIP]]) * scale

    raw_gesture = classify_gesture(finger_angles(landmark_list), thumb_index_dist, pinch_dist)

    # Release an active drag as soon as the pinch is let go, without waiting
    # for the debounce.
    if state.dragging and raw_gesture != "Dragging":
        stop_drag(state)
        if state.active_gesture == "Dragging":
            state.active_gesture = "None"

    previous_gesture = state.active_gesture
    gesture = confirm_gesture(state, raw_gesture)
    just_started = gesture != previous_gesture
    state.gesture_label = gesture

    if gesture == "Moving":
        state.armed = True
    # Click-type gestures are ignored until the hand has shown the Move pose.
    just_started = just_started and state.armed

    if gesture == "Dragging":
        start_drag(state)
        move_mouse(state, index_finger_tip)  # allow dragging while moving

    elif gesture == "Moving":
        move_mouse(state, index_finger_tip)

    elif gesture == "Scrolling":
        handle_scroll(state, landmark_list, frame)

    elif gesture == "Left Click":
        if just_started and now - state.last_click_time > CLICK_COOLDOWN:
            state.mouse.press(Button.left)
            state.mouse.release(Button.left)
            state.last_click_time = now

    elif gesture == "Right Click":
        if just_started and now - state.last_click_time > CLICK_COOLDOWN:
            state.mouse.press(Button.right)
            state.mouse.release(Button.right)
            state.last_click_time = now

    elif gesture == "Double Click":
        if just_started and now - state.last_double_click_time > DOUBLE_CLICK_COOLDOWN:
            pyautogui.doubleClick()
            state.last_double_click_time = now

    elif gesture == "Screenshot":
        if just_started and now - state.last_screenshot_time > SCREENSHOT_COOLDOWN:
            take_screenshot(frame)
            state.last_screenshot_time = now

    label = GESTURE_LABELS.get(gesture)
    if label is not None:
        draw_gesture_label(frame, *label)

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


def draw_overlay(frame, fps, state, tracking_active):
    """Render the professional status overlay: FPS, gesture, cursor/tracking status, resolution."""
    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Gesture: {state.gesture_label}",
        f"Cursor: {'PAUSED' if state.paused else 'Dragging' if state.dragging else 'Active' if tracking_active else 'Idle'}",
        f"Tracking: {'Locked' if tracking_active else 'Searching...'}",
        f"Clicks: {'Armed' if state.armed else 'Locked (show Move pose first)'}",
        f"Screen Res: {SCREEN_WIDTH}x{SCREEN_HEIGHT}",
    ]

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
    capture_exclusion_tried = False

    try:
        if not cap.isOpened():
            log.error("Could not open the camera. Check that it is connected and not in use by another app.")
            return

        while not state.quit_requested:
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

            try:
                frame = cv2.flip(frame, 1)
                state.frame_h, state.frame_w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = hands.process(frame_rgb)

                landmark_list = []
                tracking_active = False

                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    draw_tracking_visuals(frame, hand_landmarks, landmark_list, mp_draw, mp_hands.HAND_CONNECTIONS)
                    tracking_active = True

                if state.paused:
                    # Keep showing the video but don't touch the mouse.
                    stop_drag(state)
                    state.reset_hand_tracking()
                    state.gesture_label = "Paused"
                else:
                    # With no hand, landmark_list is empty and detect_gesture
                    # releases any drag and resets the tracking state.
                    detect_gesture(frame, landmark_list, state)

                now = time.time()
                elapsed = now - prev_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                prev_frame_time = now

                draw_overlay(frame, fps, state, tracking_active)
                cv2.imshow(WINDOW_NAME, frame)

                if not capture_exclusion_tried:
                    # The window only exists after the first imshow; try once.
                    capture_exclusion_tried = True
                    if not exclude_window_from_capture(WINDOW_NAME):
                        log.info("Could not hide the camera window from screenshots; it may appear in them.")

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

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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
