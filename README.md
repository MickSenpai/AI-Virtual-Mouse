<p align="center">
  <img alt="AI Virtual Mouse banner" src="images/title_image.png" width="100%">
</p>

# AI Virtual Mouse

Control your computer's cursor with nothing but your hand and a webcam. AI Virtual Mouse uses [MediaPipe Hands](https://developers.google.com/mediapipe) and OpenCV to track your hand in real time and translate a small vocabulary of gestures into full mouse functionality — movement, left/right click, double click, drag-and-drop, scrolling, and screenshots.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Hands-00A98F">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Gesture Guide](#gesture-guide)
- [Demo Overlay](#demo-overlay)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- 🖱️ **Full mouse functionality** — move, left click, right click, double click, drag & drop, scroll up/down, and screenshot capture
- 🎯 **Smooth, jitter-free cursor** — Exponential Moving Average (EMA) smoothing plus a configurable dead zone
- 🖥️ **Full-screen coordinate mapping** — reach every corner of your monitor without stretching your hand to the edge of the camera frame
- ⏱️ **Gesture cooldowns** — debounced clicks and screenshots so one gesture never fires multiple times
- 📸 **Timestamped screenshots** — auto-saved to a `Screenshots/` folder that's created for you
- 📊 **Live on-screen overlay** — FPS, current gesture, cursor status, tracking status, and screen resolution
- 🛡️ **Crash-resistant** — gracefully handles a missing camera, dropped frames, and a hand leaving the frame
- 🪶 **Minimal footprint** — just two Python files and a requirements list, no heavyweight framework or model training required

## How It Works

```
Webcam → MediaPipe Hand Detection → 21-Point Landmarks → Gesture Classification → Action Dispatch → OS Mouse Control
```

<p align="center">
  <img alt="Working structure / system architecture" src="images/system_architecture.png" width="85%">
</p>
<p align="center"><em>Figure — End-to-end working structure of the AI Virtual Mouse pipeline.</em></p>

1. **Capture** — OpenCV reads frames from your webcam and mirrors them for a natural, "look in a mirror" experience.
2. **Detect** — MediaPipe Hands locates a hand in the frame and returns 21 normalized landmark coordinates.
3. **Classify** — Simple, explainable geometry (joint angles and inter-landmark distances) determines which gesture, if any, is being made.
4. **Act** — The matched gesture triggers a cursor move, click, drag, scroll, or screenshot via PyAutoGUI/pynput.
5. **Display** — A diagnostic overlay is drawn on the video feed so you can see exactly what the system is detecting.

## Gesture Guide

<p align="center">
  <img alt="Gesture guide reference" src="images/gesture_guide.png" width="85%">
</p>
<p align="center"><em>Figure — Hand poses for each supported gesture.</em></p>

| Gesture | Hand Pose | Action |
|---|---|---|
| **Move** | Index finger extended, thumb resting near the base of the index finger | Cursor follows your fingertip |
| **Left Click** | Keep the index finger pointing and touch your thumb tip to your middle fingertip | Single left click (the cursor stays put while you pinch) |
| **Right Click** | Keep the index finger pointing and touch your thumb tip to your ring fingertip | Single right click |
| **Double Click** | Both index and middle fingers bent | OS double-click |
| **Drag & Drop** | Pinch thumb tip and index tip together, move, then release | Holds left button while pinched, releases on open |
| **Scroll Up / Down** | Index, middle, and ring fingers extended; pinky curled — move hand up or down | Scrolls the active window |
| **Screenshot** | Index and middle fingers bent, thumb close to the base of the index finger | Saves a timestamped screenshot |

> 👆 **Click pinches:** a pinch only counts after the fingertips were seen *apart* first, so start with the middle (or ring) finger away from your thumb, then touch them. A hand that rests with the thumb already touching a curled finger will not click by itself. Press **`d`** in the video window to show the live thumb-to-fingertip distances while you tune `CLICK_PINCH_THRESHOLD` / `CLICK_PINCH_RELEASE`.

> 💡 Tip: Keep gestures deliberate and hold them briefly — this gives MediaPipe a stable frame to classify and avoids accidental misfires.

## Demo Overlay

While running, the app overlays live diagnostics directly on the video feed:

```
FPS: 28.4
Gesture: Moving
Cursor: Active
Tracking: Locked
Screen Res: 1920x1080
```

## Requirements

- Python 3.9 or later
- A working webcam
- Windows, macOS, or Linux with a desktop environment (needed for cursor/screenshot control)

Python dependencies (see [`requirements.txt`](requirements.txt)):

| Package | Purpose |
|---|---|
| `opencv-python` | Video capture, frame processing, overlay rendering |
| `mediapipe` | Real-time hand landmark detection |
| `pyautogui` | Cursor movement, double-click, screenshots |
| `pynput` | Cross-platform mouse press/release and scroll events |
| `numpy` | Angle computation and coordinate interpolation |

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/muhammaddaniyal0020/AI-Virtual-Mouse
cd AI-Virtual-Mouse

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

- A window titled **"AI Virtual Mouse"** will open showing your webcam feed with hand landmarks and the status overlay.
- Perform any gesture from the [Gesture Guide](#gesture-guide) to control your cursor.
- The faint rectangle in the video is the area your **fingertip** sweeps to cover the whole screen. Keep your fingertip inside it and your wrist inside the frame; a partly visible hand is tracked poorly.
- Press **`q`** with the video window focused to quit, or **`d`** to toggle the pinch-distance readout.

Because your hand is driving the cursor, there are also **global hotkeys** that work no matter which window has focus:

| Hotkey | Action |
|---|---|
| `Ctrl` + `Alt` + `P` | Pause / resume cursor control (the video keeps running) |
| `Ctrl` + `Alt` + `Q` | Quit |

**Clicks are locked until you show the Move pose.** Whenever your hand enters the frame (or control is resumed), click, double-click and screenshot gestures are ignored until the system has seen the Move gesture once — so a hand entering the frame with curled fingers can't click by accident. The overlay shows `Clicks: Armed` or `Locked`.

Screenshots are saved automatically to a `Screenshots/` folder created in the project directory, named like `Screenshot_2025-07-17_14-32-05.png`.

## Configuration

All tunable behavior lives as named constants at the top of `main.py` — no need to touch the gesture logic to adjust feel or performance.

| Constant | Default | Description |
|---|---|---|
| `SMOOTHING` | `0.7` | EMA smoothing factor for the cursor (higher = smoother but laggier) |
| `FRAME_MARGIN_LEFT` / `_RIGHT` / `_TOP` / `_BOTTOM` | `100` / `100` / `40` / `200` px | Inset of the camera region that the fingertip maps to the full screen. The bottom margin is large so reaching the bottom of the screen doesn't push your wrist out of the frame |
| `CLICK_COOLDOWN` | `0.5` s | Minimum interval between left/right clicks |
| `CLICK_PINCH_THRESHOLD` | `40` | Thumb-to-middle/ring-fingertip distance below which a click pinch closes (normalized by hand size) |
| `CLICK_PINCH_RELEASE` | `70` | Distance above which the pinch counts as open again (hysteresis; must be larger than the threshold) |
| `CLICK_PINCH_FRAMES` | `2` | Consecutive frames the pinch must stay closed to click |
| `CLICK_FREEZE_SECONDS` | `0.3` s | After a pinch click the cursor stays put for this long (or until the pinch opens), so the click lands where you aimed |
| `MOVE_STAY_ANGLE` / `MOVE_STAY_THUMB_DIST` | `70°` / `90` | Looser limits that keep the Move pose alive once it is active, so a thumb that drifts or an index that dips a little doesn't stop the cursor |
| `STALL_LOG_SECONDS` | `0.15` s | Frames slower than this are logged with a breakdown (camera wait / hand tracking / gestures + drawing) |
| `HAND_LOST_GRACE` | `0.3` s | When tracking drops, the cursor, drag and gesture state are held this long before resetting, so brief losses near the frame edge don't interrupt control |
| `DOUBLE_CLICK_COOLDOWN` | `1.0` s | Minimum interval between double clicks |
| `SCREENSHOT_COOLDOWN` | `2.0` s | Minimum interval between screenshots |
| `SCROLL_COOLDOWN` | `0.15` s | Minimum interval between individual scroll ticks |
| `SCROLL_PIXELS_PER_TICK` | `8` px | Vertical hand movement per scroll tick (moving faster scrolls more) |
| `SCROLL_MAX_TICKS` | `5` | Maximum scroll ticks applied in a single step |
| `DEAD_ZONE` | `4` px | The cursor ignores fingertip movement smaller than this, to kill jitter |
| `SCROLL_DEAD_ZONE_PX` | `8` px | Minimum vertical movement required to register a scroll tick |
| `PINCH_THRESHOLD` | `40` | Thumb-tip/index-tip distance below which a pinch (drag) is registered (normalized by hand size) |
| `HAND_REFERENCE_SIZE` | `225` | Wrist-to-middle-knuckle distance the distance thresholds were tuned for; distances are rescaled to it so gestures work at any camera distance |
| `HAND_SCALE_MIN` / `HAND_SCALE_MAX` | `0.5` / `2.0` | Limits for the hand-size rescaling factor, so a foreshortened or very distant hand can't distort the distance thresholds |
| `HOTKEY_PAUSE` / `HOTKEY_QUIT` | `<ctrl>+<alt>+p` / `<ctrl>+<alt>+q` | Global hotkeys ([pynput format](https://pynput.readthedocs.io/en/latest/keyboard.html#global-hotkeys)) |
| `MAX_CONSECUTIVE_FRAME_ERRORS` | `30` | Frames in a row that may fail processing before the app logs the error and exits (instead of leaving a frozen window) |
| `GESTURE_CONFIRM_FRAMES` | `3` | Consecutive frames a gesture must be seen before it activates; clicks fire once per gesture |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640 × 480` | Requested webcam capture resolution |
| `MIN_DETECTION_CONFIDENCE` / `MIN_TRACKING_CONFIDENCE` | `0.7` / `0.7` | MediaPipe Hands confidence thresholds |

## Project Structure

```
AI-Virtual_Mouse/
  ├── main.py                     # App entry point: video loop, gesture dispatch, OS control, overlay
  ├── util.py                     # Pure helper functions: angles, distances, interpolation, EMA smoothing
  ├── requirements.txt            # Python dependencies
  ├── README.md                   # This file
  ├── images/
  │     ├── title_image.png         # Title banner
  │     ├── system_architecutre.png # System architecture / pipeline diagram
  │     └── gesture-guide.png       # Reference image for the Gesture Guide section
  └── docs/
        └── Project_Report.docx     # Full project report
```

`util.py` has no dependency on OpenCV, MediaPipe, or OS-control libraries, so its functions can be unit-tested in isolation. The tests live in `tests/`; run them with:

```bash
pip install -r requirements-dev.txt
pytest
```

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `[ERROR] Could not open the camera` | Camera is in use by another app, or index `0` isn't your camera | Close other apps using the camera; try `cv2.VideoCapture(1)` |
| Cursor is jittery | Poor lighting or a noisy webcam | Increase `SMOOTHING` or `DEAD_ZONE` in `main.py` |
| Clicks fire by accident | A gesture is briefly misread for one frame | Increase `GESTURE_CONFIRM_FRAMES`, or `CLICK_COOLDOWN` / `DOUBLE_CLICK_COOLDOWN` |
| Cursor doesn't reach screen edges | The `FRAME_MARGIN_*` region is too large for your reach | Lower the margins (but keep the wrist inside the frame) |
| Clicks don't register | The click pinch never "opened", or the distances differ for your hand | Press `d` to see the live distances and adjust `CLICK_PINCH_THRESHOLD` / `CLICK_PINCH_RELEASE`; start each click with the finger away from the thumb |
| Gestures aren't recognized | Hand too close/far, poor lighting, or busy background | Improve lighting, use a plain background, stay ~40–70 cm from the camera |
| Tracking lags badly when the window is minimized or behind other apps | Windows throttles background processes (slower CPU, coarser timers) | Handled automatically on Windows 10 1709+/11 (the app opts out of the throttling and never sleeps in its event loop). If it still lags, check *Power mode* / battery saver and that the process isn't in Task Manager's *Efficiency mode* |
| The cursor freezes at times | See the `Cursor:` line in the overlay: `held (click pinch)`, `held (hand lost)`, `idle (pose: …)` or `idle (no hand)` tell you why it isn't following your finger. A `[WARNING] Slow frame: …` line in the terminal shows whether the time went to the camera (low light makes webcams drop their frame rate), MediaPipe, or the rest | Fix whichever the readout points at: improve lighting, keep the hand fully in frame, or loosen the thresholds above |
| Low FPS | Camera resolution too high, or CPU-bound machine | Lower `CAMERA_WIDTH`/`CAMERA_HEIGHT`, or reduce MediaPipe `model_complexity` |

## Roadmap

- [ ] Adaptive/calibratable gesture thresholds per user
- [ ] Two-hand support with independent gesture roles
- [ ] On-screen gesture remapping UI
- [ ] Optional depth-camera support for more robust pinch detection
- [ ] Standalone packaged executables (Windows/macOS/Linux)

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

Please keep `util.py` free of OpenCV/MediaPipe/OS-control dependencies so its helpers stay easily testable.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Google MediaPipe](https://developers.google.com/mediapipe) — real-time hand landmark detection
- [OpenCV](https://opencv.org/) — video capture and image processing
- [PyAutoGUI](https://pyautogui.readthedocs.io/) & [pynput](https://pynput.readthedocs.io/) — OS-level cursor and mouse control
