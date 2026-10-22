"""Gesture HMI STEP 19 v3b: v3a voice stability + portable camera/model paths.

STEP19 v4 Windows EXE uses NumPy-only MLP with Python-free user installation.
STEP19 v3c one-click keeps STEP19 v3a gestures and voice-hold hysteresis intact.
The MLP inference uses NumPy weights (PyTorch not required at runtime).
Camera index/backend auto-scan from STEP19 Camera AutoShare; portable resources.


STEP18 v6 YouTube autocomplete fix, STEP17 media circle / STANDBY, Handy retained.

STEP17 media circle, STANDBY/LOCKED and STEP15 features preserved.

Handy v0.9.8 settings (IMPORTANT):
  Shortcut: Ctrl+Space
  Shortcut behavior: LONG PRESS / 길게 누르기 (NOT Toggle / 전환)
  Language: Korean; this program auto-starts Handy when installed.

Workflow: PEACE 1.0s -> Windows mouse ON; select a text field with the pointer
and click. Hold a THUMBS-UP ~0.45s -> begin recording; speak while maintaining
THUMBS-UP. Relax/lower thumb for ~0.55s -> release Ctrl+Space and transcribe.
After a bounded processing grace, pointer automatically resumes without PEACE.
Pinch + RIGHTWARD movement + release -> Enter/submit to focused browser field.

Uses the existing HaGRID MLP, MediaPipe, mouse, click, scroll, back, PAUSED
handling, and 0.35 click threshold; scroll clamps to 240..720 ticks.

Run from any directory (no cd required):
  python C:/path/to/Gesture_HMI/gesture_pointer_step19_v3b_portable.py --self-test
  python C:/path/to/Gesture_HMI/gesture_pointer_step19_v3b_portable.py --camera-test
  python C:/path/to/Gesture_HMI/gesture_pointer_step19_v3b_portable.py

Keyboard: ESC emergency OFF; Q quit; S beep ON/OFF; 1/2/3 pointer region;
  [ / ] swipe distance; Z / X scroll sensitivity (bounds stay 240..720); , / . click sensitivity;
  - / = pinch sensitivity; B switch back-key method; R reset pointer.
"""
from pathlib import Path
from collections import deque
import threading
import time
import sys
import urllib.request
import os
import shutil
import subprocess
import traceback

import numpy as np

# ROOT is the user-writable install folder, not the current working directory.
# Source mode: beside the .py; PyInstaller one-folder: beside the .exe.
# In bundled mode, actual --add-data resources can be in sys._MEIPASS.
ROOT = (Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False)
        else Path(__file__).resolve().parent)
BUNDLE_ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')


def _resource_candidates(relative_names, env_var=None):
    candidates = []
    if env_var:
        user_path = os.environ.get(env_var, '').strip().strip('\"')
        if user_path:
            candidates.append(Path(user_path).expanduser())
    for base in (ROOT, BUNDLE_ROOT):
        for relative in relative_names:
            candidates.append(base / relative)
    # Remove duplicates, preserve the order (installed files before bundled files).
    return list(dict.fromkeys(candidates))


def resource_locations():
    mlp = _resource_candidates(('models_v2/gesture_mlp.pth',
                                 'assets/models_v2/gesture_mlp.pth',
                                 'assets/gesture_mlp.pth'), 'GESTURE_HMI_MLP_PATH')
    hand = _resource_candidates(('hand_landmarker.task',
                                  'assets/hand_landmarker.task'),
                                 'GESTURE_HMI_HAND_MODEL_PATH')
    local = os.environ.get('LOCALAPPDATA')
    if local:
        hand.append(Path(local) / 'GestureHMI' / 'models' / 'hand_landmarker.task')
    else:
        hand.append(Path.home() / '.gesture_hmi' / 'models' / 'hand_landmarker.task')
    return mlp, list(dict.fromkeys(hand))


def find_mlp_model():
    paths, _ = resource_locations()
    for path in paths:
        if path.is_file():
            print(f'MLP MODEL: {path}')
            return path
    raise FileNotFoundError(
        'Missing v2 trained model (best_epoch=51). Checked:\n  '
        + '\n  '.join(map(str, paths))
        + '\nKeep models_v2/gesture_mlp.pth alongside the HMI script, '
          'or set GESTURE_HMI_MLP_PATH.')


def find_numpy_mlp_model():
    """Use the independently converted v2 model without requiring PyTorch."""
    candidates = _resource_candidates(
        ('models_v2/gesture_mlp_v2.npz',
         'assets/models_v2/gesture_mlp_v2.npz',
         'assets/gesture_mlp_v2.npz'), 'GESTURE_HMI_NUMPY_MODEL_PATH')
    for candidate in candidates:
        if candidate.is_file():
            print(f'NUMPY MLP MODEL: {candidate}')
            return candidate
    raise FileNotFoundError('Missing NumPy MLP model. Expected models_v2/gesture_mlp_v2.npz; checked: '
                            + ', '.join(map(str, candidates)))


def load_numpy_mlp():
    """Load fixed 42->64->32->4 ReLU weights; no pickle/code execution."""
    path = find_numpy_mlp_model()
    with np.load(path, allow_pickle=False) as data:
        weights = [np.asarray(data[name], dtype=np.float32) for name in
                   ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
        mean = np.asarray(data['mean'], dtype=np.float32)
        std = np.asarray(data['std'], dtype=np.float32)
        clip = float(data['clip'])
        classes = [str(x) for x in data['classes']]
    if [a.shape for a in weights] != [(64,42),(64,),(32,64),(32,),(4,32),(4,)]:
        raise ValueError('Invalid NumPy MLP layer shapes')
    if mean.shape != (42,) or std.shape != (42,) or not np.all(np.isfinite(std)) or np.any(std == 0):
        raise ValueError('Invalid MLP feature normalization')
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError('Invalid MLP feature clip value')
    if set(classes) != {'peace','one','stop','no_gesture'} or len(classes)!=4:
        raise ValueError('Invalid MLP classes')
    if not all(np.all(np.isfinite(a)) for a in weights):
        raise ValueError('Nonfinite MLP weights')
    return weights, mean, std, clip, classes


def numpy_mlp_probs(feat, weights):
    w1,b1,w2,b2,w3,b3 = weights
    x = np.asarray(feat, dtype=np.float32)
    x = np.maximum(x @ w1.T + b1, 0)
    x = np.maximum(x @ w2.T + b2, 0)
    z = x @ w3.T + b3
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def find_hand_model(download=True):
    _, paths = resource_locations()
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            print(f'HAND MODEL: {path}')
            return path
    if not download:
        return None
    target = paths[-1]  # LOCALAPPDATA cache; works even in read-only install dirs.
    temp = target.with_suffix('.download')
    print('HAND MODEL: missing; downloading official MediaPipe hand_landmarker.task...')
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=25) as stream, open(temp, 'wb') as out:
            shutil.copyfileobj(stream, out)
        if temp.stat().st_size == 0:
            raise RuntimeError('Empty model download')
        temp.replace(target)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            'MediaPipe hand model not found and download failed: '
            f'{type(exc).__name__}: {exc}.\n'
            'Connect to internet once, or manually put hand_landmarker.task '
            f'beside the program. Official URL: {MODEL_URL}') from exc
    print(f'HAND MODEL DOWNLOADED: {target}')
    return target


def resource_diagnostic():
    print(f'PROGRAM DIRECTORY: {ROOT}')
    print(f'BUNDLED RESOURCE DIRECTORY: {BUNDLE_ROOT}')
    mlp_paths, hand_paths = resource_locations()
    numpy_paths = _resource_candidates(('models_v2/gesture_mlp_v2.npz',
                                      'assets/models_v2/gesture_mlp_v2.npz',
                                      'assets/gesture_mlp_v2.npz'),
                                      'GESTURE_HMI_NUMPY_MODEL_PATH')
    print('NumPy MLP model candidates:')
    for p in numpy_paths:
        print(f'  {"OK " if p.is_file() else "-- "} {p}')
    print('Original PyTorch checkpoint (training-only):')
    for p in mlp_paths:
        print(f'  {"OK " if p.is_file() else "-- "} {p}')
    print('Hand Landmarker candidates:')
    for p in hand_paths:
        print(f'  {"OK " if p.is_file() else "-- "} {p}')
    print('NOTE: Missing hand_landmarker.task is downloaded on normal startup; '
          '--resource-check does NOT download or change anything.')
    has_numpy = any(p.is_file() for p in numpy_paths)
    has_hand = any(p.is_file() and p.stat().st_size > 0 for p in hand_paths)
    return has_numpy and (not getattr(sys, 'frozen', False) or has_hand)

CONFIDENCE = 0.80
TRIGGER_HOLD_S = 1.00
STOP_HOLD_S = 0.60
NO_HAND_PAUSE_S = 0.50
NO_HAND_OFF_S = 3.00
HAND_RESUME_STABLE_S = 0.20
ONE_GRACE_S = 0.40
ONE_STABLE_S = 0.08
PINCH_ENTRY_ONE_S = 0.85
PINCH_RELEASE_HOLD_S = 0.08
PINCH_LOST_S = 0.50
PINCH_MAX_S = 3.0
# Tiny but deliberate circle: height-normalized geometric footprint; rough ellipses allowed.
MEDIA_CIRCLE_MIN_ANGLE_DEG = 300.0
MEDIA_CIRCLE_MAX_ANGLE_DEG = 560.0
MEDIA_CIRCLE_MIN_DIAMETER = 0.045  # ~22px at 480p; X corrected to Y-pixel units
MEDIA_CIRCLE_MIN_PATH = 0.16
MEDIA_CIRCLE_MIN_ASPECT = 0.26   # tolerate ellipses as slender as ~1:3.8
MEDIA_CIRCLE_MIN_MONOTONICITY = 0.73
MEDIA_CIRCLE_MAX_CLOSURE_R = 1.40
MEDIA_CIRCLE_MIN_S = 0.50
MEDIA_KEY_COOLDOWN_S = 1.00
STANDBY_INDEX_GRACE_S = 1.20
PINCH_CLOSE_INITIAL = 0.42
PINCH_HYSTERESIS = 0.23
PINCH_3D_EXTRA = 0.12            # depth-based confirmation margin
PINCH_OPEN_HOLD_S = 0.12         # first show two separated fingertips
PINCH_CLOSE_HOLD_S = 0.12        # then touch and hold for three-ish frames
PINCH_ARM_WINDOW_S = 2.00       # arm expires if you do not pinch

SWIPE_THRESHOLDS = (0.055, 0.075, 0.100)  # fractions of image height
SWIPE_MIN_STRAIGHTNESS = 0.50
SWIPE_MIN_MONOTONICITY = 0.55
SWIPE_MAX_HORIZONTAL_RATIO = 1.0
SCREEN_MARGIN = 0.006
REGIONS = {1: (0.15, 0.85, 0.15, 0.85),
           2: (0.23, 0.77, 0.18, 0.82),
           3: (0.30, 0.70, 0.25, 0.75)}
SCROLL_TICKS = 240  # initial sensitivity: 7.5% vertical movement -> 240 wheel units
SCROLL_REFERENCE_DISTANCE = 0.075
SCROLL_MIN_TICKS = 240  # any accepted vertical pinch scroll is at least 240
SCROLL_MAX_TICKS = 720  # and at most 720, even when Z/X sensitivity changes
# Native Windows wheel units: actual page travel depends on app/system configuration.

# Horizontal back is *not* triggered by free pointer movement. It is only
# recognized from the palm path during a confirmed thumb-index pinch and
# only when the pinch is released, reducing accidental page navigation.
BACK_MIN_HORIZONTAL = 0.13     # fraction of camera width, not pixels
BACK_MAX_VERTICAL_RATIO = 0.50 # |dy| <= 0.5 * |dx|
BACK_MIN_STRAIGHTNESS = 0.55
BACK_MIN_MONOTONICITY = 0.65
# RIGHT pinch swipe submits Enter; strong separation from vertical scroll/back.
ENTER_MIN_HORIZONTAL = 0.13
ENTER_MAX_VERTICAL_RATIO = 0.50

# Click contact is thumb tip(4) -> proximal segment of FOLDED middle(9--10),
# not the middle fingertip(12). Distance is normalized by palm width(5--17).
CLICK_CLOSE_INITIAL = 0.35  # stricter contact than 0.40; lower means closer to middle bone
CLICK_HYSTERESIS = 0.10  # small release motion, preserve one-shot guard
CLICK_3D_EXTRA = 0.16
CLICK_OPEN_HOLD_S = 0.06
CLICK_CLOSE_HOLD_S = 0.06
CLICK_ARM_WINDOW_S = 15.0  # comfortable time to aim and click
CLICK_ENTRY_ONE_S = 3.00  # allow short MLP class drop during hand motion
CLICK_RELEASE_HOLD_S = 0.16
CLICK_RELEASE_DELTA = 0.15  # thumb lift relative to actual contact location
CLICK_RECLOSE_DELTA = 0.09  # require a new real closing move after each release   # sustained separation; avoid jitter-induced repeat
CLICK_LOST_S = 0.55
CLICK_INVALID_GRACE_S = 1.20  # tolerate transient hand-pose instability
CLICK_RELEASE_3D_MARGIN = 0.12
CLICK_MIN_BETWEEN_CLICKS_S = 0.48  # minimum interval even if a false release occurs
DRAG_HOLD_S = 1.00  # hold existing click pose this long to initiate mouseDown
DRAG_MOVE_ALPHA = 0.70  # smoothing for cursor while holding left mouse button
DRAG_RELEASE_STABLE_S = CLICK_RELEASE_HOLD_S
CLICK_MIN_THUMB_INDEX = 0.62  # don't steal existing thumb-index scrolling
CLICK_MIDDLE_FOLD_RATIO = 2.25
CLICK_MIDDLE_FOLD_COS = 0.68
DETECT_WIDTH = 480  # inference downsample: output preview stays at camera resolution
PREVIEW_EVERY_N_FRAMES = 2  # smaller UI workload; key handling stays every frame

# Handy screenshot: global hotkey Ctrl+Space; behavior MUST be '길게 누르기'.
VOICE_HOTKEY_MODIFIER_SCAN = 0x1D  # Left CTRL scan code
VOICE_HOTKEY_KEY_SCAN = 0x39       # SPACE scan code
VOICE_HOLD_S = 0.45              # prevent accidental triggering while still easy to hold
VOICE_RELEASE_S = 0.55           # explicit thumb release must persist (jitter protection)
VOICE_MIN_RECORDING_S = 1.20      # ignore brief false-release immediately after starting
VOICE_MISSING_RELEASE_S = 0.90   # brief webcam tracking losses do not end dictation
VOICE_PROCESSING_S = 3.0         # Handy transcription grace period; increase if paste is delayed
VOICE_RECORDING_MAX_S = 90.0     # failsafe: prevent indefinitely held virtual keys
VOICE_SHOW_DIAGNOSTICS = True

# STEP 15: index + middle extended TOGETHER, thumb extended APART;
# ring + pinky curled. A deliberate 0.70 s hold enters VOLUME mode.
VOLUME_POSE_HOLD_S = 0.70
VOLUME_POSE_RELEASE_S = 0.12
VOLUME_MAX_RATE_PERCENT_S = 15.0   # maximum absolute Windows-volume % points per second
VOLUME_GAIN_PER_CAMERA_HEIGHT = 155.0   # raw % points per 1.0 normalized Y motion
VOLUME_DEADZONE_PER_FRAME = 0.0025  # reject small normalized hand jitter
VOLUME_MAX_VALID_FRAME_JUMP = 0.075 # abrupt camera-tracking jump ignored
VOLUME_MAX_FRAME_DT = 0.12        # cap credit after a camera stall
VOLUME_MUTE_CHECK_INTERVAL_S = 0.30  # only check mute on deliberate volume-UP movement
VOLUME_ZERO_WAKE_PERCENT = 8.0  # one-time safe start from Windows 0% when gun first moves UP
VOLUME_ZERO_EPS_PERCENT = 0.50  # treat near-zero master level as silent
VOLUME_POSE_MIN_THUMB_SEPARATION = 0.80  # normalized by palm width
VOLUME_POSE_MAX_INDEX_MIDDLE_GAP = 0.55
VOLUME_POSE_MIN_THUMB_BASE_GAP = 0.78  # thumb separated from index root, not merely finger tips

# STEP19: GUN = common volume/seek entry. Lateral motion is deliberate
# ±10-second seek; return-to-anchor never fires the opposite direction.
GUN_AXIS_LOCK_X = 0.030     # fraction of image width
GUN_AXIS_LOCK_Y = 0.022     # fraction of image height
GUN_DIRECTION_RATIO = 1.80  # aspect-corrected displacement ratio
SEEK_STROKE_X = 0.075      # image-width fraction; ~48 px in 640px preview
SEEK_RETURN_X = 0.025      # must return close to entry point
SEEK_RETURN_Y = 0.060
SEEK_REARM_STABLE_S = 0.16
SEEK_MAX_FRAME_JUMP = 0.14
SEEK_MIN_INTERVAL_S = 0.35
SEEK_SECONDS = 10
SEEK_NATIVE_TIMEOUT_S = 2.5


def classify_gun_axis(dx, dy, aspect=4/3):
    """Pick ONE axis; diagonal or tiny movements remain undecided."""
    ax, ay = abs(float(dx)*aspect), abs(float(dy))
    if not (np.isfinite(ax) and np.isfinite(ay)):
        return None
    if abs(dx) >= GUN_AXIS_LOCK_X and ax >= GUN_DIRECTION_RATIO*max(ay, 1e-8):
        return 'SEEK'
    if abs(dy) >= GUN_AXIS_LOCK_Y and ay >= GUN_DIRECTION_RATIO*max(ax, 1e-8):
        return 'VOLUME'
    return None


class GunSeekMotion:
    """One seek per outward stroke, return/settle before another.

    The gun pose itself is gated separately (0.7 sec). It stays held during
    return motion; a sustained extreme position never auto-repeats.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 'OFF'
        self.anchor = None
        self.last = None
        self.return_since = None
        self.last_fire_at = -1e9
        self.fires = 0

    def start(self, now, palm):
        self.phase = 'READY'
        self.anchor = np.asarray(palm, dtype=np.float64).copy()
        self.last = self.anchor.copy()
        self.return_since = None
        self.last_fire_at = -1e9
        self.fires = 0

    def update(self, now, palm, aspect=4/3):
        if self.phase == 'OFF' or palm is None:
            return None
        pt = np.asarray(palm, dtype=np.float64)
        if not np.isfinite(pt).all():
            self.return_since = None
            return None
        # Skip a tracking teleport rather than interpreting it as a swipe.
        if self.last is not None and float(np.linalg.norm(pt-self.last)) > SEEK_MAX_FRAME_JUMP:
            self.last = pt.copy()
            self.return_since = None
            return None
        self.last = pt.copy()
        dx, dy = (pt-self.anchor)
        if self.phase == 'READY':
            if (abs(dx) >= SEEK_STROKE_X and
                    abs(dx*aspect) >= GUN_DIRECTION_RATIO*max(abs(dy), 1e-8) and
                    now-self.last_fire_at >= SEEK_MIN_INTERVAL_S):
                self.phase = 'RETURN'
                self.last_fire_at = now
                self.return_since = None
                self.fires += 1
                return 10 if dx > 0 else -10
        elif self.phase == 'RETURN':
            if abs(dx) <= SEEK_RETURN_X and abs(dy) <= SEEK_RETURN_Y:
                if self.return_since is None:
                    self.return_since = now
                elif now-self.return_since >= SEEK_REARM_STABLE_S:
                    self.phase = 'READY'
                    self.return_since = None
                    return 'REARMED'
            else:
                self.return_since = None
        return None


def compute_seek_position(current, start, end, seconds):
    """Return bounded target seconds, or None for unavailable timeline."""
    if not all(np.isfinite(v) for v in (current, start, end, seconds)):
        return None
    if end <= start or end <= 0:
        return None
    return max(float(start), min(float(end), float(current)+float(seconds)))


class WindowsMediaSeek:
    """Seek via Windows GSMTC; safe foreground YouTube J/L fallback ONLY.

    Requests run off the webcam thread. No fallback is ever sent to an arbitrary
    browser/text field. Package optional: winrt-Windows.Media.Control.
    """
    def __init__(self, jl_sender=None, safe_foreground=None, manager_factory=None):
        self.jl_sender = jl_sender or self._send_jl
        self.safe_foreground = safe_foreground or self._is_youtube_player_foreground
        self.manager_factory = manager_factory
        self.manager = None
        self.cached_session_id = None
        self.cached_target = None
        self.cached_target_at = -1e9
        self.last_reason = ''

    @staticmethod
    def _is_youtube_player_foreground():
        if sys.platform != 'win32':
            return False
        title = LatestMouseWorker._foreground().lower()
        if 'youtube' not in title or not any(browser in title for browser in ('chrome','edge')):
            return False
        # If UI Automation cannot prove we are outside an editable control,
        # DO NOT send J/L. This avoids writing 'j' and 'l' to input boxes.
        try:
            import uiautomation as auto
            control = auto.GetFocusedControl()
            if control is None:
                return False
            name = str(getattr(control, 'ControlTypeName', ''))
            if name not in ('DocumentControl','PaneControl','GroupControl','CustomControl'):
                return False
            # DocumentControl can be contenteditable. A read-write ValuePattern
            # is an extra reason to reject (best effort).
            try:
                vp = control.GetValuePattern()
                if vp is not None and not vp.IsReadOnly:
                    return False
            except Exception:
                pass
            return True
        except Exception as exc:
            print('  SEEK J/L BLOCKED: UIA focus unavailable:', type(exc).__name__)
            return False

    @staticmethod
    def _send_jl(delta):
        if sys.platform != 'win32':
            raise RuntimeError('J/L only runs on Windows')
        import pyautogui
        pyautogui.press('l' if delta > 0 else 'j')

    async def _winrt_seek(self, delta):
        if self.manager_factory is None:
            from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
            factory = GlobalSystemMediaTransportControlsSessionManager.request_async
        else:
            factory = self.manager_factory
        if self.manager is None:
            self.manager = await factory()
        session = self.manager.get_current_session()
        if session is None:
            return False, 'No Windows media session'
        timeline = session.get_timeline_properties()
        if timeline is None:
            return False, 'No media timeline'
        # WinRT TimeSpan is mapped to datetime.timedelta by PyWinRT.
        current = float(timeline.position.total_seconds())
        start = float(timeline.start_time.total_seconds())
        end = float(timeline.end_time.total_seconds())
        src = str(session.source_app_user_model_id)
        now = time.monotonic()
        # Some players report the old timeline briefly after previous seek.
        # Respect previously accepted target when requests arrive quickly.
        if self.cached_session_id == src and now-self.cached_target_at < 2.0 and self.cached_target is not None:
            if delta > 0:
                current = max(current, self.cached_target)
            else:
                current = min(current, self.cached_target)
        target = compute_seek_position(current,start,end,delta)
        if target is None or abs(target-current)<0.1:
            return False, 'No seekable range / already at boundary'
        accepted = bool(await session.try_change_playback_position_async(int(round(target*10_000_000))))
        if accepted:
            self.cached_session_id = src
            self.cached_target = target
            self.cached_target_at = now
            return True, f'GSMTC {src} {current:.1f}s -> {target:.1f}s (request accepted)'
        return False, f'GSMTC session {src} rejected seek'

    def seek(self, delta):
        delta = 10 if delta > 0 else -10
        # Foreground YouTube: favor the official J/L shortcut for instant response.
        # UIA must positively confirm focus is NOT in the search/edit controls.
        safe_jl = self.safe_foreground()
        native_reason=''
        if safe_jl:
            try:
                self.jl_sender(delta)
                msg=f'YouTube foreground J/L {delta:+d}s key sent (response not verified)'
                self.last_reason=msg
                return True,msg
            except Exception as exc:
                native_reason=f'J/L error {type(exc).__name__}: {exc}; '
        # Background / non-YouTube windows: seek the current Windows media
        # session, NEVER inject J/L into an arbitrary foreground application.
        try:
            import asyncio
            accepted,reason = asyncio.run(asyncio.wait_for(
                self._winrt_seek(delta), timeout=SEEK_NATIVE_TIMEOUT_S))
            native_reason += reason
            if accepted:
                self.last_reason=native_reason
                return True,native_reason
        except ImportError:
            native_reason += 'WinRT package not installed'
        except Exception as exc:
            self.manager=None  # retry discovery on the next request
            native_reason += f'GSMTC unavailable: {type(exc).__name__}: {exc}'
        msg=f'SEEK BLOCKED: {native_reason}; no safe playback-position backend'
        self.last_reason=msg
        return False,msg


class SeekCommandWorker:
    """Serial bounded queue. Revokes unsent commands when pose/state exits."""
    def __init__(self, backend=None):
        self.backend=backend or WindowsMediaSeek()
        self.lock=threading.Lock()
        self.wakeup=threading.Event()
        self.running=True
        self.allowed=False
        self.epoch=0
        self.actions=deque(maxlen=8)
        self.thread=threading.Thread(target=self._loop,daemon=True)
        self.thread.start()

    def set_authorized(self, yes):
        with self.lock:
            if self.allowed!=bool(yes):
                self.epoch+=1
                self.actions.clear()
                self.allowed=bool(yes)
        self.wakeup.set()

    def request(self, delta):
        with self.lock:
            if not self.running or not self.allowed:
                return False
            self.actions.append((self.epoch, 10 if delta > 0 else -10))
        self.wakeup.set()
        return True

    def _loop(self):
        while True:
            item=None
            with self.lock:
                if not self.running:
                    return
                if self.allowed and self.actions:
                    item=self.actions.popleft()
            if item is not None:
                token,delta=item
                with self.lock:
                    permitted=self.running and self.allowed and token==self.epoch
                if permitted:
                    ok,msg=self.backend.seek(delta)
                    print('  SEEK', 'OK' if ok else 'SKIPPED', msg)
            else:
                self.wakeup.wait(.01)
                self.wakeup.clear()

    def close(self):
        with self.lock:
            self.running=False
            self.allowed=False
            self.epoch+=1
            self.actions.clear()
        self.wakeup.set()
        self.thread.join(timeout=3.0)


def delayed_seek_test():
    if sys.platform != 'win32':
        print('This Windows media-seek test requires Windows')
        return
    print('Play YouTube, then optionally switch to another window during countdown.')
    print('Foreground YouTube: safe J/L. Background: GSMTC API (no typing).')
    for n in range(5,0,-1):
        print(n,flush=True)
        time.sleep(1)
    backend=WindowsMediaSeek()
    ok,msg=backend.seek(10)
    print('SEEK TEST:', 'REQUESTED' if ok else 'BLOCKED', msg)
    print('Verify actual YouTube playback moved +10 seconds.')



# STEP18 v5: recent HMI input-field click -> FIST 0.5s -> OK 0.5s -> clear.
# One gesture confirmation only: selection and deletion run in a single guarded
# worker action. Focus, window and input identity are checked before/after Ctrl+A.
CLEAR_FIST_HOLD_S = 0.50
CLEAR_OK_SELECT_HOLD_S = 0.50
CLEAR_CONFIRM_TIMEOUT_S = 2.50
CLEAR_RECENT_CLICK_S = 25.0
CLEAR_FIST_RELEASE_GRACE_S = 0.80  # tolerate natural fist -> OK transition; no faster delete
CLEAR_OK_CLOSE_2D = 0.43       # thumb tip(4) to index tip(8) / palm width
CLEAR_OK_CLOSE_3D = 0.63       # 3D estimate can be noisier
CLEAR_OK_OTHER_EXTENDED = 0.73 # middle/ring/pinky finger-reach ratio
CLEAR_OK_INDEX_MAX_REACH = 0.91
CLEAR_OK_INDEX_MIN_REACH = 0.30
CLEAR_OK_THUMB_STRAIGHT = 0.63
CLEAR_SELECT_LIFETIME_S = 4.0


def _finger_reach(pts, chain):
    a,b,c,d = chain
    total = sum(float(np.linalg.norm(pts[j]-pts[i]))
                for i,j in ((a,b),(b,c),(c,d)))
    return float(np.linalg.norm(pts[d]-pts[a])) / max(total, 1e-8)


def gun_volume_geometry(hand, world_hand=None, aspect=4.0/3.0):
    """A geometry-only VOLUME pose, independent of HaGRID 4-way labels.

    INDEX+MIDDLE must be straight, adjacent and parallel, THUMB open and
    apart from their fingertips; RING+PINKY curled. The estimated world
    points help with hand rotation; 2D aspect-corrected fingertip gap is
    used for the visible 'two fingers together' requirement.
    """
    diagnostics = dict(index=0.0,middle=0.0,ring=1.0,pinky=1.0,
                       gap=float('inf'),thumb_gap=0.0,thumb_base_gap=0.0,thumb_line=0.0,
                       parallel=0.0)
    if hand is None or len(hand)!=21:
        return False,diagnostics
    xy = np.asarray([[p.x*aspect,p.y] for p in hand],dtype=np.float32)
    pts = xy
    if world_hand is not None and len(world_hand)==21:
        world = np.asarray([[p.x,p.y,p.z] for p in world_hand],dtype=np.float32)
        if np.isfinite(world).all(): pts=world
    if not (np.isfinite(xy).all() and np.isfinite(pts).all()):
        return False,diagnostics
    w2 = float(np.linalg.norm(xy[5]-xy[17]))
    w3 = float(np.linalg.norm(pts[5]-pts[17]))
    if w2 < 1e-7 or w3 < 1e-7:
        return False,diagnostics
    index = _finger_reach(pts,(5,6,7,8))
    middle = _finger_reach(pts,(9,10,11,12))
    ring = _finger_reach(pts,(13,14,15,16))
    pinky = _finger_reach(pts,(17,18,19,20))
    gap = float(np.linalg.norm(xy[8]-xy[12])) / w2
    thumb_gap = min(float(np.linalg.norm(xy[4]-xy[8])),
                    float(np.linalg.norm(xy[4]-xy[12]))) / w2
    thumb_base_gap = float(np.linalg.norm(xy[4]-xy[5])) / w2
    thumb_chain = (float(np.linalg.norm(pts[4]-pts[3]))+
                   float(np.linalg.norm(pts[3]-pts[2])))
    thumb_line = float(np.linalg.norm(pts[4]-pts[2]))/max(thumb_chain,1e-7)
    vi=pts[8]-pts[5]; vm=pts[12]-pts[9]
    parallel = float(np.dot(vi,vm))/max(float(np.linalg.norm(vi)*np.linalg.norm(vm)),1e-7)
    diagnostics.update(index=index,middle=middle,ring=ring,pinky=pinky,
                       gap=gap,thumb_gap=thumb_gap,thumb_base_gap=thumb_base_gap,thumb_line=thumb_line,
                       parallel=parallel)
    yes = (index>=0.81 and middle>=0.81 and
           ring<=0.79 and pinky<=0.79 and
           gap<=VOLUME_POSE_MAX_INDEX_MIDDLE_GAP and parallel>=0.84 and
           thumb_gap>=VOLUME_POSE_MIN_THUMB_SEPARATION and
           thumb_base_gap>=VOLUME_POSE_MIN_THUMB_BASE_GAP and thumb_line>=0.81)
    return bool(yes),diagnostics


def fist_clear_geometry(hand, world_hand=None, aspect=4.0/3.0):
    """Conservative fist: FOUR curled fingers, thumb not obviously extended.

    This never uses the MLP 'no_gesture' label as a sufficient condition.
    The gate and UI Automation are additional safeguards before any deletion.
    """
    diag={'index':1.,'middle':1.,'ring':1.,'pinky':1.,'thumb_gap':float('inf')}
    if hand is None or len(hand)!=21:
        return False,diag
    xy=np.asarray([[p.x*aspect,p.y] for p in hand],dtype=np.float32)
    pts=xy
    if world_hand is not None and len(world_hand)==21:
        world=np.asarray([[p.x,p.y,p.z] for p in world_hand],dtype=np.float32)
        if np.isfinite(world).all(): pts=world
    if not np.isfinite(xy).all() or not np.isfinite(pts).all():
        return False,diag
    span=float(np.linalg.norm(pts[5]-pts[17]))
    span2=float(np.linalg.norm(xy[5]-xy[17]))
    if span<1e-7 or span2<1e-7:
        return False,diag
    fingers=[_finger_reach(pts,c) for c in ((5,6,7,8),(9,10,11,12),
                                             (13,14,15,16),(17,18,19,20))]
    palm_center=(xy[5]+xy[9]+xy[13]+xy[17])/4.0
    thumb_gap=float(np.linalg.norm(xy[4]-palm_center))/span2
    diag=dict(zip(('index','middle','ring','pinky'),fingers))
    diag['thumb_gap']=thumb_gap
    # Relaxed fist may have the thumb slightly outside the four folded fingers.
    # But thumbs-up/gun gesture, straight One, and V sign must not pass.
    return bool(max(fingers)<0.79 and thumb_gap<1.75),diag


def ok_clear_geometry(hand, world_hand=None, aspect=4.0/3.0):
    """Classic OK ring: thumb-index fingertips together, other 3 fingers OPEN.

    Deliberately distinct from scrolling pinch: middle, ring AND little finger
    must all remain extended. Screen projection and estimated depth are checked.
    Gesture is not sufficient to delete: a preceding FIST + UIA are required.
    """
    d={'tip2d':float('inf'),'tip3d':float('inf'), 'index':0.,
       'middle':0.,'ring':0.,'pinky':0.,'thumb_line':0.}
    if hand is None or len(hand)!=21: return False,d
    xy=np.asarray([[p.x*aspect,p.y] for p in hand],dtype=np.float32)
    xyz=np.asarray([[p.x*aspect,p.y,p.z*aspect] for p in hand],dtype=np.float32)
    pose=xyz
    if world_hand is not None and len(world_hand)==21:
        world=np.asarray([[p.x,p.y,p.z] for p in world_hand],dtype=np.float32)
        if np.isfinite(world).all(): pose=world
    if not (np.isfinite(xy).all() and np.isfinite(xyz).all() and np.isfinite(pose).all()):
        return False,d
    w=float(np.linalg.norm(xy[5]-xy[17]))
    wp=float(np.linalg.norm(pose[5]-pose[17]))
    if min(w,wp)<1e-7: return False,d
    finger=[_finger_reach(pose,c) for c in ((5,6,7,8),(9,10,11,12),
                                              (13,14,15,16),(17,18,19,20))]
    tip2d=float(np.linalg.norm(xy[4]-xy[8]))/w
    tip3d=float(np.linalg.norm(xyz[4]-xyz[8]))/w
    thumb=(float(np.linalg.norm(pose[4]-pose[2])) /
           max(float(np.linalg.norm(pose[4]-pose[3])+
                     np.linalg.norm(pose[3]-pose[2])),1e-7))
    d.update(tip2d=tip2d,tip3d=tip3d,index=finger[0],middle=finger[1],
             ring=finger[2],pinky=finger[3],thumb_line=thumb)
    yes=(tip2d<=CLEAR_OK_CLOSE_2D and tip3d<=CLEAR_OK_CLOSE_3D and
         CLEAR_OK_INDEX_MIN_REACH<=finger[0]<=CLEAR_OK_INDEX_MAX_REACH and
         all(q>=CLEAR_OK_OTHER_EXTENDED for q in finger[1:]) and
         thumb>=CLEAR_OK_THUMB_STRAIGHT)
    return bool(yes),d


def stop_open_palm_geometry(hand, world_hand=None, aspect=4.0/3.0):
    """Conservative STOP confirmation, independent from the 4-class MLP.

    STEP18 OK opens middle/ring/pinky but joins thumb and a BENT index.
    The MLP can call that pose STOP because the training classes did not include
    OK. Accept an MLP STOP only if the index is also extended and thumb/index
    are visibly separated. This protects real STOP without weakening LOCKED.
    """
    diag={'index_reach':0., 'middle_reach':0., 'ring_reach':0.,
          'pinky_reach':0., 'thumb_index_gap':0.}
    if hand is None or len(hand)!=21:
        return False,diag
    xy=np.asarray([[p.x*aspect,p.y] for p in hand],dtype=np.float32)
    pts=xy
    if world_hand is not None and len(world_hand)==21:
        world=np.asarray([[p.x,p.y,p.z] for p in world_hand],dtype=np.float32)
        if np.isfinite(world).all(): pts=world
    if not (np.isfinite(xy).all() and np.isfinite(pts).all()):
        return False,diag
    w=float(np.linalg.norm(xy[5]-xy[17]))
    if w<1e-7: return False,diag
    reaches=[_finger_reach(pts,c) for c in
             ((5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20))]
    gap=float(np.linalg.norm(xy[4]-xy[8]))/w
    diag=dict(zip(('index_reach','middle_reach','ring_reach','pinky_reach'),reaches))
    diag['thumb_index_gap']=gap
    # A true open-palm STOP usually has all four fingers extended. In an OK
    # sign, index reach and/or thumb-index gap fails these conditions.
    yes=(reaches[0]>=0.79 and reaches[1]>=0.72 and
         reaches[2]>=0.72 and reaches[3]>=0.69 and gap>=0.70)
    return bool(yes),diag


def effective_stop_label(label, stop_open):
    """Reject stop classification for a geometrically non-open hand."""
    return 'STOP_SHAPE_REJECTED' if label=='stop' and not stop_open else label


class ClearTextGate:
    """Fast FIST (0.5 s) -> OK (0.5 s) -> one DELETE request.

    A single FIST or a standalone OK never deletes. Short pose break (<0.35s)
    is allowed only while transitioning from ARMED to OK, not within either
    required hold. No hand or invalid input authorization cancels. After one
    request, both FIST and OK must be released before the next request.
    """
    def __init__(self):
        self.reset()

    def reset(self, lock_until_release=False):
        self.state='LOCKED' if lock_until_release else 'IDLE'
        self.since=None
        self.armed_at=None
        self.ok_since=None
        self.break_since=None

    def update(self, now, fist, ok_pose, hand_present, eligible, selected=False):
        # selected is retained for backwards-compatible callers, but no longer
        # needed: the OS worker selects and deletes after the completed OK pose.
        if not hand_present or not eligible:
            self.reset(lock_until_release=True)
            return 'CANCEL'
        if self.state=='LOCKED':
            if not fist and not ok_pose:
                self.reset()
            return None
        if self.state=='IDLE':
            if fist:
                self.state='HOLD'
                self.since=now
            return None
        if self.state=='HOLD':
            if not fist:
                self.reset(lock_until_release=True)
                return 'CANCEL'
            if now-self.since >= CLEAR_FIST_HOLD_S:
                self.state='ARMED'
                self.armed_at=now
                return 'ARMED'
            return None
        if self.state=='ARMED':
            if now-self.armed_at>CLEAR_CONFIRM_TIMEOUT_S:
                self.reset(lock_until_release=True)
                return 'CANCEL'
            if fist:
                self.ok_since=None
                self.break_since=None
            elif ok_pose:
                self.break_since=None
                if self.ok_since is None:
                    self.ok_since=now
                elif now-self.ok_since>=CLEAR_OK_SELECT_HOLD_S:
                    self.reset(lock_until_release=True)
                    return 'DELETE'
            else:
                # Give the hand a small transition window, but not a shorter
                # hold requirement or a free deletion.
                self.ok_since=None
                if self.break_since is None:
                    self.break_since=now
                elif now-self.break_since>CLEAR_FIST_RELEASE_GRACE_S:
                    self.reset(lock_until_release=True)
                    return 'CANCEL'
            return None
        raise RuntimeError('Unknown clear state '+self.state)


class FocusedEditGuard:
    """Check Windows UI Automation focus before dangerous keyboard commands.

    Fail CLOSED if uiautomation is absent, focus is uncertain, or role is not
    EditControl. Do not allow DocumentControl (web page/Word document body).
    No filesystem, clipboard, or typed text is accessed.
    """
    def inspect(self):
        if sys.platform!='win32':
            return False,'Windows only',None
        try:
            import uiautomation as auto
            control=auto.GetFocusedControl()
            if control is None:
                return False,'No focused UIA control',None
            name=str(getattr(control,'ControlTypeName',''))
            ctype=getattr(control,'ControlType',None)
            enabled=bool(getattr(control,'IsEnabled',False))
            keyboard=bool(getattr(control,'HasKeyboardFocus',False))
            title=str(getattr(control,'Name','') or '')[:90]
            description=f'{name} type={ctype} focused={keyboard} enabled={enabled} name={title!r}'
            is_edit=(name=='EditControl' or ctype==50004)
            # Editable combo boxes (some browser search/address fields) can
            # expose focused ComboBoxControl. Non-editable combos stay blocked.
            if name=='ComboBoxControl' or ctype==50003:
                try:
                    vp=control.GetValuePattern()
                    is_edit=bool(vp is not None and not vp.IsReadOnly)
                except Exception:
                    is_edit=False
            if not is_edit or not enabled or not keyboard:
                return False,'Not a focused editable text field: '+description,None
            # Same-target guard across select and delete. Never use the text
            # VALUE or Name, which can change as the user types.
            try:
                bounds=control.BoundingRectangle
                # UIAutomation Rectangle normally provides pixel coordinates.
                # Keep a parseable 4-number representation for robust comparisons.
                rect=(int(bounds.left),int(bounds.top),int(bounds.right),int(bounds.bottom))
            except Exception:
                try:
                    rect=str(control.BoundingRectangle)
                except Exception:
                    rect='?'
            try:
                aid=str(control.AutomationId)
            except Exception:
                aid='?'
            try:
                nhwnd=str(control.NativeWindowHandle)
            except Exception:
                nhwnd='?'
            # Prefer runtime identity when provider exposes it (Chrome/Windows
            # edit providers differ); fall back to role/id/hwnd/bounds.
            rid=None
            try:
                candidate=getattr(control,'GetRuntimeId',None)
                if callable(candidate): rid=tuple(candidate())
                else:
                    element=getattr(control,'Element',None)
                    candidate=getattr(element,'GetRuntimeId',None)
                    if callable(candidate): rid=tuple(candidate())
            except Exception:
                rid=None
            signature=(str(ctype),aid,nhwnd,rect,rid)
            return True,'Focused editable field: '+description,signature
        except ImportError:
            return False,'uiautomation missing: python -m pip install uiautomation',None
        except Exception as exc:
            return False,'UI Automation check failed: '+type(exc).__name__+' '+str(exc),None


def _foreground_hwnd():
    if sys.platform!='win32': return None
    import ctypes
    from ctypes import wintypes
    u=ctypes.windll.user32
    u.GetForegroundWindow.restype=wintypes.HWND
    return u.GetForegroundWindow()


def _window_root_at_point(xy):
    """Identify the actual top-level window UNDER an HMI mouse click.

    GetForegroundWindow immediately after click is subject to an asynchronous
    focus race on busy pages (notably autocomplete search UIs). Windows owns
    the hit-test independently of the subsequent focus activation.
    """
    if sys.platform != 'win32' or xy is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        class POINT(ctypes.Structure):
            _fields_=[('x',wintypes.LONG),('y',wintypes.LONG)]
        u=ctypes.windll.user32
        u.WindowFromPoint.argtypes=[POINT]
        u.WindowFromPoint.restype=wintypes.HWND
        u.GetAncestor.argtypes=[wintypes.HWND,wintypes.UINT]
        u.GetAncestor.restype=wintypes.HWND
        hit=u.WindowFromPoint(POINT(int(xy[0]),int(xy[1])))
        return u.GetAncestor(hit,2) if hit else None  # GA_ROOT
    except Exception:
        return None


def _editable_rect(signature):
    """Extract UIA focused-control bounds from an inspect() signature."""
    if not isinstance(signature,(tuple,list)) or len(signature)<4:
        return None
    value=signature[3]
    if isinstance(value,(tuple,list)) and len(value)==4:
        try:
            a=tuple(float(x) for x in value)
        except (ValueError,TypeError):
            return None
    elif isinstance(value,str):
        import re
        numbers=re.findall(r'-?\d+(?:\.\d+)?',value)
        if len(numbers)!=4:
            return None
        a=tuple(float(x) for x in numbers)
    else:
        return None
    if a[2]<=a[0] or a[3]<=a[1]:
        return None
    return a


def _point_in_edit(signature,xy,margin=6):
    bounds=_editable_rect(signature)
    if bounds is None or xy is None:
        return False
    l,t,r,b=bounds
    return (l-margin<=float(xy[0])<=r+margin and
            t-margin<=float(xy[1])<=b+margin)


def _same_focused_edit(before,after,click_xy=None):
    """Recognize the SAME on-screen edit across a UIA provider rebind.

    YouTube/Chrome may recreate UI Automation element runtime IDs on update.
    Identical identity is ideal; if it changes, require same edit role, same
    window, compatible automation id and >=85% overlapping on-screen region.
    Never accept a missing/non-edit control; callers check inspect() first.
    """
    if before==after:
        return True,'exact'
    if not (isinstance(before,(tuple,list)) and isinstance(after,(tuple,list))
            and len(before)>=5 and len(after)>=5):
        return False,'unrecognized identity'
    role1,aid1,hwnd1=map(str,before[:3])
    role2,aid2,hwnd2=map(str,after[:3])
    if role1!=role2:
        # Chrome can expose the SAME editable field as ComboBox then Edit.
        # Permit it only when the HMI clicked INSIDE both on-screen rectangles.
        if not (set((role1,role2))=={'50003','50004'} and
                _point_in_edit(before,click_xy,margin=1) and
                _point_in_edit(after,click_xy,margin=1)):
            return False,'role changed'
    if aid1 not in ('','?','None') and aid2 not in ('','?','None') and aid1!=aid2:
        return False,'automation ID changed'
    if hwnd1 not in ('0','','?','None') and hwnd2 not in ('0','','?','None') and hwnd1!=hwnd2:
        return False,'provider window changed'
    a=_editable_rect(before); b=_editable_rect(after)
    if a is None or b is None:
        return False,'missing bounding rectangle'
    width=max(a[2]-a[0],1);height=max(a[3]-a[1],1)
    width2=max(b[2]-b[0],1);height2=max(b[3]-b[1],1)
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    smaller=min(width*height,width2*height2)
    if intersection/max(smaller,1)<.85:
        return False,'focused field position changed'
    if abs((a[0]+a[2])-(b[0]+b[2]))>24 or abs((a[1]+a[3])-(b[1]+b[3]))>24:
        return False,'focused field center changed'
    return True,'same edit geometry (UIA runtime ID updated)'


def focus_diagnostic():
    print('STEP18 FOCUS DIAGNOSTIC: click inside a text input within 5 seconds.')
    for n in range(5,0,-1):
        print(n,flush=True)
        time.sleep(1)
    ok,reason,_=FocusedEditGuard().inspect()
    print('EDIT FOCUS:', 'SUPPORTED' if ok else 'BLOCKED',reason)
    print('The test never modifies or deletes text.')


def focus_stability_diagnostic():
    """Print exactly why a Chrome edit is or isn't stable (NO keys sent)."""
    print('STEP18 FOCUS STABILITY TEST: click YouTube search field in 5 seconds.')
    for n in range(5,0,-1):
        print(n,flush=True)
        time.sleep(1)
    guard=FocusedEditGuard()
    ok,reason,first=guard.inspect()
    print('FIRST:', 'SUPPORTED' if ok else 'BLOCKED',reason)
    time.sleep(.25)
    ok2,reason2,second=guard.inspect()
    print('SECOND:','SUPPORTED' if ok2 else 'BLOCKED',reason2)
    if ok and ok2:
        compatible,explanation=_same_focused_edit(first,second)
        print('SAME EDIT:',compatible,'reason:',explanation)
    else:
        print('SAME EDIT: unavailable (focused control not verified)')
    print('No Ctrl+A, no Backspace was sent.')


class VolumePoseGate:
    """Debounce gun pose. Exit within ~0.12s on pose release, instantly on no hand."""
    def __init__(self):
        self.active=False
        self.since=None
        self.release_since=None
        self.locked=False

    def reset(self,lock_until_release=False):
        self.active=False
        self.since=None
        self.release_since=None
        self.locked=lock_until_release

    def update(self,now,pose,hand_present=True):
        if not hand_present:
            old=self.active
            self.reset(lock_until_release=True)
            return 'EXIT' if old else None
        if self.locked:
            if not pose: self.locked=False
            return None
        if not self.active:
            if not pose:
                self.since=None
                return None
            if self.since is None: self.since=now
            elif now-self.since>=VOLUME_POSE_HOLD_S:
                self.active=True
                self.release_since=None
                return 'ENTER'
            return None
        if pose:
            self.release_since=None
            return None
        if self.release_since is None:
            self.release_since=now
        elif now-self.release_since>=VOLUME_POSE_RELEASE_S:
            self.reset(lock_until_release=True)
            return 'EXIT'
        return None


class WindowsVolume:
    """Lazy pycaw Windows master-output volume, 0..100 scalar percent.

    Import only on first VOLUME entry, so missing pycaw cannot break
    existing pointer/click/Handy workflows.
    """
    def __init__(self):
        self.endpoint=None

    def _ensure(self):
        if self.endpoint is None:
            if sys.platform!='win32':
                raise RuntimeError('Windows system audio required')
            try:
                import comtypes
                from pycaw.pycaw import AudioUtilities
            except ImportError as exc:
                raise RuntimeError('Install pycaw: python -m pip install pycaw') from exc
            comtypes.CoInitialize()
            speakers=AudioUtilities.GetSpeakers()
            if speakers is None:
                raise RuntimeError('No Windows output device detected')
            # Support both pycaw APIs: AudioDevice.EndpointVolume and
            # older IMMDevice.Activate(QueryInterface) releases.
            if hasattr(speakers, 'EndpointVolume'):
                self.endpoint=speakers.EndpointVolume
            else:
                from pycaw.pycaw import IAudioEndpointVolume
                self.endpoint=speakers.Activate(
                    IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None
                ).QueryInterface(IAudioEndpointVolume)

    def get_percent(self):
        self._ensure()
        return float(self.endpoint.GetMasterVolumeLevelScalar())*100.0

    def is_muted(self):
        """Windows master output mute is independent of scalar volume."""
        self._ensure()
        return bool(self.endpoint.GetMute())

    def unmute_if_needed(self):
        """Only called for deliberate gun UP motion, never for volume DOWN/idle.

        This restores sound even when the master volume is already at 100%.
        """
        if not self.is_muted():
            return False
        self.endpoint.SetMute(0,None)  # IAudioEndpointVolume: 0=unmuted
        if self.is_muted():
            raise RuntimeError('Windows output remains muted after SetMute(0)')
        print('VOLUME UNMUTE: Windows master mute OFF (gun UP)')
        return True

    def set_percent(self,percent):
        self._ensure()
        value=float(np.clip(percent,0.0,100.0))
        self.endpoint.SetMasterVolumeLevelScalar(value/100.0,None)
        return value

    def wake_from_zero(self):
        """Gun UP from silent Windows output: make an audible initial step.

        The 0->8% transition is a single, intentionally bounded wake-up
        action. Subsequent volume motion continues to honor 15 %points/sec.
        Mute is separately cleared by unmute_if_needed(), AFTER setting 8%.
        """
        current=self.get_percent()
        if current<=VOLUME_ZERO_EPS_PERCENT:
            new_level=self.set_percent(VOLUME_ZERO_WAKE_PERCENT)
            print(f'VOLUME ZERO WAKE: {current:.1f}% -> {new_level:.1f}% (gun UP)')
            return new_level
        return current


class VolumeMotion:
    """No delayed target: update only from CURRENT observed hand movement.

    Motion ceases immediately when the hand stops. A hard 15 percentage-point
    /second rate cap applies even for rapid motions; excess motion is discarded
    rather than saved as a later volume jump.
    """
    def __init__(self,backend):
        self.backend=backend
        self.active=False
        self.level=None
        self.last_y=None
        self.last_t=None
        self.ys=deque(maxlen=3)
        self.last_mute_check_t=None
        self.mute_check_error_logged=False

    def start(self,now,palm,initial_up=False):
        """Enter VOLUME; the first UP displacement that locks the axis counts.

        STEP19 v2 discarded the first movement while selecting the axis. As a
        result zero-volume users could unmute yet stay at an inaudible 0%.
        Only a confirmed UP axis wakes the volume; entry alone never does.
        """
        level=self.backend.get_percent()
        self.active=True
        self.level=float(np.clip(level,0,100))
        self.last_y=float(palm[1])
        self.last_t=float(now)
        self.ys.clear()
        self.ys.append(float(palm[1]))
        self.last_mute_check_t=None
        self.mute_check_error_logged=False
        if initial_up:
            self._handle_deliberate_up(now)
        return self.level

    def _handle_deliberate_up(self,now):
        """From 0% wake to a modest level, then remove Windows master mute.

        Errors are nonfatal, as in STEP19 v2. This is NEVER called for DOWN,
        SEEK, idle gun pose or gun acquisition before vertical movement.
        """
        if (self.last_mute_check_t is not None and
                now-self.last_mute_check_t<VOLUME_MUTE_CHECK_INTERVAL_S):
            return
        self.last_mute_check_t=float(now)
        try:
            wake=getattr(self.backend,'wake_from_zero',None)
            if callable(wake) and self.level<=VOLUME_ZERO_EPS_PERCENT:
                new_level=wake()
                self.level=float(np.clip(new_level,0,100))
            unmute=getattr(self.backend,'unmute_if_needed',None)
            if callable(unmute):
                unmute()
        except Exception as exc:
            if not self.mute_check_error_logged:
                print('VOLUME UNMUTE ERROR:',type(exc).__name__,str(exc))
                self.mute_check_error_logged=True

    def stop(self):
        self.active=False
        self.last_y=None
        self.last_t=None
        self.ys.clear()

    def update(self,now,palm):
        if not self.active or palm is None: return None
        y=float(palm[1])
        if not np.isfinite(y): return None
        self.ys.append(y)
        med=float(np.median(self.ys))
        if self.last_y is None or self.last_t is None:
            self.last_y=med; self.last_t=now
            return None
        dt=max(0.0,min(float(now-self.last_t),VOLUME_MAX_FRAME_DT))
        dy=med-self.last_y
        self.last_t=float(now)
        self.last_y=med
        if dt<=0 or abs(dy)<VOLUME_DEADZONE_PER_FRAME: return None
        if abs(dy)>VOLUME_MAX_VALID_FRAME_JUMP: return None
        raw_change=-dy*VOLUME_GAIN_PER_CAMERA_HEIGHT
        # From 0%, DOWN does nothing; the NEXT intentional UP still wakes sound.
        # Run the one-time zero wake before the normal motion increment so
        # it cannot be lost at the zero-volume boundary.
        if raw_change>0 and self.level<=VOLUME_ZERO_EPS_PERCENT:
            self._handle_deliberate_up(now)
        bounded_change=float(np.clip(raw_change,-VOLUME_MAX_RATE_PERCENT_S*dt,
                                      VOLUME_MAX_RATE_PERCENT_S*dt))
        new_level=float(np.clip(self.level+bounded_change,0,100))
        changed=abs(new_level-self.level)>=0.07
        if changed:
            self.backend.set_percent(new_level)
            self.level=new_level
        # Nonzero volume: a deliberate upward motion also unmutes if needed.
        if raw_change>0:
            self._handle_deliberate_up(now)
        return new_level if changed else None



def proportional_scroll_ticks(dy, base_ticks=SCROLL_TICKS,
                              reference=SCROLL_REFERENCE_DISTANCE):
    """Wheel amount scales with vertical pinch travel, clamped to [240,720].

    Default mapping: 0.075 -> 240, 0.150 -> 480, 0.225+ -> 720.
    Movement below the selected swipe threshold is rejected by swipe_metrics;
    it never reaches this function. Z/X only changes the slope, never the
    hard minimum or maximum. The value is wheel units, NOT screen pixels.
    """
    if not np.isfinite(dy) or reference <= 0 or base_ticks <= 0:
        raise ValueError('Invalid scroll calculation parameters')
    raw = int(round(base_ticks * abs(float(dy)) / reference))
    return int(np.clip(raw, SCROLL_MIN_TICKS, SCROLL_MAX_TICKS))


def relative_drag_position(tip, start_tip, anchor_xy, region, sw, sh):
    """Translate relative fingertip movement into screen pixels without snapping.

    Anchor is last *stable* cursor location captured at click-contact start.
    Relative movement supports varying user click hand poses.
    """
    x0, x1, y0, y1 = region
    scale_x = (sw - 1) * (1 - 2 * SCREEN_MARGIN) / (x1 - x0)
    scale_y = (sh - 1) * (1 - 2 * SCREEN_MARGIN) / (y1 - y0)
    x = round(anchor_xy[0] + (float(tip[0]) - float(start_tip[0])) * scale_x)
    y = round(anchor_xy[1] + (float(tip[1]) - float(start_tip[1])) * scale_y)
    # Keep original upper-left PyAutoGUI FAILSAFE reachable only from keyboard.
    return (int(np.clip(x, 1, sw-2)), int(np.clip(y, 1, sh-2)))


def _handy_running():
    """Check whether Handy.exe is present in Windows process list."""
    if sys.platform != 'win32':
        return False
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        raw = subprocess.check_output(
            ['tasklist', '/FI', 'IMAGENAME eq Handy.exe', '/FO', 'CSV', '/NH'],
            startupinfo=startupinfo, timeout=4)
        return 'handy.exe' in raw.decode('utf-8', errors='replace').lower()
    except (OSError, subprocess.SubprocessError):
        return False


def _handy_locations():
    """Find installed Handy without requiring it at one hardcoded path."""
    paths = []
    # Optional easy override: one line with absolute Handy.exe path.
    config_file = ROOT / 'handy_path.txt'
    if config_file.is_file():
        saved = config_file.read_text(encoding='utf-8-sig').strip().strip('\"')
        if saved:
            paths.append(Path(saved).expanduser())
    env_path = os.environ.get('GESTURE_HANDY_EXE', '').strip().strip('\"')
    if env_path:
        paths.append(Path(env_path).expanduser())
    found = shutil.which('Handy.exe') or shutil.which('handy.exe')
    if found:
        paths.append(Path(found))
    local = Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData'/'Local')
    appdata = Path(os.environ.get('APPDATA') or Path.home()/'AppData'/'Roaming')
    program = Path(os.environ.get('PROGRAMFILES') or 'C:/Program Files')
    program86 = Path(os.environ.get('PROGRAMFILES(X86)') or 'C:/Program Files (x86)')
    for base in (local/'Programs'/'Handy', local/'Programs'/'handy',
                 local/'Handy', appdata/'Handy',
                 program/'Handy', program86/'Handy'):
        paths.append(base/'Handy.exe')
        paths.append(base/'handy.exe')
    # Installed apps may offer only a Start Menu shortcut.
    for menu in (appdata/'Microsoft'/'Windows'/'Start Menu'/'Programs',
                 program/'Microsoft'/'Windows'/'Start Menu'/'Programs'):
        paths.extend((menu/'Handy.lnk', menu/'Handy'/'Handy.lnk'))
    # Registry paths are useful for installs not in standard folders.
    if sys.platform == 'win32':
        try:
            import winreg
            for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for key in (r'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Handy.exe',):
                    try:
                        with winreg.OpenKey(root, key) as reg:
                            val, _ = winreg.QueryValueEx(reg, '')
                            paths.append(Path(str(val).strip('"')))
                    except OSError:
                        pass
        except ImportError:
            pass
    return [p for p in dict.fromkeys(paths) if p.is_file()]


def ensure_handy_running():
    """Best-effort startup; never launch a second copy intentionally.

    This is called once on HMI startup, not from the per-frame camera loop.
    The user must have installed Handy and configured Ctrl+Space + LONG PRESS.
    """
    if sys.platform != 'win32':
        return 'UNSUPPORTED'
    if _handy_running():
        print('HANDY AUTO: already running')
        return 'RUNNING'
    candidates = _handy_locations()
    if not candidates:
        print('HANDY AUTO: installation not found. Start Handy manually, or set')
        print('  GESTURE_HANDY_EXE to the full path of Handy.exe.')
        return 'NOT_FOUND'
    for path in candidates:
        try:
            # os.startfile invokes .lnk files and .exe via Windows Shell.
            os.startfile(str(path))
            print('HANDY AUTO: launch requested:', path)
            return 'STARTED'
        except OSError as e:
            print('HANDY AUTO: candidate did not launch:', path, repr(e))
    return 'FAILED'


def finish_voice_controller(controller, now, hand_present):
    """Exit dictated/processing mode without requiring a second PEACE.

    If the hand is visible, resume ACTIVE immediately after the Handy grace
    period; otherwise PAUSED prevents OS commands until hand returns.
    """
    controller.stop_since = None
    controller.peace_since = None
    controller.resume_since = None
    controller.last_visible_at = now
    controller.state = 'ACTIVE' if hand_present else 'PAUSED'
    return controller.state


class ThumbPressGate:
    """Recognize a sustained THUMBS-UP press and a confirmed release.

    Events: 'PRESS' once after deliberate thumb pose, and 'RELEASE' once
    after the thumb relaxes. Missing frames are not treated as intentional
    release until a longer timeout; a held gesture never retriggers.
    """
    def __init__(self, hold_s=VOICE_HOLD_S, release_s=VOICE_RELEASE_S,
                 missing_release_s=VOICE_MISSING_RELEASE_S, min_held_s=0.0):
        self.hold_s=float(hold_s)
        self.release_s=float(release_s)
        self.missing_release_s=float(missing_release_s)
        self.min_held_s=float(min_held_s)
        self.reset()

    def reset(self, lock_until_release=False):
        self.phase='LOCKED' if lock_until_release else 'IDLE'
        self.pose_since=None
        self.release_since=None
        self.missing_since=None
        self.pressed_at=None

    def update(self, now, thumbs_up, hand_present=True):
        if self.phase == 'LOCKED':
            if hand_present and not thumbs_up:
                if self.release_since is None:
                    self.release_since=now
                elif now-self.release_since >= self.release_s:
                    self.reset()
            else:
                self.release_since=None
            return None

        if self.phase == 'IDLE':
            if thumbs_up and hand_present:
                self.phase='CANDIDATE'
                self.pose_since=now
            return None

        if self.phase == 'CANDIDATE':
            if not thumbs_up or not hand_present:
                self.reset()
            elif now-self.pose_since >= self.hold_s:
                self.phase='HELD'
                self.release_since=None
                self.missing_since=None
                self.pressed_at=now
                return 'PRESS'
            return None

        if self.phase == 'HELD':
            if thumbs_up and hand_present:
                self.release_since=None
                self.missing_since=None
                return None
            if not hand_present:
                self.release_since=None
                if self.missing_since is None:
                    self.missing_since=now
                elif now-self.missing_since >= self.missing_release_s:
                    self.reset(lock_until_release=True)
                    return 'RELEASE'
                return None
            self.missing_since=None
            # Do not immediately release the physical Ctrl+Space hotkey if the
            # hand pose momentarily flickers just after voice activation.
            # A real hand-loss timeout above still releases for safety.
            if self.pressed_at is not None and now-self.pressed_at < self.min_held_s:
                self.release_since=None
                return None
            if self.release_since is None:
                self.release_since=now
            elif now-self.release_since >= self.release_s:
                self.reset(lock_until_release=True)
                return 'RELEASE'
            return None
        raise RuntimeError('Invalid ThumbPressGate phase')



def thumbs_up_geometry(hand, world_hand=None, aspect=4.0 / 3.0):
    """Return (detected, diagnostics); don't rely on the 4-class MLP.

    Thumb must extend out of a closed fist, other four fingers curled. Use
    estimated 3D world joints when available to reduce orientation effects.
    This is a *thumbs-up hand shape*, not necessarily vertical on the screen.
    """
    if hand is None or len(hand) != 21:
        return False, {'thumb': 0., 'max_fold': 1., 'index_gap': 0.}
    xy = np.asarray([[p.x * aspect, p.y] for p in hand], dtype=np.float32)
    if (world_hand is not None and len(world_hand) == 21):
        world = np.asarray([[p.x, p.y, p.z] for p in world_hand], dtype=np.float32)
        pts = world if np.isfinite(world).all() else xy
    else:
        pts = xy
    span = float(np.linalg.norm(pts[5] - pts[17]))
    if span < 1e-6:
        return False, {'thumb': 0., 'max_fold': 1., 'index_gap': 0.}

    # True thumbs-up: extended thumb, not tucked against the palm.
    thumb_chain = (np.linalg.norm(pts[4] - pts[3]) +
                   np.linalg.norm(pts[3] - pts[2]))
    thumb_straight = float(np.linalg.norm(pts[4]-pts[2]) / max(thumb_chain,1e-6))
    thumb_projection = float(np.linalg.norm(pts[4] - pts[5]) / span)
    index_gap = float(np.linalg.norm(pts[4] - pts[8]) / span)

    def finger_reach(mcp, pip, dip, tip):
        full = (np.linalg.norm(pts[pip] - pts[mcp]) +
                np.linalg.norm(pts[dip] - pts[pip]) +
                np.linalg.norm(pts[tip] - pts[dip]))
        return float(np.linalg.norm(pts[tip] - pts[mcp]) / max(full, 1e-6))

    reaches = [finger_reach(*f) for f in ((5,6,7,8),(9,10,11,12),
                                         (13,14,15,16),(17,18,19,20))]
    max_fold = max(reaches)
    yes = (thumb_straight >= 0.87 and thumb_projection >= 0.82
           and index_gap >= 0.65 and max_fold <= 0.78)
    return bool(yes), {'thumb': thumb_projection, 'thumb_line': thumb_straight,
                       'max_fold': max_fold, 'index_gap': index_gap}


def thumbs_up_recording_hold_geometry(hand, world_hand=None, aspect=4.0 / 3.0):
    """Looser hold condition AFTER recording starts; never starts recording.

    A thumb stays raised even if estimated ring/index joint reaches fluctuate.
    Only a clearly relaxed thumb or an open hand can end the recording after
    VOICE_RELEASE_S. This is hysteresis: acquire strict, retain more tolerant.
    """
    if hand is None or len(hand) != 21:
        return False
    xy=np.asarray([[p.x*aspect,p.y] for p in hand],dtype=np.float32)
    pts=xy
    if world_hand is not None and len(world_hand)==21:
        world=np.asarray([[p.x,p.y,p.z] for p in world_hand],dtype=np.float32)
        if np.isfinite(world).all():
            pts=world
    if not (np.isfinite(pts).all() and np.isfinite(xy).all()):
        return False
    span=float(np.linalg.norm(pts[5]-pts[17]))
    if span<1e-6:
        return False
    segments=float(np.linalg.norm(pts[4]-pts[3])+np.linalg.norm(pts[3]-pts[2]))
    thumb_straight=float(np.linalg.norm(pts[4]-pts[2]))/max(segments,1e-6)
    thumb_projection=float(np.linalg.norm(pts[4]-pts[5]))/span
    fingers=[_finger_reach(pts,c) for c in ((5,6,7,8),(9,10,11,12),
                                            (13,14,15,16),(17,18,19,20))]
    # Keep only when thumb remains reasonably extended and the four fingers
    # have not obviously opened into a STOP/ONE palm.
    return bool(thumb_straight>=0.75 and thumb_projection>=0.68
                and max(fingers)<=0.90 and not index_extended(hand))


class HandyHoldKeys:
    """Keep actual Ctrl+Space key-down across webcam frames on Windows.

    Handy must be in '길게 누르기' mode. Always release keys on stop, exit,
    webcam failure, or exception to avoid a stuck Ctrl or Space key.
    """
    def __init__(self):
        self.ctrl_down=False
        self.space_down=False

    def _emit(self, scan, release=False):
        if sys.platform != 'win32':
            raise RuntimeError('Handy SendInput requires Windows')
        import ctypes
        from ctypes import wintypes

        class KEYBDINPUT(ctypes.Structure):
            _fields_=[('wVk',wintypes.WORD), ('wScan',wintypes.WORD),
                      ('dwFlags',wintypes.DWORD), ('time',wintypes.DWORD),
                      ('dwExtraInfo',ctypes.c_size_t)]
        class MOUSEINPUT(ctypes.Structure):
            _fields_=[('dx',wintypes.LONG), ('dy',wintypes.LONG),
                      ('mouseData',wintypes.DWORD), ('dwFlags',wintypes.DWORD),
                      ('time',wintypes.DWORD), ('dwExtraInfo',ctypes.c_size_t)]
        class HARDWAREINPUT(ctypes.Structure):
            _fields_=[('uMsg',wintypes.DWORD), ('wParamL',wintypes.WORD),
                      ('wParamH',wintypes.WORD)]
        class INPUT_UNION(ctypes.Union):
            _fields_=[('ki',KEYBDINPUT), ('mi',MOUSEINPUT), ('hi',HARDWAREINPUT)]
        class INPUT(ctypes.Structure):
            _fields_=[('type',wintypes.DWORD), ('u',INPUT_UNION)]
        user32=ctypes.windll.user32
        user32.SendInput.argtypes=[wintypes.UINT,ctypes.POINTER(INPUT),ctypes.c_int]
        user32.SendInput.restype=wintypes.UINT
        flags=0x0008 | (0x0002 if release else 0)  # SCANCODE | KEYUP
        ev=INPUT(type=1,u=INPUT_UNION(ki=KEYBDINPUT(0,scan,flags,0,0)))
        if user32.SendInput(1,ctypes.byref(ev),ctypes.sizeof(INPUT)) != 1:
            raise OSError('Windows SendInput failed; check permissions')

    @property
    def held(self):
        return self.ctrl_down or self.space_down

    def press(self):
        if self.held:
            return
        try:
            self._emit(VOICE_HOTKEY_MODIFIER_SCAN)
            self.ctrl_down=True
            time.sleep(0.04)
            self._emit(VOICE_HOTKEY_KEY_SCAN)
            self.space_down=True
        except BaseException:
            self.release()
            raise

    def release(self):
        # keyup is required even if dictation is cancelled via STOP/Q/ESC
        first_error=None
        if self.space_down:
            try:
                self._emit(VOICE_HOTKEY_KEY_SCAN,release=True)
            except Exception as exc:
                first_error=exc
            else:
                self.space_down=False
        if self.ctrl_down:
            try:
                self._emit(VOICE_HOTKEY_MODIFIER_SCAN,release=True)
            except Exception as exc:
                if first_error is None:
                    first_error=exc
            else:
                self.ctrl_down=False
        if first_error is not None:
            raise first_error

    def cancel(self):
        # Release held keys, then ask Handy to discard the transcript.
        # ESC cancellation is best effort (depends on Handy window/state).
        self.release()
        time.sleep(0.045)
        self._emit(0x01)       # ESC key down
        time.sleep(0.040)
        self._emit(0x01,True)  # ESC key up


def handy_shortcut_test():
    """Independent Windows smoke test: press/hold Ctrl+Space, release."""
    if sys.platform != 'win32':
        print('This test requires Windows')
        return
    print('Open Handy: Ctrl+Space, shortcut behavior=길게 누르기 (LONG PRESS).')
    print('Focus Notepad in 5 sec; while recording, speak one Korean sentence.')
    for seconds in range(5,0,-1):
        print(f'  {seconds}...',flush=True)
        time.sleep(1.0)
    k=HandyHoldKeys()
    try:
        k.press()
        print('VOICE HELD: talk now; keys stay physically held for 8 seconds.')
        time.sleep(8.0)
    finally:
        k.release()
    print('VOICE RELEASED: Handy should transcribe and paste into Notepad.')



class HandsFreeController:
    """LOCKED => Peace => ACTIVE => (0.5s missing) PAUSED => (3s missing) STANDBY.

    PAUSED recovers automatically only BEFORE the three-second deadline.
    STANDBY allows deliberately armed media-circle recognition but NOT mouse actions.
    Stop => LOCKED blocks even media until a new Peace hold; initial state is LOCKED.
    """
    def __init__(self):
        self.state='LOCKED'
        self.peace_since=None
        self.stop_since=None
        self.last_visible_at=None
        self.resume_since=None

    def force_off(self):
        # Retain call compatibility with STEP15; OFF is now explicit LOCKED.
        old=self.state
        self.state='LOCKED'
        self.peace_since=None
        self.stop_since=None
        self.last_visible_at=None
        self.resume_since=None
        return 'LOCKED' if old!='LOCKED' else None

    def update(self, now, hand_present, label, stop_allowed=True):
        if self.state in ('LOCKED','STANDBY'):
            # In STANDBY, a deliberate STOP can lock all inputs; in LOCKED it does nothing.
            if self.state=='STANDBY' and hand_present and stop_allowed and label=='stop':
                if self.stop_since is None:
                    self.stop_since=now
                elif now-self.stop_since>=STOP_HOLD_S:
                    self.force_off()
                    return 'OFF_BY_STOP'
            else:
                self.stop_since=None
            if hand_present and label=='peace':
                if self.peace_since is None:
                    self.peace_since=now
                elif now-self.peace_since>=TRIGGER_HOLD_S:
                    self.state='ACTIVE'
                    self.peace_since=None
                    self.stop_since=None
                    self.resume_since=None
                    self.last_visible_at=now
                    return 'ACTIVATED'
            else:
                self.peace_since=None
            return None

        if self.state=='ACTIVE':
            if hand_present:
                self.last_visible_at=now
                if stop_allowed and label=='stop':
                    if self.stop_since is None:
                        self.stop_since=now
                    elif now-self.stop_since>=STOP_HOLD_S:
                        self.force_off()
                        return 'OFF_BY_STOP'
                else:
                    self.stop_since=None
            else:
                self.stop_since=None
                if (self.last_visible_at is not None and
                        now-self.last_visible_at >= NO_HAND_PAUSE_S):
                    self.state='PAUSED'
                    self.resume_since=None
                    return 'PAUSED'
            return None

        if self.state=='PAUSED':
            if hand_present:
                if self.resume_since is None:
                    if (self.last_visible_at is None or
                            now-self.last_visible_at>=NO_HAND_OFF_S):
                        self.state='STANDBY'
                        self.peace_since=None
                        return 'STANDBY_BY_TIMEOUT'
                    self.resume_since=now
                elif now-self.resume_since>=HAND_RESUME_STABLE_S:
                    self.state='ACTIVE'
                    self.last_visible_at=now
                    self.resume_since=None
                    self.stop_since=None
                    return 'RESUMED'
            else:
                self.resume_since=None
                if (self.last_visible_at is None or
                        now-self.last_visible_at>=NO_HAND_OFF_S):
                    self.state='STANDBY'
                    self.peace_since=None
                    self.stop_since=None
                    return 'STANDBY_BY_TIMEOUT'
            return None
        raise RuntimeError(f'Invalid HMI state: {self.state}')


def play_state_sound(state):
    """Nonblocking Windows beep. Safe no-op if no audio device is available."""
    if sys.platform != 'win32':
        return
    def sound():
        try:
            import winsound
            patterns = {
                'ACTIVATED': [(920, 90), (1200, 100)],
                'RESUMED': [(1100, 95)],
                'PAUSED': [(520, 140)],
                'OFF_BY_STOP': [(720, 90), (400, 160)],
                'OFF_BY_TIMEOUT': [(430, 180)],
                'VOICE_STARTED': [(800, 80), (980, 70)],
                'VOICE_STOPPED': [(950, 90)],
                'VOICE_READY': [(1180, 80)],
                'STANDBY_BY_TIMEOUT': [(640, 85)],
                'MEDIA': [(1080, 75)],
                'CLEAR_ARMED': [(780, 120)],
                'CLEAR_SELECTED': [(960, 90), (1100, 90)],
                'CLEAR_DELETE': [(1210, 120)],
                'CLEAR_CANCEL': [(450, 110)],
            }
            for frequency, duration in patterns.get(state, []):
                winsound.Beep(frequency, duration)
                time.sleep(.035)
        except Exception:
            pass
    threading.Thread(target=sound, daemon=True).start()

def swipe_metrics(samples, min_vertical=0.075):
    """Evaluate timestamp,x,y path. Return (direction,reason,metrics).
    Result is direction 'UP'/'DOWN' only for a near-vertical, unidirectional path.
    No speed threshold: users can scroll comfortably, not only by flicking.
    """
    metrics = dict(dx=0.0, dy=0.0, distance=0.0, straightness=0.0,
                   monotonicity=0.0, duration=0.0)
    if len(samples) < 3:
        return None, 'few_points', metrics
    pts = np.asarray(samples, dtype=np.float64)
    xy = pts[:, 1:3]
    duration = float(pts[-1, 0] - pts[0, 0])
    dx = float(xy[-1, 0] - xy[0, 0])
    dy = float(xy[-1, 1] - xy[0, 1])
    steps = np.diff(xy, axis=0)
    path = float(np.linalg.norm(steps, axis=1).sum())
    displacement = float(np.hypot(dx, dy))
    straightness = displacement / max(path, 1e-7)
    monotonicity = abs(dy) / max(float(np.abs(steps[:, 1]).sum()), 1e-7)
    metrics.update(dx=dx, dy=dy, distance=abs(dy), duration=duration,
                   straightness=straightness, monotonicity=monotonicity)
    if abs(dy) < min_vertical:
        return None, 'small_vertical_move', metrics
    if abs(dx) > SWIPE_MAX_HORIZONTAL_RATIO * abs(dy):
        return None, 'mostly_sideways', metrics
    if straightness < SWIPE_MIN_STRAIGHTNESS:
        return None, 'path_not_straight', metrics
    if monotonicity < SWIPE_MIN_MONOTONICITY:
        return None, 'went_back_and_forth', metrics
    return ('UP' if dy < 0 else 'DOWN'), 'ok', metrics


def back_metrics(samples):
    """Detect one intentional LEFT swipe in normalized, mirrored camera space.

    Only called in PINCH mode. The action is committed on PINCH RELEASE.
    RIGHT swipes are deliberately ignored in this version.
    """
    info = dict(dx=0.0, dy=0.0, duration=0.0,
                straightness=0.0, monotonicity=0.0)
    if len(samples) < 3:
        return False, 'few_points', info
    pts = np.asarray(samples, dtype=np.float64)
    xy = pts[:, 1:3]
    steps = np.diff(xy, axis=0)
    dx = float(xy[-1, 0] - xy[0, 0])
    dy = float(xy[-1, 1] - xy[0, 1])
    distance = float(np.hypot(dx, dy))
    path = float(np.linalg.norm(steps, axis=1).sum())
    straightness = distance / max(path, 1e-8)
    monotonicity = abs(dx) / max(float(np.abs(steps[:, 0]).sum()), 1e-8)
    info.update(dx=dx, dy=dy, duration=float(pts[-1, 0] - pts[0, 0]),
                straightness=straightness, monotonicity=monotonicity)
    if dx >= 0:
        return False, 'not_left', info
    if abs(dx) < BACK_MIN_HORIZONTAL:
        return False, 'too_short', info
    if abs(dy) > BACK_MAX_VERTICAL_RATIO * abs(dx):
        return False, 'too_diagonal', info
    if straightness < BACK_MIN_STRAIGHTNESS:
        return False, 'not_straight', info
    if monotonicity < BACK_MIN_MONOTONICITY:
        return False, 'back_and_forth', info
    return True, 'ok', info



def enter_metrics(samples):
    """Confirm deliberate RIGHTWARD pinch trajectory upon release.

    Reuse the validated leftward trajectory classifier on a horizontally
    mirrored trace, so thresholds and diagonal rejection match BACK.
    """
    mirrored = [(t, 1.0 - x, y) for t, x, y in samples]
    ok, reason, info = back_metrics(mirrored)
    info['dx'] = -info['dx']
    return ok, reason.replace('not_left', 'not_right'), info

def circle_metrics(samples, aspect=4.0/3.0):
    """Circle/ellipse recognition for (time, x, y) palm-center trajectories.

    Return 'CW', 'CCW' or None; x,y MediaPipe normalized coordinates.
    X is scaled by frame aspect to prevent ordinary image aspect distortion.
    Unlike path-length alone, signed angular travel rejects line swipes.
    """
    detail=dict(angle_deg=0.0, closure=0.0, diameter=0.0, duration=0.0,
                winding=0.0, aspect=0.0)
    if len(samples)<9:
        return None,'few_points',detail
    a=np.asarray(samples,dtype=np.float64)
    p=a[:,1:3].copy()
    p[:,0]*=float(aspect)
    if not np.isfinite(p).all():
        return None,'nonfinite',detail
    duration=float(a[-1,0]-a[0,0]); detail['duration']=duration
    if duration<MEDIA_CIRCLE_MIN_S or duration>PINCH_MAX_S+0.15:
        return None,'duration',detail
    # Remove successive microjitter: rejects angular winding from noisy stationary hand.
    points=[p[0]]
    for v in p[1:]:
        if np.linalg.norm(v-points[-1])>=0.0035:
            points.append(v)
    if len(points)<9:
        return None,'no_motion',detail
    q=np.asarray(points)
    diameter=q.max(axis=0)-q.min(axis=0)
    small=float(np.min(diameter)); large=float(np.max(diameter))
    detail['diameter']=small
    detail['aspect']=small/max(large,1e-8)
    if small<MEDIA_CIRCLE_MIN_DIAMETER:
        return None,'too_small',detail
    if detail['aspect']<MEDIA_CIRCLE_MIN_ASPECT:
        return None,'too_flat',detail
    path_length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
    if path_length<MEDIA_CIRCLE_MIN_PATH:
        return None,'path_short',detail
    center=q.mean(axis=0)
    radii=np.linalg.norm(q-center,axis=1)
    if np.percentile(radii,20)<0.17*np.percentile(radii,80):
        return None,'radial_shape',detail
    # Unwrapped angular progress, centered on actual path centroid.
    theta=np.unwrap(np.arctan2(q[:,1]-center[1],q[:,0]-center[0]))
    delta=np.diff(theta)
    signed=float(delta.sum())
    magnitude=abs(signed)
    detail['angle_deg']=float(np.rad2deg(signed))
    detail['winding']=magnitude/max(float(np.abs(delta).sum()),1e-8)
    if detail['winding']<MEDIA_CIRCLE_MIN_MONOTONICITY:
        return None,'backtracking',detail
    if magnitude<np.deg2rad(MEDIA_CIRCLE_MIN_ANGLE_DEG):
        return None,'incomplete',detail
    if magnitude>np.deg2rad(MEDIA_CIRCLE_MAX_ANGLE_DEG):
        return None,'over_rotation',detail
    radius=float(np.median(radii))
    closure=float(np.linalg.norm(q[-1]-q[0]))
    detail['closure']=closure
    if closure>max(.020,MEDIA_CIRCLE_MAX_CLOSURE_R*radius):
        return None,'not_closed',detail
    # Mirrored preview coordinates: +signed degrees is clockwise (Y down).
    return ('CW' if signed>0 else 'CCW'),'ok',detail


class MediaKeyWorker:
    """State-gated media key injection, independent of mouse worker.

    Authorization is TRUE only in ACTIVE and STANDBY; never PAUSED/LOCKED/VOICE.
    A revoke clears pending requests; physical input already dispatched cannot undo.
    """
    def __init__(self, sender=None):
        self.sender=sender or self._native_media_toggle
        self.lock=threading.Lock()
        self.wakeup=threading.Event()
        self.running=True
        self.allowed=False
        self.pending=False
        self.epoch=0
        self.last_sent=-1e9
        self.thread=threading.Thread(target=self._loop,daemon=True)
        self.thread.start()

    @staticmethod
    def _native_media_toggle():
        """Win32 SendInput using VK_MEDIA_PLAY_PAUSE (0xB3), not focus-bound Space."""
        if sys.platform!='win32':
            raise RuntimeError('Media key injection requires Windows')
        import ctypes
        from ctypes import wintypes
        class KEYBDINPUT(ctypes.Structure):
            _fields_=[('wVk',wintypes.WORD),('wScan',wintypes.WORD),
                      ('dwFlags',wintypes.DWORD),('time',wintypes.DWORD),
                      ('dwExtraInfo',ctypes.c_size_t)]
        class MOUSEINPUT(ctypes.Structure):
            _fields_=[('dx',wintypes.LONG),('dy',wintypes.LONG),
                      ('mouseData',wintypes.DWORD),('dwFlags',wintypes.DWORD),
                      ('time',wintypes.DWORD),('dwExtraInfo',ctypes.c_size_t)]
        class HARDWAREINPUT(ctypes.Structure):
            _fields_=[('uMsg',wintypes.DWORD),('wParamL',wintypes.WORD),
                      ('wParamH',wintypes.WORD)]
        class UNION(ctypes.Union):
            _fields_=[('ki',KEYBDINPUT),('mi',MOUSEINPUT),('hi',HARDWAREINPUT)]
        class INPUT(ctypes.Structure):
            _fields_=[('type',wintypes.DWORD),('u',UNION)]
        user32=ctypes.windll.user32
        user32.SendInput.argtypes=[wintypes.UINT,ctypes.POINTER(INPUT),ctypes.c_int]
        user32.SendInput.restype=wintypes.UINT
        def emit(up):
            ev=INPUT(type=1,u=UNION(ki=KEYBDINPUT(0xB3,0,0x0002 if up else 0,0,0)))
            if user32.SendInput(1,ctypes.byref(ev),ctypes.sizeof(INPUT))!=1:
                raise OSError('Media Play/Pause SendInput failed')
        pressed=False
        try:
            emit(False)
            pressed=True
            time.sleep(.060)
        finally:
            if pressed:
                emit(True)

    def set_authorized(self, allowed):
        with self.lock:
            allowed=bool(allowed)
            if self.allowed!=allowed:
                self.epoch+=1
                self.pending=False
                self.allowed=allowed
        self.wakeup.set()

    def request(self):
        with self.lock:
            if not self.running or not self.allowed:
                return False
            if time.monotonic()-self.last_sent<MEDIA_KEY_COOLDOWN_S:
                return False
            self.pending=True
        self.wakeup.set()
        return True

    def _loop(self):
        while True:
            item=None
            with self.lock:
                if not self.running:
                    return
                if self.allowed and self.pending:
                    item=self.epoch
                    self.pending=False
            if item is not None:
                with self.lock:
                    permitted=self.running and self.allowed and item==self.epoch
                    if permitted:
                        self.last_sent=time.monotonic()
                if permitted:
                    try:
                        self.sender()
                        print('  WINDOWS MEDIA PLAY/PAUSE key sent (media response not verified)')
                    except Exception as exc:
                        print('  MEDIA KEY ERROR:',type(exc).__name__,str(exc))
            else:
                self.wakeup.wait(.02)
                self.wakeup.clear()

    def close(self):
        with self.lock:
            self.running=False
            self.allowed=False
            self.pending=False
            self.epoch+=1
        self.wakeup.set()
        self.thread.join(timeout=1.0)


def delayed_media_test():
    if sys.platform!='win32':
        print('This media keyboard test requires Windows')
        return
    print('Start a YouTube video, switch to another window during countdown.')
    for n in range(5,0,-1):
        print(n,flush=True)
        time.sleep(1)
    MediaKeyWorker._native_media_toggle()
    print('Play/Pause key sent: check whether YouTube actually changed state.')


def self_test():
    # Back only for sustained, clearly leftward horizontal trajectories.
    back_cases = [
        ('left back', [(0,.75,.50),(.2,.67,.51),(.4,.58,.50)], True),
        ('right ignored', [(0,.25,.50),(.2,.35,.50),(.4,.46,.50)], False),
        ('too short', [(0,.60,.50),(.2,.56,.50),(.4,.53,.50)], False),
        ('mostly vertical', [(0,.70,.70),(.2,.63,.55),(.4,.55,.39)], False),
        ('zig-zag', [(0,.70,.50),(.2,.55,.50),(.4,.66,.50),(.6,.53,.50)], False),
    ]
    for name, path, expected in back_cases:
        ok, reason, _ = back_metrics(path)
        assert ok == expected, (name, ok, reason)
    # Preserve original scroll-only recognition on vertical and reject left movement.
    vert, _, _ = swipe_metrics([(0,.5,.65),(.2,.5,.59),(.4,.5,.52)])
    assert vert == 'UP'
    horiz, _, _ = swipe_metrics([(0,.75,.50),(.2,.67,.51),(.4,.58,.50)])
    assert horiz is None
    print('6 left-back / vertical-scroll separation checks passed.')
    right_cases = [
        ('right enter', [(0,.26,.50),(.2,.34,.51),(.4,.43,.50)], True),
        ('left not enter', [(0,.75,.50),(.2,.67,.51),(.4,.58,.50)], False),
        ('right too short', [(0,.40,.50),(.2,.45,.50),(.4,.47,.50)], False),
        ('right diagonal', [(0,.25,.65),(.2,.34,.55),(.4,.44,.42)], False),
        ('right zigzag', [(0,.32,.50),(.2,.48,.50),(.4,.37,.50),(.6,.44,.50)], False),
    ]
    for name, path, expected in right_cases:
        ok, reason, _ = enter_metrics(path)
        assert ok == expected, (name, ok, reason)
    print('5 RIGHT pinch -> Enter trajectory tests passed.')

    cases = [
        ('move up', [(0.0, .5, .65), (.1, .51, .59), (.2, .50, .52)], 'UP'),
        ('move down', [(0.0, .5, .35), (.1, .50, .41), (.2, .51, .48)], 'DOWN'),
        ('small movement', [(0, .5, .5), (.2, .5, .49), (.4, .5, .47)], None),
        ('sideways', [(0, .4, .5), (.1, .5, .51), (.2, .61, .52)], None),
        ('back-and-forth', [(0,.5,.5),(.1,.5,.62),(.2,.5,.48)], None),
    ]
    for name, pts, expected in cases:
        got, reason, diag = swipe_metrics(pts)
        assert got == expected, f'{name}: expected {expected}, got {got} ({reason})'
    print('5 gesture trajectory tests passed.')
    # A static false pinch (one/fist pose) must never start.
    gate = PinchEntryGate()
    for t in (.0, .1, .2, .5, .8):
        assert not gate.update(t, .22, .30, True, .42)
    assert gate.status(.8) == 'WAIT_OPEN'
    # OPEN needs enough time, then a 3D-confirmed CLOSE needs enough time.
    gate = PinchEntryGate()
    assert not gate.update(0., .80, .82, True, .42)
    assert not gate.update(.14, .80, .82, True, .42)
    assert gate.status(.14) == 'READY'
    assert not gate.update(.2, .25, .80, True, .42)  # projection only, depth fails
    assert not gate.update(.3, .25, .30, True, .42)
    assert gate.update(.44, .25, .30, True, .42)
    # A new pinch cannot start without first separating again.
    assert not gate.update(.60, .25, .30, True, .42)
    assert gate.status(.60) == 'WAIT_OPEN'
    # No pinch when ONE wasn't recognized recently.
    gate = PinchEntryGate()
    gate.update(0., .8, .8, False, .42)
    gate.update(.2, .8, .8, False, .42)
    assert not gate.update(.4, .2, .3, False, .42)
    print('4 additional pinch-safety checks passed.')
    # Click requires open->close, a curled middle, an extended index,
    # and an index-far thumb position. A static closed pose NEVER clicks.
    click = ClickEntryGate()
    for t in (.0, .2, .4, .6):
        assert not click.update(t, .22, .30, True, True, .37)
    assert click.status(.6) == 'WAIT_OPEN'
    # Separated + middle folded, then touch proximal third segment.
    click = ClickEntryGate()
    assert not click.update(0.0, .85, .90, True, True, .37)
    assert not click.update(.15, .85, .90, True, True, .37)
    assert click.status(.15) == 'READY'
    # Projected closeness alone is NOT enough if depth is far.
    assert not click.update(.2, .20, .85, True, True, .37)
    assert not click.update(.3, .20, .28, True, True, .37)
    assert click.update(.45, .20, .28, True, True, .37)
    assert not click.update(.6, .20, .28, True, True, .37)
    assert click.status(.6) == 'CLICK_LOCKED'
    # Ineligible (index bent, middle not folded, or no recent ONE) never clicks.
    for wrong in [(False, True), (True, False)]:
        click = ClickEntryGate()
        for t, d in [(0, .9), (.2, .9), (.3, .1), (.6, .1)]:
            assert not click.update(t, d, d, wrong[0], wrong[1], .37)
    # Simple geometric projection on a segment is exact.
    a=np.array([0.,0.]); b=np.array([1.,0.])
    assert abs(point_to_segment_distance(np.array([.5,.4]), a,b)-.4)<1e-6
    # Synthetic pose: straight index, folded middle, thumb touches 9--10 bone.
    from types import SimpleNamespace
    pts=[(.50,.85)]*21
    updates={4:(.53,.53),5:(.43,.55),6:(.43,.45),7:(.43,.36),8:(.43,.25),
             9:(.52,.57),10:(.53,.46),11:(.52,.52),12:(.51,.62),17:(.68,.65)}
    for idx, xy in updates.items(): pts[idx]=xy
    hand=[SimpleNamespace(x=x,y=y,z=0.0) for x,y in pts]
    assert index_extended(hand)
    d2,d3,index_d,folded=click_metrics(hand,aspect=1.0)
    assert folded and d2<.37 and d3<.53 and index_d>.70, (d2,d3,index_d,folded)
    hand[4]=SimpleNamespace(x=.30,y=.60,z=0.0)
    d2_open,_,_,_=click_metrics(hand,aspect=1.0)
    assert d2_open>.37+CLICK_HYSTERESIS
    print('7 click geometry/one-shot gate checks passed.')
    # Click READY must survive brief misclassification while moving to Chrome,
    # but an invalid frame must never fire a click.
    click = ClickEntryGate()
    click.update(0., .9, .9, True, True, .37)
    click.update(.15, .9, .9, True, True, .37)
    assert click.status(.15) == 'READY'
    assert not click.update(.30, .18, .20, True, False, .37)
    assert click.status(.30) == 'READY'
    assert not click.update(.55, .18, .20, False, True, .37)
    assert click.status(.55) == 'READY'
    assert not click.update(.65, .18, .20, True, True, .37)
    assert click.update(.80, .18, .20, True, True, .37)
    click = ClickEntryGate()
    click.update(0., .9, .9, True, True, .37)
    click.update(.15, .9, .9, True, True, .37)
    click.update(.2, .18, .20, False, False, .37)
    click.update(1.6, .18, .20, False, False, .37)
    assert click.status(1.6) == 'WAIT_OPEN'
    assert not click.update(1.7, .18, .20, True, True, .37)
    print('2 click readiness/grace regression checks passed.')
    # Repeat-click regression: after a genuine, brief thumb separation,
    # the next click must work WITHOUT another huge open-pose gesture.
    click = ClickEntryGate()
    threshold = CLICK_CLOSE_INITIAL
    assert not click.update(0., .65, .70, True, True, threshold)
    assert not click.update(.10, .65, .70, True, True, threshold)
    assert click.status(.10) == 'READY'
    assert not click.update(.20, .20, .25, True, True, threshold)
    assert click.update(.31, .20, .25, True, True, threshold)
    assert click.status(.31) == 'CLICK_LOCKED'
    assert not click.update(.35, .20, .25, True, True, threshold)
    click.rearm_after_release(.58)  # live code calls this only after real release
    assert click.status(.58) == 'READY'
    assert not click.update(.62, .20, .25, True, True, threshold)  # cooldown
    assert not click.update(.84, .20, .25, True, True, threshold)
    assert click.update(.95, .20, .25, True, True, threshold)
    assert not click.update(1.20, .20, .25, True, True, threshold)
    assert click.status(1.20) == 'CLICK_LOCKED'
    # A held contact cannot click twice, absent a release/re-arm.
    print('3 repeat-click and debounce regression checks passed.')
    # Release -> reclose edge is required; hovering after lifting must NOT
    # click again without another inward thumb movement.
    click = ClickEntryGate()
    click.update(0.0, .8, .8, True, True, .44)
    click.update(.10, .8, .8, True, True, .44)
    click.update(.20, .20, .20, True, True, .44)
    assert click.update(.30, .20, .20, True, True, .44)
    click.rearm_after_release(.55, released_distance=.36)
    for t in (.60, .70, 1.0):
        assert not click.update(t, .36, .36, True, True, .44)
    assert not click.update(1.10, .20, .20, True, True, .44)
    assert click.update(1.20, .20, .20, True, True, .44)
    print('2 relative-release and reclose checks passed.')

    # 3D hand geometry should be rotation invariant even when 2D foreshortened.
    from types import SimpleNamespace
    points=np.zeros((21,3),dtype=np.float32)
    points[5]=[0.0,0.0,0.0]; points[6]=[0.,.08,0.]
    points[7]=[0.,.16,0.]; points[8]=[0.,.24,0.]
    points[17]=[.20,0.0,0.0]
    points[9]=[.1,0.,0.]; points[10]=[.1,.08,0.]
    points[11]=[.1,.07,0.]; points[12]=[.1,.02,0.]
    points[4]=[.10,.05,0.]
    theta=np.deg2rad(75.)
    R=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],
                [0,np.sin(theta),np.cos(theta)]],dtype=np.float32)
    tilted=points@R.T
    sample=[SimpleNamespace(x=float(x),y=float(y),z=float(z))
            for x,y,z in tilted]
    ix,mid,d,far=world_click_pose(sample)
    assert ix and mid and d<.44 and far>.62, (ix,mid,d,far)
    print('1 tilted-hand 3D orientation regression check passed.')

    # Asynchronous mouse backend works with a stub; no actual clicks here.
    class FakeMouse:
        def __init__(self): self.calls = []
        def moveTo(self, x, y, duration=0): self.calls.append(('move',x,y))
        def click(self, button='left'): self.calls.append(('click',button))
        def scroll(self, amount): self.calls.append(('scroll',amount))
        def hotkey(self, *keys): self.calls.append(('hotkey', *keys))
    fake = FakeMouse()
    worker = LatestMouseWorker(fake)
    try:
        worker.set_enabled(True)
        worker.move((20, 30))
        worker.act('click', xy=(100, 200))
        worker.act('scroll', 120)
        worker.act('back')
        deadline = time.monotonic() + 1.0
        while len(fake.calls) < 4 and time.monotonic() < deadline:
            time.sleep(.01)
        assert ('move',100,200) in fake.calls
        assert ('click','left') in fake.calls
        assert ('scroll',120) in fake.calls
        assert ('hotkey','alt','left') in fake.calls
        worker.set_enabled(False)
    finally:
        worker.close()
    print('4 asynchronous mouse / back action checks passed.')

    # STEP17: near-circle + deliberately distorted oval / tiny circle / line negative.
    def mkcircle(deg=350.,rx=.060,ry=.050,clockwise=True,duration=1.1,
                 center=(.5,.5),n=38):
        angle=np.linspace(0,np.deg2rad(deg)*(1 if clockwise else -1),n)
        return [(duration*i/(n-1),center[0]+rx*np.cos(a)/(4./3.),
                 center[1]+ry*np.sin(a)) for i,a in enumerate(angle)]
    direction,why,diag=circle_metrics(mkcircle())
    assert (direction,why)==('CW','ok'),(direction,why,diag)
    direction,why,diag=circle_metrics(mkcircle(deg=312,rx=.040,ry=.026))
    assert direction=='CW',(why,diag)
    direction,why,diag=circle_metrics(mkcircle(deg=360,rx=.080,ry=.023))
    assert direction=='CW',(why,diag)  # accept noticeably oval drawing
    direction,why,diag=circle_metrics(mkcircle(clockwise=False))
    assert direction=='CCW',(why,diag)
    direction,why,diag=circle_metrics(mkcircle(deg=260))
    assert direction is None,(why,diag)
    direction,why,diag=circle_metrics(mkcircle(rx=.013,ry=.012))
    assert direction is None,(why,diag)
    direction,why,diag=circle_metrics([(i*.05,.25+.012*i,.5+.001*i) for i in range(25)])
    assert direction is None,(why,diag)
    print('STEP17 clockwise 300deg / oval / CCW / tiny / straight rejection tests passed.')
    fired=[]
    mw=MediaKeyWorker(sender=lambda:fired.append('media'))
    try:
        assert mw.request() is False  # LOCKED
        mw.set_authorized(True)
        assert mw.request() is True
        for _ in range(70):
            if fired: break
            time.sleep(.01)
        assert fired==['media'],fired
        assert mw.request() is False  # cooldown
        mw.set_authorized(False)  # STOP/PAUSED revokes media
        assert mw.request() is False
    finally:
        mw.close()
    print('STEP17 Media worker authorization and one-shot cooldown tests passed.')

    # Hands-free controller checks with deterministic synthetic clock.
    ctl = HandsFreeController()
    assert ctl.update(.0, True, 'peace') is None
    assert ctl.update(.5, True, 'peace') is None
    assert ctl.update(1.01, True, 'peace') == 'ACTIVATED'
    assert ctl.state == 'ACTIVE'
    assert ctl.update(1.1, True, 'one') is None
    assert ctl.update(1.4, False, 'NO_HAND') is None
    assert ctl.update(1.61, False, 'NO_HAND') == 'PAUSED'
    assert ctl.state == 'PAUSED'
    assert ctl.update(1.72, True, 'one') is None
    assert ctl.state == 'PAUSED'
    assert ctl.update(1.94, True, 'one') == 'RESUMED'
    assert ctl.state == 'ACTIVE'
    assert ctl.update(2.0, True, 'stop') is None
    assert ctl.update(2.65, True, 'stop') == 'OFF_BY_STOP'
    assert ctl.state == 'LOCKED'
    ctl.update(3., True, 'peace')
    assert ctl.update(4.02, True, 'peace') == 'ACTIVATED'
    ctl.update(4.1, True, 'one')
    assert ctl.update(4.7, False, 'NO_HAND') == 'PAUSED'
    assert ctl.update(7.11, False, 'NO_HAND') == 'STANDBY_BY_TIMEOUT'
    assert ctl.state == 'STANDBY'
    assert ctl.update(7.2, True, 'one') is None
    assert ctl.state == 'STANDBY'
    # Reappearing just before the deadline must be allowed to stabilize.
    ctl = HandsFreeController()
    ctl.update(0.00, True, 'peace')
    assert ctl.update(1.01, True, 'peace') == 'ACTIVATED'
    ctl.update(1.05, True, 'one')
    assert ctl.update(1.56, False, 'NO_HAND') == 'PAUSED'
    assert ctl.update(4.00, True, 'one') is None  # seen within 3 s
    assert ctl.update(4.23, True, 'one') == 'RESUMED'
    assert ctl.state=='ACTIVE'
    ctl=HandsFreeController()
    assert ctl.update(10.,True,'peace') is None
    assert ctl.update(11.01,True,'peace')=='ACTIVATED'
    ctl.update(11.1,True,'one')
    assert ctl.update(11.7,False,'NO_HAND')=='PAUSED'
    assert ctl.update(14.11,False,'NO_HAND')=='STANDBY_BY_TIMEOUT'
    assert ctl.state=='STANDBY'
    assert ctl.update(14.5,True,'one') is None
    assert ctl.state=='STANDBY'  # no automatic pointer resume
    assert ctl.update(15.0,True,'stop') is None
    assert ctl.update(15.65,True,'stop')=='OFF_BY_STOP'
    assert ctl.state=='LOCKED'
    assert ctl.update(16.,True,'one') is None
    assert ctl.state=='LOCKED'
    assert ctl.update(16.5,True,'peace') is None
    assert ctl.update(17.55,True,'peace')=='ACTIVATED'
    print('STEP17 LOCKED / PAUSED / STANDBY / PEACE-only resume tests passed.')

    # Existing click, scroll and back commands must be revoked across PAUSED.
    class GatedMouse:
        def __init__(self):
            self.calls = []
            self.move_entered = threading.Event()
            self.unblock_move = threading.Event()
        def moveTo(self, x, y, duration=0):
            self.move_entered.set()
            self.unblock_move.wait(timeout=2.)
            self.calls.append(('move', x, y))
        def click(self, button='left'): self.calls.append(('click', button))
        def scroll(self, amt): self.calls.append(('scroll', amt))
        def hotkey(self, *keys): self.calls.append(('hotkey', *keys))
    fake = GatedMouse()
    worker = LatestMouseWorker(fake)
    try:
        worker.set_enabled(True)
        worker.move((30, 30))
        assert fake.move_entered.wait(1.0), 'Worker did not begin move'
        worker.act('click', xy=(100, 100))
        worker.act('scroll', 240)
        worker.act('back')
        worker.act('enter')
        worker.set_enabled(False)  # PAUSED cancels the queued actions
        worker.set_enabled(True)   # RESUMED must not replay previous actions
        fake.unblock_move.set()
        time.sleep(.10)
        assert not any(x[0] in ('click', 'scroll', 'hotkey', 'enter') for x in fake.calls), fake.calls
    finally:
        fake.unblock_move.set()
        worker.close()
    print('Queued OS commands revoked across PAUSED / resume check passed.')
    class EnterMouse:
        def __init__(self): self.calls = []
        def press(self, key): self.calls.append(('press', key))
        def moveTo(self, *args, **kwargs): pass
    enter_fake = EnterMouse()
    enter_worker = LatestMouseWorker(enter_fake)
    try:
        enter_worker.set_enabled(True)
        enter_worker.act('enter')
        for _ in range(40):
            if enter_fake.calls: break
            time.sleep(.01)
        assert enter_fake.calls == [('press', 'enter')], enter_fake.calls
    finally:
        enter_worker.close()
    print('Async ENTER action queued and emitted once (OS mocked).')
    ctrl = HandsFreeController()
    ctrl.state = 'ACTIVE'
    assert finish_voice_controller(ctrl, 10.0, True) == 'ACTIVE'
    assert ctrl.last_visible_at == 10.0
    assert finish_voice_controller(ctrl, 11.0, False) == 'PAUSED'
    assert ctrl.update(11.12, True, 'one') is None
    assert ctrl.update(11.35, True, 'one') == 'RESUMED'
    assert ctrl.state == 'ACTIVE'
    print('Voice finishes -> ACTIVE if hand visible; PAUSED -> RESUMED otherwise.')

    # Hold-to-talk events: one press, one release. No repeat while held.
    gate = ThumbPressGate(hold_s=.45,release_s=.30,missing_release_s=.90)
    assert gate.update(0.0,True,True) is None
    assert gate.update(.30,True,True) is None
    assert gate.update(.48,True,True) == 'PRESS'
    assert gate.update(2.0,True,True) is None
    # A short camera dropout does NOT stop recording.
    assert gate.update(2.10,False,False) is None
    assert gate.update(2.42,True,True) is None
    # Explicit thumb relaxation for 0.30s ends recording once.
    assert gate.update(2.50,False,True) is None
    assert gate.update(2.85,False,True) == 'RELEASE'
    assert gate.update(3.00,False,True) is None
    assert gate.update(3.34,False,True) is None
    assert gate.update(3.40,True,True) is None
    assert gate.update(3.88,True,True) == 'PRESS'
    # Hand loss for 0.90s ends held mic rather than leaving keys stuck.
    assert gate.update(4.00,False,False) is None
    assert gate.update(4.95,False,False) == 'RELEASE'
    # simulate error reset: ensure thumb still held cannot retrigger.
    gate.reset(lock_until_release=True)
    assert gate.update(5.10,True,True) is None
    print('Hold-to-talk press/release/no-hand-grace tests passed.')
    # Mock Win32 scan injection: CTRL/SPACE both held until release, one time.
    held_keys=HandyHoldKeys()
    emitted=[]
    held_keys._emit=lambda scan,release=False: emitted.append((scan,release))
    held_keys.press()
    held_keys.press()  # long hold must not repeat keydown
    assert held_keys.held
    held_keys.release()
    held_keys.release()  # duplicate release must be harmless
    assert not held_keys.held
    assert emitted == [(VOICE_HOTKEY_MODIFIER_SCAN,False),
                       (VOICE_HOTKEY_KEY_SCAN,False),
                       (VOICE_HOTKEY_KEY_SCAN,True),
                       (VOICE_HOTKEY_MODIFIER_SCAN,True)], emitted
    print('Windows hold-key lifecycle/idempotence mock test passed.')
    # Geometry should not classify the existing one / peace / fist click as thumb.
    from types import SimpleNamespace
    coords = {0:(.52,.88),1:(.49,.80),2:(.45,.70),3:(.41,.53),4:(.38,.37),
              5:(.45,.57),6:(.46,.50),7:(.47,.58),8:(.45,.64),
              9:(.53,.56),10:(.54,.49),11:(.55,.59),12:(.54,.64),
              13:(.61,.56),14:(.62,.49),15:(.63,.59),16:(.62,.64),
              17:(.70,.58),18:(.70,.52),19:(.71,.60),20:(.70,.64)}
    pose=[SimpleNamespace(x=coords[i][0],y=coords[i][1],z=0.0)
          for i in range(21)]
    yes, detail = thumbs_up_geometry(pose, aspect=1.0)
    assert yes, detail
    pose[8] = SimpleNamespace(x=.45, y=.28, z=0.0)  # index pointing up
    yes, _ = thumbs_up_geometry(pose, aspect=1.0)
    assert not yes
    print('Thumbs-up landmarks / one-pointer rejection tests passed.')
    # STEP19 v3a: a brief or noisy landmark classification must not immediately
    # release Handy's physically held Ctrl+Space key combination.
    gate=ThumbPressGate(hold_s=.45,release_s=.55,missing_release_s=.90,min_held_s=1.20)
    assert gate.update(0.,True,True) is None
    assert gate.update(.46,True,True)=='PRESS'
    for t in (.62,.80,1.0,1.30):
        assert gate.update(t,False,True) is None
    assert gate.update(1.38,True,True) is None  # recovers, never released
    assert gate.update(1.70,False,True) is None
    assert gate.update(2.27,False,True)=='RELEASE'
    # Genuine prolonged loss of the entire hand still releases keys safely.
    gate=ThumbPressGate(hold_s=.45,release_s=.55,missing_release_s=.90,min_held_s=1.20)
    gate.update(0.,True,True)
    assert gate.update(.46,True,True)=='PRESS'
    assert gate.update(.50,False,False) is None
    assert gate.update(1.43,False,False)=='RELEASE'
    # The loose test retains a marginal raised thumb without allowing a
    # straight index (ONE gesture) to masquerade as thumbs-up.
    sample=[SimpleNamespace(x=coords[i][0],y=coords[i][1],z=0.0)
            for i in range(21)]
    assert thumbs_up_recording_hold_geometry(sample,aspect=1.)
    sample[8]=SimpleNamespace(x=.45,y=.28,z=0.0)
    assert not thumbs_up_recording_hold_geometry(sample,aspect=1.)
    print('STEP19 v3a voice-hold hysteresis, min duration, safe release tests passed.')
    # Distance-controlled scrolling: accepted motions always stay 240..720.
    for dy, amount in ((.02,240), (.055,240), (.075,240), (.10,320),
                       (.15,480), (-.20,640), (.225,720), (.40,720)):
        assert proportional_scroll_ticks(dy) == amount, (dy, proportional_scroll_ticks(dy))
    assert proportional_scroll_ticks(.075, base_ticks=120) == 240
    assert proportional_scroll_ticks(.40, base_ticks=300) == 720
    assert proportional_scroll_ticks(-.15) == proportional_scroll_ticks(.15)
    assert SCROLL_MIN_TICKS <= proportional_scroll_ticks(.10) <= SCROLL_MAX_TICKS
    assert relative_drag_position((.50,.50),(.50,.50),(500,300),
                                  REGIONS[2],1920,1080) == (500,300)
    assert relative_drag_position((.55,.50),(.50,.50),(500,300),
                                  REGIONS[2],1920,1080)[0] > 500
    print('Proportional scroll and relative-drag coordinate tests passed.')

    class FakeDragMouse:
        def __init__(self): self.calls = []
        def moveTo(self,x,y,duration=0): self.calls.append(('move',x,y))
        def mouseDown(self,button='left'): self.calls.append(('down',button))
        def mouseUp(self,button='left'): self.calls.append(('up',button))
        def click(self,button='left'): self.calls.append(('click',button))
    fd=FakeDragMouse(); dw=LatestMouseWorker(fd)
    try:
        dw.set_enabled(True)
        dw.act('drag_start',xy=(100,200))
        for _ in range(80):
            if ('down','left') in fd.calls: break
            time.sleep(.01)
        assert ('down','left') in fd.calls, fd.calls
        dw.move((200,250))
        dw.set_enabled(False)  # PAUSED must release physical mouse button
        for _ in range(80):
            if ('up','left') in fd.calls: break
            time.sleep(.01)
        assert ('up','left') in fd.calls, fd.calls
        assert fd.calls.index(('down','left')) < fd.calls.index(('up','left'))
    finally:
        dw.close()
    print('Drag mouseDown and forced mouseUp-on-pause test passed.')
    fd=FakeDragMouse(); dw=LatestMouseWorker(fd)
    try:
        dw.set_enabled(True)
        dw.act('drag_start', xy=(300, 400))
        for _ in range(80):
            if ('down','left') in fd.calls: break
            time.sleep(.01)
        assert ('down','left') in fd.calls, fd.calls
        dw.move((320, 415))
        dw.act('drag_end')
        for _ in range(80):
            if ('up','left') in fd.calls: break
            time.sleep(.01)
        assert fd.calls.count(('down','left')) == 1
        assert fd.calls.count(('up','left')) == 1
    finally:
        dw.close()
    print('Drag release fires mouseUp once in normal flow.')
    # STEP15 volume tests: a fully extended gun pose, not Peace or THUMBS-UP.
    from types import SimpleNamespace
    pts={0:(.50,.90),1:(.38,.78),2:(.30,.74),3:(.22,.68),4:(.15,.60),
         5:(.43,.62),6:(.42,.50),7:(.42,.39),8:(.42,.28),
         9:(.51,.60),10:(.52,.48),11:(.52,.37),12:(.52,.27),
         13:(.60,.65),14:(.61,.57),15:(.60,.67),16:(.60,.76),
         17:(.68,.68),18:(.70,.60),19:(.69,.70),20:(.68,.76)}
    sample=[SimpleNamespace(x=pts[i][0],y=pts[i][1],z=0.) for i in range(21)]
    yes,diag=gun_volume_geometry(sample,aspect=1.)
    assert yes,diag
    sample[4]=SimpleNamespace(x=.44,y=.66,z=0.) # thumb tucked -> not gun
    assert not gun_volume_geometry(sample,aspect=1.)[0]
    sample[4]=SimpleNamespace(x=.40,y=.88,z=0.) # thumb looks far from tips but is NOT open
    assert not gun_volume_geometry(sample,aspect=1.)[0]
    sample[4]=SimpleNamespace(x=.15,y=.60,z=0.)
    sample[12]=SimpleNamespace(x=.70,y=.27,z=0.) # V gap too wide
    assert not gun_volume_geometry(sample,aspect=1.)[0]
    sample[12]=SimpleNamespace(x=.52,y=.27,z=0.)
    sample[8]=SimpleNamespace(x=.45,y=.64,z=0.)  # ONE-like index curled
    assert not gun_volume_geometry(sample,aspect=1.)[0]
    gate=VolumePoseGate()
    assert gate.update(0.,True) is None
    assert gate.update(.69,True) is None
    assert gate.update(.71,True)=='ENTER'
    assert gate.update(1.0,True) is None
    assert gate.update(1.01,False) is None
    assert gate.update(1.15,False)=='EXIT'
    assert gate.update(1.20,True) is None  # cannot reenter while same held pose
    gate.update(1.4,False)
    assert gate.update(1.5,True) is None
    assert gate.update(2.25,True)=='ENTER'
    assert gate.update(2.3,False,False)=='EXIT'  # lost hand -> immediate exit
    print('Volume gun-pose separation, 0.7 s hold, and release tests passed.')
    class FakeVolume:
        def __init__(self,level=30.): self.level=level; self.history=[]
        def get_percent(self): return self.level
        def set_percent(self,value): self.level=value; self.history.append(value)
    fake=FakeVolume(); motion=VolumeMotion(fake)
    assert motion.start(0.,(.5,.5))==30.0
    assert motion.update(.05,(.5,.5)) is None
    last=fake.level
    for i in range(1,41):
        t=.05+i*.05
        y=.50-i*.007  # intentionally fast up movement
        motion.update(t,(.5,y))
        assert fake.level-last <= VOLUME_MAX_RATE_PERCENT_S*.05+1e-5
        last=fake.level
    higher=fake.level
    assert higher>30.
    for i in range(41,47):
        motion.update(.05+i*.05,(.5,.50-40*.007))
    still=fake.level
    for i in range(47,52):
        motion.update(.05+i*.05,(.5,.50-40*.007))
    assert abs(fake.level-still)<1e-6, (fake.level,still)
    assert fake.level<=30+VOLUME_MAX_RATE_PERCENT_S*2.6
    motion.stop()
    assert motion.update(3.0,(.5,.2)) is None
    assert fake.level==still
    print('Volume rate-limit, movement-only, and stop/no-backlog tests passed.')
    # STEP19 v2 master mute independent of volume; moving UP clears it.
    # Inject a fake Windows endpoint so tests run without a Windows device.
    class FakeEndpoint:
        def __init__(self,level=0.40,muted=True):
            self.level=level; self.muted=muted; self.events=[]
        def GetMasterVolumeLevelScalar(self): return self.level
        def SetMasterVolumeLevelScalar(self,v,_):
            self.level=float(v); self.events.append(('volume',round(v,4)))
        def GetMute(self): return int(self.muted)
        def SetMute(self,m,_):
            self.muted=bool(m); self.events.append(('mute',int(m)))
    fake_endpoint=FakeEndpoint()
    winvol=WindowsVolume()
    winvol.endpoint=fake_endpoint
    assert winvol.get_percent()==40.
    assert winvol.is_muted()
    # DOWN adjusts scalar but keeps Windows mute ON.
    m=VolumeMotion(winvol)
    m.start(0.,(.5,.5))
    assert m.update(.1,(.5,.53)) is not None
    assert fake_endpoint.muted and not any(e[0]=='mute' for e in fake_endpoint.events)
    # UP is an explicit request to hear sound: remove mute exactly once.
    assert m.update(.5,(.5,.45)) is not None
    assert fake_endpoint.muted is False
    assert fake_endpoint.events[-1]==('mute',0),fake_endpoint.events
    m.stop()
    # Simply entering volume control does NOT remove mute.
    fake_endpoint=FakeEndpoint(level=1.0,muted=True)
    winvol=WindowsVolume(); winvol.endpoint=fake_endpoint
    m=VolumeMotion(winvol)
    assert m.start(0.,(.5,.5))==100.
    assert m.update(.1,(.5,.5)) is None
    assert fake_endpoint.muted
    # Even at 100% (no scalar can change), UP must unmute.
    assert m.update(.2,(.5,.45)) is None  # first filtered frame may be median-neutral
    assert m.update(.25,(.5,.43)) is None
    assert fake_endpoint.muted is False
    assert ('mute',0) in fake_endpoint.events
    m.stop()
    # A transient driver failure to GetMute cannot break VOLUME mode.
    class BrokenMuteEndpoint(FakeEndpoint):
        def GetMute(self): raise RuntimeError('simulated query failure')
    broken=BrokenMuteEndpoint()
    winvol=WindowsVolume(); winvol.endpoint=broken
    m=VolumeMotion(winvol)
    m.start(0.,(.5,.5))
    assert m.update(.2,(.5,.46)) is not None
    assert m.active
    # STEP19 v3: volume=0 and mute=ON is a common everyday start state.
    # First vertical axis movement UP must set a clearly nonzero level and
    # remove mute immediately, without requiring a second upward gesture.
    fake_endpoint=FakeEndpoint(level=0.0,muted=True)
    winvol=WindowsVolume(); winvol.endpoint=fake_endpoint
    m=VolumeMotion(winvol)
    assert m.start(0.,(.5,.46),initial_up=True)==VOLUME_ZERO_WAKE_PERCENT
    assert abs(fake_endpoint.level*100-VOLUME_ZERO_WAKE_PERCENT)<1e-5
    assert fake_endpoint.muted is False
    assert fake_endpoint.events[:2]==[('volume',VOLUME_ZERO_WAKE_PERCENT/100),('mute',0)]
    assert not any(e[0]=='mute' and e[1]==1 for e in fake_endpoint.events)
    first=fake_endpoint.level*100
    assert m.update(.1,(.5,.44)) is not None
    assert fake_endpoint.level*100>=first
    m.stop()
    # At 0% with mute OFF, the same UP should still make sound.
    fake_endpoint=FakeEndpoint(level=0.0,muted=False)
    winvol=WindowsVolume(); winvol.endpoint=fake_endpoint
    m=VolumeMotion(winvol)
    assert m.start(0.,(.5,.46),initial_up=True)==VOLUME_ZERO_WAKE_PERCENT
    assert fake_endpoint.muted is False
    assert fake_endpoint.events==[('volume',VOLUME_ZERO_WAKE_PERCENT/100)]
    m.stop()
    # Entering VOLUME with DOWN must not override deliberate mute or 0%.
    fake_endpoint=FakeEndpoint(level=0.0,muted=True)
    winvol=WindowsVolume(); winvol.endpoint=fake_endpoint
    m=VolumeMotion(winvol)
    assert m.start(0.,(.5,.54),initial_up=False)==0.0
    assert fake_endpoint.muted is True and fake_endpoint.level==0
    m.update(.1,(.5,.57)); m.update(.2,(.5,.60))
    assert fake_endpoint.muted and fake_endpoint.level==0
    # Reverse to UP while still in VOLUME: zero wake must happen here too.
    m.update(.3,(.5,.50)); m.update(.4,(.5,.47)); m.update(.5,(.5,.43))
    assert abs(fake_endpoint.level*100)>=VOLUME_ZERO_WAKE_PERCENT
    assert fake_endpoint.muted is False
    m.stop()
    # No immediate unmute on a gun pose with no first vertical axis movement.
    fake_endpoint=FakeEndpoint(level=0.0,muted=True)
    winvol=WindowsVolume(); winvol.endpoint=fake_endpoint
    m=VolumeMotion(winvol)
    assert m.start(0.,(.5,.5))==0.0
    assert fake_endpoint.events==[] and fake_endpoint.muted
    m.stop()
    # An UP at existing nonzero level removes mute without an 8% override.
    fake_endpoint=FakeEndpoint(level=0.36,muted=True)
    winvol=WindowsVolume(); winvol.endpoint=fake_endpoint
    m=VolumeMotion(winvol)
    assert m.start(0.,(.5,.4),initial_up=True)==36.0
    assert abs(fake_endpoint.level-0.36)<1e-6
    assert not fake_endpoint.muted and fake_endpoint.events==[('mute',0)]
    m.stop()
    print('STEP19 v3: zero-volume UP wakes 8% + unmutes; DOWN stays silent; subsequent UP wakes; nonfatal errors passed.')



    # STEP18 v4 OK needs thumb-index closed AND other THREE fingers extended.
    from types import SimpleNamespace
    pts=np.zeros((21,3),dtype=float)
    pts[5]=[0,0,0]; pts[17]=[.24,0,0]
    # index curled in near thumb; middle/ring/pinky fully extended
    pts[6]=[0,-.08,0]; pts[7]=[.04,-.09,0]; pts[8]=[.10,-.07,0]
    pts[9]=[.08,0,0]; pts[10]=[.08,-.09,0];pts[11]=[.08,-.18,0];pts[12]=[.08,-.26,0]
    pts[13]=[.16,0,0];pts[14]=[.16,-.09,0];pts[15]=[.16,-.18,0];pts[16]=[.16,-.25,0]
    pts[18]=[.24,-.07,0];pts[19]=[.24,-.14,0];pts[20]=[.24,-.20,0]
    pts[2]=[.16,.03,0];pts[3]=[.13,-.02,0];pts[4]=[.10,-.07,0]
    sample=[SimpleNamespace(x=float(x),y=float(y),z=float(z)) for x,y,z in pts]
    ok,d=ok_clear_geometry(sample,aspect=1.)
    assert ok,d
    sample[16]=SimpleNamespace(x=.16,y=-.03,z=0.)
    assert not ok_clear_geometry(sample,aspect=1.)[0]  # ring folded
    assert not ok_clear_geometry(None)[0]
    print('STEP18 v4 OK geometry positive/negative checks passed.')

    # OK shares THREE extended fingers with Stop. Even if the MLP says STOP,
    # open-palm geometry must block it; genuine open palm still shuts down.
    sample[16]=SimpleNamespace(x=.16,y=-.25,z=0.)
    ok,d=ok_clear_geometry(sample,aspect=1.)
    assert ok,d
    stop_ok,stop_d=stop_open_palm_geometry(sample,aspect=1.)
    assert not stop_ok,stop_d
    assert effective_stop_label('stop',stop_ok)=='STOP_SHAPE_REJECTED'
    openpts=np.zeros((21,3),dtype=float)
    openpts[5]=[0,0,0]; openpts[17]=[.24,0,0]
    for mcp,pip,dip,tip,x in ((5,6,7,8,0.),(9,10,11,12,.08),
                              (13,14,15,16,.16),(17,18,19,20,.24)):
        openpts[mcp]=[x,0,0]
        openpts[pip]=[x,-.08,0]
        openpts[dip]=[x,-.16,0]
        openpts[tip]=[x,-.24,0]
    openpts[2]=[-.09,-.02,0];openpts[3]=[-.13,-.09,0];openpts[4]=[-.18,-.15,0]
    openhand=[SimpleNamespace(x=float(x),y=float(y),z=float(z)) for x,y,z in openpts]
    truly_stop,stop_d=stop_open_palm_geometry(openhand,aspect=1.)
    assert truly_stop,stop_d
    assert effective_stop_label('stop',truly_stop)=='stop'
    ctl=HandsFreeController();ctl.state='ACTIVE'
    for now in (0.,.3,.7,1.0):
        ctl.update(now,True,effective_stop_label('stop',stop_ok))
    assert ctl.state=='ACTIVE','OK must not LOCK the HMI'
    assert ctl.update(1.1,True,'stop') is None
    assert ctl.update(1.8,True,'stop')=='OFF_BY_STOP'
    assert ctl.state=='LOCKED'
    print('STEP18 v4 OK-vs-STOP isolation and genuine Stop retention tests passed.')


    # FIST >= 0.5s, OK >= 0.5s, exactly one delete with release latch.
    g=ClearTextGate()
    assert g.update(0.,True,False,True,True) is None
    assert g.update(.49,True,False,True,True) is None
    assert g.update(.51,True,False,True,True)=='ARMED'
    assert g.update(.60,False,True,True,True) is None
    assert g.update(1.09,False,True,True,True) is None
    assert g.update(1.11,False,True,True,True)=='DELETE'
    assert g.update(1.60,False,True,True,True) is None  # no repeat
    assert g.update(1.70,False,False,True,True) is None  # unlock
    assert g.update(1.80,False,True,True,True) is None  # OK alone not valid
    assert g.update(2.00,False,False,True,True) is None
    # FIST too short, insufficient OK hold, missing hand, no click, timeout.
    g=ClearTextGate(); g.update(0.,True,False,True,True)
    assert g.update(.30,False,True,True,True)=='CANCEL'
    g=ClearTextGate(); g.update(0.,True,False,True,True)
    assert g.update(.52,True,False,True,True)=='ARMED'
    assert g.update(.60,False,True,True,True) is None
    assert g.update(.80,False,False,True,True) is None
    assert g.update(.90,False,True,True,True) is None
    assert g.update(1.41,False,True,True,True)=='DELETE'  # new uninterrupted OK hold
    g=ClearTextGate(); g.update(0.,True,False,True,True)
    assert g.update(.52,True,False,True,True)=='ARMED'
    assert g.update(.70,False,True,False,True)=='CANCEL'
    g=ClearTextGate(); g.update(0.,True,False,True,True)
    assert g.update(.52,True,False,True,True)=='ARMED'
    assert g.update(.70,False,True,True,False)=='CANCEL'
    g=ClearTextGate(); g.update(0.,True,False,True,True)
    assert g.update(.52,True,False,True,True)=='ARMED'
    assert g.update(3.30,True,False,True,True)=='CANCEL'
    print('STEP18 v4 FIST 0.5s -> OK 0.5s, anti-repeat/cancel regression tests passed.')

    # Fake UIA edit guard and keyboard worker: missing recent click or blocked
    # focus never calls hotkey or backspace. Successful fake focus clears once.
    class FakeEditGuard:
        def __init__(self,yes): self.yes=yes
        def inspect(self): return (self.yes,'test-edit' if self.yes else 'not-edit',None)
    class FakeEditMouse:
        def __init__(self): self.calls=[]
        def hotkey(self,*keys): self.calls.append(('hotkey',*keys))
        def press(self,key): self.calls.append(('press',key))
        def moveTo(self,*a,**k): self.calls.append(('move',))
        def click(self,button='left'): self.calls.append(('click',button))
    fake_edit=FakeEditMouse()
    wc=LatestMouseWorker(fake_edit,focus_guard=FakeEditGuard(True))
    try:
        wc.set_enabled(True)
        # Tests run on Linux; mock the foreground HWND value in module scope.
        orig_hwnd=globals()['_foreground_hwnd']
        globals()['_foreground_hwnd']=lambda: 100
        try:
            assert not wc._safe_clear_text(wc.epoch)
            wc.last_click_at=time.monotonic()
            wc.last_click_hwnd=100
            assert wc._safe_clear_text(wc.epoch)
            assert ('hotkey','ctrl','a') in fake_edit.calls
            assert ('press','backspace') in fake_edit.calls
            count=len(fake_edit.calls)
            wc.focus_guard=FakeEditGuard(False)
            assert not wc._safe_clear_text(wc.epoch)
            assert len(fake_edit.calls)==count
            wc.focus_guard=FakeEditGuard(True)
            wc.last_click_at=time.monotonic()-CLEAR_RECENT_CLICK_S-1
            assert not wc._safe_clear_text(wc.epoch)
            assert len(fake_edit.calls)==count
            wc.last_click_at=time.monotonic()
            wc.last_click_hwnd=101  # another application in foreground
            assert not wc._safe_clear_text(wc.epoch)
        finally:
            globals()['_foreground_hwnd']=orig_hwnd
    finally:
        wc.close()
    print('STEP18 worker safety tests: UIA edit, recent click, focus window, no unintended Backspace passed.')
    class SwitchFocusGuard:
        def __init__(self): self.calls=0
        def inspect(self):
            self.calls+=1
            return True,'test-edit', ('field-A' if self.calls==1 else 'field-B')
    changed_mouse=FakeEditMouse()
    changed_wc=LatestMouseWorker(changed_mouse,focus_guard=SwitchFocusGuard())
    orig_hwnd=globals()['_foreground_hwnd']
    globals()['_foreground_hwnd']=lambda: 222
    try:
        changed_wc.set_enabled(True)
        changed_wc.last_click_at=time.monotonic()
        changed_wc.last_click_hwnd=222
        assert not changed_wc._safe_clear_text(changed_wc.epoch)
        assert ('hotkey','ctrl','a') in changed_mouse.calls
        assert ('press','backspace') not in changed_mouse.calls
    finally:
        globals()['_foreground_hwnd']=orig_hwnd
        changed_wc.close()
    print('STEP18 v5 changed focused edit prevents Backspace, even after Ctrl+A.')
    # YouTube-like combobox may change UIAutomation runtime IDs after Ctrl+A.
    pre=('50003','search-input','0',(320,90,700,126),(1,10,10))
    refresh=('50003','search-input','0',(321,90,700,126),(1,99,42))
    other=('50003','search-input','0',(320,190,700,226),(1,99,42))
    assert _same_focused_edit(pre,refresh)[0]
    assert not _same_focused_edit(pre,other)[0]
    edit_child=('50004','search-input','0',(323,93,699,126),(90,12))
    assert _same_focused_edit(pre,edit_child,(400,110))[0]
    assert not _same_focused_edit(pre,edit_child,(400,300))[0]
    assert _point_in_edit(pre,(400,110))
    assert not _point_in_edit(pre,(400,300))
    class StableChromeCombo:
        def __init__(self):self.calls=0
        def inspect(self):
            self.calls+=1
            return True,'Focused editable ComboBoxControl',pre if self.calls==1 else refresh
    chrome_mouse=FakeEditMouse()
    combo_worker=LatestMouseWorker(chrome_mouse,focus_guard=StableChromeCombo())
    orig_hwnd=globals()['_foreground_hwnd']
    globals()['_foreground_hwnd']=lambda: 999
    try:
        combo_worker.set_enabled(True)
        combo_worker.last_click_at=time.monotonic()
        combo_worker.last_click_hwnd=999
        combo_worker.last_click_xy=(400,110)
        assert combo_worker._safe_clear_text(combo_worker.epoch)
        assert chrome_mouse.calls.count(('press','backspace'))==1
    finally:
        globals()['_foreground_hwnd']=orig_hwnd
        combo_worker.close()
    class ClickedChromeCombo:
        def inspect(self):return True,'Focused editable ComboBoxControl',pre
    late_worker=LatestMouseWorker(FakeEditMouse(),focus_guard=ClickedChromeCombo())
    orig_hwnd=globals()['_foreground_hwnd']
    globals()['_foreground_hwnd']=lambda: 999
    try:
        late_worker.set_enabled(True)
        late_worker.last_click_at=time.monotonic()
        late_worker.last_click_hwnd=777   # click-time foreground raced
        late_worker.last_click_xy=(400,110)
        assert late_worker.recent_input_click()
        assert late_worker.last_click_hwnd==999
        late_worker.last_click_hwnd=777
        late_worker.last_click_xy=(400,300)  # outside search control
        assert not late_worker.recent_input_click()
    finally:
        globals()['_foreground_hwnd']=orig_hwnd
        late_worker.close()
    print('STEP18 v5 YouTube ComboBox runtime refresh / delayed focus / click-anchor tests passed.')
    # STEP18 v6: Recover specifically from a YouTube autocomplete ListItem.
    # Everything below is OS-mocked: no Windows keys are sent by self_test.
    class SuggestionGuard:
        def __init__(self, restore=True):
            self.restore=restore
            self.calls=0
        def inspect(self):
            self.calls+=1
            if self.calls==1 or not self.restore:
                return False,("Not a focused editable text field: ListItemControl "
                              "type=50007 focused=True enabled=True name='suggestion'"),None
            return True,("Focused editable field: ComboBoxControl "
                         "type=50003 focused=True enabled=True name='검색'"),pre
    for restore in (True,False):
        mock_mouse=FakeEditMouse()
        g=SuggestionGuard(restore)
        w=LatestMouseWorker(mock_mouse,focus_guard=g)
        old_foreground=globals()['_foreground_hwnd']
        globals()['_foreground_hwnd']=lambda: 999
        try:
            w.set_enabled(True)
            w._foreground=lambda:'(1435) YouTube - Chrome'
            w.last_click_at=time.monotonic()
            w.last_click_hwnd=999
            w.last_click_xy=(400,110)
            result=w._safe_clear_text(w.epoch)
            assert result==restore
            assert mock_mouse.calls.count(('press','esc'))==1
            assert mock_mouse.calls.count(('hotkey','ctrl','a'))==(1 if restore else 0)
            assert mock_mouse.calls.count(('press','backspace'))==(1 if restore else 0)
        finally:
            globals()['_foreground_hwnd']=old_foreground
            w.close()
    # List items from other sites must NOT trigger Escape or delete.
    other_mouse=FakeEditMouse()
    w=LatestMouseWorker(other_mouse,focus_guard=SuggestionGuard(True))
    old_foreground=globals()['_foreground_hwnd']
    globals()['_foreground_hwnd']=lambda: 999
    try:
        w.set_enabled(True)
        w._foreground=lambda:'NAVER - Chrome'
        w.last_click_at=time.monotonic()
        w.last_click_hwnd=999
        assert not w._safe_clear_text(w.epoch)
        assert other_mouse.calls==[]
    finally:
        globals()['_foreground_hwnd']=old_foreground
        w.close()
    print('STEP18 v6 YouTube suggestions: Esc recovery / fail-closed / non-YouTube blocked tests passed.')


    # Worker stage regression: select is NOT delete; commit requires stable focus.
    fake2=FakeEditMouse()
    wc=LatestMouseWorker(fake2,focus_guard=FakeEditGuard(True))
    orig_hwnd=globals()['_foreground_hwnd']
    globals()['_foreground_hwnd']=lambda: 900
    try:
        wc.set_enabled(True)
        wc.last_click_at=time.monotonic(); wc.last_click_hwnd=900
        sequence=wc.begin_text_selection()
        for _ in range(100):
            if wc.get_text_selection()[0] in ('READY','BLOCKED'):break
            time.sleep(.01)
        assert wc.get_text_selection()[0]=='READY',wc.get_text_selection()
        assert ('hotkey','ctrl','a') in fake2.calls
        assert not any(x[0]=='press' for x in fake2.calls)
        wc.commit_text_selection(sequence)
        for _ in range(100):
            if wc.get_text_selection()[0] in ('DONE','BLOCKED'):break
            time.sleep(.01)
        assert wc.get_text_selection()[0]=='DONE'
        assert fake2.calls.count(('press','backspace'))==1
        sequence=wc.begin_text_selection()
        wc.cancel_text_selection()  # revoke old pending action
        wc.commit_text_selection(sequence) # stale confirmation must be rejected
        time.sleep(.05)
        assert fake2.calls.count(('press','backspace'))==1
        wc.last_click_at=time.monotonic(); wc.last_click_hwnd=900
        sequence=wc.begin_text_selection()
        for _ in range(100):
            if wc.get_text_selection()[0]=='READY':break
            time.sleep(.01)
        wc.focus_guard=FakeEditGuard(False)
        wc.commit_text_selection(sequence)
        for _ in range(100):
            if wc.get_text_selection()[0]=='BLOCKED':break
            time.sleep(.01)
        assert wc.get_text_selection()[0]=='BLOCKED'
        assert fake2.calls.count(('press','backspace'))==1
    finally:
        globals()['_foreground_hwnd']=orig_hwnd
        wc.close()
    print('Legacy async selection/commit/focus-change/revoke worker tests passed (not used in v4).')


def to_features(hand, mean, std, clip):
    xy = np.asarray([[p.x, p.y] for p in hand], dtype=np.float32)
    if xy.shape != (21, 2) or not np.isfinite(xy).all():
        return None
    xy = xy - xy[0].copy()
    scale = float(np.linalg.norm(xy[5] - xy[17]))
    if scale < 1e-6:
        return None
    feat = np.clip((xy / scale).reshape(42), -clip, clip)
    return ((feat - mean) / std).astype(np.float32)


def index_extended(hand):
    xy = np.asarray([[p.x, p.y] for p in hand], dtype=np.float32)
    a = xy[6] - xy[5]
    b = xy[8] - xy[6]
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    ratio = float(np.linalg.norm(xy[8] - xy[5])) / max(la, 1e-6)
    cosine = float(np.dot(a, b)) / max(la * lb, 1e-6)
    return ratio >= 1.65 and cosine >= 0.10


def pinch_ratio_and_palm(hand, aspect):
    """Measure thumb-tip (4) to index-tip (8) in 2D AND estimated 3D.

    X is aspect-corrected in frame-height units; MediaPipe z is scaled
    approximately like x, so correct z the same way. z is only an estimate,
    not calibrated physical depth; the dynamic open->close gate is primary.
    """
    xy = np.asarray([[p.x * aspect, p.y] for p in hand], dtype=np.float32)
    xyz = np.asarray([[p.x * aspect, p.y, p.z * aspect] for p in hand],
                     dtype=np.float32)
    palm_width = float(np.linalg.norm(xy[5] - xy[17]))
    if palm_width < 1e-6 or not np.isfinite(xyz).all():
        return float('inf'), float('inf'), None
    ratio_2d = float(np.linalg.norm(xy[4] - xy[8])) / palm_width
    ratio_3d = float(np.linalg.norm(xyz[4] - xyz[8])) / palm_width
    # Palm-center movement, kept unchanged from STEP 8.
    palm = np.asarray([(hand[5].x + hand[17].x) * 0.5,
                       (hand[5].y + hand[17].y) * 0.5], dtype=np.float32)
    return ratio_2d, ratio_3d, palm


class PinchEntryGate:
    """Require intentional OPEN -> CLOSE transition, not just a static fist.

    A fist with the thumb tucked may project near the index finger in 2D.
    Such a pose must NOT trigger pinch unless an observed, sustained OPEN
    state was followed by sustained contact with 3D confirmation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.open_since = None
        self.armed_until = -1e9
        self.close_since = None

    def update(self, now, dist_2d, dist_3d, one_recent, close_threshold):
        open_threshold = close_threshold + PINCH_HYSTERESIS
        if dist_2d >= open_threshold:
            if self.open_since is None:
                self.open_since = now
            elif now - self.open_since >= PINCH_OPEN_HOLD_S and one_recent:
                self.armed_until = now + PINCH_ARM_WINDOW_S
        else:
            self.open_since = None

        near = (dist_2d < close_threshold and
                dist_3d < close_threshold + PINCH_3D_EXTRA)
        ready = now <= self.armed_until
        if near and ready and one_recent:
            if self.close_since is None:
                self.close_since = now
            elif now - self.close_since >= PINCH_CLOSE_HOLD_S:
                self.reset()
                return True
        else:
            self.close_since = None
        return False

    def status(self, now):
        if self.close_since is not None:
            return 'TOUCH_VERIFY'
        if now <= self.armed_until:
            return 'READY'
        if self.open_since is not None:
            return 'OPEN_VERIFY'
        return 'WAIT_OPEN'


def point_to_segment_distance(p, a, b):
    """Distance to a finite bone segment, not just to a single joint."""
    v = b - a
    t = float(np.dot(p - a, v)) / max(float(np.dot(v, v)), 1e-10)
    closest = a + float(np.clip(t, 0.0, 1.0)) * v
    return float(np.linalg.norm(p - closest))


def click_metrics(hand, aspect):
    """Return thumb-to-middle proximal bone distances and finger configuration.

    Third segment when counting from the middle fingertip is landmark 10->9.
    MediaPipe z is an approximate monocular depth (not true physical contact).
    """
    xy = np.asarray([[p.x * aspect, p.y] for p in hand], dtype=np.float32)
    xyz = np.asarray([[p.x * aspect, p.y, p.z * aspect] for p in hand],
                     dtype=np.float32)
    width = float(np.linalg.norm(xy[5] - xy[17]))
    if width < 1e-6 or not np.isfinite(xyz).all():
        return float('inf'), float('inf'), float('inf'), False
    d2 = point_to_segment_distance(xy[4], xy[9], xy[10]) / width
    d3 = point_to_segment_distance(xyz[4], xyz[9], xyz[10]) / width
    thumb_index = float(np.linalg.norm(xy[4] - xy[8])) / width

    base = xy[10] - xy[9]       # middle MCP -> PIP
    distal = xy[12] - xy[10]    # middle PIP -> fingertip
    a = float(np.linalg.norm(base))
    b = float(np.linalg.norm(distal))
    cos = float(np.dot(base, distal)) / max(a*b, 1e-6)
    tip_reach = float(np.linalg.norm(xy[12] - xy[9])) / max(a, 1e-6)
    middle_folded = (tip_reach < CLICK_MIDDLE_FOLD_RATIO or
                     cos < CLICK_MIDDLE_FOLD_COS)
    return d2, d3, thumb_index, bool(middle_folded)


def world_click_pose(world_hand):
    """Orientation-tolerant supporting features from MediaPipe 3D *estimated* joints.

    World landmarks reduce 2D foreshortening for a tilted hand, but remain
    model estimates and do not measure real physical contact.
    Returns index_straight, middle_bent, thumb-to-middle, thumb-to-index.
    """
    if world_hand is None or len(world_hand) != 21:
        return False, False, float('inf'), float('inf')
    xyz = np.asarray([[p.x, p.y, p.z] for p in world_hand], dtype=np.float32)
    if not np.isfinite(xyz).all():
        return False, False, float('inf'), float('inf')
    width = float(np.linalg.norm(xyz[5] - xyz[17]))
    if width < 1e-6:
        return False, False, float('inf'), float('inf')
    index_chain = sum(float(np.linalg.norm(xyz[b]-xyz[a]))
                      for a,b in ((5,6),(6,7),(7,8)))
    middle_chain = sum(float(np.linalg.norm(xyz[b]-xyz[a]))
                       for a,b in ((9,10),(10,11),(11,12)))
    index_reach = float(np.linalg.norm(xyz[8]-xyz[5])) / max(index_chain,1e-6)
    middle_reach = float(np.linalg.norm(xyz[12]-xyz[9])) / max(middle_chain,1e-6)
    middle_dist = point_to_segment_distance(xyz[4], xyz[9], xyz[10]) / width
    index_dist = float(np.linalg.norm(xyz[4]-xyz[8])) / width
    return index_reach >= 0.79, middle_reach <= 0.80, middle_dist, index_dist


class LatestMouseWorker:
    """Offload Windows mouse calls from the camera loop; coalesce old moves.

    Actions (click/scroll/back) are not coalesced, but PAUSED/OFF clears them
    and invalidates previously queued work. Only one thread sends OS input.
    """
    def __init__(self, pyauto, focus_guard=None):
        self.pyauto = pyauto
        self.focus_guard=focus_guard if focus_guard is not None else FocusedEditGuard()
        self.last_click_at=-1e9
        self.last_click_hwnd=None
        self.last_click_xy=None
        self.last_click_root=None
        self.clear_sequence=0
        self.selection_status='IDLE'
        self.selection_focus=None
        self.selection_hwnd=None
        self.selection_at=-1e9
        self.selection_reason=''
        self.back_mode = "SENDINPUT"  # B toggles SENDINPUT / PYAUTOGUI
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.running = True
        self.enabled = False
        self.failed = False
        self.epoch = 0  # revoke older queued/popped work on PAUSE or OFF
        self.target = None
        self.drag_down = False        # only mouse worker changes this
        self.release_requested = False
        self.actions = deque(maxlen=10)
        self.last_move_at = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def toggle_back_mode(self):
        with self.lock:
            self.back_mode = ("PYAUTOGUI" if self.back_mode == "SENDINPUT"
                              else "SENDINPUT")
            return self.back_mode

    @staticmethod
    def _native_alt_left():
        """Send Alt+Left as real keyboard input via Win32 SendInput.

        Uses scan codes (not a window message). This targets the current
        foreground window, like a physical keyboard shortcut. Success means
        Windows accepted input, NOT that the browser navigated.
        """
        import ctypes
        from ctypes import wintypes

        ULONG_PTR = ctypes.c_size_t
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ULONG_PTR)]
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]
        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD),
                        ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]
        class INPUT_UNION(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT),
                        ("hi", HARDWAREINPUT)]
        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]

        user32 = ctypes.windll.user32
        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        user32.SendInput.restype = wintypes.UINT
        SCANCODE, EXTENDED, KEYUP = 0x0008, 0x0001, 0x0002
        def emit(scan, up=False, extended=False):
            flags = SCANCODE | (EXTENDED if extended else 0) | (KEYUP if up else 0)
            event = INPUT(type=1, u=INPUT_UNION(ki=KEYBDINPUT(0, scan, flags, 0, 0)))
            if user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
                raise OSError('SendInput could not inject a keyboard event (permissions/UIPI?)')

        # 0x38 = left Alt; E0 4B = Left Arrow.
        alt_down = False
        left_down = False
        try:
            emit(0x38)
            alt_down = True
            time.sleep(0.035)
            emit(0x4B, extended=True)
            left_down = True
            time.sleep(0.090)
        finally:
            if left_down:
                emit(0x4B, up=True, extended=True)
            if alt_down:
                emit(0x38, up=True)

    def _send_back(self):
        """Send Alt+Left to foreground app; no window-message shortcut."""
        mode = self.back_mode
        if mode == "SENDINPUT" and sys.platform == 'win32' and hasattr(self.pyauto, 'KEYBOARD_KEYS'):
            try:
                self._native_alt_left()
                print('  BACK Win32 SendInput Alt+Left injected; foreground:',
                      self._foreground(), '(navigation not verified)')
                return
            except Exception as exc:
                print('  BACK SendInput error:', type(exc).__name__, str(exc),
                      'falling back to PyAutoGUI')
        # Keep fake PyAutoGUI compatible with self-test.
        if sys.platform == 'win32' and hasattr(self.pyauto, 'KEYBOARD_KEYS'):
            self.pyauto.hotkey('alt', 'left', interval=0.09)
        else:
            self.pyauto.hotkey('alt', 'left')
        print('  BACK PyAutoGUI Alt+Left injected; foreground:',
              self._foreground(), '(navigation not verified)')

    def _send_enter(self):
        """Inject the physical Enter key to the foreground app once."""
        if sys.platform == 'win32' and hasattr(self.pyauto, 'KEYBOARD_KEYS'):
            # 0x1C is the standard keyboard Enter scan code.
            keys = HandyHoldKeys()
            down = False
            try:
                keys._emit(0x1C)
                down = True
                time.sleep(0.06)
            finally:
                if down:
                    keys._emit(0x1C, release=True)
            print('  WINDOWS ENTER sent (SendInput); foreground:', self._foreground(),
                  '(submission not verified)')
        else:
            self.pyauto.press('enter')
            print('  WINDOWS ENTER sent (fallback); foreground:', self._foreground())

    def recent_input_click(self,now=None):
        """Require a recent HMI click in this app, tolerating delayed focus.

        The click target is first anchored by WindowFromPoint, not a potentially
        stale foreground HWND. If they still disagree, only rebind when the
        actually focused editable control contains the HMI click coordinates.
        """
        now=time.monotonic() if now is None else now
        with self.lock:
            last_at=self.last_click_at
            last_hwnd=self.last_click_hwnd
            last_xy=self.last_click_xy
            last_root=self.last_click_root
            enabled=self.enabled and not self.failed
        if not enabled or now-last_at>CLEAR_RECENT_CLICK_S or last_hwnd is None:
            return False
        foreground=_foreground_hwnd()
        if last_hwnd==foreground:
            return True
        # If Windows resolved the clicked root, never silently authorize a new
        # foreground application. Fall back only for unresolved click roots.
        if (foreground is None or last_xy is None or last_root is not None or
                now-last_at>2.0):
            return False
        ok,reason,signature=self.focus_guard.inspect()
        if not ok or not _point_in_edit(signature,last_xy):
            return False
        # Only accept a delayed foreground correction for the same clicked edit.
        with self.lock:
            if self.last_click_at!=last_at or self.last_click_xy!=last_xy:
                return False
            self.last_click_hwnd=foreground
        print('  CLEAR CLICK FOCUS SYNC: confirmed clicked editable field after async focus')
        return True

    def _youtube_suggestion_candidate(self, reason):
        """Narrow recovery ONLY for YouTube's focused autocomplete list item.

        No deletion permission is granted here. The current foreground is still
        checked and a genuine editable input must regain focus before Ctrl+A.
        """
        title=self._foreground().lower()
        return ('youtube' in title and 'chrome' in title and
                'ListItemControl' in str(reason) and
                'Not a focused editable text field:' in str(reason))

    def _inspect_clear_target(self, token):
        """Verify edit focus; optionally dismiss YouTube suggestions with Esc.

        YouTube may send keyboard focus to an autocomplete ListItem after a
        pointer click. Escape is non-destructive; no Ctrl+A or Backspace is sent
        unless UI Automation confirms an editable field afterwards.
        """
        ok, reason, target=self.focus_guard.inspect()
        if ok:
            return ok,reason,target
        if not self._youtube_suggestion_candidate(reason):
            return ok,reason,target
        old_hwnd=_foreground_hwnd()
        print('  CLEAR YOUTUBE SUGGESTION: list item has focus; trying Esc to dismiss popup')
        self.pyauto.press('esc')
        time.sleep(0.09)
        if (not self._still_allowed(token) or not self.recent_input_click() or
                _foreground_hwnd()!=old_hwnd):
            return False,'YouTube focus recovery cancelled: window/HMI changed',None
        ok2,reason2,target2=self.focus_guard.inspect()
        if not ok2:
            return False,('YouTube suggestions dismissed but search edit did not regain focus: '
                          +str(reason2)),None
        # This pathway is strictly for the YouTube search combo box, not an
        # arbitrary editable element somewhere else on the page.
        if not ('ComboBoxControl' in str(reason2) and
                ("name='검색'" in str(reason2) or 'name="검색"' in str(reason2))):
            return False,'Recovered focus is not the YouTube search ComboBox: '+str(reason2),None
        print('  CLEAR YOUTUBE FOCUS RECOVERED: YouTube search box focused; safe checks continue')
        return True,reason2,target2

    def _safe_clear_text(self,token):
        """Worker-side re-check, independent of visual pose detection.

        Strict UIA EditControl + same focused target check before/after Ctrl+A.
        If the target changed, do not send the destructive key.
        """
        if not self._still_allowed(token) or self.drag_down or not self.recent_input_click():
            print('  CLEAR BLOCKED: no recent gesture click in the same foreground window')
            return False
        ok,reason,focused_target=self._inspect_clear_target(token)
        hwnd=_foreground_hwnd()
        if not ok:
            print('  CLEAR BLOCKED:',reason)
            return False
        # PyAutoGUI injects focused key events and releases modifiers by itself.
        self.pyauto.hotkey('ctrl','a')
        if not self._still_allowed(token) or not self.recent_input_click():
            print('  CLEAR STOPPED after selection: HMI state/window changed')
            return False
        ok2,reason2,target2=self.focus_guard.inspect()
        same_edit,match_reason=_same_focused_edit(focused_target,target2,self.last_click_xy)
        if not ok2 or _foreground_hwnd()!=hwnd or not same_edit:
            print('  CLEAR STOPPED before Backspace: UIA focus changed;'
                  ' initial=',focused_target,' current=',target2,
                  ' reason=',reason2,' match=',match_reason)
            return False
        if match_reason!='exact':
            print('  CLEAR FOCUS STABLE: YouTube-style UIA refresh accepted;',match_reason)
        self.pyauto.press('backspace')
        print('  CLEAR DONE: verified focused input -> Ctrl+A, Backspace (one request)')
        return True

    def cancel_text_selection(self):
        """Invalidate pending selection/commit messages; no destructive key."""
        with self.lock:
            self.clear_sequence+=1
            self.selection_status='IDLE'
            self.selection_focus=None
            self.selection_hwnd=None
            self.selection_at=-1e9
            self.selection_reason=''

    def begin_text_selection(self):
        with self.lock:
            self.clear_sequence+=1
            sequence=self.clear_sequence
            self.selection_status='PENDING'
            self.selection_focus=None
            self.selection_reason=''
        self.act('clear_select', amount=sequence)
        return sequence

    def get_text_selection(self):
        with self.lock:
            return self.selection_status, self.selection_reason, self.clear_sequence

    def _safe_select_text(self, token, sequence):
        with self.lock:
            current=sequence==self.clear_sequence and self.selection_status=='PENDING'
        if not current or not self._still_allowed(token) or self.drag_down or not self.recent_input_click():
            return False
        ok,reason,signature=self.focus_guard.inspect()
        if not ok:
            with self.lock:
                if sequence==self.clear_sequence:
                    self.selection_status='BLOCKED'; self.selection_reason=reason
            print('  CLEAR SELECT BLOCKED:',reason)
            return False
        with self.lock:
            if sequence!=self.clear_sequence: return False
        if not self._still_allowed(token) or not self.recent_input_click(): return False
        self.pyauto.hotkey('ctrl','a')  # selection only; no deletion here
        with self.lock:
            if sequence!=self.clear_sequence: return False
            self.selection_status='READY'
            self.selection_focus=signature
            self.selection_hwnd=_foreground_hwnd()
            self.selection_at=time.monotonic()
        print('  CLEAR TEXT SELECTED (not deleted): hold OK for confirmation')
        return True

    def _safe_commit_text(self, token, sequence):
        with self.lock:
            allowed=(self.selection_status=='READY' and sequence==self.clear_sequence)
            focus=self.selection_focus
            hwnd=self.selection_hwnd
            selected_at=self.selection_at
            self.selection_status='COMMITTING' if allowed else self.selection_status
        if not allowed: return False
        ok,reason,signature=self.focus_guard.inspect()
        valid=(ok and signature==focus and _foreground_hwnd()==hwnd and
               time.monotonic()-selected_at<=CLEAR_SELECT_LIFETIME_S and
               self._still_allowed(token) and self.recent_input_click() and
               not self.drag_down)
        if not valid:
            with self.lock:
                if sequence==self.clear_sequence:
                    self.selection_status='BLOCKED'
                    self.selection_reason='Focus changed or text input not verified: '+reason
            print('  CLEAR DELETE BLOCKED: selected field changed or focus unsafe:',reason)
            return False
        with self.lock:
            if sequence!=self.clear_sequence: return False
        self.pyauto.press('backspace')
        with self.lock:
            if sequence==self.clear_sequence:
                self.selection_status='DONE'
                self.selection_reason='Backspace sent to verified text field'
        print('  CLEAR DONE: one Backspace after staged OK confirmation')
        return True

    def commit_text_selection(self,sequence):
        self.act('clear_commit',amount=sequence)

    def set_enabled(self, enabled):
        with self.lock:
            self.epoch += 1
            self.enabled = bool(enabled) and not self.failed
            self.target = None
            self.actions.clear()  # discard all pending actions on OFF / PAUSED
            self.clear_sequence+=1
            self.selection_status='IDLE'
            self.selection_focus=None
            if not enabled:
                self.last_click_at=-1e9
                self.last_click_hwnd=None
                self.release_requested = True  # emergency mouseUp, even if action queue cleared
        self.wakeup.set()

    def move(self, xy):
        with self.lock:
            if self.enabled and not self.failed:
                self.target = (self.epoch, tuple(map(int, xy)))
        self.wakeup.set()

    def act(self, kind, amount=0, xy=None):
        with self.lock:
            if self.enabled and not self.failed:
                self.target = None
                self.actions.append((self.epoch, kind, amount, xy))
        self.wakeup.set()

    def _still_allowed(self, token):
        with self.lock:
            return (self.running and self.enabled and not self.failed and
                    token == self.epoch)

    def close(self):
        with self.lock:
            self.running = False
            self.enabled = False
            self.epoch += 1
            self.actions.clear()
            self.target = None
            self.release_requested = True
        self.wakeup.set()
        self.thread.join(timeout=2.0)

    def _loop(self):
        while True:
            action = None
            target = None
            emergency_release = False
            with self.lock:
                if self.release_requested:
                    emergency_release = True
                    self.release_requested = False
                if not self.running and not self.drag_down and not emergency_release:
                    return
                if self.enabled and not self.failed and not emergency_release:
                    if self.actions:
                        action = self.actions.popleft()
                    elif self.target is not None:
                        elapsed = time.monotonic() - self.last_move_at
                        if elapsed >= 1.0 / 45.0:
                            target = self.target
                            self.target = None
            try:
                if emergency_release:
                    if self.drag_down:
                        self.pyauto.mouseUp(button='left')
                        self.drag_down = False
                        print('DRAG: mouseUp on PAUSED / OFF / quit')
                    continue
                if not self.running:
                    if self.drag_down:
                        self.pyauto.mouseUp(button='left')
                        self.drag_down = False
                    return
                if action is not None:
                    token, kind, amount, xy = action
                    if not self._still_allowed(token):
                        continue  # stale work from before PAUSED/OFF
                    if kind == 'click':
                        if xy is not None:
                            self.pyauto.moveTo(*xy, duration=0)
                        self.pyauto.click(button='left')
                        clicked_at=time.monotonic()
                        clicked_root=_window_root_at_point(xy) if xy is not None else None
                        with self.lock:
                            self.last_click_at=clicked_at
                            self.last_click_hwnd=clicked_root or _foreground_hwnd()
                            self.last_click_xy=tuple(xy) if xy is not None else None
                            self.last_click_root=clicked_root
                        print('  WINDOWS CLICK; foreground:', self._foreground(),
                              ' click-root:',clicked_root)
                    elif kind == 'drag_start':
                        if xy is not None:
                            self.pyauto.moveTo(*xy, duration=0)
                        if self._still_allowed(token) and not self.drag_down:
                            self.pyauto.mouseDown(button='left')
                            self.drag_down = True
                            print('  WINDOWS DRAG START at', xy)
                    elif kind == 'drag_end':
                        if self.drag_down:
                            self.pyauto.mouseUp(button='left')
                            self.drag_down = False
                            print('  WINDOWS DRAG END')
                    elif kind == 'scroll':
                        self.pyauto.scroll(int(amount))
                    elif kind == 'back':
                        self._send_back()
                    elif kind == 'enter':
                        self._send_enter()
                    elif kind == 'clear_text':
                        self._safe_clear_text(token)  # legacy regression path
                    elif kind == 'clear_select':
                        self._safe_select_text(token,amount)
                    elif kind == 'clear_commit':
                        self._safe_commit_text(token,amount)
                elif target is not None:
                    token, xy = target
                    if not self._still_allowed(token):
                        continue
                    self.pyauto.moveTo(*xy, duration=0)
                    self.last_move_at = time.monotonic()
                else:
                    self.wakeup.wait(0.008)
                    self.wakeup.clear()
            except Exception as exc:
                if self.drag_down:
                    try:
                        self.pyauto.mouseUp(button='left')
                        self.drag_down = False
                    except Exception:
                        pass
                self.failed = True
                self.set_enabled(False)
                print('Mouse backend disabled:', type(exc).__name__, str(exc))

    @staticmethod
    def _foreground():
        if sys.platform != 'win32':
            return '(not Windows)'
        try:
            import ctypes
            u = ctypes.windll.user32
            u.GetForegroundWindow.restype = ctypes.c_void_p
            hwnd = u.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(140)
            u.GetWindowTextW.argtypes = [ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_int]
            u.GetWindowTextW(hwnd,buf,len(buf))
            return buf.value or '(unnamed window)'
        except Exception:
            return '(unknown)'


class ClickEntryGate:
    """Trigger only after deliberate AWAY -> TOUCH on a folded middle finger.

    A stationary one pose with thumb naturally resting on the fist NEVER
    clicks, because it did not show a prior separated-thumb observation.
    After firing, gate is latched until release; no repeated clicks on hold.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.open_since = None
        self.armed_until = -1e9
        self.close_since = None
        self.locked = False
        self.invalid_since = None
        self.last_click_at = -1e9
        self.reclose_cutoff = float('inf')

    def rearm_after_release(self, now, released_distance=None):
        """Confirmed small thumb lift -> ready for the NEXT click.

        Crucially do NOT call reset(): in v2 reset required a large thumb
        opening a second time, causing the user's WAIT_OPEN complaint.
        Keep last_click_at for debounce even across multiple clicks.
        """
        self.locked = False
        self.close_since = None
        self.open_since = None
        self.invalid_since = None
        self.armed_until = now + CLICK_ARM_WINDOW_S
        self.reclose_cutoff = (float(released_distance) - CLICK_RECLOSE_DELTA
                               if released_distance is not None else float('inf'))

    def update(self, now, d2, d3, index_far, eligible, threshold):
        if self.locked:
            return False
        if not eligible or not index_far:
            # While aiming at Chrome, the hand can briefly rotate or the MLP can
            # disagree with the landmark posture. Do not erase READY at once.
            # Never CLICK on an invalid frame, regardless of the grace period.
            self.close_since = None
            self.open_since = None
            if self.invalid_since is None:
                self.invalid_since = now
            elif now - self.invalid_since > CLICK_INVALID_GRACE_S:
                self.armed_until = -1e9
            return False
        self.invalid_since = None
        open_threshold = threshold + CLICK_HYSTERESIS
        if d2 >= open_threshold:
            if self.open_since is None:
                self.open_since = now
            elif now - self.open_since >= CLICK_OPEN_HOLD_S:
                self.armed_until = now + CLICK_ARM_WINDOW_S
            self.close_since = None
        else:
            self.open_since = None
            near = (d2 < threshold and d2 < self.reclose_cutoff and
                    d3 < threshold + CLICK_3D_EXTRA)
            if (near and now <= self.armed_until
                    and now - self.last_click_at >= CLICK_MIN_BETWEEN_CLICKS_S):
                if self.close_since is None:
                    self.close_since = now
                elif now - self.close_since >= CLICK_CLOSE_HOLD_S:
                    self.locked = True
                    self.last_click_at = now
                    self.close_since = None
                    self.armed_until = -1e9
                    return True
            else:
                self.close_since = None
        return False

    def status(self, now):
        if self.locked:
            return 'CLICK_LOCKED'
        if self.close_since is not None:
            return 'TOUCH_VERIFY'
        if now <= self.armed_until:
            return 'READY'
        if self.open_since is not None:
            return 'OPEN_VERIFY'
        return 'WAIT_OPEN'


def screen_coordinates(p, region, sw, sh):
    x0, x1, y0, y1 = region
    x = float(np.clip((p[0] - x0) / (x1 - x0), 0, 1))
    y = float(np.clip((p[1] - y0) / (y1 - y0), 0, 1))
    return (int((SCREEN_MARGIN + (1 - 2 * SCREEN_MARGIN) * x) * (sw - 1)),
            int((SCREEN_MARGIN + (1 - 2 * SCREEN_MARGIN) * y) * (sh - 1)))


# STEP 19 v3 camera portability: a webcam may be device 1/2 instead of 0,
# and Windows OpenCV devices may support DSHOW, MSMF or only system default.
# No hand/model inference or Windows commands run in --camera-test mode.
def camera_cli_options(args=None):
    args = list(sys.argv[1:] if args is None else args)
    choice = None
    backend = 'auto'
    for j, arg in enumerate(args):
        if arg == '--camera-index':
            if j + 1 >= len(args):
                raise ValueError('--camera-index requires 0, 1, 2, ...')
            try:
                choice = int(args[j + 1])
            except ValueError as exc:
                raise ValueError('--camera-index must be an integer') from exc
            if not 0 <= choice <= 20:
                raise ValueError('--camera-index must be between 0 and 20')
        if arg == '--camera-backend':
            if j + 1 >= len(args):
                raise ValueError('--camera-backend requires auto/dshow/msmf/any')
            backend = args[j + 1].lower()
            if backend not in ('auto', 'dshow', 'msmf', 'any'):
                raise ValueError('--camera-backend must be auto/dshow/msmf/any')
    return choice, backend


def _camera_backend_options(cv2, selected):
    choices = [
        ('DSHOW', cv2.CAP_DSHOW),
        ('MSMF', cv2.CAP_MSMF),
        ('ANY', cv2.CAP_ANY),
    ]
    if selected == 'auto':
        return choices
    return [x for x in choices if x[0].lower() == selected]


def open_compatible_camera(cv2, index=None, backend='auto', max_index=4,
                           reporter=print):
    """Require a valid frame, not only isOpened; release failed cameras."""
    indices = [index] if index is not None else list(range(max_index + 1))
    attempts = []
    for idx in indices:
        for label, api in _camera_backend_options(cv2, backend):
            cap = None
            valid = False
            try:
                cap = cv2.VideoCapture(idx, api)
                if cap is None or not cap.isOpened():
                    attempts.append((idx, label, 'not opened'))
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                valid = False
                for _ in range(5):
                    got, frame = cap.read()
                    if got and frame is not None and getattr(frame, 'size', 0) > 0:
                        valid = True
                        break
                    time.sleep(0.08)
                if valid:
                    reporter(f'CAMERA SELECTED: index={idx}, backend={label}')
                    return cap, idx, label
                attempts.append((idx, label, 'opened but no frames'))
            except Exception as exc:
                attempts.append((idx, label, type(exc).__name__))
            finally:
                # On success the function returns without releasing its capture.
                if cap is not None and not valid:
                    cap.release()
    for idx, label, reason in attempts:
        reporter(f'  camera {idx:>2} / {label:<5}: {reason}')
    raise RuntimeError(
        'No working camera (index 0..4 searched by default). Close Windows Camera, '
        'Teams, Zoom and browser camera tabs; check Windows Settings > '
        'Privacy & security > Camera > desktop app camera access; '
        'then try --camera-test or --camera-index 1 --camera-backend msmf.'
    )


def camera_diagnostic(preview=True):
    """Camera-only smoke test works before downloading gesture models."""
    import cv2
    idx, backend = camera_cli_options()
    print('CAMERA TEST: searching webcam (0..4 unless --camera-index given).')
    cap, picked, api = open_compatible_camera(cv2, idx, backend)
    print(f'CAMERA OK: index={picked} backend={api}. '
          'If this is the wrong camera use --camera-index N.')
    if not preview:
        cap.release()
        return
    print('Camera preview for up to 10 seconds. Press Q / Esc to close.')
    until=time.monotonic()+10.0
    frame_count=0
    try:
        while time.monotonic()<until:
            ok, frame=cap.read()
            if not ok or frame is None:
                print('CAMERA ERROR: capture opened but subsequent frame read failed')
                break
            frame_count+=1
            cv2.putText(frame, f'camera {picked} / {api}   press Q to quit',
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2)
            cv2.imshow('Gesture HMI - Camera Test (no HMI input)',frame)
            key=cv2.waitKey(1)&255
            if key in (27,ord('q'),ord('Q')):
                break
        print(f'CAMERA TEST COMPLETE: frames received={frame_count}')
    finally:
        cap.release()
        cv2.destroyAllWindows()



def main():
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(1)
    weights, mean, std, clip, classes = load_numpy_mlp()
    hand_model_path = find_hand_model(download=True)

    requested_camera, requested_backend = camera_cli_options()
    cap, camera_index, camera_backend = open_compatible_camera(
        cv2, requested_camera, requested_backend)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    try:
        import pyautogui as pyauto
    except ImportError as exc:
        cap.release()
        raise RuntimeError('PyAutoGUI needed: python -m pip install pyautogui') from exc
    pyauto.FAILSAFE = True  # upper-left emergency failsafe stays enabled
    pyauto.PAUSE = 0
    screen_size = pyauto.size()
    print('HANDY AUTO: checking installed Handy (Ctrl+Space / LONG PRESS required)...')
    handy_startup = ensure_handy_running()
    input_worker = LatestMouseWorker(pyauto)
    media_worker = MediaKeyWorker()
    seek_worker = SeekCommandWorker()
    handsfree = HandsFreeController()
    sounds_enabled = True
    hmi_active = False
    mouse_enabled = False
    last_screen_target = None
    mode = 'LOCKED'    # LOCKED, STANDBY, MEDIA_PINCH, POINTER, PINCH, ...
    close_threshold = PINCH_CLOSE_INITIAL
    scroll_threshold_i = 1
    scroll_ticks = SCROLL_TICKS
    region_mode = 2
    smooth_cursor = None
    last_one_at = -100.0
    one_since = None
    last_hand_at = time.monotonic()
    pinch_start_at = None
    pinch_release_since = None
    pinch_last_seen_at = None
    pinch_path = []
    pinch_smoothed_palm = None
    last_event = 'NONE'
    last_event_at = 0.0
    prob_history = deque(maxlen=3)
    pinch_gate = PinchEntryGate()
    standby_index_seen_at = -1e9
    click_gate = ClickEntryGate()
    click_threshold = CLICK_CLOSE_INITIAL
    click_release_since = None
    click_last_seen_at = None
    click_contact_d2 = None
    click_hold_started_at = None
    drag_start_tip = None
    drag_anchor = None
    drag_smooth_xy = None
    frame_number = 0
    thumb_gate = ThumbPressGate(min_held_s=VOICE_MIN_RECORDING_S)
    handy_keys = HandyHoldKeys()
    voice_state = 'NORMAL'  # NORMAL, RECORDING, PROCESSING
    voice_started_at = None
    voice_stopped_at = None
    voice_last_success = 0.0
    voice_diag = {}
    voice_stop_since = None
    volume_gate = VolumePoseGate()
    volume_backend = WindowsVolume()
    volume_motion = VolumeMotion(volume_backend)
    gun_seek = GunSeekMotion()
    gun_anchor = None
    gun_origin = None
    volume_pose = False
    volume_diag = {}
    clear_gate=ClearTextGate()
    clear_fist=False
    clear_ok=False
    clear_ok_diag={}
    clear_request_id=0
    clear_last_status='READY'

    def reset_clear():
        nonlocal clear_last_status
        input_worker.cancel_text_selection()
        clear_gate.reset()
        clear_last_status='READY'

    def reset_click():
        nonlocal click_release_since, click_last_seen_at, click_contact_d2
        nonlocal click_hold_started_at, drag_start_tip, drag_anchor, drag_smooth_xy
        click_gate.reset()
        click_release_since = None
        click_last_seen_at = None
        click_contact_d2 = None
        click_hold_started_at = None
        drag_start_tip = None
        drag_anchor = None
        drag_smooth_xy = None

    def reset_pinch():
        nonlocal pinch_start_at, pinch_release_since, pinch_last_seen_at
        nonlocal pinch_smoothed_palm, pinch_path
        pinch_start_at = None
        pinch_release_since = None
        pinch_last_seen_at = None
        pinch_smoothed_palm = None
        pinch_path = []
        pinch_gate.reset()

    def clear_tracking():
        nonlocal smooth_cursor, last_one_at, one_since, last_screen_target
        smooth_cursor = None
        last_one_at = -100.0
        one_since = None
        last_screen_target = None
        prob_history.clear()
        reset_pinch()
        reset_click()
        reset_clear()

    def leave_voice_to_pause(now, reason):
        nonlocal voice_state, voice_stopped_at, voice_started_at
        nonlocal mode, mouse_enabled, last_hand_at
        voice_state = 'NORMAL'
        voice_started_at = None
        voice_stopped_at = None
        # NEVER resume mouse while Handy might still be pasting.
        input_worker.set_enabled(False)
        mouse_enabled = False
        clear_tracking()
        handsfree.state = 'PAUSED'
        handsfree.last_visible_at = now  # fresh 3 s reacquisition window
        handsfree.resume_since = None
        mode = 'PAUSED'
        last_hand_at = now
        thumb_gate.reset(lock_until_release=True)  # held thumb must be released before next dictation
        print(reason)
        if sounds_enabled:
            play_state_sound('VOICE_READY')

    def finish_voice_processing(now, hand_present):
        nonlocal voice_state, voice_stopped_at, voice_started_at
        nonlocal mode, mouse_enabled, last_hand_at
        # Handy has no completion callback; allow configured grace before
        # permitting pointer/click to change focus away from the input field.
        voice_state = 'NORMAL'
        voice_started_at = None
        voice_stopped_at = None
        input_worker.set_enabled(False)  # clear stale actions before resuming
        media_worker.set_authorized(False)
        seek_worker.set_authorized(False)
        gun_seek.reset()
        mouse_enabled = False
        clear_tracking()
        thumb_gate.reset(lock_until_release=True)
        state = finish_voice_controller(handsfree, now, hand_present)
        last_hand_at = now
        if state == 'ACTIVE':
            mode = 'POINTER'
            input_worker.set_enabled(True)
            media_worker.set_authorized(True)
            mouse_enabled = True
            print('VOICE DONE -> POINTER ACTIVE automatically (no second PEACE).')
            if sounds_enabled:
                play_state_sound('RESUMED')
        else:
            mode = 'PAUSED'
            print('VOICE DONE -> PAUSED (no hand); show hand to resume, no PEACE needed within 3s.')
            if sounds_enabled:
                play_state_sound('PAUSED')

    def switch_off(reason):
        nonlocal hmi_active, mouse_enabled, mode, voice_state
        nonlocal voice_started_at, voice_stopped_at
        volume_motion.stop()
        volume_gate.reset(lock_until_release=True)
        gun_seek.reset()
        seek_worker.set_authorized(False)
        media_worker.set_authorized(False)
        if voice_state == 'RECORDING' or handy_keys.held:
            try:
                handy_keys.cancel()
                print('HANDY: held keys released; cancel requested on HMI shutdown')
            except Exception as exc:
                print('HANDY WARNING: key release/cancel failed:', exc)
        voice_state = 'NORMAL'
        voice_started_at = None
        voice_stopped_at = None
        thumb_gate.reset(lock_until_release=True)
        handsfree.force_off()
        hmi_active = False
        mouse_enabled = False
        mode = 'LOCKED'
        input_worker.set_enabled(False)
        clear_tracking()
        print(reason)

    def get_foreground_title():
        """Diagnostic only. Never use window focus to gate clicking."""
        if sys.platform != 'win32':
            return '(not Windows)'
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.GetForegroundWindow.restype = ctypes.c_void_p
            user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
            user32.GetWindowTextW.restype = ctypes.c_int
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(140)
            user32.GetWindowTextW(hwnd, buf, len(buf))
            return buf.value or '(unnamed window)'
        except Exception:
            return '(unknown)'

    def scroll(direction, diag, now):
        nonlocal mouse_enabled, last_event, last_event_at
        magnitude = proportional_scroll_ticks(diag['dy'], base_ticks=scroll_ticks)
        signed_ticks = magnitude if direction == 'UP' else -magnitude
        last_event = f'SCROLL_{direction}'
        last_event_at = now
        print(f'{last_event} [{"REAL" if mouse_enabled else "PREVIEW"}] '
              f'dy={diag["dy"]:+.3f} dx={diag["dx"]:+.3f} '
              f'duration={diag["duration"]:.2f}s, ticks={signed_ticks} '
              f'(min={SCROLL_MIN_TICKS}, max={SCROLL_MAX_TICKS}, scale={scroll_ticks})')
        if mouse_enabled and pyauto is not None:
            try:
                input_worker.act('scroll', signed_ticks)
            except pyauto.FailSafeException:
                mouse_enabled = False
                print('FAILSAFE: Windows mouse/scroll OFF')

    def go_back(diag, now):
        nonlocal mouse_enabled, last_event, last_event_at
        last_event = 'BACK'
        last_event_at = now
        print(f'BACK [{"REAL" if mouse_enabled else "PREVIEW"}] '
              f'dx={diag["dx"]:+.3f} dy={diag["dy"]:+.3f} '
              f'duration={diag["duration"]:.2f}s')
        if mouse_enabled and input_worker is not None:
            input_worker.act('back')  # Alt + Left, in the ACTIVE app

    def submit_enter(diag, now):
        nonlocal last_event, last_event_at
        last_event = 'ENTER'
        last_event_at = now
        print(f'ENTER [{"REAL" if mouse_enabled else "PREVIEW"}] '
              f'dx={diag["dx"]:+.3f} dy={diag["dy"]:+.3f} '
              f'duration={diag["duration"]:.2f}s')
        if mouse_enabled:
            input_worker.act('enter')

    def fire_media(diag, now, origin):
        nonlocal last_event,last_event_at
        if handsfree.state not in ('ACTIVE','STANDBY') or voice_state!='NORMAL':
            return
        last_event,last_event_at='MEDIA_PLAY_PAUSE',now
        queued=media_worker.request()
        print(f'MEDIA [{origin}] clockwise circle {diag["angle_deg"]:+.0f}deg '
              f'diameter={diag["diameter"]:.3f}, closure={diag["closure"]:.3f} '
              f'-> {"QUEUED" if queued else "COOLDOWN/BLOCKED"}')
        if queued and sounds_enabled:
            play_state_sound('MEDIA')

    print('STEP19 v2: gun UP increases volume and automatically clears Windows master MUTE.')
    print('STEP19 v3: GUN 0.7s -> LEFT -10s / RIGHT +10s; return to center -> repeat.')
    print('First motion UP/DOWN -> VOLUME (ACTIVE only); LEFT/RIGHT -> SEEK (ACTIVE/STANDBY).')
    print('SEEK: YouTube foreground -> safe instant J/L; background -> GSMTC API.')
    print('STEP18 v6: CLICK text input -> FIST 0.5s -> OK 0.5s -> Ctrl+A + Backspace.')
    print('Requires UIA: python -m pip install uiautomation')
    print('Editable Edit/ComboBox + HMI click + same focused field: fail-closed if unsafe.')
    print('Beep: fist armed=1 low, clear requested=1 high. No extra OK hold.')
    print('Release OK at any point BEFORE final confirmation to cancel deletion.')
    print('STEP17: PINCH CLOCKWISE CIRCLE -> GLOBAL MEDIA PLAY/PAUSE; STEP15 retained.')
    print('VOLUME: index+middle straight and together, thumb spread, other fingers curled.')
    print('VOLUME ZERO WAKE: first deliberate gun UP at 0% -> 8% + unmute.')
    print('Hold ~0.7s, then move whole hand UP/DOWN; max 15% points per second.')
    print('Relax hand -> POINTER; tracking lost -> PAUSED; NO backlog while still.')
    print('Install once: python -m pip install pycaw')
    print('Check sound output (non-mutating): python gesture_pointer_step19_v3b_portable.py --volume-status')
    print('PEACE 1.0s -> HMI+Windows mouse ON; hold THUMBS-UP ~0.45s -> record.')
    print(f'KEEP THUMBS-UP WHILE SPEAKING; relax thumb ~{VOICE_RELEASE_S:.2f}s -> transcribe.')
    print(f'VOICE STABILITY: min hold {VOICE_MIN_RECORDING_S:.1f}s; softer hold hysteresis during dictation.')
    print('Missing hand <0.90s while recording does NOT stop the microphone.')
    print('Recording/processing: pointer, click, scroll, back are blocked.')
    print('SHORT thumb-to-middle TOUCH + RELEASE = CLICK. HOLD ~1s = DRAG; release = DROP.')
    print('Thumb-index PINCH vertical then RELEASE -> 240..720 proportional Scroll; LEFT=BACK, RIGHT=ENTER.')
    print('LOCKED initially; PEACE 1s -> ACTIVE; missing 0.5s -> PAUSED, 3s -> STANDBY.')
    print('STANDBY: media circle + GUN horizontal seek; no mouse; PEACE to reenable pointer.')
    print('STOP 0.6s -> LOCKED (even media disabled). Circle = PINCH + CW 300+ degrees + release.')
    print('Handy auto-start is best effort. Focus a text field BEFORE THUMBS-UP.')
    print('Keys: S=beep, B=back method, Z/X=scroll strength, 1/2/3=ROI, Q=quit.')
    print(f'After thumb release allow ~{VOICE_PROCESSING_S:.0f}s for Handy to paste, then POINTER auto-resumes.')
    start = time.monotonic()
    prev_ms = -1
    last_frame_time = start
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                frame_number += 1
                ok, frame = cap.read()
                if not ok:
                    print('Unable to read frame from webcam')
                    break
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                now = time.monotonic()
                fps = 1.0 / max(now - last_frame_time, .001)
                last_frame_time = now
                # Downsample for hand detection: reduces CPU load from heavy Chrome pages.
                scale = DETECT_WIDTH / float(width)
                detect = cv2.resize(frame, (DETECT_WIDTH, max(1, int(height * scale))),
                                    interpolation=cv2.INTER_AREA) if width > DETECT_WIDTH else frame
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(detect, cv2.COLOR_BGR2RGB))
                ts = max(int((now-start)*1000), prev_ms+1)
                prev_ms = ts
                detect_begin = time.perf_counter()
                result = detector.detect_for_video(img, ts)
                detect_ms = (time.perf_counter() - detect_begin) * 1000.0
                hand = result.hand_landmarks[0] if result.hand_landmarks else None
                world_hand = (result.hand_world_landmarks[0] if
                              result.hand_world_landmarks else None)
                world_index, world_middle, world_d, world_index_d = world_click_pose(world_hand)
                label, conf, extended = 'NO_HAND', 0.0, False
                tip, palm, pinch_ratio, pinch_ratio_3d = None, None, float('inf'), float('inf')
                click_d2, click_d3, thumb_index_d, middle_folded = (float('inf'), float('inf'), float('inf'), False)
                thumb_pose = False
                voice_diag = {}
                volume_pose = False
                volume_diag = {}
                clear_fist=False
                clear_ok=False
                clear_ok_diag={}
                stop_open=False
                stop_diag={}
                if hand is not None:
                    last_hand_at = now
                    extended = index_extended(hand)
                    thumb_pose, voice_diag = thumbs_up_geometry(hand, world_hand, width/height)
                    volume_pose, volume_diag = gun_volume_geometry(hand, world_hand, width/height)
                    clear_fist, _ = fist_clear_geometry(hand,world_hand,width/height)
                    clear_ok, clear_ok_diag = ok_clear_geometry(hand,world_hand,width/height)
                    stop_open, stop_diag = stop_open_palm_geometry(hand,world_hand,width/height)
                    tip = np.asarray([hand[8].x, hand[8].y], dtype=np.float32)
                    pinch_ratio, pinch_ratio_3d, palm = pinch_ratio_and_palm(hand, width/height)
                    click_d2, click_d3, thumb_index_d, middle_folded = click_metrics(hand, width/height)
                    # A tilted hand can look curled in camera X/Y, even if the index
                    # is truly extended. Use depth-estimated world joint geometry.
                    extended = extended or world_index
                    middle_folded = middle_folded or world_middle
                    feat = to_features(hand, mean, std, clip)
                    if feat is not None:
                        probs = numpy_mlp_probs(feat, weights)
                        prob_history.append(probs)
                        smooth_probs = np.mean(np.stack(prob_history), axis=0)
                        pred = int(np.argmax(smooth_probs))
                        conf = float(smooth_probs[pred])
                        label = classes[pred] if conf >= CONFIDENCE else 'UNCERTAIN'
                        if label == 'one' and not extended:
                            label = 'ONE_REJECTED'
                    else:
                        prob_history.clear()
                        label = 'UNCERTAIN'
                else:
                    prob_history.clear()
                    pinch_gate.reset()
                    # Avoid wiping a prepared click because ONE frame was missing.
                    # Longer tracking loss still clears it before gesture input.
                    if now - last_hand_at > 0.35:
                        click_gate.reset()

                # MLP 'stop' alone is insufficient: OK can resemble open palm.
                # Open index plus separated thumb must confirm an intentional STOP.
                safe_label=effective_stop_label(label,stop_open)

                # In hold-to-talk mode the Handy key must remain physically down
                # throughout the entire THUMBS-UP, rather than sending two taps.
                # A short camera landmark dropout does not count as a release.
                if voice_state == 'RECORDING':
                    # Once Ctrl+Space is down, never let newly introduced gun/
                    # pointer mode gates interrupt a genuine held thumbs-up.
                    # Tolerate moderate landmark noise without relaxing entry.
                    voice_hold_pose=(thumb_pose or thumbs_up_recording_hold_geometry(
                        hand,world_hand,width/height))
                else:
                    voice_hold_pose=(thumb_pose and
                        mode not in ('GUN_READY','GUN_BLOCKED','VOLUME','SEEK',
                                     'CLEAR_HOLD','CLEAR_ARMED','CLEAR_OK') and
                        not volume_pose)
                thumb_event = thumb_gate.update(now,voice_hold_pose,hand is not None)
                if voice_state != 'NORMAL':
                    if hand is not None and safe_label == 'stop':
                        if voice_stop_since is None:
                            voice_stop_since = now
                        elif now - voice_stop_since >= STOP_HOLD_S:
                            switch_off('STOP: voice cancelled and HMI OFF')
                            voice_stop_since = None
                            if sounds_enabled:
                                play_state_sound('OFF_BY_STOP')
                    else:
                        voice_stop_since = None
                if (voice_state == 'NORMAL' and handsfree.state == 'ACTIVE' and
                        mode != 'VOLUME' and thumb_event == 'PRESS'):
                    # Freeze the already-selected text field while pressing CTRL+SPACE.
                    mouse_enabled = False
                    input_worker.set_enabled(False)
                    media_worker.set_authorized(False)
                    seek_worker.set_authorized(False)
                    clear_tracking()
                    mode = 'VOICE_RECORDING'
                    try:
                        handy_keys.press()
                        voice_state = 'RECORDING'
                        voice_started_at = now
                        print('VOICE START: Ctrl+Space HELD; speak while THUMBS-UP stays raised.')
                    except Exception as exc:
                        print('VOICE START FAILED:', type(exc).__name__, str(exc))
                        leave_voice_to_pause(now, 'VOICE HOTKEY FAILED: PAUSED; PEACE again if needed')
                elif voice_state == 'RECORDING':
                    if thumb_event == 'RELEASE':
                        try:
                            handy_keys.release()
                            voice_state = 'PROCESSING'
                            voice_stopped_at = now
                            mode = 'VOICE_PROCESSING'
                            print(f'VOICE RELEASE: Ctrl+Space UP after {now-(voice_started_at or now):.2f}s; '
                                  f'hand={hand is not None} strict_thumb={thumb_pose} '
                                  f'soft_hold={voice_hold_pose}; Handy transcribing; '
                                  f'POINTER resumes in ~{VOICE_PROCESSING_S:.0f}s.')
                        except Exception as exc:
                            print('VOICE RELEASE FAILED:', type(exc).__name__, str(exc))
                            switch_off('VOICE HOTKEY RELEASE ERROR: HMI OFF')
                    elif (voice_started_at is not None and
                          now - voice_started_at >= VOICE_RECORDING_MAX_S):
                        try:
                            handy_keys.release()
                            print('VOICE MAX RECORDING: safety key release')
                        except Exception as exc:
                            print('VOICE MAX RECORDING: key release failed:', exc)
                        voice_state = 'PROCESSING'
                        voice_stopped_at = now
                        mode = 'VOICE_PROCESSING'
                        thumb_gate.reset(lock_until_release=True)
                elif voice_state == 'PROCESSING':
                    if voice_stopped_at is not None and now - voice_stopped_at >= VOICE_PROCESSING_S:
                        finish_voice_processing(now, hand is not None)
                if voice_state == 'NORMAL':
                    transition = handsfree.update(
                        now, hand is not None,
                        'VOLUME_CANDIDATE' if volume_pose else safe_label,
                        stop_allowed=(mode not in ('PINCH', 'PINCH_DONE', 'MEDIA_PINCH',
                                                  'GUN_READY','GUN_BLOCKED','VOLUME','SEEK',
                                                  'CLEAR_HOLD','CLEAR_ARMED','CLEAR_OK')))
                else:
                    transition = None
                if transition in ('ACTIVATED', 'RESUMED'):
                    clear_tracking()  # resume starts with fresh One and fresh gates
                    mode = 'POINTER'
                    hmi_active = True
                    # Clear any prior queued work *before* enabling the worker.
                    input_worker.set_enabled(True)
                    media_worker.set_authorized(True)
                    seek_worker.set_authorized(False)
                    gun_seek.reset()
                    mouse_enabled = True
                    msg = ('PEACE: HMI + WINDOWS MOUSE ACTIVE' if transition == 'ACTIVATED'
                           else 'HAND RETURNED: HMI ACTIVE, WINDOWS MOUSE RESUMED')
                    print(msg)
                    if sounds_enabled:
                        play_state_sound(transition)
                elif transition == 'PAUSED':
                    # First stop background input, then cancel unfinished gestures.
                    volume_motion.stop()
                    volume_gate.reset(lock_until_release=True)
                    gun_seek.reset()
                    seek_worker.set_authorized(False)
                    mouse_enabled = False
                    input_worker.set_enabled(False)
                    media_worker.set_authorized(False)
                    mode = 'PAUSED'
                    clear_tracking()
                    print('NO HAND >= 0.5s: HMI PAUSED; all Windows input blocked')
                    if sounds_enabled:
                        play_state_sound('PAUSED')
                elif transition == 'STANDBY_BY_TIMEOUT':
                    # Never awaken Windows mouse automatically after 3s absence.
                    volume_motion.stop()
                    volume_gate.reset(lock_until_release=True)
                    gun_seek.reset()
                    seek_worker.set_authorized(False)
                    mouse_enabled=False
                    input_worker.set_enabled(False)
                    clear_tracking()
                    mode='STANDBY'
                    standby_index_seen_at=-1e9
                    media_worker.set_authorized(True)  # media ONLY
                    print('NO HAND >=3.0s: STANDBY; PINCH+CLOCKWISE CIRCLE for media, PEACE for pointer')
                    if sounds_enabled:
                        play_state_sound('STANDBY_BY_TIMEOUT')
                elif transition == 'OFF_BY_STOP':
                    switch_off('STOP: LOCKED; all Windows input including MEDIA blocked')
                    if sounds_enabled:
                        play_state_sound('OFF_BY_STOP')

                # STEP17 STANDBY: keep mouse worker disabled. Only circle and Peace/Stop.
                # Require open->close pinch from visibly extended index (not a random fist).
                if handsfree.state == 'STANDBY' and voice_state == 'NORMAL':
                    if (hand is not None and extended and palm is not None and
                            pinch_ratio > close_threshold+0.10):
                        standby_index_seen_at=now
                    if mode == 'STANDBY':
                        if (palm is not None and pinch_gate.update(
                                now,pinch_ratio,pinch_ratio_3d,
                                now-standby_index_seen_at <= STANDBY_INDEX_GRACE_S,
                                close_threshold)):
                            mode='MEDIA_PINCH'
                            pinch_start_at=now
                            pinch_last_seen_at=now
                            pinch_release_since=None
                            pinch_smoothed_palm=palm.copy()
                            pinch_path=[(now,float(palm[0]),float(palm[1]))]
                            print('STANDBY MEDIA PINCH: draw a small clockwise circle, then release')
                    elif mode == 'MEDIA_PINCH':
                        if hand is None:
                            if (pinch_last_seen_at is not None and
                                    now-pinch_last_seen_at > PINCH_LOST_S):
                                print('STANDBY MEDIA CANCEL: tracking lost')
                                reset_pinch()
                                mode='STANDBY'
                        else:
                            pinch_last_seen_at=now
                            if pinch_ratio > close_threshold+PINCH_HYSTERESIS:
                                if pinch_release_since is None:
                                    pinch_release_since=now
                                elif now-pinch_release_since >= PINCH_RELEASE_HOLD_S:
                                    media_direction,why,md = circle_metrics(pinch_path,width/height)
                                    if media_direction=='CW':
                                        fire_media(md,now,'STANDBY')
                                    else:
                                        print('STANDBY CIRCLE IGNORED:',media_direction or why,
                                              'angle=',round(md['angle_deg']))
                                    reset_pinch()
                                    mode='STANDBY'
                            else:
                                pinch_release_since=None
                                if pinch_smoothed_palm is None:
                                    pinch_smoothed_palm=palm.copy()
                                else:
                                    pinch_smoothed_palm=.68*palm+.32*pinch_smoothed_palm
                                if pinch_path:
                                    jump=float(np.linalg.norm(pinch_smoothed_palm-
                                                np.asarray(pinch_path[-1][1:3])))
                                    if jump < .28:
                                        pinch_path.append((now,float(pinch_smoothed_palm[0]),
                                                           float(pinch_smoothed_palm[1])))
                                if now-pinch_start_at>PINCH_MAX_S:
                                    print('STANDBY MEDIA CANCEL: duration exceeded')
                                    reset_pinch()
                                    mode='STANDBY'

                # STEP18 v5: FIST 0.5s arms; OK 0.5s issues one safely
                # guarded Ctrl+A -> Backspace action (no extra confirmation hold).
                if voice_state=='NORMAL' and handsfree.state=='ACTIVE' and mouse_enabled:
                    if mode=='POINTER' and clear_gate.state=='LOCKED' and not clear_fist and not clear_ok:
                        clear_gate.reset()
                    if mode=='POINTER' and clear_fist and not thumb_pose and not volume_pose and clear_gate.state!='LOCKED':
                        if input_worker.recent_input_click(now):
                            input_worker.cancel_text_selection()  # cancel stale v2 work
                            clear_gate.reset()
                            clear_gate.update(now,True,False,True,True)
                            mode='CLEAR_HOLD'
                            reset_click(); reset_pinch()
                            clear_last_status='FIST: hold 0.5s to arm'
                            print('CLEAR HOLD: FIST 0.5s, then OK sign 0.5s to ERASE')
                        elif clear_last_status!='NO RECENT INPUT CLICK':
                            clear_last_status='NO RECENT INPUT CLICK'
                            print('CLEAR NOT READY: first CLICK the editable input with gesture pointer')
                    elif mode in ('CLEAR_HOLD','CLEAR_ARMED','CLEAR_OK'):
                        valid=(input_worker.recent_input_click(now) and
                               not thumb_pose and not volume_pose)
                        clear_event=clear_gate.update(now,clear_fist,clear_ok,
                                      hand is not None,valid)
                        if clear_event=='ARMED':
                            ok_edit,detail,_=input_worker.focus_guard.inspect()
                            youtube_suggestions=(not ok_edit and
                                  input_worker._youtube_suggestion_candidate(detail))
                            if not ok_edit and not youtube_suggestions:
                                clear_last_status='BLOCKED: '+detail[:105]
                                input_worker.cancel_text_selection()
                                clear_gate.reset(lock_until_release=True)
                                mode='POINTER'
                                print('CLEAR BLOCKED ON FIST:',detail)
                                if sounds_enabled: play_state_sound('CLEAR_CANCEL')
                            else:
                                mode='CLEAR_ARMED'
                                clear_last_status=('ARMED: YouTube suggestion; Esc recovery on OK'
                                                   if youtube_suggestions else
                                                   'ARMED: show OK for 0.5s')
                                if youtube_suggestions:
                                    print('CLEAR ARMED: YouTube suggestion list focused; '
                                          'hold OK 0.5s, then try safe Esc/refocus before delete')
                                else:
                                    print('CLEAR ARMED: show OK for 0.5s -> erase once')
                                if sounds_enabled: play_state_sound('CLEAR_ARMED')
                        elif clear_event=='DELETE':
                            input_worker.act('clear_text')
                            clear_last_status='CLEAR REQUESTED: worker verifies target'
                            last_event,last_event_at='CLEAR_REQUESTED',now
                            mode='POINTER'
                            print('CLEAR REQUEST: worker rechecks same input, then Ctrl+A -> Backspace')
                            if sounds_enabled: play_state_sound('CLEAR_DELETE')
                        elif clear_event=='CANCEL':
                            clear_last_status='CANCELLED: OK not stable / timed out / hand lost'
                            print('CLEAR CANCEL: pose changed, hand lost, or confirmation timed out')
                            input_worker.cancel_text_selection()
                            clear_gate.reset(lock_until_release=True)
                            mode='POINTER'
                            if sounds_enabled: play_state_sound('CLEAR_CANCEL')
                        elif mode=='CLEAR_ARMED' and clear_gate.ok_since is not None:
                            mode='CLEAR_OK'
                            clear_last_status='OK: hold 0.5s to ERASE'
                        elif mode=='CLEAR_OK' and clear_gate.state=='ARMED' and clear_gate.ok_since is None:
                            mode='CLEAR_ARMED'
                elif mode in ('CLEAR_HOLD','CLEAR_ARMED','CLEAR_OK'):
                    reset_clear()

                # STEP19: one gun pose, axis-locked. On entering the pose,
                # freeze cursor but do NOT start pycaw until a clear vertical
                # motion. Horizontal movement switches to repeatable SEEK.
                volume_candidate = (voice_state == 'NORMAL' and
                                    handsfree.state in ('ACTIVE','STANDBY') and
                                    mode in ('POINTER','STANDBY') and volume_pose)
                if voice_state == 'NORMAL' and handsfree.state in ('ACTIVE','STANDBY'):
                    if mode in ('POINTER','STANDBY'):
                        if volume_pose and palm is not None:
                            gun_event = volume_gate.update(now, True, True)
                            if gun_event == 'ENTER':
                                gun_origin = handsfree.state
                                gun_anchor = palm.copy()
                                gun_seek.reset()
                                seek_worker.set_authorized(False)
                                input_worker.set_enabled(False)
                                media_worker.set_authorized(False)
                                mouse_enabled = False
                                clear_tracking()
                                thumb_gate.reset(lock_until_release=True)
                                mode = 'GUN_READY'
                                last_event,last_event_at = 'GUN_READY',now
                                print('GUN READY: move UP/DOWN for VOLUME or LEFT/RIGHT for SEEK +/-10s')
                        else:
                            volume_gate.update(now, False, hand is not None)
                    elif mode in ('GUN_READY','GUN_BLOCKED','VOLUME','SEEK'):
                        gun_event = volume_gate.update(now, volume_pose, hand is not None)
                        if gun_event == 'EXIT':
                            volume_motion.stop()
                            gun_seek.reset()
                            seek_worker.set_authorized(False)  # purge unsent seeks
                            last_event,last_event_at='GUN_EXIT',now
                            clear_tracking()
                            if hand is None:
                                if gun_origin=='ACTIVE':
                                    handsfree.state='PAUSED'
                                    handsfree.last_visible_at=now
                                    handsfree.resume_since=None
                                    media_worker.set_authorized(False)
                                    mode='PAUSED'
                                    print('GUN OFF: tracking lost -> PAUSED')
                                else:
                                    media_worker.set_authorized(True)
                                    mode='STANDBY'
                                    print('GUN OFF: tracking lost -> STANDBY')
                            elif gun_origin=='ACTIVE':
                                mode='POINTER'
                                input_worker.set_enabled(True)
                                media_worker.set_authorized(True)
                                mouse_enabled=True
                                print('GUN OFF -> POINTER (no PEACE)')
                            else:
                                mode='STANDBY'
                                media_worker.set_authorized(True)
                                mouse_enabled=False
                                print('GUN OFF -> STANDBY (pointer stays disabled)')
                            gun_anchor=None
                            gun_origin=None
                        elif volume_pose and palm is not None:
                            if mode == 'GUN_READY':
                                if gun_anchor is None:
                                    gun_anchor=palm.copy()
                                axis = classify_gun_axis(palm[0]-gun_anchor[0],
                                                         palm[1]-gun_anchor[1], width/height)
                                if axis == 'VOLUME':
                                    if gun_origin != 'ACTIVE':
                                        # Standby allows seeking only. Lock out any
                                        # later sideways motion until pose release.
                                        mode='GUN_BLOCKED'
                                        last_event,last_event_at='VOLUME_STANDBY_BLOCKED',now
                                        print('STANDBY: vertical gun motion blocked; release gun to retry')
                                    else:
                                        try:
                                            # The first vertical stroke selected VOLUME;
                                            # count UP immediately instead of discarding it.
                                            starting_level=volume_motion.start(
                                                now,palm,initial_up=bool(palm[1]<gun_anchor[1]))
                                        except Exception as exc:
                                            print('VOLUME UNAVAILABLE:',type(exc).__name__,str(exc))
                                            volume_gate.reset(lock_until_release=True)
                                            gun_seek.reset()
                                            mode='POINTER'
                                            input_worker.set_enabled(True)
                                            media_worker.set_authorized(True)
                                            mouse_enabled=True
                                            gun_origin=None; gun_anchor=None
                                        else:
                                            mode='VOLUME'
                                            last_event,last_event_at='VOLUME_ON',now
                                            print(f'VOLUME ON: {starting_level:.1f}%; UP/DOWN max 15 percent-points/s')
                                elif axis == 'SEEK':
                                    mode='SEEK'
                                    gun_seek.start(now,gun_anchor)
                                    seek_worker.set_authorized(True)
                                    last_event,last_event_at='SEEK_READY',now
                                    print('SEEK MODE: LEFT -10s, RIGHT +10s; return to origin to rearm')
                                    seek_event=gun_seek.update(now,palm,width/height)
                                    if isinstance(seek_event,int):
                                        if seek_worker.request(seek_event):
                                            last_event,last_event_at=f'SEEK_{seek_event:+d}s',now
                                            print('  SEEK SWIPE QUEUED:',f'{seek_event:+d}s',
                                                  'count=',gun_seek.fires)
                            elif mode == 'VOLUME' and gun_origin=='ACTIVE':
                                try:
                                    new_level=volume_motion.update(now,palm)
                                    if new_level is not None:
                                        last_event,last_event_at=f'VOLUME_{new_level:.0f}%',now
                                except Exception as exc:
                                    print('VOLUME ERROR:',type(exc).__name__,str(exc))
                                    volume_motion.stop()
                                    gun_seek.reset()
                                    seek_worker.set_authorized(False)
                                    volume_gate.reset(lock_until_release=True)
                                    input_worker.set_enabled(False)
                                    media_worker.set_authorized(False)
                                    mouse_enabled=False
                                    clear_tracking()
                                    handsfree.state='PAUSED'
                                    handsfree.last_visible_at=now
                                    handsfree.resume_since=None
                                    mode='PAUSED'
                            elif mode == 'SEEK':
                                seek_event=gun_seek.update(now,palm,width/height)
                                if isinstance(seek_event,int):
                                    if seek_worker.request(seek_event):
                                        last_event,last_event_at=f'SEEK_{seek_event:+d}s',now
                                        print('  SEEK SWIPE QUEUED:',f'{seek_event:+d}s',
                                              'count=',gun_seek.fires)
                                elif seek_event=='REARMED':
                                    last_event,last_event_at='SEEK_REARMED',now
                                    print('  SEEK REARMED: next lateral swipe can fire')
                if (handsfree.state == 'ACTIVE' and mouse_enabled and
                        not volume_candidate and mode != 'VOLUME'):
                    if (hand is not None and extended and
                        (label == 'one' or
                         (world_index and now-last_one_at < 1.5))):
                        last_one_at = now
                        if one_since is None:
                            one_since = now
                    elif now - last_one_at > ONE_GRACE_S:
                        one_since = None

                    if mode == 'POINTER':
                        # Pinch can start immediately following One. Do not require
                        # One to remain predicted after index finger curls for pinch.
                        if (palm is not None and pinch_gate.update(
                                now, pinch_ratio, pinch_ratio_3d,
                                now - last_one_at <= PINCH_ENTRY_ONE_S,
                                close_threshold)):
                            reset_click()   # scrolling gets priority over clicking
                            mode = 'PINCH'
                            pinch_start_at = now
                            pinch_last_seen_at = now
                            pinch_release_since = None
                            pinch_smoothed_palm = palm.copy()
                            pinch_path = [(now, float(palm[0]), float(palm[1]))]
                            smooth_cursor = None
                            print('PINCH START: keep tips touching; move WHOLE HAND up/down')
                        else:
                            # Separate gesture: thumb tip(4) touches proximal
                            # bone of FOLDED middle finger(9--10); index stays up.
                            eligible = (hand is not None and extended and middle_folded
                                        and now-last_one_at <= CLICK_ENTRY_ONE_S)
                            index_far = (thumb_index_d >= CLICK_MIN_THUMB_INDEX or
                                         world_index_d >= CLICK_MIN_THUMB_INDEX)
                            # Permit 3D pose to corroborate contact when tilted;
                            # do not accept arbitrary distant 2D projections.
                            eff_d2 = min(click_d2, world_d)
                            eff_d3 = min(click_d3, world_d)
                            click_triggered = click_gate.update(
                                now, eff_d2, eff_d3, index_far,
                                eligible, click_threshold)
                            if click_triggered:
                                # Defer short click until thumb RELEASE; long hold
                                # becomes mouseDown + move + mouseUp instead.
                                mode = 'CLICK_HELD'
                                click_hold_started_at = now
                                click_last_seen_at = now
                                click_release_since = None
                                click_contact_d2 = eff_d2
                                drag_start_tip = tip.copy() if tip is not None else None
                                drag_anchor = last_screen_target
                                drag_smooth_xy = None
                                pinch_gate.reset()
                                print('CLICK CONTACT: release for LEFT_CLICK, hold 1.0s for DRAG')
                            # Freeze position as soon as the two fingers are close,
                            # even while we verify the click contact.
                            elif (click_gate.status(now) != 'TOUCH_VERIFY' and
                                  hand is not None and extended and tip is not None and
                                  one_since is not None and now-one_since >= ONE_STABLE_S and
                                  now-last_one_at <= ONE_GRACE_S and
                                  label in ('one', 'UNCERTAIN', 'no_gesture')):
                                if smooth_cursor is None:
                                    smooth_cursor = tip.copy()
                                movement = float(np.linalg.norm(tip - smooth_cursor))
                                alpha = 0.75 if movement > .025 else .52
                                smooth_cursor = alpha * tip + (1-alpha) * smooth_cursor
                                if mouse_enabled and pyauto is not None:
                                    sw, sh = screen_size
                                    sx, sy = screen_coordinates(smooth_cursor,
                                                                REGIONS[region_mode], sw, sh)
                                    last_screen_target = (sx, sy)
                                    try:
                                        input_worker.move((sx, sy))
                                    except pyauto.FailSafeException:
                                        mouse_enabled = False
                                        print('FAILSAFE: Windows mouse/scroll OFF')

                    elif mode in ('CLICK_HELD', 'DRAGGING'):
                        # Hold thumb contact on folded middle bone. No cursor jump:
                        # DRAG uses relative index displacement from click start.
                        if hand is None:
                            if (click_last_seen_at is not None and
                                    now-click_last_seen_at > CLICK_LOST_S):
                                if mode == 'DRAGGING':
                                    input_worker.act('drag_end')  # mouseUp even if hand lost
                                    print('DRAG CANCEL: hand lost -> mouseUp')
                                else:
                                    print('CLICK CANCEL: hand lost (no click)')
                                reset_click()
                                mode = 'POINTER'
                                smooth_cursor = None
                        else:
                            click_last_seen_at = now
                            contact_now = min(click_d2, world_d)
                            release_limit = min(click_threshold + CLICK_HYSTERESIS,
                                                (click_contact_d2 or 0) + CLICK_RELEASE_DELTA)
                            released = (contact_now >= release_limit or
                                        (not middle_folded and contact_now >=
                                         (click_contact_d2 or 0) + 0.10))
                            if released:
                                if click_release_since is None:
                                    click_release_since = now
                                elif now-click_release_since >= CLICK_RELEASE_HOLD_S:
                                    if mode == 'DRAGGING':
                                        input_worker.act('drag_end')
                                        last_event = 'DRAG_END'
                                        print('DRAG RELEASE: mouseUp and drop')
                                    elif drag_anchor is not None:
                                        input_worker.act('click', xy=drag_anchor)
                                        last_event = 'LEFT_CLICK'
                                        print('LEFT_CLICK [REAL]: quick thumb contact + release at', drag_anchor)
                                    else:
                                        print('CLICK SKIPPED: establish pointer position first')
                                    last_event_at = now
                                    click_gate.rearm_after_release(now, contact_now)
                                    click_contact_d2 = None
                                    click_release_since = None
                                    click_hold_started_at = None
                                    drag_start_tip = None
                                    drag_anchor = None
                                    drag_smooth_xy = None
                                    pinch_gate.reset()
                                    smooth_cursor = None
                                    mode = 'POINTER'
                            else:
                                click_release_since = None
                                if (mode == 'CLICK_HELD' and
                                        click_hold_started_at is not None and
                                        now - click_hold_started_at >= DRAG_HOLD_S and
                                        drag_anchor is not None and drag_start_tip is not None):
                                    mode = 'DRAGGING'
                                    drag_smooth_xy = drag_anchor
                                    input_worker.act('drag_start', xy=drag_anchor)
                                    last_event, last_event_at = 'DRAG_START', now
                                    print('DRAG START: thumb held 1s; move INDEX while keeping contact')
                                if (mode == 'DRAGGING' and extended and tip is not None and
                                        drag_start_tip is not None and drag_anchor is not None):
                                    xy = relative_drag_position(tip, drag_start_tip,
                                         drag_anchor, REGIONS[region_mode], *screen_size)
                                    if drag_smooth_xy is None:
                                        drag_smooth_xy = xy
                                    else:
                                        drag_smooth_xy = (int(round(DRAG_MOVE_ALPHA * xy[0] +
                                                                     (1 - DRAG_MOVE_ALPHA)*drag_smooth_xy[0])),
                                                          int(round(DRAG_MOVE_ALPHA * xy[1] +
                                                                     (1 - DRAG_MOVE_ALPHA)*drag_smooth_xy[1])))
                                    input_worker.move(drag_smooth_xy)

                    elif mode in ('PINCH', 'PINCH_DONE'):
                        if hand is None:
                            if pinch_last_seen_at is not None and now-pinch_last_seen_at > PINCH_LOST_S:
                                print('PINCH CANCEL: lost hand; release and return to ONE')
                                reset_pinch()
                                mode = 'POINTER'
                        else:
                            pinch_last_seen_at = now
                            released = pinch_ratio > close_threshold + PINCH_HYSTERESIS
                            if released:
                                if pinch_release_since is None:
                                    pinch_release_since = now
                                if now - pinch_release_since >= PINCH_RELEASE_HOLD_S:
                                    if mode == 'PINCH':
                                        direction, reason, diag = swipe_metrics(
                                            pinch_path, SWIPE_THRESHOLDS[scroll_threshold_i])
                                        is_back, back_reason, back_diag = back_metrics(pinch_path)
                                        is_enter, enter_reason, enter_diag = enter_metrics(pinch_path)
                                        circle_dir,circle_reason,cdiag = circle_metrics(pinch_path,width/height)
                                        if circle_dir == 'CW':
                                            fire_media(cdiag,now,'ACTIVE')
                                        elif circle_dir == 'CCW':
                                            print('CIRCLE CCW ignored (reserved for future)')
                                        elif is_back:
                                            go_back(back_diag, now)
                                        elif is_enter:
                                            submit_enter(enter_diag, now)
                                        elif direction is not None:
                                            scroll(direction, diag, now)
                                        else:
                                            print(f'PINCH RELEASE: no action (circle={circle_reason}, scroll={reason}, '
                                                  f'back={back_reason}, enter={enter_reason}), '
                                                  f'dy={diag["dy"]:+.3f} dx={diag["dx"]:+.3f}')
                                    else:
                                        print('PINCH RELEASE: ready for next gesture')
                                    reset_pinch()
                                    mode = 'POINTER'
                            else:
                                pinch_release_since = None
                                if pinch_smoothed_palm is None:
                                    pinch_smoothed_palm = palm.copy()
                                else:
                                    pinch_smoothed_palm = 0.68 * palm + 0.32 * pinch_smoothed_palm
                                if mode == 'PINCH':
                                    # Filtering and storing palm-center motion, not fingertip motion.
                                    if pinch_path:
                                        jump = float(np.linalg.norm(
                                            pinch_smoothed_palm-np.asarray(pinch_path[-1][1:3])))
                                        if jump < 0.28:
                                            pinch_path.append((now, float(pinch_smoothed_palm[0]),
                                                               float(pinch_smoothed_palm[1])))
                                    if now - pinch_start_at > PINCH_MAX_S:
                                        print('PINCH CANCEL: held too long')
                                        mode = 'PINCH_DONE'
                                    # IMPORTANT: do NOT scroll before release.
                                    # Final displacement is needed for proportional ticks.

                if input_worker.failed and handsfree.state != 'LOCKED':
                    switch_off('WINDOWS INPUT ERROR: emergency HMI OFF (restart program)')
                # Diagnostics preview
                if tip is not None:
                    cv2.circle(frame, (int(tip[0]*width), int(tip[1]*height)),
                               8, (0, 245, 255), -1)  # yellow = index tip
                    cv2.circle(frame, (int(hand[4].x*width), int(hand[4].y*height)),
                               6, (255, 100, 0), -1)  # blue = thumb tip
                if hand is not None:
                    # 9(MCP)->10(PIP) is the target middle-finger segment.
                    p9=(int(hand[9].x*width),int(hand[9].y*height))
                    p10=(int(hand[10].x*width),int(hand[10].y*height))
                    cv2.line(frame,p9,p10,(180,0,230),4)
                    cv2.circle(frame, p10, 5, (180,0,230), -1)
                if mode in ('PINCH','PINCH_DONE','MEDIA_PINCH') and len(pinch_path)>1:
                    for j in range(1, len(pinch_path)):
                        a = (int(pinch_path[j-1][1]*width),int(pinch_path[j-1][2]*height))
                        b = (int(pinch_path[j][1]*width),int(pinch_path[j][2]*height))
                        cv2.line(frame, a, b, (240, 155, 20), 2)
                if palm is not None and mode in ('PINCH','PINCH_DONE','MEDIA_PINCH'):
                    cv2.circle(frame, (int(palm[0]*width), int(palm[1]*height)),
                               7, (10, 0, 255), -1)

                cv2.rectangle(frame, (0, 0), (width, 173), (24, 24, 24), -1)
                def txt(s, y, color=(235,235,235), scale=.52):
                    cv2.putText(frame, s, (11,y), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, color, 2, cv2.LINE_AA)
                thumb_note = ' THUMB:Y' if thumb_pose else ''
                txt(f'Class: {label}   confidence: {conf:.2f}'+thumb_note+(' STOP-BLOCK' if label=='stop' and not stop_open else ''), 23)
                state_color = ((60, 230, 190) if handsfree.state == 'ACTIVE' else
                               (60, 210, 255) if handsfree.state == 'PAUSED' else
                               (20, 180, 250) if handsfree.state == 'STANDBY' else (200,200,200))
                shown_state = ('VOICE / ' + voice_state if voice_state != 'NORMAL'
                               else handsfree.state)
                txt(f'HMI: {shown_state}  Mode: {mode}', 49, state_color)
                txt(f'Pinch: 2D={pinch_ratio:.2f} 3D={pinch_ratio_3d:.2f}'
                    f'  {pinch_gate.status(now) if mode=="POINTER" else mode}', 74)
                txt(f'FPS: {fps:.1f}  Detect: {detect_ms:.0f}ms  Mouse: {"ON" if mouse_enabled else "OFF"}'
                    f'  ROI: {region_mode}', 99)
                txt(f'Scroll: {SCROLL_MIN_TICKS}..{SCROLL_MAX_TICKS} scale={scroll_ticks} minmove={SWIPE_THRESHOLDS[scroll_threshold_i]:.3f}'
                    f'  Last: {last_event if now-last_event_at<3 else "NONE"}', 124, scale=.46)
                if mode == 'GUN_BLOCKED':
                    txt('STANDBY: vertical GUN motion disabled; release to retry',
                        150,(55,140,255),.42)
                elif mode == 'SEEK':
                    txt(f'SEEK: {gun_seek.phase}  fired={gun_seek.fires}  LEFT -10s / RIGHT +10s',
                        150,(60,230,255),.48)
                elif mode == 'GUN_READY':
                    txt('GUN READY: first direction UP/DOWN=VOLUME, LEFT/RIGHT=SEEK',
                        150,(60,230,255),.42)
                elif mode == 'VOLUME':
                    shown = volume_motion.level
                    txt(f'VOLUME: {shown:.1f}%  UP louder / DOWN quieter, max 15%/s' if shown is not None
                        else 'VOLUME: not available', 150, (80,255,180), .49)
                elif volume_candidate:
                    txt(f'VOLUME GUN: hold {max(0.0,VOLUME_POSE_HOLD_S-(now-(volume_gate.since or now))):.1f}s'
                        '  (two fingers together, thumb apart)',150, (70,230,255), .44)
                elif mode in ('CLEAR_HOLD','CLEAR_ARMED','CLEAR_OK'):
                    if mode=='CLEAR_HOLD':
                        wait=max(0.0,CLEAR_FIST_HOLD_S-(now-(clear_gate.since or now)))
                        hint=f'CLEAR: hold FIST {wait:.1f}s (beep when ARMED)'
                    elif mode=='CLEAR_ARMED':
                        hint=f'CLEAR ARMED: show OK | tip={clear_ok_diag.get("tip2d",float("inf")):.2f} | OK={"YES" if clear_ok else "NO"}'
                    elif mode=='CLEAR_OK':
                        wait=max(0.0,CLEAR_OK_SELECT_HOLD_S-(now-(clear_gate.ok_since or now)))
                        hint=f'CLEAR OK: ERASE in {wait:.1f}s'
                    txt(hint,150,(55,215,255),.43)
                elif voice_state != 'NORMAL':
                    hint = ('KEEP THUMB UP = TALK; LOWER THUMB = STOP' if voice_state == 'RECORDING'
                            else f'Handy processing: wait {max(0.0, VOICE_PROCESSING_S-(now-(voice_stopped_at or now))):.1f}s')
                    txt('VOICE: '+ hint, 150, scale=.46)
                else:
                    txt(f'Click/Drag: d2={click_d2:.2f} d3={click_d3:.2f}'
                        f' idx={"Y" if extended else "N"} mid={"Y" if middle_folded else "N"}'
                        f' gate={click_gate.status(now)}', 150, scale=.43)
                if mode in ('PINCH','MEDIA_PINCH') and len(pinch_path)>8:
                    direction,why,circle_debug=circle_metrics(pinch_path,width/height)
                    cv2.putText(frame,
                         f'MEDIA circle: {circle_debug["angle_deg"]:+.0f} deg  CW >= 300; release to send',
                         (11,height-18),cv2.FONT_HERSHEY_SIMPLEX,.52,
                         (50,240,245) if direction=='CW' else (245,245,245),2,cv2.LINE_AA)
                elif mode=='POINTER' and clear_last_status.startswith(('BLOCKED','NO RECENT')):
                    cv2.putText(frame,clear_last_status[:85],(11,height-18),
                         cv2.FONT_HERSHEY_SIMPLEX,.41,(55,140,255),2,cv2.LINE_AA)
                elif now-last_event_at<2 and last_event=='CLEAR_REQUESTED':
                    cv2.putText(frame,'CLEAR: command sent (see console for edit-focus check)',
                         (11,height-18),cv2.FONT_HERSHEY_SIMPLEX,.45,(75,230,245),2,cv2.LINE_AA)
                elif handsfree.state=='STANDBY':
                    cv2.putText(frame,'STANDBY: CIRCLE=play/pause, GUN L/R=seek; PEACE=pointer',
                        (11,height-18),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,225,255),2,cv2.LINE_AA)
                if frame_number % PREVIEW_EVERY_N_FRAMES == 0:
                    cv2.imshow('Gesture HMI STEP 19 - GUN SEEK + VOLUME', frame)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                if key == 27:
                    was_active = handsfree.state != 'LOCKED'
                    switch_off('ESC: emergency HMI & mouse OFF')
                    if was_active and sounds_enabled:
                        play_state_sound('OFF_BY_STOP')
                elif key == ord('m'):
                    print('M is no longer needed: PEACE turns on the real Windows mouse.')
                elif key == ord('s'):
                    sounds_enabled = not sounds_enabled
                    print('Sound notification:', 'ON' if sounds_enabled else 'OFF')
                elif key in (ord('1'),ord('2'),ord('3')):
                    region_mode = key - ord('0')
                    smooth_cursor = None
                    print('Pointer ROI:', region_mode)
                elif key == ord('['):
                    scroll_threshold_i = max(0, scroll_threshold_i - 1)
                    print('Scroll min displacement:', SWIPE_THRESHOLDS[scroll_threshold_i])
                elif key == ord(']'):
                    scroll_threshold_i = min(2, scroll_threshold_i + 1)
                    print('Scroll min displacement:', SWIPE_THRESHOLDS[scroll_threshold_i])
                elif key == ord('b'):
                    if input_worker is not None:
                        new_mode = input_worker.toggle_back_mode()
                        print('BACK METHOD:', new_mode,
                              '- select Chrome again before testing navigation')
                    else:
                        print('BACK METHOD: SENDINPUT (default); first activate with PEACE')
                elif key == ord('z'):
                    scroll_ticks = max(10, scroll_ticks-20)
                    print('Scroll distance sensitivity (range stays 240..720):', scroll_ticks)
                elif key == ord('x'):
                    scroll_ticks = min(480, scroll_ticks+20)
                    print('Scroll distance sensitivity (range stays 240..720):', scroll_ticks)
                elif key in (ord('-'), ord('=')):
                    diff = -0.04 if key == ord('-') else +0.04
                    close_threshold = float(np.clip(close_threshold + diff, .25, .65))
                    pinch_gate.reset()
                    print('Pinch 2D CLOSE threshold:', round(close_threshold, 2),
                          '3D threshold:', round(close_threshold+PINCH_3D_EXTRA, 2),
                          'RELEASE threshold:', round(close_threshold+PINCH_HYSTERESIS, 2))
                elif key in (ord(','), ord('.')):
                    step = -0.04 if key == ord(',') else +0.04
                    click_threshold = float(np.clip(click_threshold + step, .24, .65))
                    reset_click()
                    print('Click 2D CONTACT threshold:', round(click_threshold,2),
                          '3D:',round(click_threshold + CLICK_3D_EXTRA,2),
                          'OPEN:',round(click_threshold + CLICK_HYSTERESIS,2))
                elif key == ord('r'):
                    volume_motion.stop()
                    volume_gate.reset(lock_until_release=True)
                    gun_seek.reset()
                    seek_worker.set_authorized(False)
                    reset_click()
                    reset_pinch()
                    if handsfree.state == 'ACTIVE':
                        input_worker.set_enabled(True)
                        mouse_enabled = True
                    mode = ('POINTER' if handsfree.state == 'ACTIVE' else handsfree.state)
                    smooth_cursor = None
                    print('Pointer, Pinch, Click, Volume and Back state reset')
    finally:
        # ALWAYS release any held Ctrl/Space keys, even on camera failure / Q.
        if handy_keys.held or voice_state == 'RECORDING':
            try:
                handy_keys.cancel()
                print('VOICE: keys released on program exit; Handy cancel requested')
            except Exception as exc:
                print('WARNING: could not release Handy keys:', exc)
        media_worker.set_authorized(False)
        media_worker.close()
        seek_worker.set_authorized(False)
        seek_worker.close()
        if input_worker is not None:
            input_worker.set_enabled(False)
            input_worker.close()
        cap.release()
        cv2.destroyAllWindows()
        print('Camera closed; Windows mouse control ended')




def delayed_back_test(method='SENDINPUT'):
    """Diagnostic: send one back action 5 sec after starting, with no webcam."""
    if sys.platform != 'win32':
        print('This keyboard test is for Windows only.')
        return
    try:
        import pyautogui
    except ImportError:
        print('Install with: python -m pip install pyautogui')
        return
    worker = LatestMouseWorker(pyautogui)
    worker.back_mode = method
    try:
        print('Back-only test:', method)
        print('Within 5 seconds, CLICK a Chrome tab that has a previous page.')
        print('After the countdown, Windows will receive ONE Alt+Left.')
        for seconds in range(5, 0, -1):
            print(f'  {seconds}...', flush=True)
            time.sleep(1)
        print('Foreground BEFORE input:', worker._foreground())
        worker._send_back()
        print('Check Chrome: did it navigate back?')
    finally:
        worker.close()


def step19_self_test():
    # First strong axis wins; diagonals are never classified.
    assert classify_gun_axis(.042,.003)=='SEEK'
    assert classify_gun_axis(.003,-.035)=='VOLUME'
    assert classify_gun_axis(.050,.055) is None
    assert classify_gun_axis(.005,.002) is None
    seek=GunSeekMotion()
    seek.start(0.,(.50,.5))
    assert seek.update(.1,(.53,.50)) is None
    assert seek.update(.3,(.585,.502))==10
    assert seek.phase=='RETURN'
    # Stationary at endpoint cannot auto-repeat.
    for t in (.5,1.,2.,3.):
        assert seek.update(t,(.585,.502)) is None
    # Return motion cannot trigger negative seek.
    assert seek.update(3.2,(.545,.505)) is None
    assert seek.update(3.4,(.512,.505)) is None
    assert seek.update(3.6,(.51,.505))=='REARMED'
    assert seek.update(3.9,(.41,.503))==-10
    assert seek.fires==2
    assert seek.update(4.2,(.41,.503)) is None
    seek.reset()
    assert seek.update(4.5,(.2,.5)) is None
    # Large landmark teleports must never fire.
    seek.start(0.,(.5,.5))
    assert seek.update(.1,(.8,.5)) is None
    # Bounded playback time: no negative and no past media end.
    assert compute_seek_position(15,0,200,-10)==5
    assert compute_seek_position(5,0,200,-10)==0
    assert compute_seek_position(195,0,200,10)==200
    assert compute_seek_position(10,0,0,10) is None
    # No J/L on failed API unless real safe foreground is verified.
    calls=[]
    backend=WindowsMediaSeek(jl_sender=lambda d:calls.append(d),
                             safe_foreground=lambda:False)
    async def unsupported(_): return False,'no session'
    backend._winrt_seek=unsupported
    assert backend.seek(10)[0] is False and not calls
    backend.safe_foreground=lambda:True
    assert backend.seek(-10)[0] is True and calls==[-10]
    # Worker authorization and rapid repeat with fake backend.
    class FakeSeekBackend:
        def __init__(self): self.sent=[]
        def seek(self,delta): self.sent.append(delta); return True,'mock'
    fake=FakeSeekBackend()
    worker=SeekCommandWorker(fake)
    try:
        assert not worker.request(10)
        worker.set_authorized(True)
        assert worker.request(10)
        assert worker.request(-10)
        until=time.monotonic()+1.0
        while len(fake.sent)<2 and time.monotonic()<until:
            time.sleep(.01)
        assert fake.sent==[10,-10],fake.sent
        worker.set_authorized(False)
        assert not worker.request(10)
    finally:
        worker.close()
    print('STEP19: axis lock / seek return / repeat / bounds / safety / worker tests passed.')


def windows_distribution_doctor(camera=False):
    """Read-only Windows deployment check; camera opened only when requested.

    No model downloads or input actions are performed. This can run on CI without
    attaching a physical webcam, and can run on an end-user PC with --camera.
    """
    failures = []
    def check(label, fn):
        try:
            details = fn()
            print(f'[OK] {label}: {details if details is not None else "ready"}')
        except Exception as err:
            print(f'[FAIL] {label}: {type(err).__name__}: {err}')
            failures.append(label)

    print('GESTURE HMI WINDOWS DIAGNOSTICS')
    print('Frozen EXE:', bool(getattr(sys, 'frozen', False)))
    print('Platform:', sys.platform)
    check('NumPy gesture MLP', lambda: load_numpy_mlp()[-1])

    def check_task():
        import mediapipe as mp
        path = find_hand_model(download=False)
        if path is None:
            raise FileNotFoundError('bundled hand_landmarker.task not found')
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1)
        detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        detector.close()
        return f'loaded model ({path.name})'
    check('MediaPipe hand task', check_task)
    check('OpenCV', lambda: __import__('cv2').__version__)
    check('PyAutoGUI module', lambda: __import__('pyautogui').__version__)

    for label, module in (('UI Automation (optional)', 'uiautomation'),
                          ('Windows audio control (optional)', 'pycaw'),
                          ('COM support (optional)', 'comtypes')):
        try:
            __import__(module)
            print(f'[OK] {label}')
        except Exception as err:
            print(f'[WARN] {label}: {type(err).__name__}: {err}')

    if sys.platform == 'win32':
        try:
            print('[INFO] Handy:', 'RUNNING' if _handy_running() else
                  ('INSTALLED' if _handy_locations() else 'NOT INSTALLED (optional)'))
        except Exception as err:
            print('[WARN] Handy probe:', err)
    if camera:
        def check_camera():
            import cv2
            cam, idx, backend = open_compatible_camera(cv2)
            cam.release()
            return f'camera index={idx}, backend={backend}'
        check('Webcam', check_camera)
    print('DOCTOR:', 'PASS' if not failures else 'FAIL: '+', '.join(failures))
    return not failures


def _crash_report(exc):
    """Keep a traceback after the executable window closes."""
    base = (Path(os.environ['LOCALAPPDATA']) / 'GestureHMI' / 'logs'
            if os.environ.get('LOCALAPPDATA') else Path.home() / '.gesture_hmi' / 'logs')
    try:
        base.mkdir(parents=True, exist_ok=True)
        target = base / 'app_last.log'
        target.write_text('Gesture HMI v4 startup/runtime error\n' +
                          ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                          encoding='utf-8')
        print('Error report saved to:', target, file=sys.stderr)
    except Exception as logerr:
        print('Could not save crash report:', logerr, file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def volume_status_diagnostic():
    """Read (do not change) system volume and mute state."""
    try:
        endpoint=WindowsVolume()
        percent=endpoint.get_percent()
        muted=endpoint.is_muted()
        print(f'WINDOWS MASTER OUTPUT: level={percent:.1f}% mute={"ON" if muted else "OFF"}')
    except Exception as exc:
        print('VOLUME STATUS UNAVAILABLE:',type(exc).__name__,str(exc))


def dispatch_cli():
    if '--doctor' in sys.argv:
        raise SystemExit(0 if windows_distribution_doctor('--camera' in sys.argv) else 5)
    if '--camera-test' in sys.argv:
        camera_diagnostic()
    elif '--resource-check' in sys.argv:
        raise SystemExit(0 if resource_diagnostic() else 4)
    elif '--self-test' in sys.argv:
        self_test()
        step19_self_test()
    elif '--back-test-pyauto' in sys.argv:
        delayed_back_test('PYAUTOGUI')
    elif '--back-test' in sys.argv:
        delayed_back_test('SENDINPUT')
    elif '--media-test' in sys.argv:
        delayed_media_test()
    elif '--volume-status' in sys.argv:
        volume_status_diagnostic()
    elif '--seek-test' in sys.argv:
        delayed_seek_test()
    elif '--focus-test' in sys.argv:
        focus_diagnostic()
    elif '--focus-stability-test' in sys.argv:
        focus_stability_diagnostic()
    elif '--handy-auto-test' in sys.argv:
        print('HANDY AUTO RESULT:', ensure_handy_running())
    elif '--handy-test' in sys.argv:
        handy_shortcut_test()
    else:
        main()


if __name__ == '__main__':
    try:
        dispatch_cli()
    except SystemExit:
        raise
    except Exception as exc:
        _crash_report(exc)
        raise SystemExit(1)
