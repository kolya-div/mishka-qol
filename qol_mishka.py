"""Qo'l bilan mishkani boshqarish (kamera + MediaPipe).

Imo-ishoralar:
  1 barmoq (ko'rsatkich)                 -> kursor qo'l bilan harakatlanadi
  2 barmoq (ko'rsatkich + o'rta)         -> O'NG tugma
  3 barmoq (ko'rsatkich, o'rta, nomsiz)  -> CHAP tugma
  ko'rsatkich + jimjiloq ("rok")         -> IKKI MARTA BOSISH
  bosh + ko'rsatkich tekkizib ushlash    -> SUDRASH (qo'yib yuborsangiz tashlanadi)
  ochiq kaft + qo'lni tepaga/pastga      -> SCROLL
  ochiq kaft + qo'lni tez o'ngga/chapga  -> brauzerda OLDINGA / ORQAGA
  mushtni 1 soniya ushlash               -> dasturni o'chirish
  qo'l 5 soniya ko'rinmasa               -> dastur o'zi o'chadi
  kaftni to'liq ochib 2 soniya ushlash   -> dasturni yoqish
Tugmalar (kamera oynasida): q - chiqish, h - yashirish, m - kichik/katta, s - sozlamalar, k - kalibrovka.
Hamma joyda ishlaydigan: Ctrl+Alt+Q / H / M / S / K.
"""
import argparse
import ctypes
import os
import sys
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import sozlamalar

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# --- Quyidagi qiymatlarning ko'pi sozlamalar oynasidan o'zgartiriladi (settings.json) ---
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720   # kamera o'lchami (kengroq ko'rish maydoni; kamera qo'llamasa o'zi moslashadi)
PREVIEW_W = 1000               # ekrandagi kamera oynasi kengligi
MINI_W = 360                   # 'm' bosilganda kichik oyna kengligi
WINDOW = "Qol mishka (q - chiqish)"
MARGIN = 0.07                  # kadr chetidagi "o'lik zona" (kalibrovka bo'lmasa)
CALIB = None                   # kalibrovka: (x0, x1, y0, y1) kadr koordinatalarida

CURSOR_MIN_CUTOFF = 1.5        # One Euro: kichik = silliqroq
CURSOR_BETA = 0.012            # One Euro: katta = tez harakatda kechikish kam
RELATIVE_MODE = False          # True: kursor tezlashuvli (trekpad kabi)
ACCEL_MIN, ACCEL_MAX = 0.6, 2.5
ACCEL_SPEED = 0.8              # (ekran kengligi / soniya) shu tezlikda tezlashuv maksimal

PINCH_ON = 0.28                # bosh va ko'rsatkich uchlari orasi (kaft o'lchamiga nisbatan) < shu bo'lsa "tegdi"
PINCH_OFF = 0.42
FIST_MIN_RATIO = 0.45          # mushtda bosh va ko'rsatkich uchlari shundan uzoqroq bo'ladi
FIST_HOLD = 1.0                # mushtni shuncha soniya ushlasangiz dastur o'chadi
THUMB_OUT = 0.55               # bosh barmoq uchi - ko'rsatkich asosi / kaft uzunligi > shu bo'lsa "ochiq"
ENABLE_HOLD = 2.0              # kaftni to'liq ochib shuncha soniya ushlasangiz dastur yoqiladi
AUTO_OFF_SECONDS = 5.0         # qo'l shuncha soniya ko'rinmasa dastur o'zi o'chadi

SCROLL_GAIN = 6000
SCROLL_SMOOTH = 0.5
SCROLL_FRAMES = 2
WHEEL_NOTCH = 120

SWIPE_ENABLED = True
SWIPE_DIST = 0.22              # kadr kengligining shu qismicha tez siljish -> oldinga/orqaga
SWIPE_WINDOW = 0.4             # shu soniya ichida
SWIPE_COOLDOWN = 1.0

LOST_GRACE = 8                 # qo'l shuncha kadr ko'rinmasa ham oldingi holat saqlanadi
LEFT_FRAMES = 4
RIGHT_FRAMES = 5
DOUBLE_FRAMES = 4
CLICK_COOLDOWN = 0.4

CAL_FRAMES = 40                # kalibrovkada har burchakda shuncha kadr ushlab turish kerak

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

DEBUG = {"pinch": 0.0, "up": 0}


# ---------------------------------------------------------------- kamera
class LatestFrame:
    """Kamerani alohida oqimda o'qiydi, doim eng yangi kadrni beradi (kechikishni yo'qotadi)."""

    @staticmethod
    def _open(index):
        """Kamerani ochishga urinadi: avval MSMF (30 kadr/s), keyin DSHOW, keyin standart."""
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok = False
                for _ in range(15):          # birinchi kadrlar ba'zan bo'sh keladi
                    ok, _f = cap.read()
                    if ok:
                        break
                    time.sleep(0.05)
                if ok:
                    return cap
            cap.release()
        return None

    def __init__(self, index):
        # MSMF 30 kadr/s beradi; DSHOW esa xira yorug'likda 15 kadr/s ga tushib qotishga sabab bo'ladi
        self.cap = self._open(index)
        if self.cap is None:                 # boshqa kamera raqamlarini ham sinab ko'ramiz
            for i in range(4):
                if i != index:
                    self.cap = self._open(i)
                    if self.cap is not None:
                        print(f"Kamera {i}-raqam bilan topildi.")
                        break
        if self.cap is None:
            raise SystemExit(
                "Kamera ochilmadi.\n"
                " - Kamera ulanganini tekshiring va boshqa dastur (Zoom, Teams...) uni ishlatmayotganiga ishonch hosil qiling.\n"
                " - Windows: Sozlamalar > Maxfiylik > Kamera > 'Desktop ilovalarga ruxsat berish' yoqilgan bo'lsin.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # avtomatik ekspozitsiya (aks holda kamera qorong'i qolib ketishi mumkin)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        self.frame = None
        self.stop = False
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            f, self.frame = self.frame, None
        return f

    def release(self):
        self.stop = True
        time.sleep(0.05)
        self.cap.release()


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Model yuklanmoqda...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ---------------------------------------------------------------- Windows kiritish
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.c_void_p)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


def wheel(amount):
    """Sichqoncha g'ildiragini aylantiradi (musbat = tepaga). SendInput orqali."""
    inp = _INPUT(type=0, mi=_MOUSEINPUT(0, 0, int(amount) & 0xFFFFFFFF, 0x0800, 0, None))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def set_cursor(x, y):
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


def click(button):
    pyautogui.click(button=button)


def double_click():
    pyautogui.doubleClick()


def mouse_down():
    pyautogui.mouseDown(button="left")


def mouse_up():
    pyautogui.mouseUp(button="left")


def swipe(direction):
    """Qo'l o'ngga siltandi -> oldinga (Alt+O'ng), chapga -> orqaga (Alt+Chap)."""
    pyautogui.hotkey("alt", "right" if direction == "right" else "left")


class Hotkeys:
    """Hamma joyda ishlaydigan Ctrl+Alt+harf tugmalari (kamera oynasi yashirin bo'lsa ham)."""
    KEYS = {"quit": 0x51, "hide": 0x48, "mini": 0x4D, "settings": 0x53, "calib": 0x4B}

    def __init__(self):
        self.prev = {}

    def poll(self):
        state = ctypes.windll.user32.GetAsyncKeyState
        mods = bool(state(0x11) & 0x8000) and bool(state(0x12) & 0x8000)
        events = []
        for name, vk in self.KEYS.items():
            down = mods and bool(state(vk) & 0x8000)
            if down and not self.prev.get(name):
                events.append(name)
            self.prev[name] = down
        return events


# ---------------------------------------------------------------- imo-ishoralar
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def finger_up(lm, name):
    wrist = lm[0]
    return dist(lm[TIP[name]], wrist) > dist(lm[PIP[name]], wrist) * 1.15


def thumb_out(lm):
    size = max(dist(lm[0], lm[9]), 1e-6)
    return dist(lm[4], lm[5]) / size > THUMB_OUT


def classify(lm, pinching):
    """-> (gesture, pinching). gesture: drag / fist / scroll / left / double / right / move / none"""
    up = {n: finger_up(lm, n) for n in TIP}
    count = sum(up.values())
    size = max(dist(lm[0], lm[9]), 1e-6)
    ratio = dist(lm[4], lm[8]) / size             # bosh va ko'rsatkich uchlari orasi
    pinching = ratio < (PINCH_OFF if pinching else PINCH_ON)
    DEBUG["up"], DEBUG["pinch"] = count, ratio
    if pinching:
        return "drag", True
    if count == 0 and ratio > FIST_MIN_RATIO:
        return "fist", False
    if all(up.values()):                  # 4 ta barmoq (bosh barmoq e'tiborga olinmaydi) = ochiq kaft
        return "scroll", False
    if up["index"] and up["middle"] and up["ring"] and not up["pinky"]:
        return "left", False
    if up["index"] and up["pinky"] and not up["middle"] and not up["ring"]:
        return "double", False
    if count == 2 and up["index"] and up["middle"]:
        return "right", False
    if up["index"] and count == 1:
        return "move", False
    return "none", False


class OneEuro:
    """One Euro filtri: sekin harakatda titrashni yo'qotadi, tez harakatda kechikmaydi."""

    def __init__(self, min_cutoff, beta, dcutoff=1.0):
        self.min_cutoff, self.beta, self.dcutoff = min_cutoff, beta, dcutoff
        self.prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * 3.141592653589793 * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def prime(self, x, y, t):
        self.prev = (x, y, 0.0, 0.0, t)

    def __call__(self, x, y, t):
        if self.prev is None:
            self.prime(x, y, t)
            return x, y
        px, py, pdx, pdy, pt = self.prev
        dt = max(t - pt, 1e-3)
        ad = self._alpha(self.dcutoff, dt)
        dx = pdx + ad * ((x - px) / dt - pdx)
        dy = pdy + ad * ((y - py) / dt - pdy)
        cutoff = self.min_cutoff + self.beta * (dx * dx + dy * dy) ** 0.5
        a = self._alpha(cutoff, dt)
        fx, fy = px + a * (x - px), py + a * (y - py)
        self.prev = (fx, fy, dx, dy, t)
        return fx, fy


class Controller:
    """Ishorani aniqlab, kursor/bosish/scroll/sudrash/siltash amallarini bajaradi."""

    def __init__(self, screen_w, screen_h, cursor_pos, io=None):
        self.sw, self.sh = screen_w, screen_h
        self.cur_x, self.cur_y = cursor_pos
        self.io = io or {"cursor": set_cursor, "click": click, "wheel": wheel, "down": mouse_down,
                         "up": mouse_up, "double": double_click, "swipe": swipe}
        self.history = deque(maxlen=5)    # kursorning oxirgi joylari (bosishdagi siljishni qaytarish uchun)
        self.swipe_hist = deque()
        self.filter = OneEuro(CURSOR_MIN_CUTOFF, CURSOR_BETA)
        self.enabled = True
        self.paused = False               # kalibrovka paytida harakatlar to'xtatiladi
        self.last_seen = None
        self.last_click = float("-inf")
        self.last_swipe = float("-inf")
        self.block_scroll_until = float("-inf")
        self.fist_start, self.fist_progress = None, 0.0
        self.enable_start, self.enable_ok, self.enable_progress = None, 0.0, 0.0
        self.rel_prev = None
        self.reset()

    # ----- holatni tozalash
    def reset(self):
        self.release()
        self.pinching = False
        self.candidate, self.streak, self.confirmed = "none", 0, "none"
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.lost = 0
        self.fist_start, self.fist_progress = None, 0.0
        self.enable_start, self.enable_progress = None, 0.0
        self.rel_prev = None
        self.history.clear()
        self.swipe_hist.clear()

    def release(self):
        if getattr(self, "dragging", False):
            self.io["up"]()
        self.dragging = False
        self.drag_off = (0.0, 0.0)

    def disable(self):
        self.enabled = False
        self.reset()

    def enable(self, now):
        self.enabled = True
        self.reset()
        self.last_seen = now

    def no_hand(self, now=None):
        now = time.time() if now is None else now
        if self.last_seen is None:
            self.last_seen = now
        # qo'l 1-2 kadrga yo'qolsa holatni saqlab turamiz (uzilib-uzilib qolmasligi uchun)
        self.lost += 1
        if self.lost > LOST_GRACE:
            self.reset()
            self.lost = LOST_GRACE + 1
        if self.enabled and not self.paused and now - self.last_seen > AUTO_OFF_SECONDS:
            self.disable()

    @property
    def status(self):
        return self.confirmed if self.enabled else "off"

    # ----- asosiy yangilanish
    def update(self, lm, now=None):
        now = time.time() if now is None else now
        self.lost = 0
        self.last_seen = now
        gesture, self.pinching = classify(lm, self.pinching)

        if not self.enabled:
            self._enable_check(lm, gesture, now)
            return "off"

        if gesture == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = gesture, 1

        need = {"move": 1, "left": LEFT_FRAMES, "right": RIGHT_FRAMES, "double": DOUBLE_FRAMES,
                "scroll": SCROLL_FRAMES}.get(self.candidate, 2)
        if self.streak >= need:
            prev, self.confirmed = self.confirmed, self.candidate
            self._fist_check(now)
            if not self.enabled:
                return "off"
            if prev == "drag" and self.confirmed != "drag":
                self.release()
            if self.paused:
                pass
            elif self.confirmed == "move":
                self._move(lm, now, prev != "move")
            elif self.confirmed == "drag":
                self._drag(lm, now, prev != "drag")
            elif self.confirmed in ("right", "left", "double") and prev != self.confirmed \
                    and now - self.last_click > CLICK_COOLDOWN:
                self._click(self.confirmed, now)
            elif self.confirmed == "scroll":
                self._scroll(lm[9].y, prev != "scroll", now)
                self._swipe(lm[9].x, lm[9].y, now)
        if self.confirmed != "scroll":
            self.scroll_y = None
            self.swipe_hist.clear()
        return self.status

    # ----- dasturni yoqish / o'chirish
    def _fist_check(self, now):
        """Mushtni FIST_HOLD soniya ushlab turilsa dastur o'chadi."""
        if self.confirmed != "fist":
            self.fist_start, self.fist_progress = None, 0.0
            return
        if self.fist_start is None:
            self.fist_start = now
        self.fist_progress = min((now - self.fist_start) / FIST_HOLD, 1.0)
        if self.fist_progress >= 1.0:
            self.disable()

    def _enable_check(self, lm, gesture, now):
        """O'chiq payti kaft to'liq ochilib ENABLE_HOLD soniya ushlansa dastur yoqiladi."""
        if gesture == "scroll" and thumb_out(lm):
            self.enable_ok = now
            if self.enable_start is None:
                self.enable_start = now
            self.enable_progress = min((now - self.enable_start) / ENABLE_HOLD, 1.0)
            if self.enable_progress >= 1.0:
                self.enable(now)
        elif self.enable_start is not None and now - self.enable_ok > 0.3:
            self.enable_start, self.enable_progress = None, 0.0

    # ----- bosishlar
    def _click(self, kind, now):
        if self.history:                  # ishora boshlanishidan oldingi joyga qaytamiz
            self.cur_x, self.cur_y = self.history[0]
            self.io["cursor"](self.cur_x, self.cur_y)
        if kind == "double":
            self.io["double"]()
        else:
            self.io["click"](kind)
        self.last_click = now

    # ----- kursor
    def _target(self, lm):
        if CALIB:
            x0, x1, y0, y1 = CALIB
            nx = (lm[8].x - x0) / max(x1 - x0, 1e-6)
            ny = (lm[8].y - y0) / max(y1 - y0, 1e-6)
        else:
            nx = (lm[8].x - MARGIN) / (1 - 2 * MARGIN)
            ny = (lm[8].y - MARGIN) / (1 - 2 * MARGIN)
        nx, ny = min(max(nx, 0), 1), min(max(ny, 0), 1)
        return nx * (self.sw - 1), ny * (self.sh - 1)

    def _pointer(self, tx, ty, now, first):
        self.filter.min_cutoff, self.filter.beta = CURSOR_MIN_CUTOFF, CURSOR_BETA
        if RELATIVE_MODE:
            if first or self.rel_prev is None:
                self.filter.prime(tx, ty, now)
                self.rel_prev = (tx, ty, now)
                return
            fx, fy = self.filter(tx, ty, now)
            px, py, pt = self.rel_prev
            dx, dy = fx - px, fy - py
            speed = (dx * dx + dy * dy) ** 0.5 / max(now - pt, 1e-3) / self.sw
            gain = ACCEL_MIN + (ACCEL_MAX - ACCEL_MIN) * min(speed / ACCEL_SPEED, 1.0)
            self.cur_x = min(max(self.cur_x + dx * gain, 0), self.sw - 1)
            self.cur_y = min(max(self.cur_y + dy * gain, 0), self.sh - 1)
            self.rel_prev = (fx, fy, now)
        else:
            if first:
                self.filter.prime(self.cur_x, self.cur_y, now)
            self.cur_x, self.cur_y = self.filter(tx, ty, now)
        self.io["cursor"](self.cur_x, self.cur_y)

    def _move(self, lm, now, first):
        tx, ty = self._target(lm)
        self._pointer(tx, ty, now, first)
        self.history.append((self.cur_x, self.cur_y))

    def _drag(self, lm, now, first):
        tx, ty = self._target(lm)
        if first:
            if self.history:              # barmoqlar tekkunga qadar bo'lgan joyni bosamiz
                self.cur_x, self.cur_y = self.history[0]
                self.io["cursor"](self.cur_x, self.cur_y)
            self.drag_off = (self.cur_x - tx, self.cur_y - ty)
            self._pointer(tx + self.drag_off[0], ty + self.drag_off[1], now, True)
            self.io["down"]()
            self.dragging = True
            return
        self._pointer(tx + self.drag_off[0], ty + self.drag_off[1], now, False)

    # ----- scroll va siltash
    def _scroll(self, y, first, now):
        if first or self.scroll_y is None:
            self.scroll_y, self.scroll_acc = y, 0.0
            return
        new_y = self.scroll_y + (y - self.scroll_y) * SCROLL_SMOOTH
        dy = self.scroll_y - new_y        # qo'l ko'tarilsa musbat -> tepaga scroll
        self.scroll_y = new_y
        if now < self.block_scroll_until:
            return
        self.scroll_acc += dy * SCROLL_GAIN
        amount = int(self.scroll_acc / WHEEL_NOTCH) * WHEEL_NOTCH   # to'liq "tishcha"lar
        if amount:
            self.io["wheel"](amount)
            self.scroll_acc -= amount

    def _swipe(self, x, y, now):
        if not SWIPE_ENABLED:
            return
        h = self.swipe_hist
        h.append((now, x, y))
        while h and now - h[0][0] > SWIPE_WINDOW:
            h.popleft()
        if len(h) >= 3 and now - self.last_swipe > SWIPE_COOLDOWN:
            dx, dy = h[-1][1] - h[0][1], abs(h[-1][2] - h[0][2])
            if abs(dx) > SWIPE_DIST and dy < abs(dx) * 0.6:
                self.io["swipe"]("right" if dx > 0 else "left")
                self.last_swipe = now
                self.block_scroll_until = now + 0.6
                self.scroll_acc = 0.0
                h.clear()


# ---------------------------------------------------------------- kalibrovka
class Calibrator:
    """4 burchakni ketma-ket ko'rsatib, qo'l qaysi maydonda ishlashini o'rganadi."""
    CORNERS = [("CHAP-YUQORI", 0, 0), ("O'NG-YUQORI", 1, 0), ("O'NG-PASTKI", 1, 1), ("CHAP-PASTKI", 0, 1)]

    def __init__(self):
        self.idx, self.samples, self.points, self.result = 0, [], [], None
        self.error = ""

    @property
    def progress(self):
        return min(len(self.samples) / CAL_FRAMES, 1.0)

    @property
    def corner(self):
        return self.CORNERS[self.idx]

    def feed(self, lm, gesture):
        """Har kadrda chaqiriladi. Tugaganda True qaytaradi (natija self.result da)."""
        if lm is None or gesture != "move":
            self.samples = []
            return False
        self.samples.append((lm[8].x, lm[8].y))
        if len(self.samples) < CAL_FRAMES:
            return False
        xs = sorted(s[0] for s in self.samples)
        ys = sorted(s[1] for s in self.samples)
        self.points.append((xs[len(xs) // 2], ys[len(ys) // 2]))     # mediana
        self.samples = []
        self.idx += 1
        if self.idx < len(self.CORNERS):
            return False
        p = self.points
        x0, x1 = (p[0][0] + p[3][0]) / 2, (p[1][0] + p[2][0]) / 2
        y0, y1 = (p[0][1] + p[1][1]) / 2, (p[2][1] + p[3][1]) / 2
        if x1 - x0 < 0.1 or y1 - y0 < 0.1:
            self.error = "Burchaklar noto'g'ri tanlandi, qayta urinib ko'ring."
            self.idx, self.points = 0, []
            return False
        self.result = (x0, x1, y0, y1)
        return True


# ---------------------------------------------------------------- asosiy sikl
LABELS = {"move": "1 barmoq: HARAKAT", "right": "2 barmoq: O'NG TUGMA",
          "left": "3 barmoq: CHAP TUGMA", "scroll": "OCHIQ KAFT: SCROLL / yon tomonga siltash",
          "drag": "TEKKIZISH: SUDRASH", "fist": "MUSHT: ushlab turing (o'chirish)...",
          "double": "ROK ISHORA: IKKI MARTA BOSISH", "none": "Qo'l yo'q / noma'lum",
          "off": "O'CHIRILGAN - yoqish: kaftni TO'LIQ oching va ushlab turing"}


def parse_args():
    ap = argparse.ArgumentParser(description="Qo'l bilan mishkani boshqarish")
    ap.add_argument("--yashirin", action="store_true", help="kamera oynasiz (fonda) ishga tushirish")
    ap.add_argument("--avtoyoqish", action="store_true", help="Windows bilan birga yoqilishni yoqish")
    ap.add_argument("--avtoyoqish-ochir", action="store_true", help="Windows bilan birga yoqilishni o'chirish")
    return ap.parse_args()


def main():
    if os.name != "nt":
        raise SystemExit("Bu dastur faqat Windows uchun yozilgan.")
    args = parse_args()
    if args.avtoyoqish or args.avtoyoqish_ochir:
        sozlamalar.set_autostart(args.avtoyoqish)
        print("Windows bilan birga yoqilish:", "YOQILDI" if args.avtoyoqish else "O'CHIRILDI")
        return

    settings = sozlamalar.Settings()
    settings.apply(globals())
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    # zaxira detektor: asosiysi qo'lni yo'qotsa, kontrasti oshirilgan kadrda har safar noldan qidiradi
    rescue = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
    ))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cam = LatestFrame(CAM_INDEX)
    ctrl = Controller(*pyautogui.size(), pyautogui.position())
    hotkeys = Hotkeys()
    state = {"hidden": args.yashirin, "mini": False, "calib": None, "win": None, "size": None}

    def on_settings_change():
        settings.apply(globals())

    def start_calibration():
        ctrl.enable(time.time())
        ctrl.paused = True
        state["calib"] = Calibrator()
        state["hidden"] = False

    def clear_calibration():
        settings.calib = None
        settings.save()
        settings.apply(globals())

    panel = None
    if sozlamalar.tk is not None:
        try:
            panel = sozlamalar.Panel(settings, on_settings_change, start_calibration, clear_calibration)
        except Exception as e:                       # Tk ishlamasa ham asosiy dastur ishlayveradi
            print("Sozlamalar oynasi ochilmadi:", e)

    t0 = time.time()
    last_ts = -1

    def sync_window():
        """Kamera oynasini holatga (yashirin/ko'rinadigan/o'lcham) moslaydi."""
        if state["hidden"]:
            if state["win"]:
                cv2.destroyWindow(WINDOW)
                state["win"], state["shown"] = None, False
            return
        want_w = MINI_W if state["mini"] else PREVIEW_W
        if not state["win"]:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)   # sichqoncha bilan o'lchamini o'zgartirsa bo'ladi
            state["win"], state["size"] = True, None
        if state["size"] != want_w:
            cv2.resizeWindow(WINDOW, want_w, int(want_w * FRAME_H / FRAME_W))
            state["size"] = want_w

    running = True
    while running:
        events = hotkeys.poll()
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            events.append("quit")
        elif key == ord("h"):
            events.append("hide")
        elif key == ord("m"):
            events.append("mini")
        elif key == ord("s"):
            events.append("settings")
        elif key == ord("k"):
            events.append("calib")
        for ev in events:
            if ev == "quit":
                running = False
            elif ev == "hide":
                state["hidden"] = not state["hidden"]
            elif ev == "mini":
                state["mini"] = not state["mini"]
            elif ev == "settings" and panel:
                panel.show()
            elif ev == "calib":
                if state["calib"]:
                    state["calib"], ctrl.paused = None, False     # bekor qilish
                else:
                    start_calibration()
        if not running:
            break
        if panel:
            panel.pump()
        sync_window()
        if state["win"] and state.get("shown") and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break                                      # oyna X tugmasi bilan yopildi -> chiqish

        frame = cam.read()
        if frame is None:
            time.sleep(0.002)
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.time() - t0) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        hands = result.hand_landmarks
        if not hands:                                    # qo'l yo'qoldi -> zaxira usuli
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            boosted = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
            hands = rescue.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=boosted)).hand_landmarks

        cal = state["calib"]
        if hands:
            lm = hands[0]
            ctrl.update(lm)
            if cal and cal.feed(lm, ctrl.confirmed):
                settings.calib = cal.result
                settings.save()
                settings.apply(globals())
                state["calib"], ctrl.paused = None, False
            if not state["hidden"]:
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        else:
            ctrl.no_hand()
            if cal:
                cal.feed(None, "none")

        if state["hidden"]:
            continue

        # ---------------- chizish
        k = w / 640.0                                        # yozuvlar kadr o'lchamiga qarab kattalashadi
        font = cv2.FONT_HERSHEY_SIMPLEX
        if CALIB:
            x0, x1, y0, y1 = CALIB
            box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        else:
            box = (int(MARGIN * w), int(MARGIN * h), int((1 - MARGIN) * w), int((1 - MARGIN) * h))
        cv2.rectangle(frame, box[:2], box[2:], (255, 128, 0), 1)

        status = ctrl.status
        if status == "fist" and ctrl.fist_progress > 0:      # o'chirish uchun mushtni ushlash progressi
            prog, pcol = ctrl.fist_progress, (0, 140, 255)
        elif status == "off" and ctrl.enable_progress > 0:   # yoqish uchun kaftni ushlash progressi
            prog, pcol = ctrl.enable_progress, (0, 255, 0)
        else:
            prog, pcol = 0, None
        if prog:
            y0b, y1b, bw = int(42 * k), int(56 * k), int(240 * k)
            cv2.rectangle(frame, (10, y0b), (10 + int(bw * prog), y1b), pcol, -1)
            cv2.rectangle(frame, (10, y0b), (10 + bw, y1b), (255, 255, 255), 1)
        color = (0, 0, 255) if status == "off" else (0, 255, 255)
        cv2.putText(frame, LABELS[status], (10, int(30 * k)), font, 0.65 * k, color, max(2, int(2 * k)))

        if cal:
            name, cx, cy = cal.corner
            px, py = int(w * (0.04 + 0.92 * cx)), int(h * (0.06 + 0.88 * cy))
            cv2.circle(frame, (px, py), int(22 * k), (0, 255, 0), 3)
            cv2.ellipse(frame, (px, py), (int(30 * k), int(30 * k)), -90, 0, int(360 * cal.progress),
                        (0, 255, 255), 4)
            msg = (f"KALIBROVKA {cal.idx + 1}/4: 1 barmoq ko'rsating, uchini {name} burchakka "
                   f"eng chekka qulay joyga qo'ying va ushlab turing")
            cv2.putText(frame, msg, (10, int(h * 0.5)), font, 0.55 * k, (0, 255, 0), max(1, int(1.5 * k)))
            if cal.error:
                cv2.putText(frame, cal.error, (10, int(h * 0.56)), font, 0.55 * k, (0, 0, 255), 2)

        cv2.putText(frame, f"barmoqlar: {DEBUG['up']}  tekkizish: {DEBUG['pinch']:.2f}   "
                           f"q-chiqish h-yashirish m-kichik s-sozlamalar k-kalibrovka (Ctrl+Alt+harf)",
                    (10, h - int(12 * k)), font, 0.42 * k, (255, 255, 255), 1)
        cv2.imshow(WINDOW, frame)
        state["shown"] = True

    ctrl.release()                       # sudrash paytida chiqilsa, tugma bosilgan holda qolmasin
    cam.release()
    cv2.destroyAllWindows()
    landmarker.close()
    rescue.close()


if __name__ == "__main__":
    main()
