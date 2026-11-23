"""Qo'l bilan mishkani boshqarish (kamera + MediaPipe).

Imo-ishoralar:
  1 barmoq (ko'rsatkich)                 -> kursor qo'l bilan harakatlanadi
  2 barmoq (ko'rsatkich + o'rta)         -> CHAP tugma   (sozlamalarda almashtirish mumkin)
  3 barmoq (ko'rsatkich, o'rta, nomsiz)  -> O'NG tugma
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
import atexit
import ctypes
import math
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
CALIB_PALM = None              # kalibrovka paytidagi qo'l o'lchami (kadr balandligiga nisbatan)
FAR_MODE = True                # uzoqdan boshqarish: qo'l kichik ko'rinsa harakat kuchayadi
FAR_MAX = 3.5                  # eng ko'p kuchaytirish
REF_PALM = 0.22                # shu qo'l o'lchamida kuchaytirish yo'q (~50 sm masofa)
SIZE_SMOOTH = 0.08             # qo'l o'lchamini silliqlash (kichik = sekin va barqaror)

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
THREE_FRAMES = 6               # 3 barmoq shuncha kadr ketma-ket ushlansagina bosiladi (tasodifiy bosishga qarshi)
TWO_FRAMES = 8                 # 2 barmoq: 3 barmoqqa o'tishda oraliqdagi 2 barmoq bosilib ketmasin
SWAP_BUTTONS = True            # True: 2 barmoq = CHAP tugma, 3 barmoq = O'NG tugma; False: aksincha
DOUBLE_FRAMES = 6
DRAG_FRAMES = 3
UP_ON = 150                    # barmoq burchagi (gradus) shundan katta bo'lsa 'ochiq'
UP_OFF = 125                   # shundan kichik bo'lsa 'yopiq' (oraliqda oldingi holat saqlanadi)
LM_SMOOTH = 0.6                # ishora tanish uchun nuqtalarni silliqlash (1 = silliqlashsiz)
CLICK_COOLDOWN = 0.4

CAL_FRAMES = 40                # kalibrovkada har burchakda shuncha kadr ushlab turish kerak

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# har barmoq bo'g'imlari: (asosi MCP, PIP, DIP, uchi)
FINGER = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

DEBUG = {"pinch": 0.0, "up": 0}


# ---------------------------------------------------------------- kamera
class LatestFrame:
    """Kamerani alohida oqimda o'qiydi, doim eng yangi kadrni beradi (kechikishni yo'qotadi).

    Kamerani ochishda turli usul (MSMF/DSHOW/standart) va o'lchamlar (1280x720, 640x480, kameraning
    o'z o'lchami) sinab ko'riladi: ba'zi kameralar ayrim rejimda buzuq kadr qaytaradi."""

    BACKENDS = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)

    @staticmethod
    def _good_frame(cap):
        """Kameradan bir nechta kadr o'qib, buzuq emasligini tekshiradi. Yaroqli kadrni qaytaradi yoki None."""
        for _ in range(20):
            try:
                ok, f = cap.read()
            except cv2.error:
                return None
            if ok and f is not None and f.ndim == 3 and f.shape[0] > 0 and f.shape[1] > 0:
                try:
                    f = f.copy()               # buzuq stride bo'lsa shu yerda xato beradi
                except cv2.error:
                    return None
                return f
            time.sleep(0.05)
        return None

    @classmethod
    def _open(cls, index):
        sizes = [(FRAME_W, FRAME_H), (640, 480), None]
        for backend in cls.BACKENDS:
            for size in sizes:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    break                      # bu usulda kamera umuman ochilmadi
                if size:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # avtomatik ekspozitsiya (aks holda kamera qorong'i qolib ketishi mumkin)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
                f = cls._good_frame(cap)
                if f is not None:
                    return cap, f
                cap.release()
        return None, None

    def __init__(self, index):
        self.cap, first = self._open(index)
        if self.cap is None:                 # boshqa kamera raqamlarini ham sinab ko'ramiz
            for i in range(4):
                if i != index:
                    self.cap, first = self._open(i)
                    if self.cap is not None:
                        print(f"Kamera {i}-raqam bilan topildi.")
                        break
        if self.cap is None:
            raise SystemExit(
                "Kamera ochilmadi.\n"
                " - Kamera ulanganini tekshiring va boshqa dastur (Zoom, Teams...) uni ishlatmayotganiga ishonch hosil qiling.\n"
                " - Windows: Sozlamalar > Maxfiylik > Kamera > 'Desktop ilovalarga ruxsat berish' yoqilgan bo'lsin.")
        self.size = (first.shape[1], first.shape[0])
        self.frame = None
        self.stop = False
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        errors = 0
        while not self.stop:
            try:
                ok, f = self.cap.read()
                if ok and f is not None:
                    f = f.copy()
                    with self.lock:
                        self.frame = f
                    errors = 0
                    continue
            except cv2.error:
                pass
            errors += 1                        # buzuq kadr: dastur yiqilmaydi, biroz kutib davom etadi
            time.sleep(0.01)
            if errors > 300:
                print("Kamera kadr bermayapti (uzilib qoldimi?).")
                self.stop = True

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


def click_button(gesture):
    """Ishora -> sichqoncha tugmasi. SWAP_BUTTONS=True: 2 barmoq = chap, 3 barmoq = o'ng."""
    if gesture == "two":
        return "left" if SWAP_BUTTONS else "right"
    return "right" if SWAP_BUTTONS else "left"


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
class Pt:
    """Tekislashtirilgan (silliqlangan) nuqta."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (getattr(a, "z", 0.0) - getattr(b, "z", 0.0)) ** 2) ** 0.5


def angle(a, b, c):
    """b nuqtadagi burchak (gradus): 180 = to'g'ri, kichik = bukilgan."""
    v1 = (a.x - b.x, a.y - b.y, getattr(a, "z", 0.0) - getattr(b, "z", 0.0))
    v2 = (c.x - b.x, c.y - b.y, getattr(c, "z", 0.0) - getattr(b, "z", 0.0))
    n1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** 0.5
    n2 = (v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) ** 0.5
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def finger_score(lm, name):
    """Barmoq to'g'riligi: bo'g'imlardagi burchaklar o'rtachasi (180 = to'liq cho'zilgan)."""
    mcp, pip, dip, tip = FINGER[name]
    return 0.6 * angle(lm[mcp], lm[pip], lm[dip]) + 0.4 * angle(lm[pip], lm[dip], lm[tip])


def finger_up(lm, name, was_up=None):
    """Gisterezis bilan: oraliq burchakda oldingi holat saqlanadi (ishora "titrab" qolmaydi)."""
    s = finger_score(lm, name)
    if s >= UP_ON:
        return True
    if s <= UP_OFF:
        return False
    return bool(was_up) if was_up is not None else s >= (UP_ON + UP_OFF) / 2


def thumb_out(lm):
    size = max(dist(lm[0], lm[9]), 1e-6)
    return dist(lm[4], lm[5]) / size > THUMB_OUT


def classify(lm, pinching, fs=None):
    """-> (gesture, pinching). gesture: drag / fist / scroll / left / double / right / move / none
    fs - barmoqlarning oldingi holati (gisterezis uchun, o'zgartiriladi)."""
    fs = fs if fs is not None else {}
    up = {n: finger_up(lm, n, fs.get(n)) for n in FINGER}
    fs.update(up)
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
        return "three", False
    if up["index"] and up["pinky"] and not up["middle"] and not up["ring"]:
        return "double", False
    if count == 2 and up["index"] and up["middle"]:
        return "two", False
    if up["index"] and count == 1:
        return "move", False
    return "none", False


def hand_size(lm, aspect):
    """Qo'lning kadrdagi o'lchami (kadr balandligi birligida): kaft uzunligi va kaft eni bo'yicha."""
    palm = (((lm[9].x - lm[0].x) * aspect) ** 2 + (lm[9].y - lm[0].y) ** 2) ** 0.5
    knuckle = (((lm[5].x - lm[17].x) * aspect) ** 2 + (lm[5].y - lm[17].y) ** 2) ** 0.5
    return max(palm, knuckle * 1.25)   # qo'l burilganda biri kichrayadi - kattasini olamiz


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
        self.aspect = 16 / 9               # kadr eni / balandligi (asosiy sikl yangilab turadi)
        self.size_now = None               # qo'lning hozirgi o'lchami
        self.size_ema = None
        self.reset()

    # ----- holatni tozalash
    def reset(self):
        self.release()
        self.pinching = False
        self.fs = {}
        self.smooth = None
        self.candidate, self.streak, self.confirmed = "none", 0, "none"
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.lost = 0
        self.fist_start, self.fist_progress = None, 0.0
        self.enable_start, self.enable_progress = None, 0.0
        self.rel_prev = None
        self.size_ema = None
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
    def _smooth_lm(self, src):
        """Ishora tanish uchun nuqtalarni vaqt bo'yicha silliqlaydi (titrashdan kelib chiqadigan xatolarni kamaytiradi)."""
        pts = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in src]
        if self.smooth is None:
            self.smooth = pts
        else:
            a = LM_SMOOTH
            self.smooth = [(a * x + (1 - a) * ox, a * y + (1 - a) * oy, a * z + (1 - a) * oz)
                           for (x, y, z), (ox, oy, oz) in zip(pts, self.smooth)]
        return [Pt(*t) for t in self.smooth]

    def update(self, lm, now=None, world=None):
        """lm - kadrdagi nuqtalar (kursor uchun), world - haqiqiy 3D nuqtalar (ishora tanish uchun, bo'lsa)."""
        now = time.time() if now is None else now
        self.lost = 0
        self.last_seen = now
        self.size_now = hand_size(lm, self.aspect)
        self.size_ema = self.size_now if self.size_ema is None else \
            self.size_ema + (self.size_now - self.size_ema) * SIZE_SMOOTH
        gesture, self.pinching = classify(self._smooth_lm(world or lm), self.pinching, self.fs)

        if not self.enabled:
            self._enable_check(lm, gesture, now)
            return "off"

        if gesture == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = gesture, 1

        need = {"move": 1, "three": THREE_FRAMES, "two": TWO_FRAMES, "double": DOUBLE_FRAMES,
                "drag": DRAG_FRAMES, "scroll": SCROLL_FRAMES}.get(self.candidate, 2)
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
            elif self.confirmed in ("two", "three", "double") and prev != self.confirmed \
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
            self.io["click"](click_button(kind))
        self.last_click = now

    # ----- kursor
    @property
    def scale(self):
        """Uzoqdan boshqarish kuchaytirishi: qo'l kichik ko'rinsa (uzoq) > 1, yaqin bo'lsa 1."""
        if not FAR_MODE or not self.size_ema:
            return 1.0
        ref = CALIB_PALM or REF_PALM
        low = 0.5 if CALIB_PALM else 1.0
        return min(max(ref / self.size_ema, low), max(FAR_MAX, 1.0))

    def _target(self, lm):
        if CALIB:
            x0, x1, y0, y1 = CALIB
            nx = (lm[8].x - x0) / max(x1 - x0, 1e-6)
            ny = (lm[8].y - y0) / max(y1 - y0, 1e-6)
        else:
            nx = (lm[8].x - MARGIN) / (1 - 2 * MARGIN)
            ny = (lm[8].y - MARGIN) / (1 - 2 * MARGIN)
        s = self.scale                 # markaz atrofida harakatni kuchaytiramiz
        nx, ny = 0.5 + (nx - 0.5) * s, 0.5 + (ny - 0.5) * s
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
        self.sizes, self.all_sizes = [], []

    @property
    def progress(self):
        return min(len(self.samples) / CAL_FRAMES, 1.0)

    @property
    def corner(self):
        return self.CORNERS[min(self.idx, len(self.CORNERS) - 1)]

    def feed(self, lm, gesture, size=None):
        """Har kadrda chaqiriladi. Tugaganda True qaytaradi (natija self.result da)."""
        if lm is None or gesture != "move":
            self.samples, self.sizes = [], []
            return False
        pos = (lm[8].x, lm[8].y)
        # oldingi burchakdan qo'l siljib ketmaguncha yangi burchak uchun o'lchamaymiz
        if self.points and not self.samples and \
                ((pos[0] - self.points[-1][0]) ** 2 + (pos[1] - self.points[-1][1]) ** 2) ** 0.5 < 0.12:
            return False
        # barmoq joyidan siljisa, o'lchashni qaytadan boshlaymiz (barqaror turishi kerak)
        if self.samples and ((pos[0] - self.samples[0][0]) ** 2 + (pos[1] - self.samples[0][1]) ** 2) ** 0.5 > 0.06:
            self.samples, self.sizes = [], []
        self.samples.append(pos)
        if size:
            self.sizes.append(size)
        if len(self.samples) < CAL_FRAMES:
            return False
        xs = sorted(s[0] for s in self.samples)
        ys = sorted(s[1] for s in self.samples)
        self.points.append((xs[len(xs) // 2], ys[len(ys) // 2]))     # mediana
        self.all_sizes += self.sizes
        self.samples, self.sizes = [], []
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
        ss = sorted(self.all_sizes)
        self.result = (x0, x1, y0, y1) + ((ss[len(ss) // 2],) if ss else ())   # + qo'l o'lchami (masofa)
        return True


# ---------------------------------------------------------------- asosiy sikl
FINGER_STYLE = {   # (rang BGR, nomi, uchi nuqtasi, hamma bo'g'imlari)
    "thumb": ((0, 165, 255), "Bosh", 4, (1, 2, 3, 4)),
    "index": ((0, 220, 0), "Korsatkich", 8, (5, 6, 7, 8)),
    "middle": ((255, 200, 0), "Orta", 12, (9, 10, 11, 12)),
    "ring": ((255, 0, 200), "Nomsiz", 16, (13, 14, 15, 16)),
    "pinky": ((0, 220, 255), "Jimjiloq", 20, (17, 18, 19, 20)),
}
PALM_LINKS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


def draw_hand(frame, lm, ctrl, names=True):
    """Qo'lning hamma 21 nuqtasini, barmoqlarni (rang bilan), ularning ochiq/yopiq holatini va qo'l ramkasini chizadi."""
    h, w = frame.shape[:2]
    k = w / 640.0
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    # qo'l ramkasi
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = int(14 * k)
    cv2.rectangle(frame, (min(xs) - pad, min(ys) - pad), (max(xs) + pad, max(ys) + pad), (255, 255, 255), 1)
    # kaft
    for a, b in PALM_LINKS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), max(1, int(2 * k)))
    thumb_open = thumb_out(lm)
    for order, (name, (color, label, tip, joints)) in enumerate(FINGER_STYLE.items()):
        is_up = thumb_open if name == "thumb" else bool(ctrl.fs.get(name))
        for a, b in zip(joints, joints[1:]):
            cv2.line(frame, pts[a], pts[b], color, max(2, int(3 * k)))
        for j in joints:                                  # har bir bo'g'im
            cv2.circle(frame, pts[j], max(3, int(4 * k)), (255, 255, 255), -1)
            cv2.circle(frame, pts[j], max(3, int(4 * k)), color, 1)
        # uchi: yashil = ochiq, qizil = yopiq
        cv2.circle(frame, pts[tip], max(5, int(8 * k)), (0, 220, 0) if is_up else (0, 0, 255), -1)
        if names:
            text = label + (" +" if is_up else " -")
            # nomlar bir-birining ustiga tushmasligi uchun har barmoqniki turli balandlikda
            cv2.putText(frame, text, (pts[tip][0] + int(10 * k), pts[tip][1] - int((10 + 15 * (order % 3)) * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, color, max(1, int(k)))
    cv2.circle(frame, pts[0], max(4, int(6 * k)), (255, 255, 255), -1)   # bilak
    # kursorni boshqaradigan nuqta (ko'rsatkich uchi)
    cv2.drawMarker(frame, pts[8], (255, 255, 255), cv2.MARKER_CROSS, int(26 * k), max(1, int(k)))


LABELS = {"move": "1 barmoq: HARAKAT", "two": "2 barmoq: CHAP TUGMA", "three": "3 barmoq: O'NG TUGMA",
          "scroll": "OCHIQ KAFT: SCROLL / yon tomonga siltash",
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
    atexit.register(ctrl.release)        # xato bilan yopilsa ham sichqoncha tugmasi bosilgan holda qolmasin
    hotkeys = Hotkeys()
    state = {"hidden": args.yashirin, "mini": False, "calib": None, "win": None, "size": None, "names": True}

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
            cv2.resizeWindow(WINDOW, want_w, int(want_w * cam.size[1] / cam.size[0]))
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
        elif key == ord("l"):
            events.append("names")
        for ev in events:
            if ev == "quit":
                running = False
            elif ev == "names":
                state["names"] = not state["names"]
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
        ctrl.aspect = w / h
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.time() - t0) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        hands = result.hand_landmarks
        worlds = getattr(result, "hand_world_landmarks", None)
        if not hands:                                    # qo'l yo'qoldi -> zaxira usuli
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            boosted = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
            res2 = rescue.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=boosted))
            hands, worlds = res2.hand_landmarks, getattr(res2, "hand_world_landmarks", None)

        cal = state["calib"]
        if hands:
            lm = hands[0]
            ctrl.update(lm, world=worlds[0] if worlds else None)
            if cal and cal.feed(lm, ctrl.confirmed, ctrl.size_now):
                settings.calib = cal.result
                settings.save()
                settings.apply(globals())
                state["calib"], ctrl.paused = None, False
            cal = state["calib"]                   # tugagan bo'lsa chizishda ishlatilmasin
            if not state["hidden"]:
                draw_hand(frame, lm, ctrl, state["names"])
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
        if status in ("two", "three"):
            label = f"{'2' if status == 'two' else '3'} barmoq: " + \
                    ("CHAP" if click_button(status) == "left" else "O'NG") + " TUGMA"
        else:
            label = LABELS[status]
        cv2.putText(frame, label, (10, int(30 * k)), font, 0.65 * k, color, max(2, int(2 * k)))
        if FAR_MODE and ctrl.scale > 1.05:
            cv2.putText(frame, f"Uzoqdan boshqarish: harakat x{ctrl.scale:.1f}", (10, h - int(34 * k)), font,
                        0.5 * k, (0, 255, 0), max(1, int(k)))

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
                           f"q-chiqish h-yashirish m-kichik s-sozlamalar k-kalibrovka l-nomlar (Ctrl+Alt+harf)",
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
