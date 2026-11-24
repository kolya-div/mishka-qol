"""Qo'l bilan mishkani boshqarish (kamera + MediaPipe).

Imo-ishoralar:
  1 barmoq (ko'rsatkich)                 -> kursor qo'l bilan harakatlanadi
  2 barmoq (ko'rsatkich + o'rta)         -> CHAP tugma   (sozlamalarda almashtirish mumkin)
  3 barmoq (ko'rsatkich, o'rta, nomsiz)  -> O'NG tugma
  ko'rsatkich + jimjiloq ("rok")         -> IKKI MARTA BOSISH (bosh barmoq yopiq)
  bosh + ko'rsatkich tekkizib ushlash    -> SUDRASH (qo'yib yuborsangiz tashlanadi)
  ochiq kaft + qo'lni tepaga/pastga      -> SCROLL
  ochiq kaft + qo'lni tez o'ngga/chapga  -> brauzerda OLDINGA / ORQAGA
  telefon (bosh + jimjiloq ochiq)        -> tepaga/pastga = OVOZ, o'ngga/chapga siltash = KEYINGI/OLDINGI TREK
  "seni sevaman" (bosh+ko'rsatkich+jimjiloq) + o'ngga/chapga siltash -> ISH STOLINI almashtirish
  ikkala qo'l ochiq, uzoqlashtirish/yaqinlashtirish -> KATTALASHTIRISH (Ctrl + g'ildirak)
  mushtni 1 soniya ushlash               -> dasturni o'chirish
  qo'l 5 soniya ko'rinmasa               -> dastur o'zi o'chadi
  kaftni to'liq ochib 2 soniya ushlash   -> dasturni yoqish
Tugmalar (kamera oynasida): q - chiqish, h - yashirish, m - kichik/katta, s - sozlamalar, k - kalibrovka,
  t - ishoralarni o'rgatish, l - nomlar.
Hamma joyda ishlaydigan: Ctrl+Alt+Q / H / M / S / K / T, Ctrl+Alt+Probel - yoqish/o'chirish.
"""
import argparse
import atexit
import ctypes
import math
import os
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import sozlamalar

MODEL_PATH = os.path.join(sozlamalar.APP_DIR, "hand_landmarker.task")
if not os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(sozlamalar.RES_DIR, "hand_landmarker.task")):
    MODEL_PATH = os.path.join(sozlamalar.RES_DIR, "hand_landmarker.task")   # .exe ichiga solingan model
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
TRAIN_FRAMES = 45              # ishoralarni o'rgatishda har bosqichda shuncha kadr

LOW_LIGHT = True               # qorong'i joyda kadrni avtomatik yoritish
LOW_LIGHT_LEVEL = 70           # kadr o'rtacha yorqinligi (0-255) shundan kam bo'lsa yoritiladi
ROI_ENABLED = True             # uzoqdagi (kichik) qo'lni kattalashtirib aniqlash
ROI_MAX_HAND = 0.5             # qo'l balandligi kadrning shu qismidan kichik bo'lsa kesib kattalashtiriladi
ROI_PAD = 2.2                  # kesim o'lchami = qo'l o'lchami x shu son
ZOOM_ENABLED = True            # ikki qo'l bilan kattalashtirish
ZOOM_GAIN = 6000
MEDIA_ENABLED = True           # telefon va 'seni sevaman' ishoralari
VOLUME_STEP = 0.03             # qo'l shuncha (kadr balandligi ulushi) siljisa ovoz 1 qadam o'zgaradi
SHAKA_FRAMES = 3
LOVE_FRAMES = 4
SOUND_ENABLED = True
OVERLAY_ENABLED = True
TRAY_ENABLED = True
MONITOR = 1                    # 0 = hamma ekranlar, 1 = asosiy, 2... = boshqalari

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


def _key(vk, up=False, ext=False):
    ctypes.windll.user32.keybd_event(vk, 0, (2 if up else 0) | (1 if ext else 0), 0)


def press_keys(*keys):
    """keys: (vk, kengaytirilgan) juftliklari. Hammasi bosiladi, keyin teskari tartibda qo'yib yuboriladi."""
    for vk, ext in keys:
        _key(vk, False, ext)
    for vk, ext in reversed(keys):
        _key(vk, True, ext)


def media(action):
    """Keyingi / oldingi trek."""
    press_keys((0xB0 if action == "next" else 0xB1, False))


def volume(direction, count=1):
    """Ovozni oshirish (direction > 0) yoki kamaytirish."""
    for _ in range(max(1, min(int(count), 10))):
        press_keys((0xAF if direction > 0 else 0xAE, False))


def desktop(direction):
    """Virtual ish stolini almashtirish: Ctrl + Win + o'ng/chap strelka."""
    press_keys((0x11, False), (0x5B, False), (0x27 if direction == "right" else 0x25, True))


def zoom_wheel(amount):
    """Ctrl bosilgan holda g'ildirak: ko'p dasturda masshtabni o'zgartiradi."""
    _key(0x11)
    wheel(amount)
    _key(0x11, up=True)


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", _RECT), ("rcWork", _RECT), ("dwFlags", ctypes.c_ulong)]


def list_monitors():
    """Ekranlar ro'yxati [(x, y, eni, balandligi)], asosiy ekran birinchi."""
    mons = []
    proc_t = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_RECT), ctypes.c_void_p)

    def cb(hmon, hdc, rect, data):
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if ctypes.windll.user32.GetMonitorInfoW(ctypes.c_void_p(hmon), ctypes.byref(info)):
            r = info.rcMonitor
            mons.append((r.left, r.top, r.right - r.left, r.bottom - r.top, bool(info.dwFlags & 1)))
        return 1

    try:
        ctypes.windll.user32.EnumDisplayMonitors(None, None, proc_t(cb), 0)
    except Exception:
        pass
    mons.sort(key=lambda m: not m[4])
    return [m[:4] for m in mons]


def screen_geometry(n):
    """n = 0: hamma ekranlar (virtual ish stoli); n >= 1: n-chi ekran (1 = asosiy). -> (x, y, eni, balandligi)"""
    u = ctypes.windll.user32
    if n <= 0:
        return u.GetSystemMetrics(76), u.GetSystemMetrics(77), u.GetSystemMetrics(78), u.GetSystemMetrics(79)
    mons = list_monitors()
    if mons:
        return mons[min(int(n), len(mons)) - 1]
    return 0, 0, u.GetSystemMetrics(0), u.GetSystemMetrics(1)


def dpi_aware():
    """Kursor koordinatalari haqiqiy piksellarda bo'lishi uchun (Windows masshtabi 125%/150% bo'lsa ham)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def swipe(direction):
    """Qo'l o'ngga siltandi -> oldinga (Alt+O'ng), chapga -> orqaga (Alt+Chap)."""
    pyautogui.hotkey("alt", "right" if direction == "right" else "left")


class Hotkeys:
    """Hamma joyda ishlaydigan Ctrl+Alt+harf tugmalari (kamera oynasi yashirin bo'lsa ham)."""
    KEYS = {"quit": 0x51, "hide": 0x48, "mini": 0x4D, "settings": 0x53, "calib": 0x4B,
            "train": 0x54, "toggle": 0x20}

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
    """-> (gesture, pinching). gesture: drag / fist / scroll / three / shaka / love / double / two / move / none
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
    to = thumb_out(lm)
    if to and up["pinky"] and not up["index"] and not up["middle"] and not up["ring"]:
        return "shaka", False             # telefon: bosh barmoq + jimjiloq
    if to and up["index"] and up["pinky"] and not up["middle"] and not up["ring"]:
        return "love", False              # "seni sevaman": bosh + ko'rsatkich + jimjiloq
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

    def __init__(self, screen_w, screen_h, cursor_pos, io=None, origin=(0, 0)):
        self.sw, self.sh = screen_w, screen_h
        self.ox, self.oy = origin             # tanlangan ekranning chap-yuqori burchagi (ko'p ekranli kompyuterda)
        self.cur_x, self.cur_y = cursor_pos[0] - self.ox, cursor_pos[1] - self.oy
        self.io = io or {"cursor": set_cursor, "click": click, "wheel": wheel, "down": mouse_down,
                         "up": mouse_up, "double": double_click, "swipe": swipe, "media": media,
                         "volume": volume, "desktop": desktop, "zoom": zoom_wheel}
        self.on_event = None              # ovoz va boshqalar uchun: on_event("click") ...
        self.pl = None                    # oxirgi kadrdagi silliqlangan nuqtalar (ishora tanish uchun ishlatilgan)
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
        self.vol_y = None
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
        self.vol_y = None
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

    def _ev(self, name):
        if self.on_event:
            try:
                self.on_event(name)
            except Exception:
                pass

    def _put(self, x, y):
        """Kursorni tanlangan ekran ichidagi (x, y) nuqtaga qo'yadi."""
        self.io["cursor"](x + self.ox, y + self.oy)

    def set_screen(self, ox, oy, w, h):
        """Boshqariladigan ekranni almashtiradi (sozlamalarda 'Ekran' o'zgartirilganda)."""
        if (ox, oy, w, h) == (self.ox, self.oy, self.sw, self.sh):
            return
        self.ox, self.oy, self.sw, self.sh = ox, oy, w, h
        self.cur_x = min(max(self.cur_x, 0), w - 1)
        self.cur_y = min(max(self.cur_y, 0), h - 1)
        self.filter.prev = None
        self.rel_prev = None

    def hold(self, now):
        """Qo'l ko'rinib turibdi, lekin boshqa vazifa (masalan, kattalashtirish) bilan band."""
        self.lost = 0
        self.last_seen = now

    def disable(self):
        was = self.enabled
        self.enabled = False
        self.reset()
        if was:
            self._ev("disable")

    def enable(self, now, silent=False):
        was = self.enabled
        self.enabled = True
        self.reset()
        self.last_seen = now
        if not was and not silent:
            self._ev("enable")

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
        self.pl = self._smooth_lm(world or lm)
        gesture, self.pinching = classify(self.pl, self.pinching, self.fs)

        if not self.enabled:
            self._enable_check(lm, gesture, now)
            return "off"

        if gesture == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = gesture, 1

        need = {"move": 1, "three": THREE_FRAMES, "two": TWO_FRAMES, "double": DOUBLE_FRAMES,
                "drag": DRAG_FRAMES, "scroll": SCROLL_FRAMES, "shaka": SHAKA_FRAMES,
                "love": LOVE_FRAMES}.get(self.candidate, 2)
        if self.streak >= need:
            prev, self.confirmed = self.confirmed, self.candidate
            if self.paused:                   # kalibrovka/o'rgatish paytida mush dasturni o'chirmaydi
                self.fist_start, self.fist_progress = None, 0.0
            else:
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
                if SWIPE_ENABLED:
                    d = self._swipe_dir(lm[9].x, lm[9].y, now)
                    if d:
                        self.io["swipe"](d)
                        self._ev("swipe")
            elif self.confirmed == "shaka" and MEDIA_ENABLED:
                self._shaka(lm, now, prev != "shaka")
            elif self.confirmed == "love" and MEDIA_ENABLED:
                self._love(lm, now, prev != "love")
        if self.confirmed != "scroll":
            self.scroll_y = None
        if self.confirmed != "shaka":
            self.vol_y = None
        if self.confirmed not in ("scroll", "shaka", "love"):
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
            self._put(self.cur_x, self.cur_y)
        if kind == "double":
            self.io["double"]()
            self._ev("double")
        else:
            self.io["click"](click_button(kind))
            self._ev("click")
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
        self._put(self.cur_x, self.cur_y)

    def _move(self, lm, now, first):
        tx, ty = self._target(lm)
        self._pointer(tx, ty, now, first)
        self.history.append((self.cur_x, self.cur_y))

    def _drag(self, lm, now, first):
        tx, ty = self._target(lm)
        if first:
            if self.history:              # barmoqlar tekkunga qadar bo'lgan joyni bosamiz
                self.cur_x, self.cur_y = self.history[0]
                self._put(self.cur_x, self.cur_y)
            self.drag_off = (self.cur_x - tx, self.cur_y - ty)
            self._pointer(tx + self.drag_off[0], ty + self.drag_off[1], now, True)
            self.io["down"]()
            self.dragging = True
            self._ev("drag")
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

    def _swipe_dir(self, x, y, now):
        """Qo'l tez yon tomonga siltandimi? -> "right" / "left" / None."""
        h = self.swipe_hist
        h.append((now, x, y))
        while h and now - h[0][0] > SWIPE_WINDOW:
            h.popleft()
        if len(h) >= 3 and now - self.last_swipe > SWIPE_COOLDOWN:
            dx, dy = h[-1][1] - h[0][1], abs(h[-1][2] - h[0][2])
            if abs(dx) > SWIPE_DIST and dy < abs(dx) * 0.6:
                self.last_swipe = now
                self.block_scroll_until = now + 0.6
                self.scroll_acc = 0.0
                h.clear()
                return "right" if dx > 0 else "left"
        return None

    def _shaka(self, lm, now, first):
        """Telefon ishorasi: tepaga/pastga = ovoz, yon tomonga siltash = keyingi/oldingi trek."""
        if first:
            self.swipe_hist.clear()
            self.vol_y = None
        d = self._swipe_dir(lm[9].x, lm[9].y, now)
        if d:
            self.io["media"]("next" if d == "right" else "prev")
            self._ev("media")
            self.vol_y = None
            return
        y = lm[9].y
        if self.vol_y is None or now < self.block_scroll_until:
            self.vol_y = y
            return
        steps = int((self.vol_y - y) / VOLUME_STEP)        # qo'l ko'tarilsa musbat -> ovoz oshadi
        if steps:
            self.io["volume"](1 if steps > 0 else -1, abs(steps))
            self.vol_y -= steps * VOLUME_STEP

    def _love(self, lm, now, first):
        """"Seni sevaman" ishorasi + yon tomonga siltash = virtual ish stolini almashtirish."""
        if first:
            self.swipe_hist.clear()
        d = self._swipe_dir(lm[9].x, lm[9].y, now)
        if d:
            self.io["desktop"](d)
            self._ev("desktop")


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


# ---------------------------------------------------------------- ikki qo'l bilan kattalashtirish
class Zoomer:
    """Ikkala qo'l ochiq bo'lib uzoqlashsa/yaqinlashsa Ctrl + g'ildirak yuboradi (masshtabni o'zgartirish)."""
    STABLE = 4          # ikkala qo'l shuncha kadr ochiq turgandan keyin boshlanadi
    GRACE = 4           # qo'llardan biri shuncha kadr ko'rinmasa ham davom etadi

    def __init__(self, io):
        self.io = io
        self.active = False
        self.reset()

    def reset(self):
        self.active, self.prev, self.acc, self.stable, self.miss = False, None, 0.0, 0, 0

    @staticmethod
    def hand_open(lm):
        mid = (UP_ON + UP_OFF) / 2
        return all(finger_score(lm, n) >= mid for n in FINGER)

    def update(self, pair, aspect):
        """pair: ikkita qo'l nuqtalari yoki None. Hozir kattalashtirish rejimidami (True/False)?"""
        if pair is None or not (self.hand_open(pair[0]) and self.hand_open(pair[1])):
            self.miss += 1
            self.stable = 0
            if self.miss > self.GRACE:
                self.reset()
            return self.active
        self.miss = 0
        a, b = pair[0][9], pair[1][9]
        d = (((a.x - b.x) * aspect) ** 2 + (a.y - b.y) ** 2) ** 0.5
        self.stable += 1
        if self.stable < self.STABLE:
            self.prev = d
            return self.active
        if not self.active:
            self.active, self.prev, self.acc = True, d, 0.0
            return True
        self.acc += (d - self.prev) * ZOOM_GAIN
        self.prev = d
        amount = int(self.acc / WHEEL_NOTCH) * WHEEL_NOTCH
        if amount:
            self.io["zoom"](amount)
            self.acc -= amount
        return True


# ---------------------------------------------------------------- ishoralarni o'rgatish
class Trainer:
    """Ochiq kaft va mushtni ko'rsatib, barmoq ochiq/yopiq chegaralarini sizning qo'lingizga moslaydi."""

    def __init__(self):
        self.phase, self.samples, self.pause_until = 0, [], None
        self.open_min = None
        self.result, self.error = None, ""

    @property
    def progress(self):
        return min(len(self.samples) / TRAIN_FRAMES, 1.0)

    def prompt(self):
        if self.pause_until is not None:
            return "Yaxshi! Endi tayyorlaning..."
        if self.phase == 0:
            return "O'RGATISH 1/2: kaftni TO'LIQ oching (barmoqlar tik) va ushlab turing"
        return "O'RGATISH 2/2: MUSHT qiling (hamma barmoqni mahkam yuming) va ushlab turing"

    def feed(self, lm, now):
        """lm: ishora tanishda ishlatilgan nuqtalar yoki None. Tugaganda True qaytaradi (natija self.result da)."""
        if lm is None:
            self.samples = []
            return False
        if self.pause_until is not None:
            if now < self.pause_until:
                return False
            self.pause_until = None
        scores = [finger_score(lm, n) for n in FINGER]
        self.samples.append(min(scores) if self.phase == 0 else max(scores))
        if len(self.samples) < TRAIN_FRAMES:
            return False
        med = sorted(self.samples)[len(self.samples) // 2]
        self.samples = []
        if self.phase == 0:
            self.open_min, self.phase, self.pause_until = med, 1, now + 1.5
            return False
        d = self.open_min - med
        if d < 40:
            self.error = "Ochiq va yopiq qo'l deyarli farq qilmadi, qayta urinib ko'ring."
            self.phase, self.open_min = 0, None
            return False
        on = min(med + 0.7 * d, self.open_min - 5)
        off = max(min(med + 0.4 * d, on - 10), med + 5)
        self.result, self.error = (round(on, 1), round(off, 1)), ""
        return True


# ---------------------------------------------------------------- qo'lni topish (kesish, yoritish, zaxira)
def crop_window(box, w, h, pad):
    """Qo'l atrofidagi kvadrat oyna (x, y, tomon) piksellarda; kadr ichida qoladi."""
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    side = max((box[2] - box[0]) * w, (box[3] - box[1]) * h) * pad
    side = int(min(max(side, 160), h, w))
    x = int(min(max(cx - side / 2, 0), w - side))
    y = int(min(max(cy - side / 2, 0), h - side))
    return x, y, side


def remap(lms, x, y, side, w, h):
    """Kesilgan oyna ichidagi nuqtalarni butun kadr koordinatalariga qaytaradi."""
    return [Pt((x + p.x * side) / w, (y + p.y * side) / h, p.z) for p in lms]


class Detector:
    """Qo'lni topadi: uzoqdagi kichik qo'lni kesib kattalashtirib, qorong'i kadrni yoritib, yo'qolsa zaxira usul bilan."""

    def __init__(self, full, crop, rescue, clahe=None):
        self.full, self.crop, self.rescue = full, crop, rescue
        self.clahe = clahe or cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.idx, self.box, self.n_last, self.last_wrist = 0, None, 0, None
        self.mode, self.gamma, self.luts = "full", 1.0, {}
        self.t0, self.last_ts = time.time(), -1

    def _brighten(self, frame):
        self.gamma = 1.0
        if not LOW_LIGHT:
            return frame
        mean = float(frame[::6, ::6].mean())
        if mean >= LOW_LIGHT_LEVEL:
            return frame
        g = max(0.35, math.log(110 / 255) / math.log(max(mean, 1.0) / 255))
        key = round(g, 2)
        if key not in self.luts:
            self.luts[key] = np.array([((i / 255.0) ** key) * 255 for i in range(256)], dtype=np.uint8)
        self.gamma = key
        return cv2.LUT(frame, self.luts[key])

    @staticmethod
    def _img(rgb):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def detect(self, frame):
        """frame: BGR kadr. -> (hands, worlds): qo'llar ro'yxati (asosiy qo'l birinchi), worlds yoki None."""
        self.idx += 1
        src = self._brighten(frame)
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        hands, worlds, self.mode = [], None, "full"

        small = self.box is not None and (self.box[3] - self.box[1]) < ROI_MAX_HAND
        if ROI_ENABLED and small and self.n_last == 1 and self.idx % 4:
            x, y, side = crop_window(self.box, w, h, ROI_PAD)
            piece = rgb[y:y + side, x:x + side]
            if side < 384:
                piece = cv2.resize(piece, (384, 384), interpolation=cv2.INTER_LINEAR)
            res = self.crop.detect(self._img(piece))
            if res.hand_landmarks:
                hands = [remap(res.hand_landmarks[0], x, y, side, w, h)]
                worlds = getattr(res, "hand_world_landmarks", None)
                self.mode = "kesilgan"
        if not hands:
            ts = max(int((time.time() - self.t0) * 1000), self.last_ts + 1)
            self.last_ts = ts
            res = self.full.detect_for_video(self._img(rgb), ts)
            hands, worlds = res.hand_landmarks, getattr(res, "hand_world_landmarks", None)
        if not hands:                                    # qo'l yo'qoldi -> zaxira usuli (kontrastni oshirish)
            lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            boosted = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
            res = self.rescue.detect(self._img(boosted))
            hands, worlds = res.hand_landmarks, getattr(res, "hand_world_landmarks", None)
            self.mode = "zaxira"
        return self._finish(hands, worlds)

    def _finish(self, hands, worlds):
        hands = list(hands) if hands else []
        worlds = list(worlds) if worlds else None
        if len(hands) > 1 and self.last_wrist:           # asosiy qo'l: oldingi joyga eng yaqini
            lw = self.last_wrist
            order = sorted(range(len(hands)), key=lambda i: (hands[i][0].x - lw[0]) ** 2 + (hands[i][0].y - lw[1]) ** 2)
            hands = [hands[i] for i in order]
            if worlds and len(worlds) == len(order):
                worlds = [worlds[i] for i in order]
        if hands:
            xs, ys = [p.x for p in hands[0]], [p.y for p in hands[0]]
            self.box = (min(xs), min(ys), max(xs), max(ys))
            self.last_wrist = (hands[0][0].x, hands[0][0].y)
        else:
            self.box = None
        self.n_last = len(hands)
        return hands, worlds


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
          "shaka": "TELEFON: tepaga/pastga = OVOZ, yon tomonga = TREK",
          "love": "SENI SEVAMAN: yon tomonga siltash = ISH STOLI",
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

    dpi_aware()
    settings = sozlamalar.Settings()
    settings.apply(globals())
    ensure_model()
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,                                     # ikkinchi qo'l kattalashtirish ishorasi uchun
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    ))
    image_opts = dict(base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                      running_mode=vision.RunningMode.IMAGE, num_hands=1)
    crop_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, **image_opts))
    rescue = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        min_hand_detection_confidence=0.25, min_hand_presence_confidence=0.25, **image_opts))
    detector = Detector(landmarker, crop_lm, rescue)

    cam = LatestFrame(CAM_INDEX)
    try:
        geom = screen_geometry(MONITOR)
    except Exception:
        geom = (0, 0) + tuple(pyautogui.size())
    ctrl = Controller(geom[2], geom[3], pyautogui.position(), origin=geom[:2])
    zoomer = Zoomer(ctrl.io)
    sound = sozlamalar.Sound()
    ctrl.on_event = sound.play
    atexit.register(ctrl.release)        # xato bilan yopilsa ham sichqoncha tugmasi bosilgan holda qolmasin
    hotkeys = Hotkeys()
    state = {"hidden": args.yashirin, "mini": False, "calib": None, "train": None, "win": None, "size": None,
             "names": True, "ui": None}
    pending = []                         # tray menyusidan keladigan buyruqlar

    def on_settings_change():
        settings.apply(globals())
        try:
            ctrl.set_screen(*screen_geometry(MONITOR))
        except Exception:
            pass
        state["ui"] = None               # belgi qayta chizilsin

    def start_calibration():
        ctrl.enable(time.time(), silent=True)
        ctrl.paused, state["train"] = True, None
        state["calib"] = Calibrator()
        state["hidden"] = False

    def clear_calibration():
        settings.calib = None
        settings.save()
        settings.apply(globals())

    def start_training():
        ctrl.enable(time.time(), silent=True)
        ctrl.paused, state["calib"] = True, None
        state["train"] = Trainer()
        state["hidden"] = False

    def clear_training():
        settings.train = None
        settings.save()
        settings.apply(globals())

    panel = None
    if sozlamalar.tk is not None:
        try:
            panel = sozlamalar.Panel(settings, on_settings_change, start_calibration, clear_calibration,
                                     start_training, clear_training)
        except Exception as e:                       # Tk ishlamasa ham asosiy dastur ishlayveradi
            print("Sozlamalar oynasi ochilmadi:", e)
    tray = None
    if TRAY_ENABLED and sozlamalar.HAS_TRAY:
        tray = sozlamalar.Tray(pending.append)
        tray.start()

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

    keymap = {ord("h"): "hide", ord("m"): "mini", ord("s"): "settings", ord("k"): "calib",
              ord("t"): "train", ord("l"): "names"}
    was_zooming = False
    running = True
    while running:
        events = hotkeys.poll()
        while pending:
            events.append(pending.pop(0))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            events.append("quit")
        elif key in keymap:
            events.append(keymap[key])
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
            elif ev == "toggle":
                if ctrl.enabled:
                    ctrl.disable()
                else:
                    ctrl.enable(time.time())
            elif ev == "calib":
                if state["calib"]:
                    state["calib"], ctrl.paused = None, False     # bekor qilish
                else:
                    start_calibration()
            elif ev == "train":
                if state["train"]:
                    state["train"], ctrl.paused = None, False
                else:
                    start_training()
        if not running:
            break
        sound.enabled = SOUND_ENABLED
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
        now = time.time()
        hands, worlds = detector.detect(frame)

        # ---- ikki qo'l bilan kattalashtirish
        pair = hands[:2] if (ZOOM_ENABLED and len(hands) >= 2) else None
        zooming = zoomer.update(pair, ctrl.aspect) if ZOOM_ENABLED else False
        if not ZOOM_ENABLED:
            zoomer.reset()
        if zooming and not was_zooming:
            ctrl.reset()
        was_zooming = zooming

        cal, train = state["calib"], state["train"]
        if hands:
            lm = hands[0]
            if zooming:
                ctrl.hold(now)
            else:
                ctrl.update(lm, now, world=worlds[0] if worlds else None)
            if cal and cal.feed(lm, ctrl.confirmed, ctrl.size_now):
                settings.calib = cal.result
                settings.save()
                settings.apply(globals())
                state["calib"], ctrl.paused = None, False
                sound.play("done")
            if train and train.feed(ctrl.pl, now):
                settings.train = train.result
                settings.save()
                settings.apply(globals())
                state["train"], ctrl.paused = None, False
                sound.play("done")
            cal, train = state["calib"], state["train"]       # tugagan bo'lsa chizishda ishlatilmasin
            if not state["hidden"]:
                draw_hand(frame, lm, ctrl, state["names"])
                if len(hands) > 1:                              # ikkinchi qo'l: oddiy skelet
                    pts2 = [(int(p.x * w), int(p.y * h)) for p in hands[1]]
                    for a_, b_ in CONNECTIONS:
                        cv2.line(frame, pts2[a_], pts2[b_], (200, 120, 255), 2)
        else:
            ctrl.no_hand(now)
            if cal:
                cal.feed(None, "none")
            if train:
                train.feed(None, now)

        # ---- ekrandagi belgi va tray rangi
        ui = "paused" if ctrl.paused else ctrl.status
        if ui != state["ui"]:
            state["ui"] = ui
            if panel:
                panel.set_status(ui, OVERLAY_ENABLED)
            if tray:
                tray.set_color(sozlamalar.STATUS_COLOR.get(ui, sozlamalar.STATUS_COLOR["none"]))

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
        if zooming:
            label = "IKKI QO'L: KATTALASHTIRISH (uzoqlashtiring / yaqinlashtiring)"
        elif status in ("two", "three"):
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
        if train:
            cv2.putText(frame, train.prompt(), (10, int(h * 0.5)), font, 0.6 * k, (0, 255, 0), max(1, int(1.5 * k)))
            bw = int(300 * k)
            cv2.rectangle(frame, (10, int(h * 0.54)), (10 + int(bw * train.progress), int(h * 0.54) + int(14 * k)),
                          (0, 255, 255), -1)
            cv2.rectangle(frame, (10, int(h * 0.54)), (10 + bw, int(h * 0.54) + int(14 * k)), (255, 255, 255), 1)
            if train.error:
                cv2.putText(frame, train.error, (10, int(h * 0.62)), font, 0.55 * k, (0, 0, 255), 2)

        extra = f"  aniqlash: {detector.mode}" + (f" (yoritish x{detector.gamma})" if detector.gamma != 1.0 else "")
        cv2.putText(frame, f"barmoqlar: {DEBUG['up']}  tekkizish: {DEBUG['pinch']:.2f}{extra}   "
                           f"q h m s k t l (Ctrl+Alt+harf, Probel = yoqish/o'chirish)",
                    (10, h - int(12 * k)), font, 0.42 * k, (255, 255, 255), 1)
        cv2.imshow(WINDOW, frame)
        state["shown"] = True

    ctrl.release()                       # sudrash paytida chiqilsa, tugma bosilgan holda qolmasin
    if tray:
        tray.stop()
    cam.release()
    cv2.destroyAllWindows()
    landmarker.close()
    crop_lm.close()
    rescue.close()


def run():
    """Dasturni ishga tushiradi; kutilmagan xato bo'lsa xato.log fayliga yozadi (.exe da konsol yo'q)."""
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        try:
            with open(os.path.join(sozlamalar.APP_DIR, "xato.log"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc() + "\n")
        except OSError:
            pass
        raise


if __name__ == "__main__":
    run()
