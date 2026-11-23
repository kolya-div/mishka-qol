"""Qo'l bilan mishkani boshqarish (kamera + MediaPipe).

Imo-ishoralar:
  1 barmoq (ko'rsatkich)                 -> kursor qo'l bilan harakatlanadi
  2 barmoq (ko'rsatkich + o'rta)         -> O'NG tugma bosiladi
  3 barmoq (ko'rsatkich, o'rta, nomsiz)  -> CHAP tugma bosiladi
  5 barmoq (ochiq kaft) + qo'lni tepaga/pastga siljitish -> SCROLL
Chiqish: oynada 'q' yoki Esc.
"""
import ctypes
import os
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
MARGIN = 0.15            # kadr chetidagi "o'lik zona"
MIN_ALPHA = 0.25         # sekin harakatda silliqlash (titrashni yo'qotadi)
MAX_ALPHA = 1.0          # tez harakatda silliqlash yo'q (qotish yo'q)
FAST_DIST = 0.08         # ekran kengligining shu qismidan katta siljish "tez" hisoblanadi

SCROLL_GAIN = 6000       # qo'l siljishi -> scroll tezligi (katta = tezroq)
SCROLL_SMOOTH = 0.5      # qo'l balandligini silliqlash (1 = silliqlashsiz)
SCROLL_FRAMES = 2
WHEEL_NOTCH = 120

LOST_GRACE = 5           # qo'l shuncha kadr ko'rinmasa ham oldingi holat saqlanadi
LEFT_FRAMES = 4          # chap tugma uchun ishora ushlanishi kerak bo'lgan kadrlar
RIGHT_FRAMES = 5         # o'ng tugma uchun (3 barmoqqa o'tishda adashmaslik uchun ko'proq)
CLICK_COOLDOWN = 0.4     # bosishlar orasidagi minimal vaqt (soniya)

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

DEBUG = {"gap": 0.0, "up": 0}


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


# ---------------------------------------------------------------- imo-ishoralar
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def finger_up(lm, name):
    wrist = lm[0]
    return dist(lm[TIP[name]], wrist) > dist(lm[PIP[name]], wrist) * 1.15


def classify(lm, together):
    """-> (gesture, together). gesture: move / right / left / scroll / none"""
    up = {n: finger_up(lm, n) for n in TIP}
    count = sum(up.values())
    DEBUG["up"] = count
    if all(up.values()):                  # 4 ta barmoq (bosh barmoq e'tiborga olinmaydi) = ochiq kaft
        return "scroll", False
    if up["index"] and up["middle"] and up["ring"] and not up["pinky"]:
        return "left", False
    if count == 2 and up["index"] and up["middle"]:
        return "right", False
    if up["index"] and count == 1:
        return "move", False
    return "none", False


class Controller:
    """Ishorani aniqlab, kursor/bosish/scroll amallarini bajaradi."""

    def __init__(self, screen_w, screen_h, cursor_pos, io=None):
        self.sw, self.sh = screen_w, screen_h
        self.cur_x, self.cur_y = cursor_pos
        self.io = io or {"cursor": set_cursor, "click": click, "wheel": wheel}
        self.history = deque(maxlen=5)    # kursorning oxirgi joylari (bosishdagi siljishni qaytarish uchun)
        self.reset()
        self.last_click = float("-inf")

    def reset(self):
        self.together = False
        self.candidate, self.streak, self.confirmed = "none", 0, "none"
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.anchor_y = None              # 3 barmoq ochiq turganda qo'l boshlang'ich balandligi
        self.lost = 0
        self.history.clear()

    def no_hand(self):
        # qo'l 1-2 kadrga yo'qolsa holatni saqlab turamiz (uzilib-uzilib qolmasligi uchun)
        self.lost += 1
        if self.lost > LOST_GRACE:
            self.reset()
            self.lost = LOST_GRACE + 1

    def update(self, lm, now=None):
        now = time.time() if now is None else now
        self.lost = 0
        gesture, self.together = classify(lm, self.together)
        palm_y = lm[9].y

        if gesture == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = gesture, 1

        need = {"move": 1, "left": LEFT_FRAMES, "right": RIGHT_FRAMES,
                "scroll": SCROLL_FRAMES}.get(self.candidate, 2)
        if self.streak >= need:
            prev, self.confirmed = self.confirmed, self.candidate
            if self.confirmed == "move":
                self._move(lm)
            elif self.confirmed in ("right", "left") and prev != self.confirmed \
                    and now - self.last_click > CLICK_COOLDOWN:
                if self.history:          # ishora boshlanishidan oldingi joyga qaytamiz
                    self.cur_x, self.cur_y = self.history[0]
                    self.io["cursor"](self.cur_x, self.cur_y)
                self.io["click"](self.confirmed)
                self.last_click = now
            elif self.confirmed == "scroll":
                self._scroll(palm_y, prev != "scroll")
        if self.confirmed != "scroll":
            self.scroll_y = None
        return self.confirmed

    def _move(self, lm):
        nx = min(max((lm[8].x - MARGIN) / (1 - 2 * MARGIN), 0), 1)
        ny = min(max((lm[8].y - MARGIN) / (1 - 2 * MARGIN), 0), 1)
        tx, ty = nx * (self.sw - 1), ny * (self.sh - 1)
        d = ((tx - self.cur_x) ** 2 + (ty - self.cur_y) ** 2) ** 0.5 / self.sw
        alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * min(d / FAST_DIST, 1)
        self.cur_x += (tx - self.cur_x) * alpha
        self.cur_y += (ty - self.cur_y) * alpha
        self.io["cursor"](self.cur_x, self.cur_y)
        self.history.append((self.cur_x, self.cur_y))

    def _scroll(self, y, first):
        if first or self.scroll_y is None:
            self.scroll_y, self.scroll_acc = y, 0.0
            return
        new_y = self.scroll_y + (y - self.scroll_y) * SCROLL_SMOOTH
        dy = self.scroll_y - new_y        # qo'l ko'tarilsa musbat -> tepaga scroll
        self.scroll_y = new_y
        self.scroll_acc += dy * SCROLL_GAIN
        amount = int(self.scroll_acc / WHEEL_NOTCH) * WHEEL_NOTCH   # to'liq "tishcha"lar
        if amount:
            self.io["wheel"](amount)
            self.scroll_acc -= amount


# ---------------------------------------------------------------- asosiy sikl
def main():
    if os.name != "nt":
        raise SystemExit("Bu dastur faqat Windows uchun yozilgan.")
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cam = LatestFrame(CAM_INDEX)

    ctrl = Controller(*pyautogui.size(), pyautogui.position())
    t0 = time.time()
    last_ts = -1
    labels = {"move": "1 barmoq: HARAKAT", "right": "2 barmoq: O'NG TUGMA",
              "left": "3 barmoq: CHAP TUGMA", "scroll": "5 barmoq: SCROLL (qo'lni tepaga/pastga)",
              "none": "Qo'l yo'q / noma'lum"}

    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.time() - t0) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            ctrl.update(lm)
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        else:
            ctrl.no_hand()

        cv2.rectangle(frame, (int(MARGIN * w), int(MARGIN * h)),
                      (int((1 - MARGIN) * w), int((1 - MARGIN) * h)), (255, 128, 0), 1)
        cv2.putText(frame, labels[ctrl.confirmed], (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"barmoqlar: {DEBUG['up']}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Qol mishka (q - chiqish)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cam.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
