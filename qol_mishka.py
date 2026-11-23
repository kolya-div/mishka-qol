"""Qo'l bilan mishkani boshqarish (kamera + MediaPipe).

Imo-ishoralar:
  1 barmoq (ko'rsatkich)                 -> kursor qo'l bilan harakatlanadi
  2 barmoq (ko'rsatkich + o'rta)         -> O'NG tugma bosiladi
  3 barmoq (ko'rsatkich, o'rta, nomsiz)  -> CHAP tugma bosiladi
  5 barmoq (ochiq kaft) + qo'lni tepaga/pastga siljitish -> SCROLL
  bosh + ko'rsatkich barmoq tekkizib ushlash -> SUDRASH (qo'yib yuborsangiz tashlanadi)
  mushtni 1 soniya ushlash                -> dasturni O'CHIRISH / YOQISH
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
FRAME_W, FRAME_H = 1280, 720   # kamera o'lchami (kengroq ko'rish maydoni; kamera qo'llamasa o'zi moslashadi)
PREVIEW_W = 1000               # ekrandagi kamera oynasi kengligi (piksel). Oynani sichqoncha bilan ham kattalashtirsa bo'ladi
WINDOW = "Qol mishka (q - chiqish)"
MARGIN = 0.07            # kadr chetidagi "o'lik zona" (kichik = qo'lni kadrning deyarli butun maydonida ishlatsa bo'ladi)
CURSOR_MIN_CUTOFF = 1.5  # One Euro: kichik = sekin harakatda silliqroq (titrash kam), lekin sekinroq
CURSOR_BETA = 0.012      # One Euro: katta = tez harakatda kechikish kamroq

PINCH_ON = 0.28          # bosh va ko'rsatkich uchlari orasi (kaft o'lchamiga nisbatan) < shu bo'lsa "tegdi" -> sudrash
PINCH_OFF = 0.42         # > shu bo'lsa "ajraldi" -> qo'yib yuborish
FIST_MIN_RATIO = 0.45    # mushtda bosh va ko'rsatkich uchlari shundan uzoqroq bo'lishi kerak (sudrashdan farqlash)
FIST_HOLD = 1.0          # mushtni shuncha soniya ushlab tursangiz dastur yoqiladi/o'chadi

SCROLL_GAIN = 6000       # qo'l siljishi -> scroll tezligi (katta = tezroq)
SCROLL_SMOOTH = 0.5      # qo'l balandligini silliqlash (1 = silliqlashsiz)
SCROLL_FRAMES = 2
WHEEL_NOTCH = 120

LOST_GRACE = 8          # qo'l shuncha kadr ko'rinmasa ham oldingi holat saqlanadi
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


def mouse_down():
    pyautogui.mouseDown(button="left")


def mouse_up():
    pyautogui.mouseUp(button="left")


# ---------------------------------------------------------------- imo-ishoralar
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def finger_up(lm, name):
    wrist = lm[0]
    return dist(lm[TIP[name]], wrist) > dist(lm[PIP[name]], wrist) * 1.15


def classify(lm, pinching):
    """-> (gesture, pinching). gesture: drag / fist / move / right / left / scroll / none"""
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
    """Ishorani aniqlab, kursor/bosish/scroll/sudrash amallarini bajaradi."""

    def __init__(self, screen_w, screen_h, cursor_pos, io=None):
        self.sw, self.sh = screen_w, screen_h
        self.cur_x, self.cur_y = cursor_pos
        self.io = io or {"cursor": set_cursor, "click": click, "wheel": wheel,
                         "down": mouse_down, "up": mouse_up}
        self.history = deque(maxlen=5)    # kursorning oxirgi joylari (bosishdagi siljishni qaytarish uchun)
        self.filter = OneEuro(CURSOR_MIN_CUTOFF, CURSOR_BETA)
        self.enabled = True
        self.fist_start = None
        self.fist_latched = False
        self.fist_progress = 0.0
        self.last_click = float("-inf")
        self.reset()

    # ----- holatni tozalash
    def reset(self):
        self.release()
        self.pinching = False
        self.candidate, self.streak, self.confirmed = "none", 0, "none"
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.lost = 0
        self.fist_start, self.fist_latched, self.fist_progress = None, False, 0.0
        self.history.clear()

    def release(self):
        if getattr(self, "dragging", False):
            self.io["up"]()
        self.dragging = False
        self.drag_off = (0.0, 0.0)

    def no_hand(self):
        # qo'l 1-2 kadrga yo'qolsa holatni saqlab turamiz (uzilib-uzilib qolmasligi uchun)
        self.lost += 1
        if self.lost > LOST_GRACE:
            self.reset()
            self.lost = LOST_GRACE + 1

    @property
    def status(self):
        return self.confirmed if self.enabled else "off"

    # ----- asosiy yangilanish
    def update(self, lm, now=None):
        now = time.time() if now is None else now
        self.lost = 0
        gesture, self.pinching = classify(lm, self.pinching)
        palm_y = lm[9].y

        if gesture == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = gesture, 1

        need = {"move": 1, "left": LEFT_FRAMES, "right": RIGHT_FRAMES,
                "scroll": SCROLL_FRAMES}.get(self.candidate, 2)
        if self.streak >= need:
            prev, self.confirmed = self.confirmed, self.candidate
            self._fist_toggle(now)
            if prev == "drag" and self.confirmed != "drag":
                self.release()
            if not self.enabled:
                pass
            elif self.confirmed == "move":
                self._move(lm, now, prev != "move")
            elif self.confirmed == "drag":
                self._drag(lm, now, prev != "drag")
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
        return self.status

    # ----- mushtni ushlab turish: dasturni yoqish/o'chirish
    def _fist_toggle(self, now):
        if self.confirmed != "fist":
            self.fist_start, self.fist_latched, self.fist_progress = None, False, 0.0
            return
        if self.fist_start is None:
            self.fist_start = now
        self.fist_progress = min((now - self.fist_start) / FIST_HOLD, 1.0)
        if self.fist_progress >= 1.0 and not self.fist_latched:
            self.fist_latched = True          # mushtni ochmaguncha qayta almashmaydi
            self.enabled = not self.enabled
            self.release()
            self.history.clear()

    # ----- kursor
    def _target(self, lm):
        nx = min(max((lm[8].x - MARGIN) / (1 - 2 * MARGIN), 0), 1)
        ny = min(max((lm[8].y - MARGIN) / (1 - 2 * MARGIN), 0), 1)
        return nx * (self.sw - 1), ny * (self.sh - 1)

    def _move(self, lm, now, first):
        if first:
            self.filter.prime(self.cur_x, self.cur_y, now)
        tx, ty = self._target(lm)
        self.cur_x, self.cur_y = self.filter(tx, ty, now)
        self.io["cursor"](self.cur_x, self.cur_y)
        self.history.append((self.cur_x, self.cur_y))

    def _drag(self, lm, now, first):
        tx, ty = self._target(lm)
        if first:
            if self.history:              # barmoqlar tekkunga qadar bo'lgan joyni bosamiz
                self.cur_x, self.cur_y = self.history[0]
                self.io["cursor"](self.cur_x, self.cur_y)
            self.drag_off = (self.cur_x - tx, self.cur_y - ty)
            self.filter.prime(self.cur_x, self.cur_y, now)
            self.io["down"]()
            self.dragging = True
            return
        self.cur_x, self.cur_y = self.filter(tx + self.drag_off[0], ty + self.drag_off[1], now)
        self.io["cursor"](self.cur_x, self.cur_y)

    # ----- scroll
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
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)           # oyna o'lchamini sichqoncha bilan o'zgartirsa bo'ladi
    cv2.resizeWindow(WINDOW, PREVIEW_W, int(PREVIEW_W * FRAME_H / FRAME_W))

    ctrl = Controller(*pyautogui.size(), pyautogui.position())
    t0 = time.time()
    last_ts = -1
    labels = {"move": "1 barmoq: HARAKAT", "right": "2 barmoq: O'NG TUGMA",
              "left": "3 barmoq: CHAP TUGMA", "scroll": "5 barmoq: SCROLL (qo'lni tepaga/pastga)",
              "drag": "TEKKIZISH: SUDRASH", "fist": "MUSHT: ushlab turing...",
              "none": "Qo'l yo'q / noma'lum",
              "off": "O'CHIRILGAN - yoqish uchun mushtni 1 soniya ushlang"}

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

        hands = result.hand_landmarks
        if not hands:                                    # qo'l yo'qoldi -> zaxira usuli
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            boosted = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
            hands = rescue.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=boosted)).hand_landmarks

        if hands:
            lm = hands[0]
            ctrl.update(lm)
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        else:
            ctrl.no_hand()

        k = w / 640.0                                        # yozuvlar kadr o'lchamiga qarab kattalashadi
        cv2.rectangle(frame, (int(MARGIN * w), int(MARGIN * h)),
                      (int((1 - MARGIN) * w), int((1 - MARGIN) * h)), (255, 128, 0), 1)
        status = ctrl.status
        if status == "fist" and ctrl.fist_progress > 0:      # mushtni ushlash progressi
            y0, y1, bw = int(42 * k), int(56 * k), int(200 * k)
            cv2.rectangle(frame, (10, y0), (10 + int(bw * ctrl.fist_progress), y1), (0, 255, 255), -1)
            cv2.rectangle(frame, (10, y0), (10 + bw, y1), (255, 255, 255), 1)
        color = (0, 0, 255) if status == "off" else (0, 255, 255)
        cv2.putText(frame, labels[status], (10, int(30 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * k, color, max(2, int(2 * k)))
        cv2.putText(frame, f"barmoqlar: {DEBUG['up']}  tekkizish: {DEBUG['pinch']:.2f}", (10, h - int(12 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (255, 255, 255), max(1, int(k)))
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    ctrl.release()                       # sudrash paytida chiqilsa, tugma bosilgan holda qolmasin
    cam.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
