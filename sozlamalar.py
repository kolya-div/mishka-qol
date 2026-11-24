"""Sozlamalar: saqlash/yuklash, profillar, sozlamalar oynasi (tkinter), ekrandagi holat belgisi va kursor halqasi,
ovoz, ishora jurnali, tizim trayi va Windows bilan birga yoqilish."""
import json
import os
import queue
import re
import sys
import threading
import time

try:
    import tkinter as tk
except Exception:          # tkinter o'rnatilmagan bo'lsa ham dastur ishlayveradi (faqat panel bo'lmaydi)
    tk = None

try:
    import winsound
except Exception:          # Windows bo'lmasa
    winsound = None

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except Exception:          # pystray o'rnatilmagan bo'lsa ham dastur ishlayveradi (tray belgisi bo'lmaydi)
    pystray = None
    HAS_TRAY = False

SETTINGS_VERSION = 3          # 2: yoqish vaqti 2 -> 1 soniya; 3: bosh barmoq chegarasi 0.25 -> 0.30
FROZEN = bool(getattr(sys, "frozen", False))
# APP_DIR: sozlamalar saqlanadigan papka (.exe yonida yoki loyiha papkasi); RES_DIR: .exe ichiga solingan fayllar
APP_DIR = os.path.dirname(sys.executable) if FROZEN else os.path.dirname(os.path.abspath(__file__))
RES_DIR = getattr(sys, "_MEIPASS", APP_DIR)
BASE = APP_DIR
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
PROFILE_DIR = os.path.join(APP_DIR, "profillar")
LOG_PATH = os.path.join(APP_DIR, "ishora_jurnali.txt")

# (kalit, yorliq, min, max, qadam, standart, dastur ichidagi nomi)
NUMERIC = [
    ("scroll_gain", "Scroll tezligi", 1000, 15000, 250, 6000, "SCROLL_GAIN"),
    ("cursor_smooth", "Kursor silliqligi (kichik = silliqroq)", 0.3, 5.0, 0.1, 1.5, "CURSOR_MIN_CUTOFF"),
    ("cursor_speed", "Kursor javobi (katta = kechikishsiz)", 0.002, 0.05, 0.002, 0.012, "CURSOR_BETA"),
    ("deadband", "Kursor bo'sh zonasi, piksel (mayda tugmaga aniq tegish uchun)", 0, 15, 1, 4, "DEADBAND"),
    ("margin", "Chekka o'lik zona (kalibrovkasiz)", 0.0, 0.3, 0.01, 0.07, "MARGIN"),
    ("far_max", "Uzoqdan boshqarish: eng ko'p kuchaytirish", 1.0, 6.0, 0.5, 3.5, "FAR_MAX"),
    ("ref_palm", "Kuchaytirishsiz masofa (qo'l o'lchami, kichik = uzoqroq)", 0.08, 0.4, 0.01, 0.22, "REF_PALM"),
    ("pinch", "Sudrash sezgirligi (katta = oson tegadi)", 0.15, 0.5, 0.01, 0.28, "PINCH_ON"),
    ("accel_max", "Tezlashuv kuchi (nisbiy rejimda)", 1.0, 4.0, 0.1, 2.5, "ACCEL_MAX"),
    ("zoom_gain", "Ikki qo'l bilan kattalashtirish tezligi", 1000, 15000, 500, 6000, "ZOOM_GAIN"),
    ("auto_off", "Qo'l ko'rinmasa o'chish (soniya)", 2, 30, 1, 5, "AUTO_OFF_SECONDS"),
    ("enable_hold", "Yoqish uchun kaftni ushlash (soniya)", 0.5, 5.0, 0.5, 1.0, "ENABLE_HOLD"),
    ("thumb_off", "Bosh barmoq 'yopiq' chegarasi (kichik = yashil uzoqroq turadi)", 0.1, 0.5, 0.01, 0.30, "THUMB_OFF"),
    ("dwell_time", "Tuxtab turib bosish: kutish vaqti (soniya)", 0.6, 3.0, 0.1, 1.2, "DWELL_TIME"),
    ("cursor_hand", "Ikki qo'l rejimida kursor qo'li (0 = o'ng, 1 = chap)", 0, 1, 1, 0, "CURSOR_HAND"),
    ("monitor", "Ekran: 0 = hamma ekranlar, 1 = asosiy, 2... = boshqalari", 0, 8, 1, 1, "MONITOR"),
    ("preview_w", "Kamera oynasi kengligi", 400, 1600, 50, 1000, "PREVIEW_W"),
    ("cam_index", "Kamera raqami (qayta ishga tushirish kerak)", 0, 5, 1, 0, "CAM_INDEX"),
]
BOOLEAN = [
    ("sure_only", "Faqat ANIQ ishoraga buyruq (barmoq yarim bukilgan bo'lsa hech narsa qilmaydi)", True, "SURE_ONLY"),
    ("still_click", "Bosish uchun qo'l qimirlamay turishi kerak (tasodifiy bosishga qarshi)", True, "STILL_CLICK"),
    ("far_mode", "Uzoqdan boshqarish (qo'l kichik ko'rinsa harakat kuchayadi)", True, "FAR_MODE"),
    ("roi", "Uzoqdagi qo'lni kattalashtirib aniqlash", True, "ROI_ENABLED"),
    ("low_light", "Qorong'i joyda kadrni avtomatik yoritish", True, "LOW_LIGHT"),
    ("swap_buttons", "2 barmoq = CHAP tugma, 3 barmoq = O'NG tugma (o'chirilsa aksincha)", True, "SWAP_BUTTONS"),
    ("relative", "Tezlashuvli (nisbiy) kursor rejimi", False, "RELATIVE_MODE"),
    ("swipe", "Orqaga/oldinga ishorasi (kaftni yon tomonga siltash)", True, "SWIPE_ENABLED"),
    ("media", "Media ishoralari (telefon: ovoz/trek, 'seni sevaman': ish stoli)", True, "MEDIA_ENABLED"),
    ("two_hands", "Ikki qo'l bilan kattalashtirish (Ctrl + g'ildirak)", True, "ZOOM_ENABLED"),
    ("two_hand_mode", "Ikki qo'l rejimi: bir qo'l kursor, ikkinchisi bosish", False, "TWO_HAND_MODE"),
    ("dwell", "Tuxtab turib bosish (kursorni ushlab tursangiz o'zi bosadi)", False, "DWELL_ENABLED"),
    ("lock_enabled", "Qulflash ishorasi (bosh barmoqni tepaga ko'tarib ushlash)", True, "LOCK_ENABLED"),
    ("low_power", "Kam quvvat rejimi (qo'l ko'rinmaganda protsessor yuki kamayadi)", True, "LOW_POWER"),
    ("log", "Ishora jurnalini yozish (aniqlikni sozlash uchun, ishora_jurnali.txt)", True, "LOG_ENABLED"),
    ("sound", "Ovozli signallar (bosish, yoqish, o'chirish)", True, "SOUND_ENABLED"),
    ("overlay", "Ekran burchagida holat belgisi (doira)", True, "OVERLAY_ENABLED"),
    ("ring", "Kursor yonida holat halqasi", True, "RING_ENABLED"),
]

# holat -> rang (ekrandagi doira, kursor halqasi va tray belgisi uchun)
STATUS_COLOR = {
    "off": "#d32f2f", "none": "#43a047", "move": "#43a047", "two": "#fdd835", "three": "#fdd835",
    "double": "#fdd835", "drag": "#1e88e5", "scroll": "#fb8c00", "fist": "#e64a19",
    "shaka": "#8e24aa", "love": "#8e24aa", "paused": "#00acc1", "like": "#9e9e9e", "lock": "#9e9e9e",
}


def _clean_name(name):
    """Profil nomini fayl nomi uchun xavfsiz ko'rinishga keltiradi."""
    return re.sub(r"[^\w\- ]", "", str(name), flags=re.UNICODE).strip()[:40]


class Settings:
    def __init__(self, path=SETTINGS_PATH):
        self.path = path
        self.values = {k: d for k, _l, _lo, _hi, _s, d, _n in NUMERIC}
        self.values.update({k: d for k, _l, d, _n in BOOLEAN})
        self.calib = None                # (x0, x1, y0, y1[, qo'l o'lchami]) kadr koordinatalarida yoki None
        self.train = None                # (UP_ON, UP_OFF[, THUMB_ON, THUMB_OFF]): o'rgatish natijasi yoki None
        self.load()

    # ----- yuklash / saqlash
    def _read(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _load_data(self, data):
        old_ver = data.get("version", 1)
        reset_keys = set()
        if old_ver < 2:
            reset_keys.add("enable_hold")        # eski qiymat (2 s) o'rniga yangi standart (1 s)
        if old_ver < 3:
            reset_keys.add("thumb_off")          # yangi standart (0.30)
        for k, _l, lo, hi, _s, d, _n in NUMERIC:
            if k in reset_keys:
                continue
            try:
                v = float(data.get(k, d))
                v = min(max(v, lo), hi)
                self.values[k] = int(round(v)) if isinstance(d, int) else v
            except (TypeError, ValueError):
                pass
        for k, _l, d, _n in BOOLEAN:
            self.values[k] = bool(data.get(k, d))
        c = data.get("calib")
        if isinstance(c, list) and len(c) in (4, 5) and all(isinstance(x, (int, float)) for x in c):
            self.calib = tuple(float(x) for x in c)
        t = data.get("train")
        if isinstance(t, list) and len(t) in (2, 4) and all(isinstance(x, (int, float)) for x in t) \
                and 60 <= t[1] < t[0] <= 178 and (len(t) == 2 or 0.05 <= t[3] < t[2] <= 2.0):
            self.train = tuple(float(x) for x in t)

    def load(self):
        data = self._read(self.path)
        if data is not None:
            self._load_data(data)

    def _dump(self):
        data = dict(self.values)
        data["version"] = SETTINGS_VERSION
        data["calib"] = list(self.calib) if self.calib else None
        data["train"] = list(self.train) if self.train else None
        return data

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._dump(), f, indent=2)
        except OSError:
            pass

    def reset_defaults(self):
        for k, _l, _lo, _hi, _s, d, _n in NUMERIC:
            self.values[k] = d
        for k, _l, d, _n in BOOLEAN:
            self.values[k] = d

    def apply(self, ns):
        """Qiymatlarni dastur o'zgaruvchilariga (modul nomlar maydoniga) yozadi."""
        for k, _l, _lo, _hi, _s, _d, name in NUMERIC:
            ns[name] = self.values[k]
        for k, _l, _d, name in BOOLEAN:
            ns[name] = self.values[k]
        ns["PINCH_OFF"] = ns["PINCH_ON"] * 1.5
        ns["THUMB_ON"] = ns["THUMB_OFF"] + 0.20      # bosh barmoq: ochiq/yopiq orasida oraliq zona
        c = self.calib
        ns["CALIB"] = tuple(c[:4]) if c else None
        ns["CALIB_PALM"] = c[4] if c and len(c) == 5 else None
        t = self.train
        ns["UP_ON"], ns["UP_OFF"] = (t[0], t[1]) if t else (150.0, 125.0)
        if t and len(t) == 4:                        # o'rgatishda bosh barmoq ham o'rganilgan
            ns["THUMB_ON"], ns["THUMB_OFF"] = t[2], t[3]

    # ----- profillar (turli odam/kompyuter uchun alohida sozlamalar)
    @staticmethod
    def profiles_dir():
        return os.environ.get("QOL_PROFILE_DIR") or PROFILE_DIR

    def profile_path(self, name):
        return os.path.join(self.profiles_dir(), _clean_name(name) + ".json")

    def list_profiles(self):
        try:
            return sorted(os.path.splitext(f)[0] for f in os.listdir(self.profiles_dir()) if f.lower().endswith(".json"))
        except OSError:
            return []

    def save_profile(self, name):
        name = _clean_name(name)
        if not name:
            return False
        try:
            os.makedirs(self.profiles_dir(), exist_ok=True)
            with open(self.profile_path(name), "w", encoding="utf-8") as f:
                json.dump(self._dump(), f, indent=2)
        except OSError:
            return False
        return True

    def load_profile(self, name):
        data = self._read(self.profile_path(name))
        if data is None:
            return False
        self.reset_defaults()
        self.calib = self.train = None
        self._load_data(data)
        self.save()
        return True

    def delete_profile(self, name):
        try:
            os.remove(self.profile_path(name))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------- Windows bilan birga yoqilish
def startup_file():
    folder = os.environ.get("QOL_STARTUP_DIR") or os.path.join(
        os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
    return os.path.join(folder, "qol_mishka_avtoyoqish.bat")


def is_autostart():
    return os.path.exists(startup_file())


def set_autostart(on):
    path = startup_file()
    if on:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if FROZEN:                                   # .exe: to'g'ridan-to'g'ri o'zini ishga tushiradi
            line = f'start "" /min "{sys.executable}" --yashirin'
        else:
            line = f'start "" /min "{os.path.join(BASE, "ishga_tushir.bat")}" --yashirin'
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'@echo off\r\ncd /d "{BASE}"\r\n{line}\r\n')
    elif os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------- ovoz
class Sound:
    """Qisqa ovozli signallar (alohida oqimda, asosiy siklni to'xtatmaydi)."""
    PATTERNS = {
        "click": [(900, 35)], "double": [(900, 30), (1100, 30)], "drag": [(600, 40)],
        "enable": [(700, 70), (1000, 90)], "disable": [(1000, 70), (600, 90)],
        "swipe": [(800, 25)], "media": [(1200, 40)], "desktop": [(500, 40), (700, 40)],
        "lock": [(500, 70), (350, 90)], "unlock": [(350, 70), (600, 90)],
        "done": [(800, 60), (1000, 60), (1300, 90)],
    }

    def __init__(self):
        self.enabled = True
        self.q = queue.Queue(maxsize=4)
        if winsound is not None:
            threading.Thread(target=self._run, daemon=True).start()

    def play(self, name):
        if not self.enabled or winsound is None or name not in self.PATTERNS:
            return
        try:
            self.q.put_nowait(name)
        except queue.Full:
            pass

    def _run(self):
        while True:
            name = self.q.get()
            for freq, ms in self.PATTERNS[name]:
                try:
                    winsound.Beep(freq, ms)
                except Exception:
                    break


# ---------------------------------------------------------------- ishora jurnali
class GestureLog:
    """Muhim hodisalarni (bosish, sudrash, noaniq ishora...) barmoq ballari bilan faylga yozadi.
    Aniqlik muammosi bo'lsa shu fayl yordamida chegaralarni aniq sozlash mumkin."""
    MAX_BYTES = 1_500_000

    def __init__(self, path=None):
        self.path = path or LOG_PATH
        self.enabled = True
        self.last = {}

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        if isinstance(v, dict):
            return ",".join(f"{k}:{x:.0f}" if isinstance(x, (int, float)) else f"{k}:{x}" for k, x in v.items())
        return str(v)

    def write(self, event, throttle=0.0, **fields):
        if not self.enabled:
            return
        now = time.time()
        if throttle and now - self.last.get(event, 0.0) < throttle:
            return
        self.last[event] = now
        line = time.strftime("%H:%M:%S") + f" {event} " + " ".join(f"{k}={self._fmt(v)}" for k, v in fields.items())
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) > self.MAX_BYTES:
                os.replace(self.path, self.path + ".old")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------- tizim trayi
def make_icon(color):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=color, outline="white", width=4)
    d.rectangle((26, 18, 38, 30), fill="white")          # sichqoncha belgisi
    d.rectangle((30, 18, 34, 30), fill=color)
    return img


class Tray:
    """Soat yonidagi belgi: sozlamalar, kamera, yoqish/o'chirish, qulflash, profillar, kalibrovka, chiqish."""

    def __init__(self, push, settings=None):
        self.push, self.color, self.icon, self.settings = push, STATUS_COLOR["none"], None, settings
        if not HAS_TRAY:
            return
        menu = pystray.Menu(
            pystray.MenuItem("Sozlamalar", lambda: push("settings")),
            pystray.MenuItem("Kamerani ko'rsatish / yashirish", lambda: push("hide")),
            pystray.MenuItem("Yoqish / o'chirish", lambda: push("toggle")),
            pystray.MenuItem("Qulflash / ochish", lambda: push("lock")),
            pystray.MenuItem("Profillar", pystray.Menu(self._profile_items)),
            pystray.MenuItem("Kalibrovka", lambda: push("calib")),
            pystray.MenuItem("Ishoralarni o'rgatish", lambda: push("train")),
            pystray.MenuItem("Chiqish", lambda: push("quit")),
        )
        self.icon = pystray.Icon("qol_mishka", make_icon(self.color), "Qo'l mishka", menu)

    def _profile_items(self):
        names = self.settings.list_profiles() if self.settings else []
        if not names:
            yield pystray.MenuItem("(profil yo'q: sozlamalarda saqlang)", None, enabled=False)
            return
        for n in names:
            yield pystray.MenuItem(n, (lambda name: lambda: self.push(("profile", name)))(n))

    def start(self):
        if self.icon is not None:
            try:
                self.icon.run_detached()
            except Exception:
                self.icon = None

    def set_color(self, color):
        if self.icon is not None and color != self.color:
            self.color = color
            try:
                self.icon.icon = make_icon(color)
            except Exception:
                pass

    def stop(self):
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:
                pass


# ---------------------------------------------------------------- ekrandagi belgilar (bosishlarga to'sqinlik qilmaydi)
def make_click_through(win):
    """Oynani "shaffof" qilamiz: sichqoncha bosishlari uning ostidagi oynaga o'tadi, fokus olmaydi."""
    try:
        import ctypes
        win.update_idletasks()
        u = ctypes.windll.user32
        hwnd = u.GetParent(win.winfo_id()) or win.winfo_id()
        ex = u.GetWindowLongW(hwnd, -20)
        u.SetWindowLongW(hwnd, -20, ex | 0x80000 | 0x20 | 0x80 | 0x08000000)
    except Exception:
        pass


class Overlay:
    """Ekranning yuqori o'ng burchagida kichik rangli doira."""
    SIZE = 24
    KEY = "#010101"

    def __init__(self, root):
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.config(bg=self.KEY)
        try:
            self.win.attributes("-transparentcolor", self.KEY)
        except tk.TclError:
            pass
        s = self.SIZE
        self.win.geometry(f"{s}x{s}+{root.winfo_screenwidth() - s - 8}+8")
        self.canvas = tk.Canvas(self.win, width=s, height=s, highlightthickness=0, bg=self.KEY)
        self.canvas.pack()
        self.dot = self.canvas.create_oval(2, 2, s - 2, s - 2, fill=STATUS_COLOR["none"], outline="white", width=2)
        self.color = None
        self.visible = True
        make_click_through(self.win)

    def set(self, color):
        if color != self.color:
            self.color = color
            self.canvas.itemconfig(self.dot, fill=color)

    def show(self, on):
        if on != self.visible:
            self.visible = on
            (self.win.deiconify if on else self.win.withdraw)()


class CursorRing:
    """Sichqoncha ko'rsatkichi yonidagi rangli halqa: holatni ko'rsatadi; "tuxtab turib bosish" paytida to'lib boradi."""
    SIZE = 46
    KEY = "#010101"

    def __init__(self, root):
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.config(bg=self.KEY)
        try:
            self.win.attributes("-transparentcolor", self.KEY)
        except tk.TclError:
            pass
        s = self.SIZE
        self.win.geometry(f"{s}x{s}+0+0")
        self.canvas = tk.Canvas(self.win, width=s, height=s, highlightthickness=0, bg=self.KEY)
        self.canvas.pack()
        self.circle = self.canvas.create_oval(5, 5, s - 5, s - 5, outline=STATUS_COLOR["none"], width=3)
        self.arc = self.canvas.create_arc(2, 2, s - 2, s - 2, start=90, extent=0, style="arc",
                                          outline="white", width=4, state="hidden")
        self.color, self.pos, self.progress, self.visible = None, None, 0.0, True
        make_click_through(self.win)

    def update(self, x, y, color, progress=0.0, show=True):
        if show != self.visible:
            self.visible = show
            (self.win.deiconify if show else self.win.withdraw)()
        if not show:
            return
        s = self.SIZE
        pos = (int(x) - s // 2, int(y) - s // 2)
        if pos != self.pos:
            self.pos = pos
            self.win.geometry(f"+{pos[0]}+{pos[1]}")
        if color != self.color:
            self.color = color
            self.canvas.itemconfig(self.circle, outline=color)
        if progress != self.progress:
            self.progress = progress
            if progress > 0:
                self.canvas.itemconfig(self.arc, state="normal", extent=-min(359.0, 360.0 * progress))
            else:
                self.canvas.itemconfig(self.arc, state="hidden")


# ---------------------------------------------------------------- sozlamalar oynasi
class Panel:
    def __init__(self, settings, on_change, on_calibrate, on_clear_calib, on_train=None, on_clear_train=None,
                 on_profile=None):
        self.settings, self.on_change, self.on_profile = settings, on_change, on_profile
        self.visible = False
        self.root = tk.Tk()
        self.root.title("Qo'l mishka - sozlamalar")
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.root.attributes("-topmost", True)
        self.root.withdraw()
        self.scales, self.checks = {}, {}
        try:
            self.overlay = Overlay(self.root)
        except Exception:
            self.overlay = None
        try:
            self.ring = CursorRing(self.root)
        except Exception:
            self.ring = None

        # ichidagi hamma narsa aylantiriladigan (scroll) maydonda: oyna ekranga sig'adi
        outer = tk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        body = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.root.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

        half = (len(NUMERIC) + 1) // 2                   # sozlamalar ikki ustunda (oyna past bo'lsin)
        for i, (k, label, lo, hi, step, _d, _n) in enumerate(NUMERIC):
            cell = tk.Frame(body)
            cell.grid(row=i % half, column=i // half, sticky="w", padx=8, pady=1)
            tk.Label(cell, text=label, anchor="w", justify="left", wraplength=250).pack(anchor="w")
            sc = tk.Scale(cell, from_=lo, to=hi, resolution=step, orient="horizontal", length=250,
                          command=lambda v, key=k: self._set(key, v))
            sc.set(settings.values[k])
            sc.pack(anchor="w")
            self.scales[k] = sc
        row = half
        for k, label, _d, _n in BOOLEAN:
            var = tk.BooleanVar(value=settings.values[k])
            tk.Checkbutton(body, text=label, variable=var, justify="left", anchor="w",
                           command=lambda key=k, v=var: self._set(key, v.get())
                           ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8)
            self.checks[k] = var
            row += 1
        self.auto_var = tk.BooleanVar(value=is_autostart())
        tk.Checkbutton(body, text="Windows bilan birga yoqilish (kamera oynasi yashirin)",
                       variable=self.auto_var,
                       command=lambda: set_autostart(self.auto_var.get())
                       ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8)
        row += 1

        # ----- profillar
        prof = tk.LabelFrame(body, text="Profillar (turli odam / kompyuter uchun alohida sozlamalar)")
        prof.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=6)
        self.prof_name = tk.StringVar()
        tk.Entry(prof, textvariable=self.prof_name, width=22).pack(side="left", padx=4, pady=4)
        tk.Button(prof, text="Hozirgisini saqlash", command=self._save_profile).pack(side="left", padx=4)
        self.prof_pick = tk.StringVar()
        self.prof_menu = tk.OptionMenu(prof, self.prof_pick, "")
        self.prof_menu.pack(side="left", padx=4)
        tk.Button(prof, text="Yuklash", command=self._load_profile).pack(side="left", padx=2)
        tk.Button(prof, text="O'chirish", command=self._delete_profile).pack(side="left", padx=2)
        self._refresh_profiles()
        row += 1

        bar = tk.Frame(body)
        bar.grid(row=row, column=0, columnspan=2, pady=8)
        tk.Button(bar, text="Kalibrovka boshlash", command=lambda: (self.hide(), on_calibrate())
                  ).pack(side="left", padx=4)
        tk.Button(bar, text="Kalibrovkani o'chirish", command=on_clear_calib).pack(side="left", padx=4)
        if on_train:
            tk.Button(bar, text="Ishoralarni o'rgatish", command=lambda: (self.hide(), on_train())
                      ).pack(side="left", padx=4)
            tk.Button(bar, text="O'rgatishni o'chirish", command=on_clear_train).pack(side="left", padx=4)
        tk.Button(bar, text="Standart", command=self._defaults).pack(side="left", padx=4)

        self.root.update_idletasks()                     # oyna balandligi ekrandan oshmasin
        self.canvas.config(width=body.winfo_reqwidth(),
                           height=min(body.winfo_reqheight(), max(300, self.root.winfo_screenheight() - 140)))

    # ----- sozlamalar
    def _set(self, key, value):
        d = next((x[5] for x in NUMERIC if x[0] == key), None)
        if d is not None:
            value = int(round(float(value))) if isinstance(d, int) else float(value)
        self.settings.values[key] = value
        self.settings.save()
        self.on_change()

    def refresh(self):
        """Sozlamalar tashqaridan o'zgargan bo'lsa (masalan profil yuklandi), oynadagi qiymatlarni yangilaydi."""
        for k, sc in self.scales.items():
            sc.set(self.settings.values[k])
        for k, var in self.checks.items():
            var.set(self.settings.values[k])

    def _defaults(self):
        self.settings.reset_defaults()
        self.refresh()
        self.settings.save()
        self.on_change()

    # ----- profillar
    def _refresh_profiles(self, select=None):
        names = self.settings.list_profiles()
        menu = self.prof_menu["menu"]
        menu.delete(0, "end")
        for n in names:
            menu.add_command(label=n, command=lambda v=n: self.prof_pick.set(v))
        self.prof_pick.set(select if select in names else (names[0] if names else ""))

    def _save_profile(self):
        name = _clean_name(self.prof_name.get())
        if name and self.settings.save_profile(name):
            self._refresh_profiles(select=name)

    def _load_profile(self):
        name = self.prof_pick.get()
        if name and self.on_profile:
            self.on_profile(name)

    def _delete_profile(self):
        name = self.prof_pick.get()
        if name:
            self.settings.delete_profile(name)
            self._refresh_profiles()

    # ----- ko'rinish
    def show(self):
        self.visible = True
        self.root.deiconify()
        self.root.lift()

    def hide(self):
        self.visible = False
        self.root.withdraw()

    def set_status(self, status, show=True):
        if self.overlay is not None:
            self.overlay.show(show)
            self.overlay.set(STATUS_COLOR.get(status, STATUS_COLOR["none"]))

    def set_ring(self, x, y, status, progress=0.0, show=True):
        if self.ring is not None:
            self.ring.update(x, y, STATUS_COLOR.get(status, STATUS_COLOR["none"]), progress, show)

    def pump(self):
        try:
            self.root.update()                    # belgi va halqa ham yangilanishi uchun doim
        except tk.TclError:
            self.visible = False
