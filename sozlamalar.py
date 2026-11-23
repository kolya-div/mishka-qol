"""Sozlamalar: saqlash/yuklash, sozlamalar oynasi (tkinter) va Windows bilan birga yoqilish."""
import json
import os

try:
    import tkinter as tk
except Exception:          # tkinter o'rnatilmagan bo'lsa ham dastur ishlayveradi (faqat panel bo'lmaydi)
    tk = None

BASE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE, "settings.json")

# (kalit, yorliq, min, max, qadam, standart, dastur ichidagi nomi)
NUMERIC = [
    ("scroll_gain", "Scroll tezligi", 1000, 15000, 250, 6000, "SCROLL_GAIN"),
    ("cursor_smooth", "Kursor silliqligi (kichik = silliqroq)", 0.3, 5.0, 0.1, 1.5, "CURSOR_MIN_CUTOFF"),
    ("cursor_speed", "Kursor javobi (katta = kechikishsiz)", 0.002, 0.05, 0.002, 0.012, "CURSOR_BETA"),
    ("margin", "Chekka o'lik zona (kalibrovkasiz)", 0.0, 0.3, 0.01, 0.07, "MARGIN"),
    ("far_max", "Uzoqdan boshqarish: eng ko'p kuchaytirish", 1.0, 6.0, 0.5, 3.5, "FAR_MAX"),
    ("ref_palm", "Kuchaytirishsiz masofa (qo'l o'lchami, kichik = uzoqroq)", 0.08, 0.4, 0.01, 0.22, "REF_PALM"),
    ("pinch", "Sudrash sezgirligi (katta = oson tegadi)", 0.15, 0.5, 0.01, 0.28, "PINCH_ON"),
    ("accel_max", "Tezlashuv kuchi (nisbiy rejimda)", 1.0, 4.0, 0.1, 2.5, "ACCEL_MAX"),
    ("auto_off", "Qo'l ko'rinmasa o'chish (soniya)", 2, 30, 1, 5, "AUTO_OFF_SECONDS"),
    ("enable_hold", "Yoqish uchun kaftni ushlash (soniya)", 0.5, 5.0, 0.5, 2.0, "ENABLE_HOLD"),
    ("preview_w", "Kamera oynasi kengligi", 400, 1600, 50, 1000, "PREVIEW_W"),
    ("cam_index", "Kamera raqami (qayta ishga tushirish kerak)", 0, 5, 1, 0, "CAM_INDEX"),
]
BOOLEAN = [
    ("far_mode", "Uzoqdan boshqarish (qo'l kichik ko'rinsa harakat kuchayadi)", True, "FAR_MODE"),
    ("swap_buttons", "2 barmoq = CHAP tugma, 3 barmoq = O'NG tugma (o'chirilsa aksincha)", True, "SWAP_BUTTONS"),
    ("relative", "Tezlashuvli (nisbiy) kursor rejimi", False, "RELATIVE_MODE"),
    ("swipe", "Orqaga/oldinga ishorasi (kaftni yon tomonga siltash)", True, "SWIPE_ENABLED"),
]


class Settings:
    def __init__(self, path=SETTINGS_PATH):
        self.path = path
        self.values = {k: d for k, _l, _lo, _hi, _s, d, _n in NUMERIC}
        self.values.update({k: d for k, _l, d, _n in BOOLEAN})
        self.calib = None                # (x0, x1, y0, y1[, qo'l o'lchami]) kadr koordinatalarida yoki None
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for k, _l, lo, hi, _s, d, _n in NUMERIC:
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

    def save(self):
        data = dict(self.values)
        data["calib"] = list(self.calib) if self.calib else None
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
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
        c = self.calib
        ns["CALIB"] = tuple(c[:4]) if c else None
        ns["CALIB_PALM"] = c[4] if c and len(c) == 5 else None


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
        launcher = os.path.join(BASE, "ishga_tushir.bat")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'@echo off\r\ncd /d "{BASE}"\r\nstart "" /min "{launcher}" --yashirin\r\n')
    elif os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------- sozlamalar oynasi
class Panel:
    def __init__(self, settings, on_change, on_calibrate, on_clear_calib):
        self.settings, self.on_change = settings, on_change
        self.visible = False
        self.root = tk.Tk()
        self.root.title("Qo'l mishka - sozlamalar")
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.root.attributes("-topmost", True)
        self.root.withdraw()
        self.scales, self.checks = {}, {}

        for row, (k, label, lo, hi, step, _d, _n) in enumerate(NUMERIC):
            tk.Label(self.root, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=8, pady=2)
            sc = tk.Scale(self.root, from_=lo, to=hi, resolution=step, orient="horizontal", length=240,
                          command=lambda v, key=k: self._set(key, v))
            sc.set(settings.values[k])
            sc.grid(row=row, column=1, padx=8)
            self.scales[k] = sc
        row = len(NUMERIC)
        for k, label, _d, _n in BOOLEAN:
            var = tk.BooleanVar(value=settings.values[k])
            tk.Checkbutton(self.root, text=label, variable=var,
                           command=lambda key=k, v=var: self._set(key, v.get())
                           ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8)
            self.checks[k] = var
            row += 1
        self.auto_var = tk.BooleanVar(value=is_autostart())
        tk.Checkbutton(self.root, text="Windows bilan birga yoqilish (kamera oynasi yashirin)",
                       variable=self.auto_var,
                       command=lambda: set_autostart(self.auto_var.get())
                       ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8)
        row += 1
        bar = tk.Frame(self.root)
        bar.grid(row=row, column=0, columnspan=2, pady=8)
        tk.Button(bar, text="Kalibrovka boshlash", command=lambda: (self.hide(), on_calibrate())
                  ).pack(side="left", padx=4)
        tk.Button(bar, text="Kalibrovkani o'chirish", command=on_clear_calib).pack(side="left", padx=4)
        tk.Button(bar, text="Standart", command=self._defaults).pack(side="left", padx=4)

    def _set(self, key, value):
        d = next((x[5] for x in NUMERIC if x[0] == key), None)
        if d is not None:
            value = int(round(float(value))) if isinstance(d, int) else float(value)
        self.settings.values[key] = value
        self.settings.save()
        self.on_change()

    def _defaults(self):
        self.settings.reset_defaults()
        for k, sc in self.scales.items():
            sc.set(self.settings.values[k])
        for k, var in self.checks.items():
            var.set(self.settings.values[k])
        self.settings.save()
        self.on_change()

    def show(self):
        self.visible = True
        self.root.deiconify()
        self.root.lift()

    def hide(self):
        self.visible = False
        self.root.withdraw()

    def pump(self):
        if self.visible:
            try:
                self.root.update()
            except tk.TclError:
                self.visible = False
