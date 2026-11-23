# Qo'l bilan mishkani boshqarish

Kamera qo'lingizni ko'radi va sichqonchani boshqaradi (MediaPipe + OpenCV). Faqat **Windows**.

## Ishga tushirish

Talab: Python 3.10–3.13 (64-bit), veb-kamera, birinchi marta internet.

1. Loyihani yuklab oling (`git clone ...` yoki ZIP).
2. `ishga_tushir.bat` ni ikki marta bosing. Birinchi marta u o'zi virtual muhit yaratib, kutubxonalarni o'rnatadi.

Qo'lda ishga tushirish:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python qol_mishka.py
```

Chiqish: kamera oynasida `q` yoki `Esc`.

## Ishoralar

| Ishora | Natija |
|---|---|
| 1 barmoq (ko'rsatkich) | Kursor harakati |
| 2 barmoq (ko'rsatkich + o'rta) | O'ng tugma |
| 3 barmoq (ko'rsatkich + o'rta + nomsiz) | Chap tugma |
| Ochiq kaft + qo'lni tepaga/pastga | Scroll |

## Muammolar

- **Kamera ochilmadi:** boshqa dastur (Zoom, Teams) kamerani ishlatmayotganini tekshiring; Windows → Sozlamalar → Maxfiylik → Kamera → desktop ilovalarga ruxsat yoqilgan bo'lsin. Boshqa kamera tanlash uchun `qol_mishka.py` boshidagi `CAM_INDEX` ni o'zgartiring.
- **Qo'l yomon taniladi:** yorug'lik qo'lingizga tushsin (orqangizdan emas), qo'l kameradan 30–60 sm uzoqlikda bo'lsin.
- **Scroll tez/sekin:** `SCROLL_GAIN` ni o'zgartiring. Kursor titrasa/sekin bo'lsa `MIN_ALPHA`.
- `hand_landmarker.task` — qo'l modeli (7,8 MB), repoda bor. Yo'q bo'lib qolsa dastur uni o'zi yuklab oladi.
