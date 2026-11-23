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

Parametrlar: `--yashirin` (kamera oynasiz, fonda), `--avtoyoqish` / `--avtoyoqish-ochir` (Windows bilan birga yoqilish).

## Ishoralar

| Ishora | Natija |
|---|---|
| 1 barmoq (ko'rsatkich) | Kursor harakati |
| 2 barmoq (ko'rsatkich + o'rta) | O'ng tugma |
| 3 barmoq (ko'rsatkich + o'rta + nomsiz) | Chap tugma |
| Ko'rsatkich + jimjiloq ("rok") | Ikki marta bosish |
| Bosh va ko'rsatkich barmoqni tekkizib ushlash | Sudrash (qo'yib yuborsangiz tashlanadi) |
| Ochiq kaft + qo'lni tepaga/pastga | Scroll |
| Ochiq kaft + qo'lni tez o'ngga/chapga siltash | Brauzerda oldinga / orqaga (Alt+→ / Alt+←) |
| Mushtni 1 soniya ushlash | Dasturni o'chirish |
| Qo'l 5 soniya ko'rinmasa | Dastur o'zi o'chadi |
| Kaftni **to'liq** (bosh barmoq bilan) ochib 2 soniya ushlash | Dasturni yoqish |

O'chiq payti hech narsa bosilmaydi, kamera oynasida qizil yozuv chiqadi.

## Tugmalar

Kamera oynasida: `q` chiqish · `h` yashirish · `m` kichik/katta oyna · `s` sozlamalar · `k` kalibrovka.
Hamma joyda (oyna yashirin bo'lsa ham): `Ctrl+Alt+Q / H / M / S / K`.

## Sozlamalar

`s` (yoki `Ctrl+Alt+S`) bosilsa sozlamalar oynasi ochiladi: scroll tezligi, kursor silliqligi va tezligi, sudrash sezgirligi, avto-o'chish vaqti, yoqish vaqti, oyna kengligi, kamera raqami, tezlashuvli (nisbiy) kursor rejimi, siltash ishorasi va Windows bilan birga yoqilish. O'zgarishlar darrov qo'llanadi va `settings.json` ga saqlanadi (bu fayl GitHub'ga yuklanmaydi).

**Kalibrovka** (`k`): dastur 4 burchakni ketma-ket ko'rsatadi. 1 barmoq bilan uchini har burchakka (eng chekka qulay joyga) qo'yib ushlab turing. Shundan keyin qo'lingizning qulay maydoni butun ekranga yoyiladi. Sozlamalar oynasidan o'chirish mumkin.

## Muammolar

- **Kamera ochilmadi:** boshqa dastur (Zoom, Teams) kamerani ishlatmayotganini tekshiring; Windows → Sozlamalar → Maxfiylik → Kamera → desktop ilovalarga ruxsat yoqilgan bo'lsin.
- **Qo'l yomon taniladi:** yorug'lik qo'lingizga tushsin (orqangizdan emas), qo'l kameradan 30–60 sm uzoqlikda bo'lsin. Dastur qo'l yo'qolsa kontrastni oshirib qayta qidiradi.
- **Mush/sudrash adashadi:** sozlamalarda "Sudrash sezgirligi" ni o'zgartiring.
- **Sozlamalar oynasi ochilmasa:** Python'ni o'rnatishda "tcl/tk" belgilanmagan bo'lishi mumkin; dastur baribir ishlaydi, sozlamalar `settings.json` orqali o'zgaradi.
- `hand_landmarker.task` — qo'l modeli (7,8 MB), repoda bor. Yo'q bo'lib qolsa dastur uni o'zi yuklab oladi.
