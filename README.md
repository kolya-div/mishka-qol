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

### `.exe` (Python o'rnatmasdan)

`build_exe.bat` ni ishga tushiring: `dist\QolMishka\QolMishka.exe` yig'iladi. Butun `dist\QolMishka` papkasini boshqa kompyuterga ko'chirib, `QolMishka.exe` ni bosish yetarli (Python kerak emas). Sozlamalar (`settings.json`) va xatolar (`xato.log`) shu papkada saqlanadi. Dastur doim tizim trayida (soat yonida) turadi.

## Ishoralar

| Ishora | Natija |
|---|---|
| 1 barmoq (ko'rsatkich) | Kursor harakati |
| 2 barmoq (ko'rsatkich + o'rta) | Chap tugma |
| 3 barmoq (ko'rsatkich + o'rta + nomsiz) | O'ng tugma |
| Ko'rsatkich + jimjiloq ("rok"), bosh barmoq yopiq | Ikki marta bosish |
| Bosh va ko'rsatkich barmoqni tekkizib ushlash | Sudrash (qo'yib yuborsangiz tashlanadi) |
| Ochiq kaft + qo'lni tepaga/pastga | Scroll |
| Ochiq kaft + qo'lni tez o'ngga/chapga siltash | Brauzerda oldinga / orqaga (Alt+→ / Alt+←) |
| Telefon (bosh barmoq + jimjiloq ochiq) + qo'lni tepaga/pastga | Ovozni oshirish / kamaytirish |
| Telefon + qo'lni tez o'ngga/chapga siltash | Keyingi / oldingi trek |
| "Seni sevaman" (bosh + ko'rsatkich + jimjiloq) + o'ngga/chapga siltash | Virtual ish stolini almashtirish (Ctrl+Win+←/→) |
| Ikkala qo'l ochiq, bir-biridan uzoqlashtirish / yaqinlashtirish | Kattalashtirish / kichraytirish (Ctrl + g'ildirak) |
| Mushtni 1 soniya ushlash | Dasturni o'chirish |
| Qo'l 5 soniya ko'rinmasa | Dastur o'zi o'chadi |
| Kaftni **to'liq** (bosh barmoq bilan) ochib 1 soniya ushlash | Dasturni yoqish |

O'chiq payti hech narsa bosilmaydi, kamera oynasida qizil yozuv chiqadi.

## Tugmalar

Kamera oynasida: `q` chiqish · `h` yashirish · `m` kichik/katta oyna · `s` sozlamalar · `k` kalibrovka · `t` ishoralarni o'rgatish · `l` nomlarni yoqish/o'chirish.
Hamma joyda (oyna yashirin bo'lsa ham): `Ctrl+Alt+Q / H / M / S / K / T`, `Ctrl+Alt+Probel` = yoqish/o'chirish.
Tizim trayidagi belgi (o'ng tugma): sozlamalar, kamera, yoqish/o'chirish, kalibrovka, o'rgatish, chiqish.

## Holat belgisi va ovoz

Ekranning yuqori o'ng burchagida kichik doira turadi (kamera oynasi yashirin bo'lsa ham): yashil = ishlayapti, qizil = o'chiq, sariq = bosish, ko'k = sudrash, to'q sariq = scroll, binafsha = ovoz/trek/ish stoli, havorang = kalibrovka/o'rgatish. U bosishlarga to'sqinlik qilmaydi. Bosish, yoqish va o'chirishda qisqa ovoz chiqadi. Ikkalasini ham sozlamalarda o'chirish mumkin.

## Sozlamalar

`s` (yoki `Ctrl+Alt+S`) bosilsa sozlamalar oynasi ochiladi: scroll tezligi, kursor silliqligi va tezligi, sudrash sezgirligi, uzoqdan boshqarish, kattalashtirish tezligi, avto-o'chish va yoqish vaqti, **ekran tanlash** (0 = hamma ekranlar, 1 = asosiy, 2... = boshqalari), oyna kengligi, kamera raqami, tezlashuvli (nisbiy) kursor rejimi, siltash/media/ikki qo'l ishoralari, ovoz, holat belgisi va Windows bilan birga yoqilish. O'zgarishlar darrov qo'llanadi va `settings.json` ga saqlanadi (bu fayl GitHub'ga yuklanmaydi).

**Uzoqdan boshqarish.** Qo'l kameradan uzoqlashib kichik ko'rinsa, kursor harakati avtomatik kuchayadi (kadrdagi qo'l o'lchamiga qarab), shuning uchun uzoqdan ham joyingizdan siljimasdan butun ekranga yetasiz. Kamera oynasida "Uzoqdan boshqarish: harakat xN" yoziladi. Qo'l kadrda kichik bo'lsa, dastur uning atrofini kesib kattalashtirib tahlil qiladi ("Uzoqdagi qo'lni kattalashtirib aniqlash"), shuning uchun uzoqdan ham barmoqlar aniqroq ajraladi.

**Kalibrovka** (`k`): dastur 4 burchakni ketma-ket ko'rsatadi. 1 barmoq bilan uchini har burchakka (eng chekka qulay joyga) qo'yib ushlab turing. Shundan keyin qo'lingizning qulay maydoni butun ekranga yoyiladi va o'sha masofa eslab qolinadi (shu masofada kuchaytirish 1 bo'ladi). Uzoqdan o'tirib boshqarmoqchi bo'lsangiz, kalibrovkani shu joyda qiling.

**Aniq ishora.** Buyruq (bosish, ikki marta bosish, scroll, ovoz...) faqat hamma barmoq **aniq** ochiq yoki **aniq** yopiq bo'lganda ishlaydi. Kamera oynasida barmoq uchi: **yashil** = aniq ochiq, **qizil** = aniq yopiq, **sariq (?)** = noaniq (yarim bukilgan); sariq barmoq bilan hech narsa bosilmaydi. Bosh barmoq boshqa barmoqlarga yaqinlashsa ham yashil turadi (faqat mahkam yopilganda qizil bo'ladi). Bosish uchun qo'l qimirlamay turishi kerak, uzoqda esa ishora biroz uzoqroq ushlanishi kerak (shovqin ko'p). Sudrash faqat bosh va ko'rsatkich barmoq avval ochiq turib, keyin tekkanda boshlanadi. Sozlamalarda "Faqat ANIQ ishoraga buyruq", "Bosish uchun qo'l qimirlamay turishi kerak" va "Bosh barmoq 'yopiq' chegarasi" bor.

**Ishoralarni o'rgatish** (`t`): avval kaftni to'liq ochib, keyin mushtni mahkam yumib 1,5 soniyadan ushlab turasiz. Dastur barmoq ochiq/yopiq chegaralarini sizning qo'lingizga moslaydi. Buyruqlar tez-tez almashib ketsa, shuni qiling.

**Qorong'i joy.** Kadr qorong'i bo'lsa, qo'lni topish uchun u avtomatik yoritiladi (kamera oynasida "yoritish xN" yoziladi).

## Muammolar

- **Kamera ochilmadi:** boshqa dastur (Zoom, Teams) kamerani ishlatmayotganini tekshiring; Windows → Sozlamalar → Maxfiylik → Kamera → desktop ilovalarga ruxsat yoqilgan bo'lsin.
- **Qo'l yomon taniladi:** yorug'lik qo'lingizga tushsin (orqangizdan emas). Ishoralarni o'rgating (`t`).
- **Mush/sudrash adashadi:** sozlamalarda "Sudrash sezgirligi" ni o'zgartiring yoki ishoralarni o'rgating.
- **Sozlamalar oynasi ochilmasa:** Python'ni o'rnatishda "tcl/tk" belgilanmagan bo'lishi mumkin; dastur baribir ishlaydi, sozlamalar `settings.json` orqali o'zgaradi.
- **Kutilmagan xato:** `xato.log` fayliga yoziladi; shu fayl matnini yuboring.
- `hand_landmarker.task` — qo'l modeli (7,8 MB), repoda bor. Yo'q bo'lib qolsa dastur uni o'zi yuklab oladi.
