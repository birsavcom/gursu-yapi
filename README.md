# Gursu Yapi Degisim Tespiti

Gursu ilcesi icin Esri World Imagery Wayback kaynakli 512x512 uydu goruntusu indirme, 2021-2026 yeni yapi adayi tespiti ve harita uzerinde inceleme projesi.

## Calisma Alani

- BBox lon: 29.131191 -> 29.306497
- BBox lat: 40.198367 -> 40.339645
- Zoom: 18
- Goruntu boyutu: 512x512
- Beklenen ham goruntu sayisi: 4420 / yil
- Karsilastirma: 2021 -> 2026

## GitHub Icerigi

- index.html: Harita ve popup arayuzu
- data/imarsiz-gursu.geojson: Imarsiz/plansiz alan poligonlari
- data/ruhsatli-yapi-parseller-gursu.geojson: Yapi ruhsati bulunan parseller
- results/gursu_change_verified_segmentation_2021_2026.jsonl: Nokta ve koordinat kayitlari
- results/masks_segmentation_verified_2021_2026/: Popup icin before / after / detected uclu gorseller
- build_gursu_html.py: JSONL sonuclarindan haritayi yeniden uretir
- gursu_change_detection.py: 2021-2026 yapi fark tespit scripti
- download_gursu_wayback.py: Esri Wayback goruntu indirme scripti

## GitHub Disinda Tutulanlar

- dataset/: Ham uydu goruntuleri buyuk oldugu icin repoya eklenmez
- models/*.pt: Model dosyalari GitHub 100 MB sinirini astigi icin repoya eklenmez
- logs/: Calisma loglari yereldedir

## Imar Siniflandirmasi

Haritadaki tespit noktalari `data/ruhsatli-yapi-parseller-gursu.geojson` ve `data/imarsiz-gursu.geojson` kaynaklarina gore siniflandirilir. Oncelik sirasi ruhsatli parsel, imarsiz/plansiz alan, ardindan yapi farki seklindedir.

- Yapi Ruhsatli: Tespit noktasi yapi ruhsati bulunan parsel icindedir ve haritada kahverengi nokta ile gosterilir.
- Kacak Yapi: Tespit noktasi ruhsatli parsel disinda, imarsiz/plansiz alan poligonu icindedir ve haritada kirmizi nokta ile gosterilir.
- Yapi Farki: Tespit noktasi ruhsatli veya imarsiz alana denk gelmez; imarli/yapi farki olarak haritada sari nokta ile gosterilir.

`maks-ruhsat.json` ruhsat kayitlarini MAKS `kimlik_no` ve ada/parsel bilgileriyle tutar; dosyada dogrudan koordinat yoktur. `data/ruhsatli-yapi-parseller-gursu.geojson`, yapi ruhsati kayitlarindaki ada/parsel bilgilerinin GeoServer `adaparselv1` parselleriyle eslestirilmesiyle uretilmistir.
