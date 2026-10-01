# Grain Analyzer

Analiza wielkości ziaren ze zdjęcia — **bez trenowania**, na gotowych modelach:

| Etap | Narzędzie |
|---|---|
| Segmentacja ziaren | SAM (`facebook/sam-vit-*`, HF `mask-generation`) |
| Głębia | Depth Anything V2 (`depth-anything/Depth-Anything-V2-*-hf`) |
| Skala px→mm | znacznik ArUco o znanym boku (lub ręcznie mm/px) |
| UI | Streamlit: kontury, histogram, D10/D50/D90, eksport CSV |

## Uruchomienie
```bash
pip install -r requirements.txt
streamlit run app.py
pytest            # (pip install -r requirements-dev.txt) testy bez modeli (syntetyczna scena + ArUco)
```
Modele pobierają się z Hugging Face przy pierwszym uruchomieniu (SAM ViT-B ≈ 375 MB).

## Jak zrobić zdjęcie
Wydrukuj znacznik ArUco (domyślnie słownik 4x4_50), zmierz **bok czarnego kwadratu** w mm i
wpisz go w panelu. Połóż go w tej samej płaszczyźnie co ziarna, kamera możliwie prostopadle do podłoża.

## Układ
```
app.py                       UI
src/grain_analyzer/
  scale.py                   detekcja ArUco → mm/px
  segmentation.py            SAM + selekcja masek (dedup, rozmiar, wykluczenie znacznika)
  depth.py                   Depth Anything V2 (mapa względna 0–1)
  measure.py                 ECD, Feret min/max, elipsa, kołowość, cechy głębi
  stats.py                   D10/D50/D90 (liczbowe lub objętościowe ~d³), histogram
  pipeline.py                analyze(): całość, modele wstrzykiwane (łatwe testy)
  viz.py / export.py         nakładka konturów, histogram, CSV
tests/
```

## Ograniczenia (świadome)
- Depth Anything V2 daje głębię **względną**, nie metryczną — nie przelicza px→mm. Służy do
  odrzucania obiektów z tła (robust z-score) i kolumny `rel_height` (bez jednostki).
- Pomiar jest 2D (rzut ziarna); D-percentyle „objętościowe” zakładają podobny kształt i gęstość
  ziaren, więc nie są wprost równoważne analizie sitowej.
- Skala jest jedna na cały kadr — działa przy płaskiej warstwie ziaren i kamerze prostopadłej.
- Gęsto stykające się ziarna mogą być sklejone przez SAM; sprawdź podgląd konturów.
