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

## Tryb „Dron” (hałdy urobku, zdjęcia ukośne)
Cała hałda na jednym zdjęciu (np. DJI Matrice 3D/4D). Skala musi wynikać z czegoś o znanym
rozmiarze w kadrze — do wyboru w panelu:
- **człowiek** (domyślnie 1,75 m): klikasz czubek głowy i stopy,
- **ściana o znanej wysokości**: klikasz górną i dolną krawędź (jedną pionową linią), podajesz wysokość
  i nachylenie ściany (pochylona od kamery daje inną skalę niż pionowa),
- **znacznik ArUco na ziemi**: wykrywany automatycznie.

Wszystkie trzy sprowadzają się do wysokości kamery nad terenem; z niej, z kąta kamery i ogniskowej
(metadane DJI w oryginalnym pliku, inaczej ręcznie) liczona jest skala dla każdego wiersza zdjęcia.
Dalej: klikasz prostokąt hałdy; SAM działa na kafelkach; pomiar z lokalną skalą.

**Drobnica** (kamienie mniejsze niż próg pomiaru) nie jest widoczna na zdjęciu, więc wynik ma 3 warstwy:
1. *Skład powierzchni według frakcji* — zmierzone udziały klas + jawny wiersz „niezmierzone”.
2. *Granice niepewności* — dolna (niezmierzone pominięte) i górna (całe niezmierzone = drobnica).
3. *Szacunek dla całej hałdy* — dopasowanie rozkładu Rosina–Rammlera do górnej krzywej i ekstrapolacja
   poniżej progu (oznaczone jako szacunek; podane R²; przy złym dopasowaniu aplikacja go nie podaje).

Uwagi: skala zakłada, że hałda leży w płaszczyźnie terenu punktu odniesienia (wyższe partie wychodzą
lekko zawyżone). Człowiek/znacznik < ~15 px daje mało dokładną kalibrację. Wynik jest orientacyjny.

## Jak zrobić zdjęcie (tryb ze znacznikiem)
Wydrukuj znacznik ArUco (domyślnie słownik 4x4_50), zmierz **bok czarnego kwadratu** w mm i
wpisz go w panelu. Połóż go w tej samej płaszczyźnie co ziarna, kamera możliwie prostopadle do podłoża.

## Układ
```
app.py                       UI
src/grain_analyzer/
  scale.py                   detekcja ArUco → mm/px
  camera.py / drone.py       metadane DJI, skala z człowieka, kafelkowanie, pomiar z lokalną skalą
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
