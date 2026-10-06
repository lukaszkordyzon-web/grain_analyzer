"""Streamlit UI:  streamlit run app.py"""
import dataclasses
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent / "src"))

from grain_analyzer.camera import DEFAULT_FOCAL_35MM, read_camera_meta  # noqa: E402
from grain_analyzer.depth import DEPTH_MODELS, DepthEstimator  # noqa: E402
from grain_analyzer.export import grains_csv, summary_csv  # noqa: E402
from grain_analyzer.measure import SIZE_METRICS  # noqa: E402
from grain_analyzer.pipeline import (REFERENCES, DroneParams, Params, analyze,  # noqa: E402
                                     measure_drone, plan_drone, preview_drone,
                                     segment_drone, working_side_for)
from grain_analyzer.scale import ARUCO_DICTS  # noqa: E402
from grain_analyzer.segmentation import available_segmenters, make_segmenter  # noqa: E402
from grain_analyzer.stats import WEIGHTINGS, rr_x50  # noqa: E402
from grain_analyzer.viz import depth_preview, draw_overlay, plot_histogram, plot_psd  # noqa: E402

st.set_page_config(page_title="Analiza ziaren", layout="wide")
st.title("Analiza wielkości ziaren")

def version_info() -> dict:
    """What this deployment is actually running (branch, commit, key packages)."""
    import importlib.metadata as md
    import subprocess

    def pkg(name):
        try:
            return md.version(name)
        except Exception:
            return "brak"

    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=3,
                                  cwd=Path(__file__).parent).stdout.strip() or "?"
        except Exception:
            return "?"

    return {"gałąź (git)": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "--short", "HEAD"),
            "python": sys.version.split()[0], "ultralytics": pkg("ultralytics"),
            "torch": pkg("torch"), "transformers": pkg("transformers"), "streamlit": pkg("streamlit")}


MODE_CLOSE = "Zbliżenie ze znacznikiem ArUco"
MODE_DRONE = "Dron — zdjęcie całej hałdy"


@st.cache_resource(max_entries=1, show_spinner="Ładowanie modelu segmentacji…")
def get_segmenter(key: str):
    return make_segmenter(key)


def get_segmenter_exclusive(key: str):
    """Keep ONE segmentation model in memory: free hosts have ~1 GB, and two models plus torch
    got the app killed. Switching model drops the previous one first."""
    last = st.session_state.get("last_segmenter")
    if last and last != key:
        get_segmenter.clear()
        gc.collect()
    st.session_state.last_segmenter = key
    return get_segmenter(key)


def pick_segmenter(label="Model segmentacji"):
    av = available_segmenters()
    key = st.selectbox(label, list(av), format_func=lambda k: av[k].label
                       + ("" if av[k].tested else " (nieprzetestowany)"))
    if av[key].note:
        st.caption(av[key].note)
    import importlib.util
    if importlib.util.find_spec("ultralytics") is None:
        st.caption("FastSAM, MobileSAM i SAM 2.1 nie są dostępne na tym serwerze: wymagają "
                   "pakietu `ultralytics`, którego tu nie zainstalowano (patrz README).")
    return key


@st.cache_resource(show_spinner="Ładowanie modelu głębi…")
def get_depth(model_id: str) -> DepthEstimator:
    return DepthEstimator(model_id)


def _fmt(v, floor):
    return f"< {floor:.0f}" if not np.isfinite(v) else f"{v:.0f}"


def kuzram_panel(est, res):
    """Calibration of the Kuz-Ram model: correction factors ka (rock factor) and kn (uniformity index).
    Always shown in drone mode; without a usable Rosin-Rammler fit only the model prediction is computed."""
    import pandas as pd
    from grain_analyzer.blast import BlastInputs, calibrate

    fit = est["fit"] if est else None
    x50_mm, n_meas = (rr_x50(fit), fit["n"]) if fit else (None, None)
    with st.expander("Kalibracja modelu odpału Kuz-Ram: współczynniki ka i kn", expanded=est is None):
        if est is None:
            st.warning(
                "Brak zmierzonego rozkładu Rosina–Rammlera, więc ka i kn nie da się policzyć. "
                + ("Wybierz w panelu bocznym ważenie **powierzchniowe (d²)** i uruchom analizę ponownie."
                   if not res.bounds else
                   "Dopasowanie do wiarygodnej części krzywej nie było wiarygodne (R² poniżej 0,9 albo za mało "
                   "punktów). Spróbuj zmniejszyć „Najmniejszy kamień do zmierzenia” albo zmienić model "
                   "segmentacji.")
                + " Możesz już teraz policzyć predykcję modelu z danych odpału.")
        st.caption("Wpisz dane odpału, którego dotyczy zdjęcie. Aplikacja policzy, co przewiduje model "
                   "Kuz-Ram, i porówna z rozkładem Rosina–Rammlera zmierzonym ze zdjęcia: "
                   "**ka = A zmierzone / A przyjęte**, **kn = n zmierzone / n z modelu**. "
                   "Wzory są z literatury (Cunningham); zweryfikuj je z normą u siebie.")
        c1, c2, c3 = st.columns(3)
        D = c1.number_input("Średnica otworu D [mm]", 0.0, 500.0, 0.0, 1.0, key="kr_D")
        B = c1.number_input("Nadkład B [m]", 0.0, 20.0, 0.0, 0.1, key="kr_B")
        S = c1.number_input("Rozstaw S [m]", 0.0, 20.0, 0.0, 0.1, key="kr_S")
        H = c2.number_input("Wysokość ławy H [m]", 0.0, 60.0, 0.0, 0.5, key="kr_H")
        W = c2.number_input("Błąd wiercenia W (odch. std.) [m]", 0.0, 5.0, 0.0, 0.05, key="kr_W")
        Q = c2.number_input("Ładunek na otwór Q [kg]", 0.0, 5000.0, 0.0, 1.0, key="kr_Q")
        E = c3.number_input("Siła względna MW (ANFO = 100)", 1.0, 300.0, 100.0, 1.0, key="kr_E")
        BCL = c3.number_input("Ładunek denny BCL [m]", 0.0, 60.0, 0.0, 0.1, key="kr_BCL")
        CCL = c3.number_input("Ładunek kolumnowy CCL [m]", 0.0, 60.0, 0.0, 0.1, key="kr_CCL")
        A = c3.number_input("Współczynnik skały A (przyjęty)", 0.0, 30.0, 0.0, 0.1, key="kr_A",
                            help="Z opisu górotworu; 0 = nie podano (wtedy policzę tylko A zmierzone).")
        inputs = BlastInputs(B, S, D, H, Q, E, W, BCL, CCL, A)
        out = calibrate(inputs, x50_mm, n_meas)
        # X50 below the measurement limit is an extrapolation: also show the extreme scenario
        # "everything unmeasured is fines" (smaller X50 -> smaller A)
        d50_low = est["D_low"]["D50"] if est else float("nan")
        out_low = calibrate(inputs, d50_low, None) if np.isfinite(d50_low) else None

        def f(v, fmt="{:.2f}"):
            return fmt.format(v) if v is not None else "—"

        rows = [("X50 zmierzone (dopasowanie Rosina–Rammlera) [cm]", f(x50_mm / 10 if x50_mm else None, "{:.1f}")),
                ("n zmierzone (dopasowanie)", f(n_meas)),
                ("Współczynnik ładowania K [kg/m³]", f(out["K"])),
                ("X50 z modelu [cm]", f(out["X50_model_cm"], "{:.1f}")),
                ("n z modelu", f(out["n_model"])),
                ("A zmierzone (wsteczne) [–]", f(out["A_measured"])),
                ("A zmierzone — skrajnie (niezmierzone = sama drobnica) [–]",
                 f(out_low["A_measured"]) if out_low else "—"),
                ("A przyjęte [–]", f(out["A_assumed"])),
                ("ka", f(out["ka"])),
                ("ka — skrajnie", f(out_low["ka"]) if out_low else "—"),
                ("kn", f(out["kn"]))]
        table = pd.DataFrame(rows, columns=["Wielkość", "Wartość"]).set_index("Wielkość")
        show_table(table)
        if out["ka"] is None and out["kn"] is None:
            st.info("Uzupełnij dane odpału (co najmniej D, B, S, H, Q oraz ładunki BCL/CCL dla kn; "
                    "dla ka także współczynnik A).")
        if est and x50_mm < est["floor_mm"]:
            st.warning(f"X50 z dopasowania ({x50_mm:.0f} mm) leży poniżej zasięgu wiarygodnej ekstrapolacji "
                       f"({est['floor_mm']:.0f} mm): ka jest niepewne.")
        if out["ka"] is not None and 0 < A < 3:
            st.caption(f"ka zależy wprost od wpisanego A (tu {A:.1f}): wpisz współczynnik skały z opisu "
                       "górotworu, inaczej ka nie ma sensu. Samo A zmierzone nie zależy od tego pola.")
        if W == 0:
            st.caption("W = 0 oznacza idealne wiercenie. W praktyce błąd 0,1–0,3 m obniża n z modelu, "
                       "a więc podnosi kn.")
        st.caption("Uwaga: zdjęcie pokazuje powierzchnię hałdy, więc X50 bywa zawyżone, a n jest wrażliwe na "
                   "zakres dopasowania i model segmentacji. Współczynniki z jednego odpału są orientacyjne; "
                   "rzetelniejsza kalibracja wymaga kilkunastu odpałów w tych samych warunkach.")
        st.download_button("Pobierz kalibrację (CSV)", table.to_csv().encode("utf-8-sig"),
                           "kalibracja_kuzram.csv", "text/csv")


def comparison_table():
    """Side-by-side summary of all analysed photos (2+), with one CSV."""
    import pandas as pd
    done = [(k, v[0]) for k, v in st.session_state.get("results", {}).items() if k[0] == MODE_DRONE]
    if len(done) < 2:
        return
    rows = []
    for (_, name, _), r in done:
        est = r.estimate
        D = est["D"] if est else {}
        fit = est["fit"] if est else None
        rows.append({
            "Zdjęcie": name,
            "D10 [mm]": D.get("D10", r.percentiles.get("D10")),
            "D50 [mm]": D.get("D50", r.percentiles.get("D50")),
            "D90 [mm]": D.get("D90", r.percentiles.get("D90")),
            "X50 RR [mm]": rr_x50(fit) if fit else None,
            "n RR": fit["n"] if fit else None,
            "R²": fit["r2"] if fit else None,
            "dopasowano od [cm]": (r.reliable_mm / 10) if (est and r.reliable_mm) else None,
            "kamieni": len(r.grains),
        })
    df = pd.DataFrame(rows).set_index("Zdjęcie")
    with st.expander(f"Porównanie zdjęć ({len(done)})", expanded=True):
        show_table(df.round(2))
        st.caption("Aby porównanie było uczciwe, ustaw w „Zaawansowane” ten sam próg dopasowania "
                   "Rosina–Rammlera dla wszystkich zdjęć i przelicz je ponownie.")
        st.download_button("Pobierz porównanie (CSV)", df.round(3).to_csv().encode("utf-8-sig"),
                           "porownanie.csv", "text/csv")


def show_table(df):
    """Mała tabela z jawną wysokością — automatyczna ucina ostatnie wiersze."""
    st.dataframe(df, width="content", height=(len(df) + 1) * 35 + 3)


def results_view(res, label, wt, bins):
    if res.grains.empty:
        st.warning("Nie wykryto ziaren — zmniejsz minimalny rozmiar lub zmień ustawienia.")
        return
    est = res.estimate
    heads = est["D"] if est else res.percentiles
    c = st.columns(5)
    c[0].metric("Zmierzone kamienie", len(res.grains))

    def d_text(k, v):
        if not est:
            return f"{v:.1f} mm"
        return f"{v:.0f} mm" if np.isfinite(v) else f"< {est['floor_mm']:.0f} mm"

    for col, (k, v) in zip(c[1:4], heads.items()):
        col.metric(k + (" (szacunek)" if est else ""), d_text(k, v))
    c[4].metric("Skala (mediana)", f"{res.scale.mm_per_px:.3f} mm/px", res.scale.method,
                delta_color="off")
    if est:
        lo = est["D_low"]
        extreme = ", ".join(f"{k} ≥ {v:.0f} mm" for k, v in lo.items() if np.isfinite(v)
                            and abs(v - est["D"][k]) > 0.03 * v)
        st.caption("D10/D50/D90 to **szacunek dla całej powierzchni hałdy**, w tym drobnicy, której nie "
                   "widać na zdjęciu: rozkład Rosina–Rammlera dopasowany do wiarygodnej części krzywej "
                   f"(kamienie ≥ {est['reliable_mm'] / 10 if est['reliable_mm'] else res.min_size_mm / 10:.0f} cm; "
                   f"R² = {est['fit']['r2']:.2f}) i przedłużony w dół. Zakłada, że rozkład ma kształt "
                   "Rosina–Rammlera w całym zakresie, więc kamienie, których brakuje tuż nad progiem, "
                   "są niewykryte, a nie drobniejsze."
                   + (f" W skrajnym przypadku (całe niezmierzone to sama drobnica): {extreme}." if extreme else "")
                   + f" Wartości poniżej {est['floor_mm']:.0f} mm nie są podawane (zbyt daleka ekstrapolacja).")
    if res.n_rejected_depth:
        st.caption(f"Odrzucono wg głębi: {res.n_rejected_depth}")
    for n in res.notes:
        st.warning(n)
    if res.fractions:
        import pandas as pd
        st.markdown("**Skład powierzchni według frakcji** (udział analizowanego obszaru)")
        show_table(pd.DataFrame({"Frakcja": [r["label"] for r in res.fractions],
                                   "Udział powierzchni": [f"{r['fraction']:.0%}" for r in res.fractions]}
                                  ).set_index("Frakcja"))
    if est:
        import pandas as pd
        f = est["fit"]
        with st.expander("Parametry rozkładu Rosina–Rammlera (do kalibracji modelu odpału)"):
            show_table(pd.DataFrame({
                "Parametr": ["x_c — rozmiar charakterystyczny (63,2% przechodzi) [mm]",
                             "n — wskaźnik jednorodności [–]",
                             "X50 = x_c·(ln 2)^(1/n) [mm]",
                             "R² dopasowania", "dopasowano do kamieni ≥ [cm]", "liczba punktów"],
                "Wartość": [f"{f['xc']:.0f}", f"{f['n']:.2f}", f"{rr_x50(f):.0f}", f"{f['r2']:.3f}",
                            f"{(est['reliable_mm'] or res.min_size_mm) / 10:.0f}", f"{f['n_points']}"]}
            ).set_index("Parametr"))
            st.caption("Parametry dotyczą **powierzchni** hałdy widocznej ze zdjęcia, więc X50 bywa zawyżone "
                       "(grubsze kamienie na wierzchu, drobniejsze ukryte). n jest wrażliwe na zakres "
                       "dopasowania i wybór modelu segmentacji.")
    if res.annotations:                                   # drone mode
        kuzram_panel(est, res)
    if res.bounds:
        import pandas as pd
        with st.expander("Szczegóły: zmierzone kamienie i granice niepewności", expanded=not est):
            u = res.bounds["unmeasured_fraction"]
            st.markdown(
                f"**{u:.0%} powierzchni to materiał poniżej progu pomiaru (< {res.min_size_mm:.0f} mm), "
                "szczeliny, cień lub niewykryte kamienie.** Prawdziwe D leży między granicą dolną "
                "(niezmierzone pominięte — tylko zmierzone kamienie) a górną (całe niezmierzone "
                "to drobnica)." + (" Szacunek wykorzystuje górną granicę jako punkt odniesienia."
                                   if est else " Dopasowanie rozkładu nie było wiarygodne, więc "
                                   "szacunku dla drobnicy nie podaję."))
            d_lo, d_up = res.bounds["D_lower"], res.bounds["D_upper"]
            table = {"D": list(d_lo), "dolna granica [mm]": [f"{v:.0f}" for v in d_lo.values()],
                     "górna granica [mm]": [_fmt(d_up[k], res.min_size_mm) for k in d_lo]}
            if est:
                table["szacunek (Rosin–Rammler) [mm]"] = [_fmt(est["D"][k], est["floor_mm"]) for k in d_lo]
                table["skrajnie: sama drobnica [mm]"] = [_fmt(est["D_low"][k], est["floor_mm"]) for k in d_lo]
            show_table(pd.DataFrame(table).set_index("D"))

    t1, t2, t3 = st.tabs(["Kontury", "Krzywa uziarnienia", "Dane"])
    with t1:
        c1, c2, c3 = st.columns([1, 1, 2])
        ids = c1.checkbox("Numery ziaren")
        by_size = c2.checkbox("Kolor wg rozmiaru", True,
                              help="Zielony = małe kamienie, czerwony = duże (skala logarytmiczna).")
        thick = c3.slider("Grubość konturu", 1, 8, 3)
        st.image(draw_overlay(res, ids, thick, by_size), width="stretch")
        if res.depth is not None and not res.annotations:
            with st.expander("Mapa głębi (względna)"):
                st.image(depth_preview(res.depth), width="stretch")
    with t2:
        log_x = st.checkbox("Skala logarytmiczna osi X", True)
        st.pyplot(plot_psd(res, label, log_x))
        if res.bounds:
            st.caption("**Jak czytać:** to jedna krzywa dla całej powierzchni hałdy. Niebieska część to "
                       "zmierzone kamienie; na granicy pomiaru zaczyna się od udziału tego, czego nie "
                       "zmierzono (traktowanego jako drobnica). Pomarańczowa to jej szacowany ciąg dalszy "
                       "poniżej granicy. Jasny pas to niepewność: dolna linia oznacza, że niezmierzony "
                       "obszar to w rzeczywistości szczeliny i cień, a nie drobnica.")
        with st.expander("Histogram częstości (pomocniczy)"):
            st.pyplot(plot_histogram(res.sizes, res.percentiles, label, wt, bins))
    with t3:
        st.dataframe(res.grains.round(3), width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("Pobierz ziarna (CSV)", grains_csv(res), "ziarna.csv", "text/csv")
        d2.download_button("Pobierz podsumowanie (CSV)", summary_csv(res, label, wt),
                           "podsumowanie.csv", "text/csv")


with st.sidebar:
    from grain_analyzer.device import describe_device
    st.caption(f"Obliczenia: {describe_device()}")
    with st.expander("Informacje o wersji"):
        st.json(version_info())
    mode = st.radio("Tryb", [MODE_CLOSE, MODE_DRONE])

uploads = st.file_uploader("Wgraj zdjęcie (może być kilka)", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
                           accept_multiple_files=True)
if not uploads:
    st.info("Wgraj zdjęcie, aby rozpocząć. Dla zdjęć z drona użyj oryginału (z metadanymi).")
    st.stop()
if len(uploads) > 1:
    names = [u.name for u in uploads]
    upload = uploads[st.radio("Które zdjęcie analizujesz?", range(len(uploads)), horizontal=True,
                              format_func=lambda i: names[i] if names.count(names[i]) == 1
                              else f"{i + 1}. {names[i]}")]
else:
    upload = uploads[0]
FILE_KEY = (mode, upload.name, upload.size)


def set_result(res, meta):
    """Results are kept per photo (light objects, no masks), so photos can be compared."""
    st.session_state.setdefault("results", {})[FILE_KEY] = (res, meta)


raw = upload.getvalue()
image = np.array(ImageOps.exif_transpose(Image.open(upload)).convert("RGB"))
H, W = image.shape[:2]

# =========================================================================== close-up mode
if mode == MODE_CLOSE:
    with st.sidebar:
        st.header("Skala")
        manual = st.checkbox("Podaj skalę ręcznie (zamiast znacznika)")
        if manual:
            mm_per_px = st.number_input("mm na piksel (oryginalne zdjęcie)", 0.001, 10.0, 0.05,
                                        format="%.4f")
            marker_mm, marker_dict, marker_id = 20.0, "4x4_50", None
        else:
            mm_per_px = None
            marker_mm = st.number_input("Bok znacznika ArUco [mm]", 1.0, 500.0, 20.0)
            marker_dict = st.selectbox("Słownik ArUco", list(ARUCO_DICTS))
            marker_id = st.number_input("ID znacznika (-1 = dowolny)", -1, 999, -1)
            marker_id = None if marker_id < 0 else int(marker_id)

        st.header("Modele")
        sam_name = pick_segmenter("Segmentacja")
        use_depth = st.checkbox("Użyj głębi (Depth Anything V2) — więcej RAM", False)
        depth_name = st.selectbox("Model głębi", list(DEPTH_MODELS), disabled=not use_depth)
        reject = st.checkbox("Odrzucaj obiekty z tła (wg głębi)", True, disabled=not use_depth)

        st.header("Filtrowanie masek")
        max_side = st.slider("Maks. rozdzielczość [px]", 800, 3000, 1200, 100)
        min_area = st.slider("Min. pole ziarna [px]", 20, 2000, 150)
        max_frac = st.slider("Maks. pole ziarna [% kadru]", 0.5, 30.0, 5.0) / 100

        st.header("Statystyki")
        metric = st.selectbox("Miara wielkości", list(SIZE_METRICS), format_func=SIZE_METRICS.get)
        weighting = st.radio("Ważenie krzywej i D10/D50/D90", list(WEIGHTINGS), index=0,
                             format_func=WEIGHTINGS.get)
        bins = st.slider("Liczba przedziałów histogramu pomocniczego", 5, 60, 20)

    st.image(image, caption=f"{W}×{H} px", width=400)
    if st.button("Analizuj", type="primary"):
        import gc
        gc.collect()
        params = Params(max_side=max_side, marker_size_mm=marker_mm, marker_dict=marker_dict,
                        marker_id=marker_id, manual_mm_per_px=mm_per_px, min_area_px=min_area,
                        max_area_frac=max_frac, use_depth=use_depth,
                        depth_reject_sigma=3.0 if (use_depth and reject) else None,
                        size_metric=metric, weighting=weighting)
        try:
            with st.spinner("Segmentacja… (na CPU może potrwać minutę)"):
                set_result(analyze(
                    image, params, get_segmenter_exclusive(sam_name),
                    get_depth(DEPTH_MODELS[depth_name]) if use_depth else None),
                    (SIZE_METRICS[metric], weighting, bins))
        except ValueError as e:
            st.error(str(e))
            st.stop()

# =========================================================================== drone mode
else:
    from streamlit_image_coordinates import streamlit_image_coordinates

    meta = read_camera_meta(raw)
    with st.sidebar:
        st.header("Skala zdjęcia")
        ref = st.radio("Skąd wziąć skalę?", list(REFERENCES), format_func=REFERENCES.get,
                       help="Zaznaczasz na zdjęciu coś o znanym rozmiarze. Człowiek i ściana "
                            "wymagają kliknięcia dwóch punktów; znacznik jest wykrywany sam.")
        person_h, wall_h, wall_slope, marker_mm, marker_dict = 1.75, 10.0, 90.0, 200.0, "4x4_50"
        if ref == "person":
            person_h = st.number_input("Wzrost człowieka w kadrze [m]", 1.2, 2.2, 1.75, 0.05)
        elif ref == "wall":
            wall_h = st.number_input("Wysokość ściany [m]", 1.0, 100.0, 10.0, 0.5)
            wall_slope = {"Pionowa": 90.0, "Lekko pochylona (ok. 80°)": 80.0,
                          "Mocno pochylona (ok. 70°)": 70.0}[
                st.selectbox("Nachylenie ściany", ["Pionowa", "Lekko pochylona (ok. 80°)",
                                                   "Mocno pochylona (ok. 70°)"],
                             help="Ściana pochylona od kamery daje inną skalę niż pionowa.")]
        else:
            marker_mm = st.number_input("Bok znacznika ArUco [mm]", 20.0, 5000.0, 200.0)
            marker_dict = st.selectbox("Słownik ArUco", list(ARUCO_DICTS))
        st.header("Kamera")
        pitch0 = abs(meta.gimbal_pitch_deg) if meta.gimbal_pitch_deg is not None else 45.0
        pitch = st.number_input(
            "Kąt kamery w dół [°] (90 = prosto w dół)", 5.0, 90.0, float(pitch0), 1.0,
            help="Z metadanych DJI (GimbalPitchDegree); jeśli ich brak, wpisz z aplikacji DJI Pilot.")
        st.caption("✓ z metadanych zdjęcia" if meta.gimbal_pitch_deg is not None
                   else "⚠ brak w metadanych — wpisz ręcznie")
        focal = st.number_input("Ogniskowa ekw. 35 mm [mm]", 8.0, 200.0,
                                float(meta.focal_35mm or DEFAULT_FOCAL_35MM), 1.0)
        st.caption("✓ z EXIF" if meta.focal_35mm else
                   f"⚠ brak w EXIF — domyślnie {DEFAULT_FOCAL_35MM:.0f} mm (kamera szerokokątna Matrice)")

        st.header("Jak drobne kamienie mierzyć?")
        target_cm = st.select_slider(
            "Najmniejszy kamień do zmierzenia [cm]", [10, 15, 20, 25, 30, 40, 50, 60, 80], value=40,
            help="Aplikacja sama dobiera rozdzielczość. Drobniejsze kamienie = dużo więcej "
                 "obliczeń i wymagają zdjęcia o odpowiednio wysokiej rozdzielczości.")
        with st.expander("Zaawansowane"):
            sam_name = pick_segmenter()
            spec = available_segmenters()[sam_name]
            tile = st.slider("Rozmiar kafelka [px]", 480, 1280, spec.tile, 32, key=f"tile_{sam_name}",
                             help="SAM pracuje natywnie na 1024 px. Mniejszy kafelek = więcej kafelków "
                                  "(dłużej), ale drobniejsze kamienie i mniej pamięci na kafelek. Dla "
                                  "FastSAM domyślnie 640 px.")
            min_d = st.slider("Min. średnica kamienia [px]", 6, 60, 12,
                              help="Mniejsze obiekty nie są liczone (nierozróżnialne).")
            max_frac = st.slider("Maks. pole kamienia [% obszaru]", 0.2, 10.0, 2.0) / 100
            shadow_on = st.checkbox("Odrzucaj cienie (ciemne, jednolite plamy)", True,
                                    help="Rzucany cień obok głazu bywa obrysowany jak kamień. "
                                         "Ciemny kamień w zacienionym miejscu zostaje.")
            fit_cm = st.number_input(
                "Próg dopasowania Rosina–Rammlera [cm] (0 = automatycznie)", 0, 200, 0, 5,
                help="Rozkład Rosina–Rammlera jest dopasowywany tylko do kamieni większych niż ten próg. "
                     "Ustaw ten sam próg (np. 25 cm) dla wszystkich zdjęć, aby wyniki były porównywalne.")
            max_stone_m = st.number_input("Maks. rozmiar kamienia [m]", 0.5, 20.0, 3.0, 0.5,
                                          help="Większe „kamienie” to zwykle maski cienia lub ściany.")

        st.header("Statystyki")
        metric = st.selectbox("Miara wielkości", list(SIZE_METRICS), format_func=SIZE_METRICS.get)
        weighting = st.radio("Ważenie krzywej i D10/D50/D90", list(WEIGHTINGS), index=1,
                             format_func=WEIGHTINGS.get,
                             help="Dla zdjęć z góry standardem jest udział powierzchni. "
                                  "Drobnica (frakcje, szacunek, granice) jest liczona tylko dla tego ważenia.")
        bins = st.slider("Liczba przedziałów histogramu pomocniczego", 5, 60, 20)

    # ---- click-to-mark -------------------------------------------------------------
    up_id = (upload.name, upload.size)
    if st.session_state.get("up_id") != up_id:     # another photo: its own points; heavy cache dropped
        st.session_state.update(up_id=up_id, last_click=None, _next_step=0)
        st.session_state.pop("seg_cache", None)
    pts = st.session_state.setdefault("pts_by", {}).setdefault(up_id, {})
    ROI_STEPS = [("roi0", "Obszar hałdy: lewy górny róg"), ("roi1", "Obszar hałdy: prawy dolny róg")]
    STEPS = {
        "person": [("head", "Czubek głowy człowieka"), ("feet", "Stopy człowieka")] + ROI_STEPS,
        "wall": [("head", "Górna krawędź ściany"),
                 ("feet", "Dolna krawędź ściany (pod górną, w tym samym pionie)")] + ROI_STEPS,
        "marker": ROI_STEPS,
    }[ref]
    STEPS = [(k, f"{i + 1}. {lab}") for i, (k, lab) in enumerate(STEPS)]
    if st.session_state.get("ref_prev") != ref:          # other reference -> its own points
        st.session_state.ref_prev = ref
        pts.pop("head", None)
        pts.pop("feet", None)
        st.session_state._next_step = 0
    if "_next_step" in st.session_state:                  # auto-advance after a click
        st.session_state.step_radio = min(st.session_state.pop("_next_step"), len(STEPS) - 1)
    if st.session_state.get("step_radio", 0) >= len(STEPS):
        st.session_state.step_radio = 0
    step = st.radio("Co teraz klikasz na zdjęciu?", range(len(STEPS)), horizontal=True,
                    format_func=lambda i: STEPS[i][1], key="step_radio")

    DW = min(1000, W)
    sc = DW / W
    prev = cv2.resize(image, (DW, round(H * sc)), interpolation=cv2.INTER_AREA)
    for key, col in (("head", (255, 0, 255)), ("feet", (255, 0, 255)),
                     ("roi0", (255, 160, 0)), ("roi1", (255, 160, 0))):
        if key in pts:
            cv2.circle(prev, (int(pts[key][0] * sc), int(pts[key][1] * sc)), 6, col, 2)
    if "head" in pts and "feet" in pts:
        cv2.line(prev, tuple(int(v * sc) for v in pts["head"]),
                 tuple(int(v * sc) for v in pts["feet"]), (255, 0, 255), 2)
    if "roi0" in pts and "roi1" in pts:
        cv2.rectangle(prev, tuple(int(v * sc) for v in pts["roi0"]),
                      tuple(int(v * sc) for v in pts["roi1"]), (255, 160, 0), 2)

    click = streamlit_image_coordinates(Image.fromarray(prev), key="img_click", width=DW)
    if click and (click["x"], click["y"]) != st.session_state.last_click:
        st.session_state.last_click = (click["x"], click["y"])
        pts[STEPS[step][0]] = (click["x"] / sc, click["y"] / sc)
        if step < len(STEPS) - 1:
            st.session_state._next_step = step + 1
        st.rerun()
    st.caption({
        "person": "Człowiek jest tylko linijką skali: kliknij dokładnie czubek głowy i stopy. ",
        "wall": "Kliknij górną i dolną krawędź ściany jedną pionową linią; dół musi leżeć na "
                "tym samym terenie co hałda. ",
        "marker": "Znacznik zostanie znaleziony automatycznie. ",
    }[ref] + "Obszar hałdy zawęża analizę do kamieni (bez ścian, kałuż i podłoża).")

    needed = [k for k, _ in STEPS]
    ready = all(k in pts for k in needed)
    st.caption(f"Zdjęcie: {W}×{H} px")
    max_side, _roi, app_key = 2000, None, None
    base = DroneParams(reference=ref, person_height_m=person_h, wall_height_m=wall_h,
                       wall_slope_deg=wall_slope, marker_size_mm=marker_mm, marker_dict=marker_dict,
                       pitch_deg=pitch, focal_35mm=focal, tile=tile, min_diameter_px=min_d,
                       max_area_frac=max_frac, size_metric=metric, weighting=weighting,
                       shadow_ratio=0.6 if shadow_on else 0.0, fit_from_mm=fit_cm * 10.0, max_stone_mm=max_stone_m * 1000,
                       max_tiles=max(DroneParams().max_tiles, spec.max_tiles))
    if "roi0" in pts and "roi1" in pts:
        _roi = (min(pts["roi0"][0], pts["roi1"][0]), min(pts["roi0"][1], pts["roi1"][1]),
                max(pts["roi0"][0], pts["roi1"][0]), max(pts["roi0"][1], pts["roi1"][1]))
        pv = preview_drone((H, W), base, pts.get("head"), pts.get("feet"), _roi, image)
        if pv is None:
            st.info("Zaznacz wszystkie punkty, aby zobaczyć, jak drobne kamienie da się zmierzyć.")
        else:
            long_side = max(H, W)
            s_orig = pv["mm_per_px_orig"]

            def plan_for(t_cm):
                need = working_side_for(t_cm * 10, s_orig, long_side, min_d)
                side = int(min(max(need, 1000), long_side))
                n, _ = plan_drone((H, W), _roi, dataclasses.replace(base, max_side=side))
                return need, side, n

            need, max_side, n_t = plan_for(target_cm)
            # The cache is tied to what the user chose (target size, area, tiling), NOT to the
            # derived resolution: that depends on the scale and would change with every tweak
            # of the reference, defeating the point of caching.
            app_key = (target_cm, tile, min_d, round(max_frac, 5),
                       tuple(round(v) for v in _roi), sam_name)
            cache = st.session_state.get("seg_cache")
            cached = bool(cache and cache[0] == app_key)
            if cached:
                max_side = cache[2]
                n_t, _ = plan_drone((H, W), _roi, dataclasses.replace(base, max_side=max_side))
            finest_cm = min_d * s_orig / 10
            if need > long_side:
                st.warning(f"To zdjęcie ma za małą rozdzielczość, by mierzyć kamienie od {target_cm} cm "
                           f"(przy pełnej rozdzielczości najmniejszy mierzony kamień to ok. "
                           f"{finest_cm:.0f} cm). Użyję pełnej rozdzielczości.")
            achieved_cm = min_d * s_orig * long_side / max_side / 10
            sec = st.session_state.get("sec_per_tile")
            msg = (f"Najmniejszy mierzony kamień ≈ **{achieved_cm:.0f} cm** · rozdzielczość robocza "
                   f"{max_side} px · liczba kafelków: **{n_t}** (limit {base.max_tiles})"
                   + (f" · ok. {n_t * sec / 60:.1f} min (wg poprzedniej analizy)" if sec and not cached else "")
                   + ".")
            if cached:
                msg += " Segmentacja jest już policzona — zmiana skali i statystyk jest natychmiastowa."
            if n_t > base.max_tiles:
                fits = next((t for t in (10, 15, 20, 25, 30, 40, 50, 60, 80)
                             if plan_for(t)[2] <= base.max_tiles), None)
                st.warning(msg + " To za dużo dla tego serwera. "
                           + (f"Wybierz kamienie od ≥ {fits} cm" if fits else "Zaznacz mniejszy obszar")
                           + " albo zaznacz mniejszy obszar hałdy.")
                ready = False
            else:
                st.caption(msg)
    else:
        ready = False

    if st.button("Analizuj", type="primary", disabled=not ready):
        import gc
        p = dataclasses.replace(base, max_side=max_side)
        cache = st.session_state.get("seg_cache")
        bar = st.progress(0.0, "Start…")
        try:
            if cache and cache[0] == app_key:
                seg = cache[1]
            else:
                st.session_state.pop("seg_cache", None)
                gc.collect()
                seg = segment_drone(image, p, get_segmenter_exclusive(sam_name), _roi,
                                    progress=lambda f, t: bar.progress(f, t), model=sam_name)
                st.session_state.seg_cache = (app_key, seg, max_side)
                st.session_state.sec_per_tile = seg.seconds / max(seg.n_tiles, 1)
            set_result(measure_drone(seg, p, pts.get("head"), pts.get("feet")),
                       (SIZE_METRICS[metric], weighting, bins))
        except ValueError as e:
            st.error(str(e))
            st.stop()
        finally:
            bar.empty()

comparison_table()
_stored = st.session_state.get("results", {}).get(FILE_KEY)
if _stored is not None:
    res, (label, wt, bins) = _stored
    results_view(res, label, wt, bins)
