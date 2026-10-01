"""Streamlit UI:  streamlit run app.py"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent / "src"))

from grain_analyzer.camera import DEFAULT_FOCAL_35MM, read_camera_meta  # noqa: E402
from grain_analyzer.depth import DEPTH_MODELS, METRIC_DEPTH_MODELS, DepthEstimator  # noqa: E402
from grain_analyzer.export import grains_csv, summary_csv  # noqa: E402
from grain_analyzer.measure import SIZE_METRICS  # noqa: E402
from grain_analyzer.pipeline import DroneParams, Params, analyze, analyze_drone  # noqa: E402
from grain_analyzer.scale import ARUCO_DICTS  # noqa: E402
from grain_analyzer.segmentation import SAM_MODELS, SamSegmenter  # noqa: E402
from grain_analyzer.viz import depth_preview, draw_overlay, plot_histogram  # noqa: E402

st.set_page_config(page_title="Analiza ziaren", layout="wide")
st.title("Analiza wielkości ziaren")

MODE_CLOSE = "Zbliżenie ze znacznikiem ArUco"
MODE_DRONE = "Dron — człowiek w kadrze jako skala"


@st.cache_resource(show_spinner="Ładowanie modelu SAM…")
def get_segmenter(model_id: str) -> SamSegmenter:
    return SamSegmenter(model_id)


@st.cache_resource(show_spinner="Ładowanie modelu głębi…")
def get_depth(model_id: str, metric: bool = False) -> DepthEstimator:
    return DepthEstimator(model_id, metric=metric)


def results_view(res, label, wt, bins):
    if res.grains.empty:
        st.warning("Nie wykryto ziaren — zmniejsz minimalny rozmiar lub zmień ustawienia.")
        return
    c = st.columns(5)
    c[0].metric("Ziarna", len(res.grains))
    for col, (k, v) in zip(c[1:4], res.percentiles.items()):
        col.metric(k, f"{v:.1f} mm")
    c[4].metric("Skala (mediana)", f"{res.scale.mm_per_px:.3f} mm/px", res.scale.method,
                delta_color="off")
    if res.n_rejected_depth:
        st.caption(f"Odrzucono wg głębi: {res.n_rejected_depth}")
    for n in res.notes:
        st.warning(n)
    t1, t2, t3 = st.tabs(["Kontury", "Histogram", "Dane"])
    with t1:
        ids = st.checkbox("Numery ziaren")
        st.image(draw_overlay(res, ids), width="stretch")
        if res.depth is not None and res.scale.method != "person":
            with st.expander("Mapa głębi (względna)"):
                st.image(depth_preview(res.depth), width="stretch")
    with t2:
        st.pyplot(plot_histogram(res.sizes, res.percentiles, label, wt, bins))
    with t3:
        st.dataframe(res.grains.round(3), width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("Pobierz ziarna (CSV)", grains_csv(res), "ziarna.csv", "text/csv")
        d2.download_button("Pobierz podsumowanie (CSV)", summary_csv(res, label, wt),
                           "podsumowanie.csv", "text/csv")


with st.sidebar:
    mode = st.radio("Tryb", [MODE_CLOSE, MODE_DRONE])

upload = st.file_uploader("Wgraj zdjęcie", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
if upload is None:
    st.info("Wgraj zdjęcie, aby rozpocząć. Dla zdjęć z drona użyj oryginału (z metadanymi).")
    st.stop()
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
        sam_name = st.selectbox("Segmentacja", list(SAM_MODELS))
        use_depth = st.checkbox("Użyj głębi (Depth Anything V2) — więcej RAM", False)
        depth_name = st.selectbox("Model głębi", list(DEPTH_MODELS), disabled=not use_depth)
        reject = st.checkbox("Odrzucaj obiekty z tła (wg głębi)", True, disabled=not use_depth)

        st.header("Filtrowanie masek")
        max_side = st.slider("Maks. rozdzielczość [px]", 800, 3000, 1200, 100)
        min_area = st.slider("Min. pole ziarna [px]", 20, 2000, 150)
        max_frac = st.slider("Maks. pole ziarna [% kadru]", 0.5, 30.0, 5.0) / 100

        st.header("Statystyki")
        metric = st.selectbox("Miara wielkości", list(SIZE_METRICS), format_func=SIZE_METRICS.get)
        weighting = st.radio("Ważenie D10/D50/D90", ["number", "volume"], horizontal=True,
                             format_func={"number": "liczbowe", "volume": "objętościowe (d³)"}.get)
        bins = st.slider("Liczba przedziałów histogramu", 5, 60, 20)

    st.image(image, caption=f"{W}×{H} px", width=400)
    if st.button("Analizuj", type="primary"):
        import gc
        st.session_state.pop("result", None)
        gc.collect()
        params = Params(max_side=max_side, marker_size_mm=marker_mm, marker_dict=marker_dict,
                        marker_id=marker_id, manual_mm_per_px=mm_per_px, min_area_px=min_area,
                        max_area_frac=max_frac, use_depth=use_depth,
                        depth_reject_sigma=3.0 if (use_depth and reject) else None,
                        size_metric=metric, weighting=weighting)
        try:
            with st.spinner("Segmentacja… (na CPU może potrwać minutę)"):
                st.session_state.result = analyze(
                    image, params, get_segmenter(SAM_MODELS[sam_name]),
                    get_depth(DEPTH_MODELS[depth_name]) if use_depth else None)
            st.session_state.meta = (SIZE_METRICS[metric], weighting, bins)
        except ValueError as e:
            st.error(str(e))
            st.stop()

# =========================================================================== drone mode
else:
    from streamlit_image_coordinates import streamlit_image_coordinates

    meta = read_camera_meta(raw)
    with st.sidebar:
        st.header("Kamera i człowiek")
        person_h = st.number_input("Wzrost człowieka w kadrze [m]", 1.2, 2.2, 1.75, 0.05)
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

        st.header("Segmentacja")
        sam_name = st.selectbox("Model SAM", list(SAM_MODELS))
        depth_name = st.selectbox("Model głębi (metryczny)", list(METRIC_DEPTH_MODELS))
        max_side = st.slider("Rozdzielczość robocza [px]", 1000, 4000, 2000, 250)
        tile = st.slider("Rozmiar kafelka [px]", 500, 1200, 800, 50,
                         help="Mniejszy kafelek = drobniejsze kamienie, ale dłużej.")
        min_d = st.slider("Min. średnica kamienia [px]", 6, 60, 12,
                          help="Mniejsze obiekty nie są liczone (nierozróżnialne).")
        max_frac = st.slider("Maks. pole kamienia [% obszaru]", 0.2, 10.0, 2.0) / 100

        st.header("Statystyki")
        metric = st.selectbox("Miara wielkości", list(SIZE_METRICS), format_func=SIZE_METRICS.get)
        weighting = st.radio("Ważenie D10/D50/D90", ["number", "volume"], index=1, horizontal=True,
                             format_func={"number": "liczbowe", "volume": "objętościowe (d³)"}.get)
        bins = st.slider("Liczba przedziałów histogramu", 5, 60, 20)

    # ---- click-to-mark -------------------------------------------------------------
    up_id = (upload.name, upload.size)
    if st.session_state.get("up_id") != up_id:
        st.session_state.update(up_id=up_id, pts={}, last_click=None, step=0)
        st.session_state.pop("result", None)
    pts = st.session_state.pts
    STEPS = [("head", "1. Czubek głowy człowieka"), ("feet", "2. Stopy człowieka"),
             ("roi0", "3. Obszar hałdy: lewy górny róg"), ("roi1", "4. Obszar hałdy: prawy dolny róg")]
    step = st.radio("Co teraz klikasz na zdjęciu?", range(len(STEPS)), horizontal=True,
                    format_func=lambda i: STEPS[i][1], index=st.session_state.step, key="step_radio")
    st.session_state.step = step

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
            st.session_state.step = step + 1
        st.rerun()
    st.caption("Człowiek jest tylko liniką skali: kliknij dokładnie czubek głowy i stopy. "
               "Obszar hałdy zawęża analizę do kamieni (bez ścian, kałuż i podłoża).")

    ready = all(k in pts for k in ("head", "feet", "roi0", "roi1"))
    if st.button("Analizuj", type="primary", disabled=not ready):
        import gc
        st.session_state.pop("result", None)
        gc.collect()
        x0, x1 = sorted((pts["roi0"][0], pts["roi1"][0]))
        y0, y1 = sorted((pts["roi0"][1], pts["roi1"][1]))
        p = DroneParams(max_side=max_side, person_height_m=person_h, pitch_deg=pitch,
                        focal_35mm=focal, tile=tile, min_diameter_px=min_d, max_area_frac=max_frac,
                        size_metric=metric, weighting=weighting)
        bar = st.progress(0.0, "Start…")
        try:
            st.session_state.result = analyze_drone(
                image, p, get_segmenter(SAM_MODELS[sam_name]),
                get_depth(METRIC_DEPTH_MODELS[depth_name], metric=True),
                pts["head"], pts["feet"], (x0, y0, x1, y1), progress=lambda f, t: bar.progress(f, t))
            st.session_state.meta = (SIZE_METRICS[metric], weighting, bins)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        finally:
            bar.empty()

res = st.session_state.get("result")
if res is not None:
    label, wt, bins = st.session_state.meta
    results_view(res, label, wt, bins)
