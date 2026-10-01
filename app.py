"""Streamlit UI:  streamlit run app.py"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent / "src"))

from grain_analyzer.depth import DEPTH_MODELS, DepthEstimator  # noqa: E402
from grain_analyzer.export import grains_csv, summary_csv  # noqa: E402
from grain_analyzer.measure import SIZE_METRICS  # noqa: E402
from grain_analyzer.pipeline import Params, analyze  # noqa: E402
from grain_analyzer.scale import ARUCO_DICTS  # noqa: E402
from grain_analyzer.segmentation import SAM_MODELS, SamSegmenter  # noqa: E402
from grain_analyzer.viz import depth_preview, draw_overlay, plot_histogram  # noqa: E402

st.set_page_config(page_title="Analiza ziaren", layout="wide")
st.title("Analiza wielkości ziaren")


@st.cache_resource(show_spinner="Ładowanie modelu SAM…")
def get_segmenter(model_id: str) -> SamSegmenter:
    return SamSegmenter(model_id)


@st.cache_resource(show_spinner="Ładowanie Depth Anything V2…")
def get_depth(model_id: str) -> DepthEstimator:
    return DepthEstimator(model_id)


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
    use_depth = st.checkbox("Użyj głębi (Depth Anything V2)", True)
    depth_name = st.selectbox("Model głębi", list(DEPTH_MODELS), disabled=not use_depth)
    reject = st.checkbox("Odrzucaj obiekty z tła (wg głębi)", True, disabled=not use_depth)

    st.header("Filtrowanie masek")
    max_side = st.slider("Maks. rozdzielczość [px]", 800, 3000, 1600, 100)
    min_area = st.slider("Min. pole ziarna [px]", 20, 2000, 150)
    max_frac = st.slider("Maks. pole ziarna [% kadru]", 0.5, 30.0, 5.0) / 100

    st.header("Statystyki")
    metric = st.selectbox("Miara wielkości", list(SIZE_METRICS), format_func=SIZE_METRICS.get)
    weighting = st.radio("Ważenie D10/D50/D90", ["number", "volume"], horizontal=True,
                         format_func={"number": "liczbowe", "volume": "objętościowe (d³)"}.get)
    bins = st.slider("Liczba przedziałów histogramu", 5, 60, 20)

upload = st.file_uploader("Wgraj zdjęcie ziaren ze znacznikiem w kadrze",
                          type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
if upload is None:
    st.info("Wgraj zdjęcie, aby rozpocząć.")
    st.stop()

image = np.array(ImageOps.exif_transpose(Image.open(upload)).convert("RGB"))
st.image(image, caption=f"{image.shape[1]}×{image.shape[0]} px", width=400)

if st.button("Analizuj", type="primary"):
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
        st.session_state.meta = (SIZE_METRICS[metric], weighting)
    except ValueError as e:
        st.error(str(e))
        st.stop()

res = st.session_state.get("result")
if res is None:
    st.stop()
label, wt = st.session_state.meta
if res.grains.empty:
    st.warning("Nie wykryto ziaren — zmniejsz minimalne pole lub zmień model.")
    st.stop()

c = st.columns(5)
c[0].metric("Ziarna", len(res.grains))
for col, (k, v) in zip(c[1:4], res.percentiles.items()):
    col.metric(k, f"{v:.2f} mm")
c[4].metric("Skala", f"{res.scale.mm_per_px:.4f} mm/px", res.scale.method, delta_color="off")
if res.n_rejected_depth:
    st.caption(f"Odrzucono wg głębi: {res.n_rejected_depth}")

t1, t2, t3 = st.tabs(["Kontury", "Histogram", "Dane"])
with t1:
    ids = st.checkbox("Numery ziaren")
    st.image(draw_overlay(res, ids), use_container_width=True)
    if res.depth is not None:
        with st.expander("Mapa głębi (względna)"):
            st.image(depth_preview(res.depth), use_container_width=True)
with t2:
    st.pyplot(plot_histogram(res.sizes, res.percentiles, label, wt, bins))
with t3:
    st.dataframe(res.grains.round(3), use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("Pobierz ziarna (CSV)", grains_csv(res), "ziarna.csv", "text/csv")
    d2.download_button("Pobierz podsumowanie (CSV)", summary_csv(res, label, wt),
                       "podsumowanie.csv", "text/csv")
