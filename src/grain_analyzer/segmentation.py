"""Grain instance segmentation with SAM (automatic mask generation, no training)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SAM_MODELS = {      # kept for backward compatibility (Hugging Face SAM v1)
    "SAM ViT-B (szybki)": "facebook/sam-vit-base",
    "SAM ViT-L": "facebook/sam-vit-large",
    "SAM ViT-H (najdokładniejszy)": "facebook/sam-vit-huge",
}


class Segmenter(Protocol):
    def segment(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return [(bool mask HxW, quality score)] for every candidate region."""


class SamSegmenter:
    """Wraps the HF ``mask-generation`` pipeline (SAM). Model is loaded lazily."""

    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str | None = None,
                 points_per_batch: int = 32, pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.92):
        self.model_id = model_id
        self.device = device
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            from .device import pick_device

            device = self.device or pick_device()
            self._pipe = pipeline("mask-generation", model=self.model_id, device=device)
        return self._pipe

    def segment(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        from PIL import Image

        out = self._load()(
            Image.fromarray(image_rgb),
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        scores = out.get("scores", [1.0] * len(out["masks"]))
        return [(np.asarray(m, bool), float(s)) for m, s in zip(out["masks"], scores)]


def select_grain_masks(candidates: list[tuple[np.ndarray, float]], *, min_area_px: int,
                       max_area_frac: float, exclude: np.ndarray | None = None,
                       max_overlap: float = 0.5) -> list[np.ndarray]:
    """Turn raw SAM candidates into a clean, non-duplicated set of grain masks.

    SAM returns nested/overlapping masks (grain, part of grain, grain+neighbour).
    Keep the best-scoring ones greedily and drop any mask that overlaps an already
    accepted one by more than ``max_overlap`` of its own area.
    """
    h, w = candidates[0][0].shape if candidates else (0, 0)
    max_area = max_area_frac * h * w
    taken = np.zeros((h, w), bool)
    kept: list[np.ndarray] = []
    for mask, _ in sorted(candidates, key=lambda c: -c[1]):
        area = int(mask.sum())
        if area < min_area_px or area > max_area:
            continue
        if exclude is not None and (mask & exclude).sum() > 0.2 * area:
            continue
        if (mask & taken).sum() > max_overlap * area:
            continue
        kept.append(mask)
        taken |= mask
    return kept


# ----------------------------------------------------------------------------- alternatives
def weights_dir() -> str:
    d = os.environ.get("GRAIN_WEIGHTS_DIR") or os.path.join(os.path.expanduser("~"), ".cache",
                                                              "grain_analyzer", "weights")
    os.makedirs(d, exist_ok=True)
    return d


def _fit_masks(masks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize (N, h, w) bool masks to the image shape if they differ."""
    if masks.shape[1:] == tuple(shape):
        return masks
    import cv2

    return np.stack([cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                                interpolation=cv2.INTER_NEAREST) for m in masks]).astype(bool)


class UltralyticsSegmenter:
    """SAM 1 / SAM 2.1 / MobileSAM / FastSAM through the ``ultralytics`` package. Weights are
    fetched from the project's GitHub releases into ``weights_dir()`` on first use."""

    def __init__(self, weights: str, family: str = "sam", device: str | None = None,
                 imgsz: int = 1024, conf: float = 0.2, iou: float = 0.7):
        self.weights, self.family, self.device = weights, family, device
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import SAM, FastSAM
            from ultralytics.utils.downloads import attempt_download_asset

            path = os.path.join(weights_dir(), self.weights)
            if not os.path.exists(path):
                attempt_download_asset(path)
            self._model = (FastSAM if self.family == "fastsam" else SAM)(path)
        return self._model

    def segment(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        from .device import pick_device

        model = self._load()
        kw = dict(verbose=False, device=self.device or pick_device())
        if self.family == "fastsam":
            kw.update(imgsz=self.imgsz, conf=self.conf, iou=self.iou, retina_masks=True)
        r = model(np.ascontiguousarray(image_rgb[..., ::-1]), **kw)[0]     # ultralytics expects BGR
        if r.masks is None:
            return []
        # no copy when the tensor is already bool on the CPU: hundreds of tile-sized masks add up
        masks = _fit_masks(r.masks.data.cpu().numpy().astype(bool, copy=False), image_rgb.shape[:2])
        conf = (r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(masks)))
        return [(m, float(c)) for m, c in zip(masks, conf)]


class Sam3TextSegmenter:
    """SAM 3 with a text prompt (e.g. "rock"): finds instances of the concept directly.
    UNTESTED here - the weights are gated (access request) and were not downloadable. Needs a
    local weights file: set ``GRAIN_SAM3_WEIGHTS`` to its path."""

    def __init__(self, weights: str, prompt: str = "rock", conf: float = 0.25):
        self.weights, self.prompt, self.conf = weights, prompt, conf
        self._pred = None

    def segment(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        if self._pred is None:
            from ultralytics.models.sam import SAM3SemanticPredictor

            self._pred = SAM3SemanticPredictor(overrides=dict(
                conf=self.conf, task="segment", mode="predict", model=self.weights,
                save=False, verbose=False))
        self._pred.set_image(np.ascontiguousarray(image_rgb[..., ::-1]))
        r = self._pred(text=[self.prompt])[0]
        if r.masks is None:
            return []
        masks = _fit_masks(r.masks.data.cpu().numpy().astype(bool), image_rgb.shape[:2])
        conf = (r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(masks)))
        return [(m, float(c)) for m, c in zip(masks, conf)]


@dataclass(frozen=True)
class SegmenterSpec:
    label: str
    backend: str            # "hf" | "ultralytics" | "sam3"
    ref: str                # HF model id, weights file name, ...
    family: str = "sam"
    note: str = ""
    tested: bool = True     # exercised in this project with the real weights


SEGMENTERS: dict[str, SegmenterSpec] = {
    "sam-b": SegmenterSpec("SAM ViT-B (domyślny)", "hf", "facebook/sam-vit-base"),
    "sam-l": SegmenterSpec("SAM ViT-L", "hf", "facebook/sam-vit-large", tested=False),
    "sam-h": SegmenterSpec("SAM ViT-H (najdokładniejszy, wolny)", "hf", "facebook/sam-vit-huge",
                           tested=False),
    "sam2.1-t": SegmenterSpec("SAM 2.1 tiny", "ultralytics", "sam2.1_t.pt"),
    "sam2.1-b": SegmenterSpec("SAM 2.1 base", "ultralytics", "sam2.1_b.pt"),
    "mobile": SegmenterSpec("MobileSAM", "ultralytics", "mobile_sam.pt"),
    "fastsam": SegmenterSpec("FastSAM (bardzo szybki, mniej dokładny)", "ultralytics",
                             "FastSAM-s.pt", family="fastsam"),
    "sam-hq": SegmenterSpec("SAM-HQ (ostrzejsze krawędzie)", "hf",
                            "syscv-community/sam-hq-vit-base", tested=False,
                            note="Nieprzetestowany: wagi tylko z Hugging Face."),
    "sam3": SegmenterSpec("SAM 3 (opis tekstowy: „rock”)", "sam3", "sam3.pt", tested=False,
                          note="Wymaga ręcznie pobranych wag: ustaw GRAIN_SAM3_WEIGHTS."),
}


def available_segmenters() -> dict[str, SegmenterSpec]:
    """Registry minus entries that cannot work here: Ultralytics-backed models need the optional
    ``ultralytics`` package (not in the default requirements - it drags in a second OpenCV),
    SAM 3 needs its weights file."""
    import importlib.util

    out = dict(SEGMENTERS)
    if importlib.util.find_spec("ultralytics") is None:
        out = {k: v for k, v in out.items() if v.backend != "ultralytics"}
    w = os.environ.get("GRAIN_SAM3_WEIGHTS")
    if not (w and os.path.exists(w)) or importlib.util.find_spec("ultralytics") is None:
        out.pop("sam3", None)
    return out


def make_segmenter(key: str) -> Segmenter:
    spec = SEGMENTERS[key]
    if spec.backend == "hf":
        return SamSegmenter(spec.ref)
    if spec.backend == "ultralytics":
        return UltralyticsSegmenter(spec.ref, spec.family)
    if spec.backend == "sam3":
        return Sam3TextSegmenter(os.environ["GRAIN_SAM3_WEIGHTS"])
    raise KeyError(key)
