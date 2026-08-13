"""Overlay per-patch ABMIL attention weights on a WSI in napari.

Written by Claude.

Loads a slide's extracted UNI2-h features (features/labels/coords, as saved
by extract_features.py) plus a trained ABMILite checkpoint, computes the
raw pre-softmax per-instance attention scores, and renders one rectangle
per patch on top of the WSI, colored by attention value.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th
import napari
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import cm
from matplotlib.colors import Normalize

# PyGaulix-Suite houses the napari WSI-loading helpers (multiscale pyramid
# via tiffslide + dask) used across the rest of the annotation tooling.
sys.path.insert(0, str(Path(r"D:\Projects\PyGaulix-Suite")))

from src.model import ABMILite
from src.napari_utils import load_big_wsi_v2, get_topmost_visible_image

# Must match the architecture the checkpoint was trained with
# (see experiments/2026-06-01 CAMELYON-UNI2-ABMIL.ipynb).
EMBEDDING_DIM = 1536
HIDDEN_DIM = EMBEDDING_DIM // 4
ATTENTION_BRANCHES = 3
ATTENTION_MODE = "simple"

# openslide.mpp-x / mpp-y read directly off a CAMELYON16 slide (level 0).
# Re-check this against slide.properties if pointing at a different scanner/dataset.
MICRONS_PER_PIXEL = 0.243094


def load_model(checkpoint_path: Path, device: str) -> ABMILite:
    model = ABMILite(
        EMBEDDING_DIM,
        HIDDEN_DIM,
        use_feature_extractor=False,
        feature_extraction_layers=3,
        attention_branches=ATTENTION_BRANCHES,
        attention_mode=ATTENTION_MODE,
    )
    state_dict = th.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def compute_attention(model: ABMILite, features: th.Tensor, device: str) -> np.ndarray:
    """Raw, pre-softmax attention score per instance -- comparable across
    instances regardless of bag size, unlike the softmax-normalized weight
    (which shrinks as bag size grows). Matches the convention used for
    instance-level analysis in 2026-06-01 CAMELYON-UNI2-ABMIL.ipynb."""
    with th.no_grad():
        attn = model.attention_module(features.to(device), softmax=False, transpose=False)
    return attn.squeeze(-1).cpu().numpy()


def compute_instance_logits(model: ABMILite, features: th.Tensor, device: str) -> np.ndarray:
    """Raw per-instance classifier logit (pre-sigmoid), i.e. `abmilite(x,
    mode='instance')` -- positive leans tumor, negative leans benign,
    independent of the attention weighting."""
    with th.no_grad():
        logits = model(features.to(device), mode="instance")
    return logits.squeeze(-1).cpu().numpy()


def add_colorbar(viewer: napari.Viewer, name: str, colormap_name: str, vmin: float, vmax: float, label: str) -> None:
    """Dock a labeled colorbar legend next to the viewer, mapping colors to
    values -- for reference when building figures/screenshots."""
    fig = Figure(figsize=(1.6, 4.5))
    ax = fig.add_axes((0.4, 0.08, 0.25, 0.88))
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap_name), cax=ax, label=label)
    canvas = FigureCanvasQTAgg(fig)
    viewer.window.add_dock_widget(canvas, name=name, area="right")


def coords_to_rectangles(coords: th.Tensor) -> list[np.ndarray]:
    """Convert level-0 pixel boxes (x0, y0, x1, y1) into napari rectangles.

    napari shapes are in (row, col) = (y, x) order, matching the flip
    convention already used for polygons elsewhere in pygaulix/PyGaulix-Suite.
    """
    coords_np = coords.numpy() if isinstance(coords, th.Tensor) else np.asarray(coords)
    return [
        np.array([[y0, x0], [y1, x1]], dtype=float)
        for x0, y0, x1, y1 in coords_np
    ]


def main():
    parser = argparse.ArgumentParser(description="Overlay per-patch ABMIL attention on a WSI in napari.")
    parser.add_argument("-w", "--wsi-path", type=Path, required=True, help="Path to the WSI (.tif) to overlay.")
    parser.add_argument("-f", "--features-path", type=Path, required=True, help="Slide's extracted features .pt (features/labels/coords).")
    parser.add_argument("-c", "--checkpoint", type=Path, required=True, help="Trained ABMILite state_dict .pt.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--opacity", type=float, default=0.6, help="Layer opacity for the attention/logit overlays.")
    parser.add_argument("--colormap", type=str, default="magma", help="napari colormap for attention intensity.")
    parser.add_argument("--logit-colormap", type=str, default="coolwarm", help="napari colormap for instance logit (diverging, centered at 0).")
    parser.add_argument("--edge-color", type=str, default="white", help="Outline color for the edges-only 'tile edges' layer.")
    parser.add_argument("--edge-width", type=float, default=1.0, help="Outline width for the edges-only 'tile edges' layer.")
    args = parser.parse_args()

    data = th.load(args.features_path, map_location="cpu")
    features, coords = data["features"], data["coords"]

    model = load_model(args.checkpoint, args.device)
    attention = compute_attention(model, features, args.device)
    instance_logits = compute_instance_logits(model, features, args.device)

    viewer = napari.Viewer()
    load_big_wsi_v2(viewer, args.wsi_path, with_annotations=False)

    # Tag the WSI layer with physical pixel size so the scale bar reads in
    # real units instead of raw pixels.
    image_layer = get_topmost_visible_image(viewer)
    image_layer.scale = (MICRONS_PER_PIXEL, MICRONS_PER_PIXEL)
    image_layer.units = ("um", "um")

    rectangles = coords_to_rectangles(coords)
    vmin, vmax = float(attention.min()), float(attention.max())

    viewer.add_shapes(
        rectangles,
        shape_type="rectangle",
        name="attention",
        properties={"attention": attention},
        face_color="attention",
        face_colormap=args.colormap,
        face_contrast_limits=(vmin, vmax),
        edge_width=0,
        opacity=args.opacity,
        scale=(MICRONS_PER_PIXEL, MICRONS_PER_PIXEL),
        units=("um", "um"),
    )
    add_colorbar(viewer, "Attention colorbar", args.colormap, vmin, vmax, "Attention weight")

    # Instance logit overlay (positive leans tumor, negative leans benign),
    # hidden by default -- toggle on in the layer list. Colormap is centered
    # at 0 since the sign is meaningful, unlike attention.
    logit_abs_max = float(np.abs(instance_logits).max())
    logit_vmin, logit_vmax = -logit_abs_max, logit_abs_max
    viewer.add_shapes(
        rectangles,
        shape_type="rectangle",
        name="instance logit",
        properties={"logit": instance_logits},
        face_color="logit",
        face_colormap=args.logit_colormap,
        face_contrast_limits=(logit_vmin, logit_vmax),
        edge_width=0,
        opacity=args.opacity,
        scale=(MICRONS_PER_PIXEL, MICRONS_PER_PIXEL),
        units=("um", "um"),
        visible=False,
    )
    add_colorbar(viewer, "Instance logit colorbar", args.logit_colormap, logit_vmin, logit_vmax, "Instance logit")

    # Outline-only view of the same tiles, hidden by default -- toggle this
    # layer on and the color overlays off in the layer list for
    # edges-without-heatmap figures.
    viewer.add_shapes(
        rectangles,
        shape_type="rectangle",
        name="tile edges",
        face_color=[0, 0, 0, 0],
        edge_color=args.edge_color,
        edge_width=args.edge_width,
        opacity=1.0,
        scale=(MICRONS_PER_PIXEL, MICRONS_PER_PIXEL),
        units=("um", "um"),
        visible=False,
    )

    viewer.scale_bar.visible = True

    napari.run()


if __name__ == "__main__":
    main()
