"""
Image loading/encoding for VLM calls, plus cheap local quality heuristics
that don't need a model call (saves cost/latency before even hitting the
VLM).
"""

import base64
import hashlib
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum dimensions below which an image is probably useless for damage assessment
_MIN_DIMENSION = 100


def load_and_encode(image_path: Path, max_dimension: int = 1024) -> str:
    """
    Open with PIL, downscale if larger than max_dimension (cuts token
    cost on most VLM APIs with negligible quality loss for damage
    detection), encode as base64 JPEG string for the API call.

    Raises a clear error (don't silently skip) if the file is
    unreadable/corrupt -- the caller should turn that into
    valid_image=false + a risk flag.
    """
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  # normalize palette/CMYK/etc.
    except Exception as exc:
        raise ValueError(f"Cannot open image {image_path}: {exc}") from exc

    # Downscale if needed (preserves aspect ratio)
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug("Downscaled %s from %dx%d to %dx%d", image_path, w, h, new_w, new_h)

    # Encode as base64 JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return encoded


def cheap_quality_heuristics(image_path: Path) -> list[str]:
    """
    Fast, local-only checks before spending a model call:
    - file size near zero -> likely corrupt
    - image dimensions absurdly small -> not usable
    - basic blur-variance check (Laplacian via PIL) to pre-flag blurry images

    Return a list of preliminary notes (e.g. "blurry_image", "small").
    Final risk flagging still happens in risk_assessor.py.
    """
    from PIL import Image
    import numpy as np

    notes: list[str] = []

    # File size check
    file_size = image_path.stat().st_size
    if file_size < 1024:  # less than 1KB is suspicious
        notes.append("tiny_file")
        return notes  # no point checking further

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        notes.append("unreadable")
        return notes

    w, h = img.size
    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        notes.append("small")
        return notes

    # Blur check (Laplacian variance)
    try:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)
        # Simple Laplacian convolution
        laplacian = (
            arr[1:-1, 1:-1] * 4
            - arr[:-2, 1:-1]
            - arr[2:, 1:-1]
            - arr[1:-1, :-2]
            - arr[1:-1, 2:]
        )
        variance = float(laplacian.var())
        if variance < 50:  # very blurry
            notes.append("blurry_image")
    except Exception:
        pass  # blur check is best-effort

    return notes


def image_hash(image_path: Path) -> str:
    """Stable content hash (sha256 of file bytes) for caching."""
    sha = hashlib.sha256()
    with open(image_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()
