"""Image helpers shared by the server."""

import hashlib
import numpy as np
from PIL import Image


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def size_of(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def road_pixels(path):
    return int((load_gray(path) > 128).sum())


def isolate_noise(clean_path, noisy_path, out_path):
    """
    Noise on its own = material with noise - material alone.

    Subtracting removes the base material's own colour and grain, which the
    material tree already covers, so what is left is the cracks and wear. That
    is what the noise tree studies, and it is what lets generated wear vary
    instead of repeating one stamped shape.
    """
    a = np.array(Image.open(clean_path).convert("RGB")).astype(np.int16)
    b = np.array(Image.open(noisy_path).convert("RGB")).astype(np.int16)
    if a.shape != b.shape:
        return {"ok": False,
                "error": f"sizes differ: {a.shape[1]}x{a.shape[0]} and {b.shape[1]}x{b.shape[0]}"}
    diff = np.abs(a.sum(axis=2) - b.sum(axis=2))
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff, "L").save(out_path)
    mean = float(diff.mean())
    covered = float((diff > 20).mean())
    return {"ok": True, "mean": round(mean, 2), "max": int(diff.max()),
            "covered_fraction": round(covered, 4),
            "note": ("almost no difference: check the second image really shows cracks"
                     if mean < 6 else "difference found across the surface")}
