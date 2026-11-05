"""Image Fixer - the simple edit chain (no GUI code).
One picture = one Item.  Settings (Item.p):
  clean  0..100  remove shadow + grey / green tint of the paper
  white  0..100  push light paper to pure white
  text   0..100  sharper text and lines
  dark   0..100  darker, bolder text (faint writing gets stronger, paper stays white)
  sat    100..250 colour strength (100 = unchanged)
  adjs   list    picked colours: each has a slider "k" (100 = same, 0 = gone, 200 = stronger) and optional "to" colour
"""
import copy
import os

import numpy as np
from PIL import Image

import core

core.PREVIEW_MAX = 1800

PRESET = {"clean": 100, "white": 60, "text": 35, "dark": 25, "sat": 100}


def default_params():
    return {"clean": 0, "white": 0, "text": 0, "dark": 0, "sat": 100, "adjs": []}


def is_plain(p):
    return (p["clean"] == 0 and p["white"] == 0 and p["text"] == 0 and p["dark"] == 0 and p["sat"] == 100
            and not core.active_adjs(p["adjs"]) and not any(a.get("to") for a in p["adjs"]))


def copy_params(p):
    return copy.deepcopy(p)


class Item:
    def __init__(self, path, img):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.p = default_params()
        self.version = 0                 # grows with every change (used for the export cache)
        self.undo = []                   # older versions of base (rotate, erase, move, crop)
        self.view = None                 # processed preview (uint8 array)
        self.thumb = None                # QImage, set by the window
        self.export = None               # (version, path)
        self.set_base(img)
        self.original = img

    # ---- picture data
    def set_base(self, img):
        self.base = img
        self.prev, self.scale = core.make_preview(img)
        self.cache = {}
        self.view = self.prev
        self.version += 1

    def push_undo(self):
        self.undo.append(self.base)
        del self.undo[:-8]

    def do_undo(self):
        if not self.undo:
            return False
        self.set_base(self.undo.pop())
        return True

    def reset(self):
        self.p = default_params()
        self.undo = []
        self.set_base(self.original)


# ------------------------------------------------------------------ the chain
def _recolor(x, orig, a, lab, cache):
    """Gives the ink of one picked colour another colour (a["to"]) and/or another strength."""
    m = core.get_mask(orig, a, lab, cache)
    mg = np.maximum(m, core._grow(m))
    old = np.array(a["rgb"], np.float32)
    xf = x.astype(np.float32)
    u = 255.0 - old
    al = np.clip(((255.0 - xf) * u).sum(2) / (float((u * u).sum()) + 1e-3), 0.0, 1.2)
    w = (mg * al)[..., None]
    if a.get("to"):
        xf = xf + (np.array(a["to"], np.float32) - old) * w
    k = a["k"]
    if k > 100:
        xf = xf * (1.0 - 0.4 * min((k - 100) / 100.0, 1.0) * w)
    elif k < 100:
        xf = xf + (255.0 - xf) * w * ((100 - k) / 100.0)
    return np.clip(xf + 0.5, 0, 255).astype(np.uint8)


def _tone(arr, p):
    """Darker text + whiter paper in one table lookup."""
    dark, white = p["dark"], p["white"]
    if dark <= 0 and white <= 0:
        return arr
    v = np.arange(256, dtype=np.float32) / 255.0
    v = v ** (1.0 + dark / 100.0 * 1.1)
    hi = 255.0 - 0.30 * white
    lut = np.clip(np.round(v * 255.0 * (255.0 / hi)), 0, 255).astype(np.uint8)
    return lut[arr]


def apply(arr, p, lab=None, cache=None, chunk_rows=0):
    """arr = paper already cleaned (uint8). Returns the finished uint8 picture."""
    # recolour rows are handled below: here they count as "unchanged" (k=100) so their masks stay in the cache
    adjs = [dict(a, k=100) if a.get("to") else a for a in p["adjs"]]
    out = core.render(arr, adjs, p["sat"], p["text"], 0, lab=lab, cache=cache, chunk_rows=chunk_rows)
    for a in p["adjs"]:
        if a.get("to"):
            out = _recolor(out, arr, a, lab, cache)
    return _tone(out, p)


def make_view(it):
    """Live preview for the window (runs in a worker thread). Returns (view array, cleaned array)."""
    p = copy_params(it.p)
    c = it.cache
    if p["clean"] > 0:
        if c.get("ck") != p["clean"]:
            c.clear()
            c["ck"] = p["clean"]
            c["arr"] = np.asarray(core.remove_shadow(Image.fromarray(it.prev), p["clean"] / 100.0))
        arr = c["arr"]
    else:
        if c.get("ck") != 0:
            c.clear()
            c["ck"] = 0
            c["arr"] = it.prev
        arr = c["arr"]
    if p["adjs"] and "lab" not in c:
        c["lab"] = core.rgb_to_lab(arr)
    c.setdefault("m", {})
    return apply(arr, p, c.get("lab"), c["m"] if p["adjs"] else None), arr


def full_image(base, p):
    """Full-resolution result for saving / dragging out (runs in a worker thread)."""
    if is_plain(p):
        return base
    img = core.remove_shadow(base, p["clean"] / 100.0) if p["clean"] > 0 else base
    return Image.fromarray(apply(np.asarray(img), p, None, None, chunk_rows=700))
