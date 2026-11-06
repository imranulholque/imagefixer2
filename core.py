"""Image Fixer - all picture maths (no GUI code here, so it can be tested on its own)."""
import colorsys
import math
import os
import uuid

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageOps
from PIL import ImageFilter as _PilFilter


class _RankFilter(_PilFilter.Filter):
    """Min / Max filter that does not keep Python's lock while it works (Pillow's does, which froze the window)."""
    def __init__(self, size, use_max):
        self.size, self.use_max = int(size), use_max

    def filter(self, image):
        a = np.asarray(Image.Image()._new(image))
        return Image.fromarray(min_filter(a, self.size, self.use_max)).im


class ImageFilter:                                  # the names used in this file; the rank filters are our own
    GaussianBlur = _PilFilter.GaussianBlur

    @staticmethod
    def MaxFilter(size=3):
        return _RankFilter(size, True)

    @staticmethod
    def MinFilter(size=3):
        return _RankFilter(size, False)


Image.MAX_IMAGE_PIXELS = 400_000_000

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
try:  # iPhone photos (optional)
    import pillow_heif

    pillow_heif.register_heif_opener()
    IMG_EXT |= {".heic", ".heif"}
except Exception:
    pass

PREVIEW_MAX = 2400          # long side of the on-screen working copy
NEUTRAL_C = 14.0            # chroma below this counts as black / gray
NEUTRAL_SOFT = 12.0
BAND_EDGE = 4.0             # softness between two neighbouring gray bands (L units)


# ----------------------------------------------------------------- loading
def load_image(path):
    """Open any picture as plain RGB. WhatsApp / phone photos are rotated by EXIF, so fix that."""
    im = Image.open(path)
    im.load()
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        arr = np.asarray(im, dtype=np.float32)
        top = float(arr.max()) or 1.0
        arr = arr / (65535.0 if top > 255 else 255.0) * 255.0
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return im.convert("RGB")


def make_preview(img):
    """(uint8 array, scale). scale = preview size / real size (1.0 for normal pictures)."""
    w, h = img.size
    m = max(w, h)
    if m <= PREVIEW_MAX:
        return np.array(img), 1.0
    s = PREVIEW_MAX / m
    small = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
    return np.array(small), small.size[0] / w


# ------------------------------------------------------------------- colour
_x = np.arange(256, dtype=np.float32) / 255.0
_LUT = np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4).astype(np.float32)
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_MT = np.ascontiguousarray((_M / _WHITE[:, None]).T)


def rgb_to_lab(rgb):
    """uint8 (...,3) -> float32 Lab (...,3), D65."""
    shape = rgb.shape
    lin = _LUT[rgb.reshape(-1, 3)]
    xyz = lin @ _MT
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0).astype(np.float32)
    out = np.empty_like(f)
    out[:, 0] = 116.0 * f[:, 1] - 16.0
    out[:, 1] = 500.0 * (f[:, 0] - f[:, 1])
    out[:, 2] = 200.0 * (f[:, 1] - f[:, 2])
    return out.reshape(shape)


def L_to_gray(L):
    fy = (L + 16.0) / 116.0
    y = fy ** 3 if fy ** 3 > 0.008856 else (fy - 16.0 / 116.0) / 7.787
    v = 12.92 * y if y <= 0.0031308 else 1.055 * (y ** (1 / 2.4)) - 0.055
    return int(round(255 * min(1.0, max(0.0, v))))


def hex_of(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def color_name(rgb):
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    d = h * 360.0
    if 15 <= d < 50 and v < 0.55:
        return "Brown"
    for lim, name in ((15, "Red"), (45, "Orange"), (70, "Yellow"), (165, "Green"), (200, "Cyan"),
                      (255, "Blue"), (290, "Purple"), (340, "Pink"), (361, "Red")):
        if d < lim:
            return name
    return "Color"


def band_name(L):
    for lim, name in ((12, "Deep black"), (26, "Black"), (42, "Dark ash"), (58, "Ash"),
                      (74, "Light ash"), (88, "Pale ash"), (101, "Near white")):
        if L < lim:
            return name
    return "Near white"


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# -------------------------------------------------------------- adjustments
# An adjustment is a plain dict, so it can be copied for undo:
#   colour : {"type":"color","target":[L,a,b],"tol":30,"k":100,"feather":0,...}
#   band   : {"type":"band","lo":..,"hi":..,"center":L,"k":100,"feather":0,...}   (black / gray shades)
# k = 100 -> unchanged
# k < 100 -> that colour is removed and replaced by what surrounds it (0 = fully gone)
# k > 100 -> colours get stronger and clearer;  black / gray shades get darker
def new_color_adj(rgb, lab_pt, tol=30.0, feather=0):
    rgb = [int(v) for v in rgb]
    return {"id": uuid.uuid4().hex[:8], "type": "color",
            "target": [float(lab_pt[0]), float(lab_pt[1]), float(lab_pt[2])],
            "rgb": rgb, "tol": float(tol), "k": 100, "feather": int(feather),
            "label": "%s  %s" % (color_name(rgb), hex_of(rgb))}


def new_band_adj(center, lo, hi, feather=0):
    g = L_to_gray(center)
    return {"id": uuid.uuid4().hex[:8], "type": "band", "lo": lo, "hi": hi,
            "center": float(center), "rgb": [g, g, g], "k": 100, "feather": int(feather),
            "label": band_name(center)}


def adj_key(a):
    f = int(a.get("feather", 0))
    if a["type"] == "color":
        t = a["target"]
        return ("c", round(t[0], 1), round(t[1], 1), round(t[2], 1), round(a["tol"], 1), f)
    return ("b", None if a["lo"] is None else round(a["lo"], 1), None if a["hi"] is None else round(a["hi"], 1), f)


def _neutral_factor(lab):
    C = np.hypot(lab[..., 1], lab[..., 2])
    return 1.0 - _smooth((C - NEUTRAL_C) / NEUTRAL_SOFT)


def color_match(lab, t, tol):
    """Pixels that have the SAME COLOUR as target t (Lab), whatever their lightness: same hue, enough colour strength.
    A black pixel with a hint of red, or a grey one, is NOT matched (it has too little colour)."""
    Ct = math.hypot(t[1], t[2])
    a, b = lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    dh = np.abs((np.arctan2(b, a) - math.atan2(t[2], t[1]) + math.pi) % (2 * math.pi) - math.pi)
    dh = np.degrees(dh)
    hw = 12.0 + tol * 0.5
    hue_g = 1.0 - _smooth((dh - hw * 0.75) / (hw * 0.5))
    chroma_g = _smooth((C - 0.32 * Ct) / (0.25 * Ct))
    dL = np.abs(lab[..., 0] - t[0])
    light_g = 1.0 - _smooth((dL - tol * 1.6) / (tol * 0.8))
    return (hue_g * chroma_g * light_g).astype(np.float32)


def color_match(lab, t, tol):
    """0..1: how much each pixel IS colour t=(L,a,b).  Hue and chroma decide first, lightness counts less, and
    grey / black pixels (even with a faint colour cast) never match a strong colour, so selecting red does not
    take the black writing with it."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    da, db = a - t[1], b - t[2]
    dL = (L - t[0]) * 0.5
    d = np.sqrt(dL * dL + da * da + db * db)
    inner, outer = tol * 0.7, tol * 1.3
    m = 1.0 - _smooth((d - inner) / (outer - inner))
    Ct = math.hypot(t[1], t[2])
    if Ct > 18.0:
        C = np.hypot(a, b)
        m = m * _smooth((C - 0.2 * Ct) / (0.2 * Ct))                      # must be coloured too
        cosang = (a * t[1] + b * t[2]) / (C * Ct + 1e-3)
        dh = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        m = m * (1.0 - _smooth((dh - (22.0 + tol * 0.5)) / 15.0))            # same hue
    return m.astype(np.float32)


def adj_mask(lab, a):
    """float32 0..1 : how strongly each pixel belongs to this colour / shade (no feather)."""
    if a["type"] == "color":
        t = a["target"]
        Ct = math.hypot(t[1], t[2])
        if Ct >= 22.0:
            return color_match(lab, t, a["tol"])
        dL = (lab[..., 0] - t[0]) * 0.8
        da = lab[..., 1] - t[1]
        db = lab[..., 2] - t[2]
        d = np.sqrt(dL * dL + da * da + db * db)
        inner, outer = a["tol"] * 0.7, a["tol"] * 1.3
        return (1.0 - _smooth((d - inner) / (outer - inner))).astype(np.float32)
    L = lab[..., 0]
    m = _neutral_factor(lab)
    if a["lo"] is not None:
        m = m * _smooth((L - (a["lo"] - BAND_EDGE / 2)) / BAND_EDGE)
    if a["hi"] is not None:
        m = m * (1.0 - _smooth((L - (a["hi"] - BAND_EDGE / 2)) / BAND_EDGE))
    return m.astype(np.float32)


def feather_mask(m, amount, long_side):
    """Soft edge like Photoshop's Feather. amount 0..100, relative to the picture size."""
    if amount <= 0:
        return m
    sigma = amount / 100.0 * 0.02 * long_side
    if sigma < 0.4:
        return m
    im = Image.fromarray((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(im, dtype=np.float32) / 255.0


def get_mask(rgb, a, lab=None, cache=None):
    """Selection of one colour / shade, with feather. Cached for the live preview."""
    key = ("m",) + adj_key(a)
    if cache is not None and key in cache:
        return cache[key]
    H, W = rgb.shape[:2]
    if lab is not None:
        m = adj_mask(lab, a)
    else:                                   # big picture: no full Lab copy, work in strips
        m = np.empty((H, W), dtype=np.float32)
        for y in range(0, H, 500):
            m[y:y + 500] = adj_mask(rgb_to_lab(rgb[y:y + 500]), a)
    m = feather_mask(m, a.get("feather", 0), max(H, W))
    if cache is not None:
        cache[key] = m
    return m


# ------------------------------------------------------ fill from surroundings
def _pool2(a):
    return a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]


def _up(a, h, w):
    """bilinear resize of a float32 (h0,w0) or (h0,w0,C) array."""
    if a.ndim == 3:
        return np.stack([_up(a[..., c], h, w) for c in range(a.shape[2])], -1)
    im = Image.fromarray(np.ascontiguousarray(a, dtype=np.float32))
    return np.asarray(im.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)


def _fill_pyramid(S0, W0, want):
    """Multi-scale 'average of the trusted pixels around'. Returns {level: estimate (padded size)}."""
    H, W = W0.shape
    L, hh, ww = 0, H, W
    while min(hh, ww) >= 8 and L < 9:
        hh, ww, L = (hh + 1) // 2, (ww + 1) // 2, L + 1
    mult = 1 << L
    ph, pw = -(-H // mult) * mult, -(-W // mult) * mult
    S = np.pad(S0, ((0, ph - H), (0, pw - W), (0, 0)))
    Wt = np.pad(W0, ((0, ph - H), (0, pw - W)))
    levels = [(S, Wt)]
    for _ in range(L):
        S, Wt = _pool2(S), _pool2(Wt)
        levels.append((S, Wt))
    total = float(levels[-1][1].sum())
    gmean = (levels[-1][0].sum((0, 1)) / total).astype(np.float32) if total > 1e-6 else np.full(3, 255.0, np.float32)
    out, F = {}, None
    low = min(want)
    for l in range(L, low - 1, -1):
        S_l, W_l = levels[l]
        has = (W_l > 1e-6)[..., None]
        avg = S_l / np.maximum(W_l, 1e-6)[..., None]
        if F is None:
            F = np.where(has, avg, gmean).astype(np.float32)
        else:
            up = _up(F, S_l.shape[0], S_l.shape[1])
            frac = np.clip(W_l / (4.0 ** l), 0, 1)[..., None]
            F = np.where(has, frac * avg + (1 - frac) * up, up).astype(np.float32)
        out[l] = F
    return out, L


def fill_background(rgb, valid):
    """What should be where pixels were removed?  rgb uint8 (H,W,3); valid (H,W) 1 = trusted background pixel.
    Returns uint8 (H,W,3): a smooth picture of the surroundings (colour of paper, table, wall ...)."""
    H, W = valid.shape
    v = np.clip(valid, 0, 1).astype(np.float32)
    rgbf = rgb.astype(np.float32)
    base = 1 if max(H, W) > 700 else 0
    out, L = _fill_pyramid(rgbf * v[..., None], v, {base})
    nb_level = max(base, min(3, L))
    nb = _up(out[nb_level][: -(-H // (1 << nb_level)), : -(-W // (1 << nb_level))], H, W)
    # second pass: trust only pixels that look like the background around them (ignores nearby text / lines)
    d = np.abs(rgbf - nb).sum(2)
    w2 = (v * np.exp(-(d / 45.0) ** 2)).astype(np.float32)
    out2, _ = _fill_pyramid(rgbf * w2[..., None], w2, {base})
    F = out2[base]
    F = _up(F[: -(-H // (1 << base)), : -(-W // (1 << base))], H, W) if base else F[:H, :W]
    return np.clip(F + 0.5, 0, 255).astype(np.uint8)


def get_fill(rgb, masks, keys, cache=None):
    key = ("f",) + tuple(keys)
    if cache is not None and key in cache:
        return cache[key]
    H, W = rgb.shape[:2]
    hole = np.maximum.reduce(masks) > 0.03
    hole_img = Image.fromarray((hole * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    valid = 1.0 - np.asarray(hole_img, dtype=np.float32) / 255.0
    if H * W > 6_000_000:                   # huge picture: the fill is smooth, so work on a smaller copy
        f = (H * W / 3_000_000.0) ** 0.5
        sw, sh = max(8, int(W / f)), max(8, int(H / f))
        small = np.asarray(Image.fromarray(rgb).resize((sw, sh), Image.Resampling.BOX))
        sv = np.asarray(Image.fromarray((valid * 255).astype(np.uint8)).resize((sw, sh), Image.Resampling.BOX),
                        dtype=np.float32) / 255.0
        sv = (sv > 0.999).astype(np.float32)
        F = fill_background(small, sv)
        F = np.asarray(Image.fromarray(F).resize((W, H), Image.Resampling.BILINEAR))
    else:
        F = fill_background(rgb, valid)
    if cache is not None:
        cache[key] = F
    return F


# ------------------------------------------------------- saturation / clarity
def _saturate(x, sat):
    """Saturation of the whole picture. White and gray never change when it goes up;
    when it goes down, colours get lighter and lighter until they are white."""
    if sat == 100:
        return x
    if sat > 100:
        g = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)[..., None]
        return g + (x - g) * (sat / 100.0)
    mx, mn = x.max(2), x.min(2)
    S = (mx - mn) / np.maximum(mx, 1.0)
    w = _smooth(S / 0.35)                          # grays (black text, paper) stay as they are
    f = (100 - sat) / 100.0
    return x + (255.0 - x) * (f * w)[..., None]


# ---- rank filters in pure numpy.  Pillow's Min / Max / Median filters keep the Python lock while they run, so the whole
# window froze during a text / shadow preview.  numpy releases it, so the window stays alive.
def _shift_stack(a, k):
    r = k // 2
    p = np.pad(a, [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    return p, r


def min_filter(a, k, use_max=False):
    """k x k minimum (or maximum) of a 2-D / 3-D array (same shape back)."""
    if k <= 1:
        return a
    op = np.maximum if use_max else np.minimum
    p, r = _shift_stack(a, k)
    H, W = a.shape[:2]
    rows = p[0:H]
    for i in range(1, k):                               # vertical pass
        rows = op(rows, p[i:i + H])
    out = rows[:, 0:W]
    for j in range(1, k):                               # horizontal pass
        out = op(out, rows[:, j:j + W])
    return out


def max_filter(a, k):
    return min_filter(a, k, True)


def median3(a):
    """3 x 3 median (2-D array), sorting network."""
    p = np.pad(a, 1, mode="edge")
    H, W = a.shape
    v = [p[i:i + H, j:j + W] for i in range(3) for j in range(3)]

    def s(i, j):
        lo, hi = np.minimum(v[i], v[j]), np.maximum(v[i], v[j])
        v[i], v[j] = lo, hi
    for i, j in ((1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8), (0, 3), (5, 8), (4, 7),
                 (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4), (4, 2)):
        s(i, j)
    return v[4]


def _mask_filter(m, k, use_max=True):
    """Min / max filter of a uint8 / bool mask given as a numpy array."""
    return min_filter(np.asarray(m), k, use_max)


def clarity(rgb, look):
    """Text / line / shape sharpness, faint strokes, darker text, cleaner specks.  Only the lightness is changed,
    with a limit against halos, so shapes are never bent or moved."""
    sharp, objs, detail = look["sharp"], look["objs"], look["detail"]
    dark, clear = look["dark"], look["clear"]
    if not clarity_active(look):
        return rgb
    H, W = rgb.shape[:2]
    long = max(H, W)
    x = rgb.astype(np.float32)
    Y = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    Yf = Y.copy()

    def u8(a):
        return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))

    def unsharp(Yf, r, amount, noise):
        im = u8(Yf)
        u = np.asarray(im)
        blur = np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
        d = Yf - blur
        if noise:
            d = np.sign(d) * np.maximum(np.abs(d) - noise, 0.0)
        Ys = Yf + amount * d
        k = max(3, int(2 * math.ceil(r) + 1))
        mn = min_filter(u, k).astype(np.float32)
        mx = max_filter(u, k).astype(np.float32)
        margin = 0.3 * (mx - mn) + 2.0
        return np.clip(Ys, mn - margin, mx + margin)

    if clear > 0:                                   # remove specks / noise
        med = median3(np.asarray(u8(Yf))).astype(np.float32)
        a = clear / 100.0
        Yf = Yf * (1 - a) + med * a
    if detail > 0:                                  # local contrast with a wide radius = faint strokes
        r2 = max(6.0, 0.02 * long)
        blur = np.asarray(u8(Yf).filter(ImageFilter.GaussianBlur(r2)), dtype=np.float32)
        Yf = Yf + (detail / 100.0 * 1.6) * (Yf - blur)
    if objs > 0:                                    # shapes / thick lines: medium radius
        Yf = unsharp(Yf, max(1.5, 0.004 * long), objs / 100.0 * 1.4, 0.0)
    if sharp > 0:                                   # text edges: small radius
        Yf = unsharp(Yf, float(np.clip(0.0008 * long, 0.6, 3.0)), sharp / 100.0 * 2.5, 2.0)
    if dark > 0:                                    # text darker, paper untouched
        t = 1.0 - _smooth(Yf / 200.0)
        Yf = Yf * (1.0 - 0.7 * (dark / 100.0) * t)
    out = x + (Yf - Y)[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _grow(m):
    """Selection grown by ~2 px so anti-aliased edge pixels of a stamp are included."""
    im = Image.fromarray(max_filter((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8), 5))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255.0


def unmix_ink(x, P, ink, w, r, tol):
    """Takes ink out of the pixels: pixel = paper*(1-a) + ink*a, so we find a and give back that part of the paper.
    Edges fade out smoothly, paper texture stays, nothing gets a new colour.  r = 0..1 how much to remove."""
    u = ink[None, None, :] - P
    v = x - P
    uu = (u * u).sum(2) + 1e-3
    al = (v * u).sum(2) / uu
    perp = np.sqrt(np.maximum(((v - al[..., None] * u) ** 2).sum(2), 0.0))
    t = max(14.0, tol * 1.1)
    gate = 1.0 - _smooth((perp - t) / t)
    gate *= _smooth(np.sqrt(uu) / 45.0)
    al = np.clip((al - 0.04) / 0.96, 0.0, 1.5)
    return x - u * (al * gate * w * r)[..., None]


def adj_active(a):
    return a["k"] != 100 or bool(a.get("to"))


def active_adjs(adjs):
    return [a for a in adjs if adj_active(a)]


# ----------------------------------------------------------------- look (global sliders)
# Every value is "0 = unchanged" except sat (100 = unchanged).  They act ONLY inside the selection (if there is one).
DEFAULT_LOOK = {"sat": 100, "bright": 0, "contrast": 0, "hue": 0, "warm": 0,
                "sharp": 0, "objs": 0, "detail": 0, "dark": 0, "clear": 0}


def new_look(**kw):
    d = dict(DEFAULT_LOOK)
    d.update(kw)
    return d


def tone_active(l):
    return l["sat"] != 100 or l["bright"] != 0 or l["contrast"] != 0 or l["hue"] != 0 or l["warm"] != 0


def clarity_active(l):
    return l["sharp"] > 0 or l["objs"] > 0 or l["detail"] > 0 or l["dark"] > 0 or l["clear"] > 0


def look_edited(l):
    return tone_active(l) or clarity_active(l)


def apply_tone(x, l):
    """x float32 (...,3) 0..255 -> warmth, hue, saturation, brightness, contrast."""
    if l["warm"]:
        w = l["warm"] * 0.6
        x = x + np.array([w, 0.0, -w], np.float32)
    if l["hue"]:
        a = -math.radians(l["hue"])
        c, s_ = math.cos(a), math.sin(a)
        Y = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
        I = x[..., 0] * 0.596 - x[..., 1] * 0.274 - x[..., 2] * 0.322
        Q = x[..., 0] * 0.211 - x[..., 1] * 0.523 + x[..., 2] * 0.312
        I2, Q2 = I * c - Q * s_, I * s_ + Q * c
        x = np.stack([Y + 0.956 * I2 + 0.621 * Q2, Y - 0.272 * I2 - 0.647 * Q2, Y - 1.106 * I2 + 1.703 * Q2], -1)
    x = _saturate(x, l["sat"])
    if l["bright"]:
        t = np.clip(x, 0, 255) / 255.0
        x = (t ** (2.0 ** (-l["bright"] / 50.0))) * 255.0
    if l["contrast"]:
        c = l["contrast"]
        f = (100.0 + c) / 100.0 if c < 0 else 1.0 + c / 50.0
        x = (x - 128.0) * f + 128.0
    return x


def sel_bbox(sel, pad=0, thr=0.004):
    """(y0, y1, x0, x1) of the selected area (None if empty)."""
    ys = np.nonzero(sel.max(1) > thr)[0]
    xs = np.nonzero(sel.max(0) > thr)[0]
    if ys.size == 0 or xs.size == 0:
        return None
    H, W = sel.shape
    return (max(0, int(ys[0]) - pad), min(H, int(ys[-1]) + 1 + pad),
            max(0, int(xs[0]) - pad), min(W, int(xs[-1]) + 1 + pad))


def render(rgb, adjs, look=None, lab=None, cache=None, chunk_rows=0, sel=None):
    """The whole edit chain on a uint8 picture.  lab / cache let the live preview reuse earlier work.
    sel = optional float mask (same size) 0..1: EVERYTHING (colour edits, brightness, saturation, sharpness ...) acts
    only inside it.  Rows that no edit touches are copied as they are, so a small selection costs almost nothing."""
    look = look or DEFAULT_LOOK
    act = active_adjs(adjs)
    dec = [a for a in act if a["k"] < 100]
    if cache is not None:
        keep = {("m",) + adj_key(a) for a in adjs}
        keep.add(("f",) + tuple(adj_key(a) for a in dec))
        keep |= {("g",) + adj_key(a) for a in dec}
        for k in [k for k in cache if k not in keep]:
            del cache[k]
    tone_on, clar_on = tone_active(look), clarity_active(look)
    if not act and not tone_on and not clar_on:
        return rgb
    H, W = rgb.shape[:2]
    masks = {a["id"]: get_mask(rgb, a, lab, cache) for a in act}
    # the fill must avoid ALL matching pixels, also the ones outside the selection
    F = get_fill(rgb, [masks[a["id"]] for a in dec], [adj_key(a) for a in dec], cache) if dec else None
    grown = {}
    for a in dec:
        if a["type"] == "color":
            gk = ("g",) + adj_key(a)
            g = cache.get(gk) if cache is not None else None
            if g is None:
                g = np.maximum(masks[a["id"]], _grow(masks[a["id"]]))
                if cache is not None:
                    cache[gk] = g
            grown[a["id"]] = g
    rowmax = {k: m.max(1) for k, m in masks.items()}
    sel_rows = None if sel is None else sel.max(1)
    out8 = np.empty_like(rgb)
    rows = chunk_rows or 256
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        touched = tone_on or any(float(r[y0:y1].max()) > 0.004 for r in rowmax.values())
        if touched and sel_rows is not None and float(sel_rows[y0:y1].max()) <= 0.004:
            touched = False
        if not touched:
            out8[y0:y1] = rgb[y0:y1]
            continue
        x = rgb[y0:y1].astype(np.float32)
        s = None if sel is None else sel[y0:y1]
        for a in act:
            m = masks[a["id"]][y0:y1]
            if s is not None:
                m = m * s
            if a.get("to"):                       # change this colour into another one (shading stays)
                x = x + (np.array(a["to"], np.float32) - np.array(a["rgb"], np.float32)) * m[..., None]
            if a["k"] < 100 and a["type"] == "color":
                gm = grown[a["id"]][y0:y1]
                if s is not None:
                    gm = gm * s
                x = unmix_ink(x, F[y0:y1].astype(np.float32), np.array(a["rgb"], np.float32),
                              gm, (100 - a["k"]) / 100.0, a["tol"])
            elif a["k"] < 100:
                x += (F[y0:y1].astype(np.float32) - x) * (m * ((100 - a["k"]) / 100.0))[..., None]
            elif a["k"] > 100:
                t = min((a["k"] - 100) / 100.0, 1.0)
                mm = (m * t)[..., None]
                if a["type"] == "color":      # clearer, stronger colour
                    g = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)[..., None]
                    x = g + (x - g) * (1.0 + 1.3 * mm)
                    x *= (1.0 - 0.28 * mm)
                else:                          # black / gray shade: darker
                    x *= (1.0 - mm)
        if tone_on:
            xt = apply_tone(x, look)
            x = xt if s is None else x + (xt - x) * s[..., None]
        out8[y0:y1] = np.clip(x + 0.5, 0, 255).astype(np.uint8)
    if clar_on:
        if sel is None:
            out8 = clarity(out8, look)
        else:
            bb = sel_bbox(sel, pad=16)
            if bb is not None:
                y0, y1, x0, x1 = bb
                crop = out8[y0:y1, x0:x1]
                c = clarity(crop, look).astype(np.float32)
                s = sel[y0:y1, x0:x1][..., None]
                out8[y0:y1, x0:x1] = np.clip(crop.astype(np.float32) + (c - crop) * s + 0.5, 0, 255).astype(np.uint8)
    return out8


def render_full(img, adjs, look=None, sel=None):
    """Full resolution result as a PIL image.  sel = selection mask at preview size (or None)."""
    look = look or DEFAULT_LOOK
    if not active_adjs(adjs) and not look_edited(look):
        return img
    arr = np.asarray(img)
    full_sel = None if sel is None else resize_mask(sel, img.size)
    return Image.fromarray(render(arr, adjs, look, chunk_rows=700, sel=full_sel))


# -------------------------------------------------------- shade detection
def detect_bands(lab, max_bands=6):
    """Find the different black / gray shades of this picture (deep black ... light ash).
    Returns band adjustments that split the whole 0..100 lightness range without overlap."""
    L = lab[..., 0].reshape(-1)
    C = np.hypot(lab[..., 1], lab[..., 2]).reshape(-1)
    Ln = L[C < NEUTRAL_C + 6]
    if Ln.size < 200:
        return []
    if Ln.size > 400000:
        Ln = Ln[::Ln.size // 400000]
    lo_p, hi_p = np.percentile(Ln, [0.5, 99.5])
    if hi_p - lo_p < 6:
        return [new_band_adj(float(Ln.mean()), None, None)]
    k = max_bands
    centers = np.linspace(lo_p, hi_p, k).astype(np.float64)
    for _ in range(30):
        idx = np.abs(Ln[:, None] - centers[None, :]).argmin(1)
        new = centers.copy()
        for j in range(k):
            sel = idx == j
            if sel.any():
                new[j] = Ln[sel].mean()
        if np.allclose(new, centers, atol=0.05):
            centers = new
            break
        centers = new
    idx = np.abs(Ln[:, None] - centers[None, :]).argmin(1)
    counts = np.bincount(idx, minlength=k)
    keep = counts >= max(3, 0.002 * Ln.size)
    centers, counts = centers[keep], counts[keep]
    order = np.argsort(centers)
    merged = []
    for c, n in zip(centers[order], counts[order]):
        if merged and c - merged[-1][0] < 7.0:
            tot = merged[-1][1] + n
            merged[-1][0] = (merged[-1][0] * merged[-1][1] + c * n) / tot
            merged[-1][1] = tot
        else:
            merged.append([float(c), int(n)])
    cs = [m[0] for m in merged]
    bands = []
    for i, c in enumerate(cs):
        lo = None if i == 0 else (cs[i - 1] + c) / 2.0
        hi = None if i == len(cs) - 1 else (c + cs[i + 1]) / 2.0
        bands.append(new_band_adj(c, lo, hi))
    return bands


def band_index_for(bands, L):
    for i, b in enumerate(bands):
        lo = -1e9 if b["lo"] is None else b["lo"]
        hi = 1e9 if b["hi"] is None else b["hi"]
        if lo <= L < hi:
            return i
    return 0


def classify_pick(lab_img, rgb):
    """What did the user click on?  ('bands', [...], L)  for black / gray,  ('color', [adj], L)  otherwise."""
    pt = rgb_to_lab(np.array([[list(rgb)]], dtype=np.uint8))[0, 0]
    if math.hypot(float(pt[1]), float(pt[2])) < NEUTRAL_C:
        bands = detect_bands(lab_img)
        if not bands:
            L = float(pt[0])
            bands = [new_band_adj(L, L - 10, L + 10)]
        return "bands", bands, float(pt[0])
    return "color", [new_color_adj(rgb, pt)], float(pt[0])


# ------------------------------------------------------------ shadow removal
def _max_filter_bool(m, r):
    """grow a boolean mask by r pixels (fast: PIL does the work)."""
    if r <= 0:
        return m
    im = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * int(r) + 1))
    return np.asarray(im) > 127


def _block_max(a, f):
    h, w = a.shape[:2]
    ph, pw = -(-h // f) * f, -(-w // f) * f
    pad = np.pad(a, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    return pad.reshape(ph // f, f, pw // f, f, a.shape[2]).max(axis=(1, 3))


def _filter_rgb(a, filt):
    """PIL filter on each channel of a float picture (values 0..255)."""
    return np.stack([np.asarray(Image.fromarray(np.clip(a[..., c] + 0.5, 0, 255).astype(np.uint8)).filter(filt),
                                dtype=np.float32) for c in range(a.shape[2])], -1)


def estimate_paper(small, progress=None):
    """Smooth picture of the paper lighting (float32 h,w,3) for a working-size uint8 picture.
    Pass 1 = coarse guess from block maxima.  Pass 2 = throw away everything that is ink OR sits on an edge
    (text, stamps, lines, and a margin around them where soft shadows / blur live) and fill the holes
    from the clean paper around - so ink and its fringe never leak into the lighting estimate."""
    h, w = small.shape[:2]
    a = small.astype(np.float32)
    f = max(4, int(round(max(h, w) / 60)))
    cm = _block_max(a, f)
    cm = _filter_rgb(cm, ImageFilter.MaxFilter(3))
    bg1 = _filter_rgb(_up(cm, h, w), ImageFilter.GaussianBlur(f * 0.6))
    bg1 = np.maximum(bg1, 40.0)
    if progress:
        progress(0.35, "Looking for text and edges")
    lum = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    lum1 = bg1[..., 0] * 0.299 + bg1[..., 1] * 0.587 + bg1[..., 2] * 0.114
    ratio = lum / lum1
    ink = ratio < 0.90
    sm = np.asarray(Image.fromarray(np.clip(lum + 0.5, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)),
                    dtype=np.float32)
    gy, gx = np.gradient(sm)
    edge = np.hypot(gx, gy) / lum1 > 0.035                       # relative edge strength: works in dark corners too
    bad = _max_filter_bool(ink | edge, max(2, int(round(0.004 * max(h, w)))))
    if progress:
        progress(0.55, "Rebuilding the paper lighting")
    if bad.mean() > 0.92:                                        # almost no clean paper: keep the coarse guess
        return bg1
    valid = (~bad).astype(np.float32)
    bg = fill_background(small, valid).astype(np.float32)
    # where the paper is dark because of a real shadow the fill follows it; never brighter than the lightest paper
    return np.maximum(bg, 40.0)


def _sliding_max(a, k, axis):
    """Maximum over a window of k pixels along one axis (doubling trick: only log2(k) passes)."""
    out = a
    span = 1
    while span * 2 <= k:
        sh = np.roll(out, -span, axis=axis)
        # roll wraps around: put the edge values back instead of the wrapped ones
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(-span, None)
        sh[tuple(idx)] = out[tuple(idx)]
        out = np.maximum(out, sh)
        span *= 2
    if span < k:
        sh = np.roll(out, -(k - span), axis=axis)
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(-(k - span), None)
        sh[tuple(idx)] = out[tuple(idx)]
        out = np.maximum(out, sh)
    return out


def _centered_max(a, k):
    """max filter, window k (odd), both axes, centred."""
    h = k // 2
    p = np.pad(a, ((h, h), (h, h)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    p = _sliding_max(_sliding_max(p, k, 0), k, 1)
    return p[:a.shape[0], :a.shape[1]] if h == 0 else p[h - h:h - h + a.shape[0], h - h:h - h + a.shape[1]]


def paper_light(small, progress=None, kfrac=0.022):
    """Brightness of the blank paper everywhere (float32 h,w,3) - soft AND hard-edged shadows (window blinds, a hand).
    Method: 'closing' of the picture: the lightest value in a window wider than any pen stroke removes the writing,
    shrinking it back again keeps shadow edges exactly where they are."""
    h, w = small.shape[:2]
    a = small.astype(np.float32)
    k = int(max(9, round(max(h, w) * kfrac))) | 1
    sm = np.stack([np.asarray(Image.fromarray(small[..., c]).filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
                   for c in range(3)], -1)
    mx = np.stack([_centered_max(sm[..., c], k) for c in range(3)], -1)
    if progress:
        progress(0.4, "Measuring the light")
    neg = 255.0 - mx
    bg = 255.0 - np.stack([_centered_max(neg[..., c], k) for c in range(3)], -1)   # erosion = min
    # the lightest value over-estimates a little (noise): take the typical paper level of the neighbourhood
    bg = np.stack([np.asarray(Image.fromarray(np.clip(bg[..., c] + 0.5, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(1.0, k / 6.0))), dtype=np.float32) for c in range(3)], -1)
    return np.maximum(bg * 0.985, 30.0)


def photo_mask(img):
    """Where are photos (a face on an ID card, a picture in a form)?  float32 mask (picture size, 0..1).
    A photo is a big, dense, roughly rectangular area that is not paper-white; writing and stamps are thin."""
    W, H = img.size
    sc = min(1.0, 400.0 / max(W, H))
    small = np.asarray(img.resize((max(16, round(W * sc)), max(16, round(H * sc))), Image.Resampling.BOX)) \
        if sc < 1.0 else np.asarray(img)
    h, w = small.shape[:2]
    lum = small.astype(np.float32).mean(2)
    # paper level: 90th percentile in big blocks, then the lightest neighbour block (a photo fills whole blocks)
    gb = 6
    paper = np.zeros((gb, gb), np.float32)
    for i in range(gb):
        for j in range(gb):
            blk = lum[i * h // gb:(i + 1) * h // gb, j * w // gb:(j + 1) * w // gb]
            paper[i, j] = np.percentile(blk, 90) if blk.size else 255.0
    paper = np.asarray(Image.fromarray(paper).filter(ImageFilter.MaxFilter(3)), dtype=np.float32)
    paper = _up(paper, h, w)
    nonpaper = (lum / np.maximum(paper, 30.0) < 0.88).astype(np.float32)
    cell = 12
    gh, gw = -(-h // cell), -(-w // cell)
    pad = np.pad(nonpaper, ((0, gh * cell - h), (0, gw * cell - w)))
    dens = pad.reshape(gh, cell, gw, cell).mean((1, 3))
    dense = Image.fromarray(((dens > 0.55) * 255).astype(np.uint8))
    dense = dense.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    g = np.asarray(dense) > 127
    # skin: a face / neck / hands. Medium dark and warm; the pale gold of a watermark does not count
    f = small.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    ratio = lum / np.maximum(paper, 30.0)
    skin = (Cb > 77) & (Cb < 127) & (Cr > 133) & (Cr < 173) & (ratio > 0.22) & (ratio < 0.80) & (R > G) & (G > B * 0.85)
    seen = np.zeros_like(g)
    out = np.zeros((h, w), np.float32)
    for y in range(gh):
        for x in range(gw):
            if g[y, x] and not seen[y, x]:
                stack, cells = [(y, x)], []
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    cells.append((cy, cx))
                    for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                        if 0 <= ny < gh and 0 <= nx < gw and g[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ys = [c_[0] for c_ in cells]
                xs = [c_[1] for c_ in cells]
                box = (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1)
                if not (len(cells) >= max(6, 0.012 * gh * gw) and len(cells) / box >= 0.55):
                    continue
                cy0, cy1, cx0, cx1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
                # where inside this block is a person?  cells that are mostly skin, grouped; the photo is the
                # rectangle around such a group (a watermark / text next to the photo is not part of it)
                spad = np.pad(skin, ((0, gh * cell - h), (0, gw * cell - w)))
                sd = spad.reshape(gh, cell, gw, cell).mean((1, 3))
                sk = sd > 0.25
                sk[:cy0], sk[cy1:], sk[:, :cx0], sk[:, cx1:] = False, False, False, False
                sk = np.asarray(Image.fromarray((sk * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 127
                seen2 = np.zeros_like(sk)
                for sy in range(gh):
                    for sx in range(gw):
                        if not sk[sy, sx] or seen2[sy, sx]:
                            continue
                        st2, cl = [(sy, sx)], []
                        seen2[sy, sx] = True
                        while st2:
                            qy, qx = st2.pop()
                            cl.append((qy, qx))
                            for ny in (qy - 1, qy, qy + 1):
                                for nx in (qx - 1, qx, qx + 1):
                                    if 0 <= ny < gh and 0 <= nx < gw and sk[ny, nx] and not seen2[ny, nx]:
                                        seen2[ny, nx] = True
                                        st2.append((ny, nx))
                        if len(cl) < 6:
                            continue
                        qys = [q[0] for q in cl]
                        qxs = [q[1] for q in cl]
                        y0 = max(cy0, min(qys) - 1) * cell
                        y1 = min(h, min(cy1, max(qys) + 2) * cell)
                        x0 = max(cx0, min(qxs) - 1) * cell
                        x1 = min(w, min(cx1, max(qxs) + 2) * cell)
                        npx = nonpaper[y0:y1, x0:x1] > 0.5
                        while y1 - y0 > 8 and npx[0].mean() < 0.5:                 # trim paper-coloured rim
                            y0 += 1; npx = nonpaper[y0:y1, x0:x1] > 0.5
                        while y1 - y0 > 8 and npx[-1].mean() < 0.5:
                            y1 -= 1; npx = nonpaper[y0:y1, x0:x1] > 0.5
                        while x1 - x0 > 8 and npx[:, 0].mean() < 0.5:
                            x0 += 1; npx = nonpaper[y0:y1, x0:x1] > 0.5
                        while x1 - x0 > 8 and npx[:, -1].mean() < 0.5:
                            x1 -= 1; npx = nonpaper[y0:y1, x0:x1] > 0.5
                        out[y0:y1, x0:x1] = 1.0
    if not out.any():
        return np.zeros((H, W), np.float32)
    big = Image.fromarray((out * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
    big = big.filter(ImageFilter.GaussianBlur(max(0.8, 0.0025 * max(W, H))))
    return np.asarray(big, dtype=np.float32) / 255.0


SHADOW_DEFAULTS = {"strength": 100, "white": 50, "ink": 50, "scale": 50, "mode": "color", "protect": 1}


def remove_shadow(img, strength=1.0, progress=None, opts=None):
    """'Scan' look: shadows and grey paper become white, writing / lines / shapes stay and get clearer.
    strength 0..1: how far the paper is pushed to white and the ink darkened.
    opts: white 0..100 (paper whiteness), ink 0..100 (text darkness), scale 0..100 (small shadows ... big shadows),
          mode 'color' | 'gray' | 'bw'."""
    o = dict(SHADOW_DEFAULTS)
    o.update(opts or {})
    arr = np.asarray(img)
    H, W = arr.shape[:2]
    ws = 1400.0 / max(H, W)
    small = np.asarray(img.resize((max(8, round(W * ws)), max(8, round(H * ws))), Image.Resampling.BOX)) \
        if ws < 1.0 else arr
    if progress:
        progress(0.1, "Measuring the light")
    bg = paper_light(small, progress, 0.022 * 2.0 ** ((o["scale"] - 50) / 40.0))
    s = float(max(0.0, min(1.0, strength)))
    gain = np.clip(255.0 / bg, 1.0, 6.0).astype(np.float32)
    gh, gw = gain.shape[:2]
    lum_small = small.astype(np.float32).mean(2) * 255.0 / np.maximum(bg.mean(2), 1.0)
    bp = float(np.clip(np.percentile(lum_small, 0.5), 0, 90))
    wp = 250.0 - 38.0 * s - (o["white"] - 50) * 0.45
    wp = float(max(bp + 20.0, min(254.0, wp)))
    gam = (1.0 + 0.35 * s) * (1.0 + (o["ink"] - 50) / 100.0 * 0.9)
    out = np.empty_like(arr)
    pm = photo_mask(img) if o.get("protect", 1) else None      # photos (faces ...): only the light is evened out
    if pm is not None and float(pm.max()) < 0.05:
        pm = None
    step = 600
    for y0 in range(0, H, step):
        y1 = min(H, y0 + step)
        box = (0.0, y0 * gh / H, float(gw), y1 * gh / H)
        g = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(gain[..., c])).resize(
            (W, y1 - y0), Image.Resampling.BILINEAR, box=box), dtype=np.float32) for c in range(3)], -1)
        x = np.minimum(arr[y0:y1].astype(np.float32) * g, 255.0)
        x = np.clip((x - bp) / max(1.0, wp - bp), 0.0, 1.0) ** gam
        if o["mode"] in ("gray", "bw"):
            gray = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)
            if o["mode"] == "bw":
                gray = np.clip((gray - 0.45) / 0.12 + 0.5, 0.0, 1.0)      # hard but anti-aliased threshold
            x = np.repeat(gray[..., None], 3, 2)
        res = np.clip(x * 255.0 + 0.5, 0, 255)
        if pm is not None and float(pm[y0:y1].max()) > 0.01:
            gp = np.minimum(1.0 + 0.6 * (g - 1.0), 1.3)           # soft correction: no contrast / ink changes
            xp = np.minimum(arr[y0:y1].astype(np.float32) * gp, 255.0)
            if o["mode"] in ("gray", "bw"):
                xp = np.repeat((xp[..., 0] * 0.299 + xp[..., 1] * 0.587 + xp[..., 2] * 0.114)[..., None], 3, 2)
            a_ = pm[y0:y1, :, None]
            res = res * (1.0 - a_) + xp * a_
        out[y0:y1] = np.clip(res + 0.5, 0, 255).astype(np.uint8)
        if progress:
            progress(0.6 + 0.4 * y1 / H, "Applying")
    res = Image.fromarray(out)
    return res


# --------------------------------------------------------------------- crop
def order_quad(pts):
    """Any 4 points -> top-left, top-right, bottom-right, bottom-left."""
    p = [(float(x), float(y)) for x, y in pts]
    s = [x + y for x, y in p]
    d = [x - y for x, y in p]
    tl, br = p[int(np.argmin(s))], p[int(np.argmax(s))]
    tr, bl = p[int(np.argmax(d))], p[int(np.argmin(d))]
    return [tl, tr, br, bl]


def _coeffs(out_pts, in_pts):
    A, B = [], []
    for (x, y), (X, Y) in zip(out_pts, in_pts):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        B.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        B.append(Y)
    return np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)).tolist()


def warp_quad(img, pts):
    """Straighten the area inside 4 points into a perfect rectangle (like Photoshop's perspective crop)."""
    tl, tr, br, bl = order_quad(pts)
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    w = max(1, int(round(max(dist(tl, tr), dist(bl, br)))))
    h = max(1, int(round(max(dist(tl, bl), dist(tr, br)))))
    if w < 4 or h < 4:
        raise ValueError("The 4 points are too close together")
    try:
        co = _coeffs([(0, 0), (w, 0), (w, h), (0, h)], [tl, tr, br, bl])
    except np.linalg.LinAlgError:
        raise ValueError("The 4 points do not make a proper shape")
    return img.transform((w, h), Image.Transform.PERSPECTIVE, co, Image.Resampling.BICUBIC)


def crop_box(img, box):
    l, t, r, b = box
    W, H = img.size
    l, t = max(0, int(round(l))), max(0, int(round(t)))
    r, b = min(W, int(round(r))), min(H, int(round(b)))
    if r - l < 2 or b - t < 2:
        raise ValueError("The crop area is too small")
    return img.crop((l, t, r, b))


# ------------------------------------------------------------- eraser / clone
def make_mask(size, shape, data, brush=20.0):
    """Mask (PIL 'L', 255 = erase) from what the user drew, in real picture pixels.
    shape: 'brush' (list of points, brush = width), 'rect' / 'ellipse' ((x0,y0,x1,y1)), 'lasso' (list of points)."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if shape == "brush":
        w = max(1.0, float(brush))
        r = w / 2.0
        pts = [(float(x), float(y)) for x, y in data]
        if len(pts) > 1:
            d.line(pts, fill=255, width=int(round(w)), joint="curve")
        for x, y in pts[:: max(1, len(pts) // 400)] + pts[-1:]:
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    elif shape == "rect":
        x0, y0, x1, y1 = data
        d.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), fill=255)
    elif shape == "ellipse":
        x0, y0, x1, y1 = data
        d.ellipse((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), fill=255)
    elif shape == "lasso":
        pts = [(float(x), float(y)) for x, y in data]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    return m


def _gray_from_L(L):
    fy = (L + 16.0) / 116.0
    y = np.where(fy ** 3 > 0.008856, fy ** 3, (fy - 16.0 / 116.0) / 7.787)
    v = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 1e-6), 1 / 2.4) - 0.055)
    return np.clip(v, 0, 1) * 255.0


def _keep_dark_ink(reg, hole_img, res):
    """A coloured stamp lies on top of black writing: where the removed pixels are clearly DARKER than the stamp ink,
    the writing is still there - keep it (as neutral dark) instead of painting paper over it."""
    hm = np.asarray(hole_img) > 127
    if not hm.any():
        return res
    lab = rgb_to_lab(reg)
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    ink = hm & (C > 18.0)
    if int(ink.sum()) < 20 or int(ink.sum()) < 0.04 * int(hm.sum()):
        return res                                   # what is removed is not coloured ink: nothing to keep
    Lref = float(np.median(L[ink]))
    s = np.clip((Lref - L - 6.0) / (0.4 * Lref + 1.0), 0.0, 1.0) ** 0.7
    s = (s * hm)[..., None].astype(np.float32)
    gray = np.repeat(_gray_from_L(L)[..., None], 3, 2)
    return res * (1 - s) + gray * s


def _ink_like(reg, F):
    """Pixels that clearly differ from the paper estimate F (dark / coloured: writing, lines, stamps)."""
    d = np.abs(reg.astype(np.float32) - F.astype(np.float32)).sum(2)
    return d > 22.0


def _grow_over_ink(reg, hole):
    """The brush covers most of a letter, but its soft anti-aliased edge sticks out. Grow the hole 1-3 px, but only
    over pixels that are ink-like, so it follows the strokes and does not eat paper."""
    h0 = hole > 0.5
    F = fill_background(reg, (~h0).astype(np.float32))
    ink = _ink_like(reg, F)
    cur = h0
    for _ in range(3):
        grow = np.asarray(Image.fromarray((cur * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 127
        cur = cur | (grow & ink)
    out = cur.astype(np.float32)
    return out, Image.fromarray((cur * 255).astype(np.uint8))


def _paper_like(reg, valid):
    """1 where a valid pixel looks like plain paper (not ink, not the dim halo around ink)."""
    F = fill_background(reg, valid)
    ink = _ink_like(reg, F)
    ink = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 127
    return (~ink).astype(np.float32)


def erase_region(img, mask, offset=None, feather=1.5, keep_ink=False):
    """Removes what is under the mask and puts there what surrounds it, so it looks as if nothing was there.
    offset=(dx,dy): copy from that spot instead (clone / heal): the texture comes from (x+dx, y+dy),
    the brightness and colour are matched to the surroundings."""
    hard = mask.point(lambda v: 255 if v > 16 else 0)
    bbox = hard.getbbox()
    if bbox is None:
        return img
    W, H = img.size
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    m = max(24, int(0.6 * max(bw, bh)))
    box = (max(0, bbox[0] - m), max(0, bbox[1] - m), min(W, bbox[2] + m), min(H, bbox[3] + m))
    full = np.asarray(img)
    reg = full[box[1]:box[3], box[0]:box[2]].copy()
    hole_img = hard.crop(box).filter(ImageFilter.MaxFilter(5))
    hole = np.asarray(hole_img, dtype=np.float32) / 255.0
    if offset is None:
        hole, hole_img = _grow_over_ink(reg, hole)           # the soft edge of the writing goes too: no grey ghost
    valid = 1.0 - hole
    if offset is None:                                       # only real paper counts for the fill, never ink / halo
        valid = valid * _paper_like(reg, valid)
    F = fill_background(reg, valid).astype(np.float32)

    # soft edge: grow a little, then blur, so nothing of the old picture is left at the border
    f = max(0.8, float(feather))
    k = 2 * int(math.ceil(f)) + 1
    soft = np.asarray(mask.crop(box).filter(ImageFilter.MaxFilter(max(3, k))).filter(ImageFilter.GaussianBlur(f)),
                      dtype=np.float32) / 255.0
    soft = np.clip(soft, 0, 1)[..., None]

    ring = (np.asarray(hole_img.filter(ImageFilter.MaxFilter(25)), dtype=np.float32) / 255.0 > 0.5) & (hole < 0.5)
    if offset is None:
        # grain: add noise like the surrounding picture has, so the patch does not look too flat
        sm = np.asarray(Image.fromarray(reg).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
        hf = reg.astype(np.float32) - sm
        res = F
        if int(ring.sum()) > 80:
            h = hf[ring]
            sig = 1.4826 * np.median(np.abs(h - np.median(h, axis=0)), axis=0)     # robust: text edges do not count
            rng = np.random.default_rng(12345)
            nz = rng.standard_normal(F.shape).astype(np.float32)
            p_ = np.pad(nz, ((1, 1), (1, 1), (0, 0)), mode="edge")             # slight blur: grain, not white noise
            nz = (2 * nz + p_[:-2, 1:-1] + p_[2:, 1:-1] + p_[1:-1, :-2] + p_[1:-1, 2:]) / 6.0
            nz /= max(1e-3, float(nz.std()))
            res = F + nz * sig * 0.9
    else:
        dx, dy = int(round(offset[0])), int(round(offset[1]))
        ys = np.clip(np.arange(box[1], box[3]) + dy, 0, H - 1)
        xs = np.clip(np.arange(box[0], box[2]) + dx, 0, W - 1)
        src = full[np.ix_(ys, xs)].astype(np.float32)
        res = src
        if int(ring.sum()) > 50:                   # heal: same texture, but the tone of the new surroundings
            res = src + (np.median(reg[ring].astype(np.float32), axis=0) - np.median(src[ring], axis=0))
    if keep_ink and offset is None:
        res = _keep_dark_ink(reg, hard.crop(box), res)
    out = reg.astype(np.float32) * (1 - soft) + res * soft
    result = img.copy()
    result.paste(Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)), (box[0], box[1]))
    return result


# --------------------------------------------------------------------- save
SAVE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}


def out_ext(src_name):
    e = os.path.splitext(src_name)[1].lower()
    return e if e in SAVE_EXT else ".png"


def unique_path(folder, base, ext):
    p = os.path.join(folder, base + ext)
    n = 2
    while os.path.exists(p):
        p = os.path.join(folder, "%s (%d)%s" % (base, n, ext))
        n += 1
    return p


def save_image(img, path):
    e = os.path.splitext(path)[1].lower()
    if e in (".jpg", ".jpeg"):
        img.save(path, "JPEG", quality=95, subsampling=0, optimize=True)   # 4:4:4 keeps thin text sharp
    elif e == ".webp":
        img.save(path, "WEBP", quality=95)
    elif e in (".tif", ".tiff"):
        img.save(path, "TIFF", compression="tiff_lzw")
    elif e == ".bmp":
        img.save(path, "BMP")
    elif e == ".pdf":
        save_pdf([img], path)
    else:
        img.save(path, "PNG")


# ------------------------------------------------------------------- selection
# A selection is a float32 mask (0..1) at preview size, or None = nothing selected (edits act on the whole picture).
def resize_mask(m, size):
    """size = (width, height) in pixels."""
    w, h = size
    if m.shape[1] == w and m.shape[0] == h:
        return m
    im = Image.fromarray(np.ascontiguousarray(m, dtype=np.float32))
    return np.asarray(im.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)


def sel_combine(old, new, mode):
    """mode: 'new' | 'add' | 'sub'.  Returns the new selection (None when empty)."""
    if mode == "add" and old is not None:
        out = np.maximum(old, new)
    elif mode == "int" and old is not None:
        out = old * new
    elif mode == "sub":
        if old is None:
            return None
        out = old * (1.0 - new)
    else:
        out = new
    return out.astype(np.float32) if float(out.max()) > 0.02 else None


def sel_coverage(m):
    return 0.0 if m is None else float((m > 0.5).mean())


def shape_mask(size, shape, data, brush=20.0):
    """Float mask from a drawn rectangle / ellipse / lasso / brush (coordinates in the same pixels as size=(w,h))."""
    return np.asarray(make_mask(size, shape, data, brush), dtype=np.float32) / 255.0


def color_range_mask(lab, rgb, tol=30.0):
    """Every pixel whose colour matches rgb (like Photoshop's Color Range). Coloured targets match by hue + colour
    strength (so a dark pixel with a faint tint is not taken); white / grey / black match by lightness."""
    pt = rgb_to_lab(np.array([[list(rgb)]], dtype=np.uint8))[0, 0]
    return adj_mask(lab, {"type": "color", "target": [float(pt[0]), float(pt[1]), float(pt[2])], "tol": float(tol)})


def quick_select(lab, stroke, tol=28.0):
    """Brush over an object: the selection grows from the stroke over connected pixels that look like what was
    brushed, and stops at the object's boundary (where the colour changes)."""
    H, W = stroke.shape
    ys, xs = np.nonzero(stroke > 0.5)
    if ys.size == 0:
        return np.zeros((H, W), np.float32)
    margin = int(max(60, 0.3 * max(H, W)))
    y0, y1 = max(0, int(ys.min()) - margin), min(H, int(ys.max()) + 1 + margin)
    x0, x1 = max(0, int(xs.min()) - margin), min(W, int(xs.max()) + 1 + margin)
    sub = lab[y0:y1, x0:x1]
    st = stroke[y0:y1, x0:x1] > 0.5
    vals = sub[st]
    ref = np.median(vals, axis=0)
    spread = float(np.mean(1.4826 * np.median(np.abs(vals - ref), axis=0)))
    thr = tol + min(18.0, 2.0 * spread)
    d = np.sqrt(((sub[..., 0] - ref[0]) * 0.8) ** 2 + (sub[..., 1] - ref[1]) ** 2 + (sub[..., 2] - ref[2]) ** 2)
    soft = 1.0 - _smooth((d - thr * 0.75) / (thr * 0.5))
    cand = soft > 0.5
    seed = st & cand
    if not seed.any():
        seed = st
    cur = Image.fromarray((seed * 255).astype(np.uint8))
    allow = Image.fromarray((cand * 255).astype(np.uint8))
    prev_n = int(seed.sum())
    for _ in range(max(6, margin // 2)):               # grow only through matching pixels (stays connected)
        nxt = ImageChops.multiply(cur.filter(ImageFilter.MaxFilter(5)), allow)
        n = int(np.count_nonzero(np.asarray(nxt)))
        cur = nxt
        if n == prev_n:
            break
        prev_n = n
    region = np.asarray(cur, dtype=np.float32) / 255.0
    region = np.maximum(region * soft, st.astype(np.float32))
    region = np.asarray(Image.fromarray((region * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)),
                        dtype=np.float32) / 255.0
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = region
    return out


def shift_mask(m, dx, dy):
    """The same selection moved by (dx, dy) pixels; what leaves the picture is cut off."""
    H, W = m.shape
    out = np.zeros_like(m)
    sx0, sx1 = max(0, -dx), min(W, W - dx)
    sy0, sy1 = max(0, -dy), min(H, H - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = m[sy0:sy1, sx0:sx1]
    return out


def move_region(img, mask, dx, dy, copy=False, feather=1.5):
    """Takes the selected part of the picture to another place.  mask = PIL 'L' mask at the picture's size.
    copy=False: the old place is filled from its surroundings (like erasing it); copy=True: it stays."""
    hard = mask.point(lambda v: 255 if v > 16 else 0)
    bbox = hard.getbbox()
    if bbox is None:
        return img
    base = img.copy() if copy else erase_region(img, mask, None, feather)
    patch = img.crop(bbox)
    pm = mask.crop(bbox).filter(ImageFilter.GaussianBlur(0.8))
    base.paste(patch, (bbox[0] + int(dx), bbox[1] + int(dy)), pm)
    return base


# --------------------------------------------------------------------- rotate
def border_color(img):
    a = np.asarray(img)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], 0)
    return tuple(int(v) for v in np.median(edge, axis=0))


def rotate_image(img, deg_ccw):
    """Turn the picture counter-clockwise by deg_ccw degrees. Multiples of 90 are lossless; any other angle
    makes the picture bigger and fills the corners with the colour of the picture's border."""
    d = float(deg_ccw) % 360.0
    if abs(d) < 1e-6 or abs(d - 360.0) < 1e-6:
        return img
    for angle, t in ((90.0, Image.Transpose.ROTATE_90), (180.0, Image.Transpose.ROTATE_180),
                     (270.0, Image.Transpose.ROTATE_270)):
        if abs(d - angle) < 1e-6:
            return img.transpose(t)
    return img.rotate(d, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=border_color(img))


# ------------------------------------------------------------------------ PDF
PAGE_SIZES = {"a4": (595.276, 841.89), "letter": (612.0, 792.0)}


def pdf_page(w, h, mode):
    """(page width, page height, image x, y, drawn width, drawn height) in PDF points.
    'fit'  : the page has the picture's own proportions (nothing is cut or padded)
    'a4' / 'letter' : standard paper, turned to match the picture, picture centred and as big as it fits."""
    if mode in PAGE_SIZES:
        pw, ph = PAGE_SIZES[mode]
        if w > h:
            pw, ph = ph, pw
        sc = min(pw / w, ph / h)
        dw, dh = w * sc, h * sc
        return pw, ph, (pw - dw) / 2, (ph - dh) / 2, dw, dh
    long = 841.89                                      # long side = A4 long side, short side follows the picture
    sc = long / max(w, h)
    return w * sc, h * sc, 0.0, 0.0, w * sc, h * sc


def save_pdf(sources, path, mode="fit", quality=95, progress=None):
    """One PDF, one page per picture, in the given order.  sources = list of PIL images or functions that return one
    (so a long list never sits in memory all at once).  The picture is embedded at full resolution as high-quality
    JPEG (4:4:4, nothing is resized)."""
    import io
    n = len(sources)
    if n == 0:
        raise ValueError("Nothing to save")
    offsets = {}
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        def put(num, body):
            offsets[num] = f.tell()
            f.write(b"%d 0 obj\n" % num + body + b"\nendobj\n")

        for i, src in enumerate(sources):
            if progress:
                progress(i / n, "Page %d of %d" % (i + 1, n))
            im = src() if callable(src) else src
            im = im.convert("RGB")
            w, h = im.size
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
            data = buf.getvalue()
            pw, ph, x, y, dw, dh = pdf_page(w, h, mode)
            base = 3 + 3 * i
            put(base, ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /XObject << /Im0 %d 0 R >> "
                       "/ProcSet [/PDF /ImageC] >> /Contents %d 0 R >>" % (pw, ph, base + 2, base + 1)).encode())
            content = ("q %.3f 0 0 %.3f %.3f %.3f cm /Im0 Do Q" % (dw, dh, x, y)).encode()
            put(base + 1, b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
            put(base + 2, b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                          b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n" % (w, h, len(data))
                + data + b"\nendstream")
            del im, data, buf
        kids = " ".join("%d 0 R" % (3 + 3 * i) for i in range(n))
        put(2, ("<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n)).encode())
        put(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        total = 3 + 3 * n
        xref = f.tell()
        f.write(b"xref\n0 %d\n0000000000 65535 f \n" % total)
        for num in range(1, total):
            f.write(b"%010d 00000 n \n" % offsets[num])
        f.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (total, xref))
    if progress:
        progress(1.0, "Done")


def sel_invert(m, shape):
    """Everything that is not selected.  With no selection: the whole picture."""
    if m is None:
        return np.ones(shape, np.float32)
    out = (1.0 - m).astype(np.float32)
    return out if float(out.max()) > 0.02 else None


def sel_grow(m, px):
    """Selection bigger (px > 0) or smaller (px < 0) by about px pixels."""
    if m is None or px == 0:
        return m
    im = Image.fromarray((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8))
    rem = abs(int(px))
    filt = ImageFilter.MaxFilter if px > 0 else ImageFilter.MinFilter
    while rem > 0:
        r = min(rem, 6)
        im = im.filter(filt(2 * r + 1))
        rem -= r
    im = im.filter(ImageFilter.GaussianBlur(0.7))
    out = np.asarray(im, dtype=np.float32) / 255.0
    return out if float(out.max()) > 0.02 else None


def fill_region(img, mask, rgb):
    """Paints the masked area with one colour. mask = PIL 'L' at the picture's size."""
    solid = Image.new("RGB", img.size, tuple(int(v) for v in rgb))
    return Image.composite(solid, img, mask)


def transform_region(img, mask, cx, cy, sx, sy, ang_cw, soft=1.0):
    """Scales / rotates the selected part around its own centre and puts it at (cx, cy) (picture pixels).
    The old place is filled from its surroundings.  mask = PIL 'L'.  Returns (new picture, new mask 'L')."""
    hard = mask.point(lambda v: 255 if v > 16 else 0)
    bbox = hard.getbbox()
    if bbox is None:
        return img, mask
    base = erase_region(img, mask, None, soft)
    patch = img.crop(bbox).convert("RGBA")
    pm = mask.crop(bbox)
    patch.putalpha(pm)
    w, h = patch.size
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    patch = patch.resize((nw, nh), Image.Resampling.BICUBIC)
    patch = patch.rotate(-ang_cw, resample=Image.Resampling.BICUBIC, expand=True)
    px, py = int(round(cx - patch.width / 2.0)), int(round(cy - patch.height / 2.0))
    base.paste(patch.convert("RGB"), (px, py), patch.split()[3])
    newmask = Image.new("L", img.size, 0)
    newmask.paste(patch.split()[3], (px, py))
    return base, newmask


def homography(src, dst):
    """3x3 matrix H with  dst ~ H @ src  for 4 point pairs."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A, np.float64), np.array(b, np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def warp_region(img, mask, grid, n, soft=1.0):
    """Perspective / distort / skew / warp of the selected part.
    grid = (n+1)*(n+1) points (x, y) in picture pixels, row by row: where the corners of the selection (n = 1) or
    the points of a 4x4 net (n = 3) are now.  The selected part is bent onto them; the old place is filled from the
    surroundings.  Returns (new picture, new mask 'L')."""
    hard = mask.point(lambda v: 255 if v > 16 else 0)
    bbox = hard.getbbox()
    if bbox is None:
        return img, mask
    base = erase_region(img, mask, None, soft).convert("RGBA")
    patch = img.crop(bbox).convert("RGBA")
    patch.putalpha(mask.crop(bbox))
    pw, ph = patch.size
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = [[grid[i * (n + 1) + j] for j in range(n + 1)] for i in range(n + 1)]
    for i in range(n):
        for j in range(n):
            sx0, sx1 = j * pw / n, (j + 1) * pw / n
            sy0, sy1 = i * ph / n, (i + 1) * ph / n
            dq = [g[i][j], g[i][j + 1], g[i + 1][j + 1], g[i + 1][j]]
            xs = [p[0] for p in dq]; ys = [p[1] for p in dq]
            bx0, by0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
            bx1, by1 = min(W, int(math.ceil(max(xs)))), min(H, int(math.ceil(max(ys))))
            if bx1 - bx0 < 1 or by1 - by0 < 1:
                continue
            m = 0 if n == 1 else 2                       # a little overlap so the cells have no seams
            cx0, cy0 = max(0, int(sx0) - m), max(0, int(sy0) - m)
            cx1, cy1 = min(pw, int(math.ceil(sx1)) + m), min(ph, int(math.ceil(sy1)) + m)
            crop = patch.crop((cx0, cy0, cx1, cy1))
            sq = [(sx0 - cx0, sy0 - cy0), (sx1 - cx0, sy0 - cy0), (sx1 - cx0, sy1 - cy0), (sx0 - cx0, sy1 - cy0)]
            dloc = [(p[0] - bx0, p[1] - by0) for p in dq]
            try:
                Hm = homography(dloc, sq)                # output -> source, what PIL wants
            except np.linalg.LinAlgError:
                continue
            Hm = Hm / Hm[2, 2]
            co = (Hm[0, 0], Hm[0, 1], Hm[0, 2], Hm[1, 0], Hm[1, 1], Hm[1, 2], Hm[2, 0], Hm[2, 1])
            piece = crop.transform((bx1 - bx0, by1 - by0), Image.Transform.PERSPECTIVE, co, Image.Resampling.BICUBIC)
            layer.alpha_composite(piece, (bx0, by0))
    out = Image.alpha_composite(base, layer).convert("RGB")
    return out, layer.split()[3]


def remove_stamp(img, mask):
    """Removes a stamp / seal (the selected part) but KEEPS a black signature that crosses it.
    The stamp becomes paper again (filled from around); dark neutral strokes inside the selection are kept, their
    blue tint is taken out, and small gaps in the strokes are bridged so the signature looks complete.
    mask = PIL 'L' selection at the picture's size."""
    hard = mask.point(lambda v: 255 if v > 40 else 0)
    bbox = hard.getbbox()
    if bbox is None:
        return img
    W, H = img.size
    m = 14
    box = (max(0, bbox[0] - m), max(0, bbox[1] - m), min(W, bbox[2] + m), min(H, bbox[3] + m))
    full = np.asarray(img)
    reg = full[box[1]:box[3], box[0]:box[2]].copy()
    hole_img = hard.crop(box).filter(ImageFilter.MaxFilter(7))
    hole = np.asarray(hole_img) > 127
    F = fill_background(reg, (~hole).astype(np.float32)).astype(np.float32)
    lab = rgb_to_lab(reg)
    Lp = np.maximum(rgb_to_lab(np.clip(F, 0, 255).astype(np.uint8))[..., 0], 40.0)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    sig = hole & (L < 0.58 * Lp) & (C < 45.0)                    # dark, hardly coloured = the signature
    gray = (reg[..., 0] * 0.299 + reg[..., 1] * 0.587 + reg[..., 2] * 0.114).astype(np.float32)
    out = F.copy()
    if sig.any():
        a0 = np.clip((0.75 * Lp - L) / (0.30 * Lp), 0.0, 1.0)
        a0 = np.where(sig, a0, 0.0)[..., None]
        sg = np.repeat(gray[..., None], 3, 2)
        out = out * (1 - a0) + sg * a0
        # bridge small gaps in the strokes (the stamp ink was hiding them)
        s_img = Image.fromarray((sig * 255).astype(np.uint8))
        closed = np.asarray(s_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))) > 127
        gap = closed & ~sig & hole
        if gap.any():
            col = float(np.median(gray[sig]))
            gsoft = np.asarray(Image.fromarray((gap * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)),
                               dtype=np.float32)[..., None] / 255.0 * 0.85
            out = out * (1 - gsoft) + col * gsoft
    soft = np.asarray(mask.crop(box).filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.5)),
                      dtype=np.float32)[..., None] / 255.0
    res = reg.astype(np.float32) * (1 - soft) + out * soft
    result = img.copy()
    result.paste(Image.fromarray(np.clip(res + 0.5, 0, 255).astype(np.uint8)), (box[0], box[1]))
    return result
