"""Document scanning maths (no GUI): find the paper in a photo, and straighten it."""
import math

import numpy as np
from PIL import Image, ImageFilter

import core

LABEL_SIDE = 220            # long side of the picture used to find the paper


def _otsu(g):
    hist = np.bincount(g.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = (np.arange(256) * hist).sum()
    wb = 0.0
    sb = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
        mb, mf = sb / wb, (sum_all - sb) / wf
        between = wb * wf * (mb - mf) ** 2
        if between > best:
            best, thr = between, t
    return thr


def _components(mask):
    """Connected regions of a boolean mask (4-neighbours). Returns (labels (h,w) int32, 0 = background)."""
    h, w = mask.shape
    big = h * w + 1
    lab = np.where(mask, np.arange(1, h * w + 1, dtype=np.int32).reshape(h, w), big).astype(np.int32)
    for _ in range(h + w + 10):
        new = lab.copy()
        np.minimum(new[1:], lab[:-1], out=new[1:])
        np.minimum(new[:-1], lab[1:], out=new[:-1])
        np.minimum(new[:, 1:], lab[:, :-1], out=new[:, 1:])
        np.minimum(new[:, :-1], lab[:, 1:], out=new[:, :-1])
        new = np.where(mask, new, big)
        # pointer jumping speeds up long thin shapes
        flat = new.reshape(-1)
        idx = np.where(flat < big, flat - 1, 0)
        jumped = np.where(flat < big, np.minimum(flat, flat[idx]), big)
        new = jumped.reshape(h, w)
        if np.array_equal(new, lab):
            break
        lab = new
    lab = np.where(mask, lab, 0)
    return lab


def _hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def _poly_area(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2.0


def _simplify_to_quad(hull):
    """Remove the least important corner of the convex outline until 4 remain (Visvalingam)."""
    p = list(hull)
    while len(p) > 4:
        best, bi = 1e18, 0
        for i in range(len(p)):
            a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
            ar = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
            if ar < best:
                best, bi = ar, i
        p.pop(bi)
    return p


def _refine_quad(quad, outline):
    """Fit a straight line to the outline pixels along each side and use the lines' crossings as corners
    (the simplified outline cuts corners; this puts them back)."""
    pts = np.array(outline, dtype=np.float64)
    q = np.array(quad, dtype=np.float64)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = np.hypot(*d)
        if L < 1e-6:
            return quad
        n = np.array([-d[1], d[0]]) / L
        dist = np.abs((pts - a) @ n)
        t = ((pts - a) @ d) / (L * L)
        sel = pts[(dist < max(2.5, 0.03 * L)) & (t > -0.05) & (t < 1.05)]
        if len(sel) < 6:
            return quad
        c = sel.mean(0)
        _, _, vt = np.linalg.svd(sel - c)
        lines.append((c, vt[0]))
    out = []
    for i in range(4):
        c1, d1 = lines[i - 1]
        c2, d2 = lines[i]
        M = np.array([d1, -d2]).T
        if abs(np.linalg.det(M)) < 1e-6:
            return quad
        t = np.linalg.solve(M, c2 - c1)
        out.append(tuple(c1 + t[0] * d1))
    return out


def _score(mask):
    h, w = mask.shape
    lab = _components(mask)
    counts = np.bincount(lab.reshape(-1))
    counts[0] = 0
    if counts.max() == 0:
        return None
    best = None
    for cid in np.argsort(counts)[::-1][:3]:
        area = int(counts[cid])
        f = area / float(h * w)
        if f < 0.10:
            break
        comp = lab == cid
        # outline pixels only (fast hull)
        er = comp.copy()
        er[1:-1, 1:-1] &= comp[:-2, 1:-1] & comp[2:, 1:-1] & comp[1:-1, :-2] & comp[1:-1, 2:]
        ys, xs = np.nonzero(comp & ~er)
        hull = _hull(list(zip(xs.tolist(), ys.tolist())))
        if len(hull) < 3:
            continue
        ha = _poly_area(hull)
        solidity = area / max(ha, 1.0)
        sides = int(comp[0].any()) + int(comp[-1].any()) + int(comp[:, 0].any()) + int(comp[:, -1].any())
        if f > 0.97 or sides >= 4:
            continue
        sc = (min(1.0, solidity) ** 3) * (f ** 0.5) * (1.0 - 0.12 * sides)
        if best is None or sc > best[0]:
            best = (sc, hull, list(zip(xs.tolist(), ys.tolist())))
    return best


def detect_document(img):
    """Finds the 4 corners of the paper. Returns (corners [(x,y) TL,TR,BR,BL in picture pixels], found: bool).
    If no paper is found the corners are the picture's own corners with a small margin (the user fixes them by hand)."""
    W, H = img.size
    s = LABEL_SIDE / float(max(W, H))
    small = img.convert("L").resize((max(8, round(W * s)), max(8, round(H * s))), Image.Resampling.BOX)
    small = small.filter(ImageFilter.GaussianBlur(1.2))
    g = np.asarray(small)
    t = _otsu(g)
    cands = []
    for dt in (0, -18, 18, -36, 36):
        thr = int(min(250, max(5, t + dt)))
        for inv in (False, True):
            m = (g <= thr) if inv else (g > thr)
            sc = _score(m)
            if sc is not None:
                cands.append(sc)
    gh, gw = g.shape
    if cands:
        sc, hull, outline = max(cands, key=lambda c: c[0])
        quad = _simplify_to_quad(hull)
        if len(quad) == 4:
            quad = _refine_quad(core.order_quad(quad), outline)
        if len(quad) == 4 and _poly_area(quad) > 0.08 * gh * gw:
            pts = core.order_quad(quad)
            return [(x / s, y / s) for x, y in pts], True
    mx, my = 0.04 * W, 0.04 * H
    return [(mx, my), (W - mx, my), (W - mx, H - my), (mx, H - my)], False


def aspect_ratio(quad, size):
    """width / height of the real sheet, from its photographed corners (TL,TR,BR,BL) - the camera angle is worked out
    from the four corners, so a tilted photo of an A5 page comes out as A5, not as a stretched rectangle."""
    W, H = size
    u0, v0 = W / 2.0, H / 2.0
    tl, tr, br, bl = quad
    m1, m2, m3, m4 = [np.array([x, y, 1.0]) for x, y in (tl, tr, bl, br)]
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    simple = (dist(tl, tr) + dist(bl, br)) / max(1e-6, (dist(tl, bl) + dist(tr, br)))
    try:
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1
        den = n2[2] * n3[2]
        if abs(den) < 1e-9:                      # camera looks straight at the page
            return simple
        f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0) +
               (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / den
        if not np.isfinite(f2) or f2 <= 0:
            return simple
        f = math.sqrt(f2)
        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])
        Ai = np.linalg.inv(A)
        M = Ai.T @ Ai
        r = math.sqrt(max(1e-12, float(n2 @ M @ n2)) / max(1e-12, float(n3 @ M @ n3)))
        if not np.isfinite(r) or r < 0.2 or r > 5.0 or abs(math.log(r / simple)) > 0.45:
            return simple                        # estimate is unreliable: fall back to the edge lengths
        return r
    except Exception:
        return simple


def straighten(img, quad):
    """Crop the paper out and remove the perspective. Keeps the sheet's real proportions (A4, A5, Letter, ID card ...)."""
    tl, tr, br, bl = core.order_quad(quad)
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    ratio = aspect_ratio((tl, tr, br, bl), img.size)
    w0 = max(dist(tl, tr), dist(bl, br))
    h0 = max(dist(tl, bl), dist(tr, br))
    if w0 / max(h0, 1e-6) > ratio:               # use the larger side as the measure so nothing loses detail
        h = h0
        w = h0 * ratio
        if w < w0:
            w, h = w0, w0 / ratio
    else:
        w = w0
        h = w0 / ratio
        if h < h0:
            w, h = h0 * ratio, h0
    w, h = max(8, int(round(w))), max(8, int(round(h)))
    co = core._coeffs([(0, 0), (w, 0), (w, h), (0, h)], [tl, tr, br, bl])
    return img.transform((w, h), Image.Transform.PERSPECTIVE, co, Image.Resampling.BICUBIC)
