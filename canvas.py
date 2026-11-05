"""The big picture ("card") with its tools.  Works in preview pixels; the window converts to real pixels."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

import core
from theme import dropper_cursor

HANDLE = 12                 # screen pixels around a crop edge that grab it


def rgb_qimage(arr):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888), arr


def rgba_qimage(arr):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888), arr


class Canvas(QWidget):
    picked = Signal(int, int)               # eyedropper: x, y (preview pixels)
    eraseDone = Signal(str, object, float)  # shape, data, brush size
    moveDone = Signal(int, int, bool)       # dx, dy (preview pixels), copy?
    slide = Signal(int)                     # -1 / +1 (Shift + wheel)
    selChanged = Signal(bool)               # has a selection
    message = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.arr = None
        self.orig = None
        self.show_orig = False
        self._img = None
        self._keep = None
        self.z, self.ox, self.oy = 1.0, 0.0, 0.0
        self.tool = "hand"
        self.shape = "brush"
        self.brush = 24.0
        self.selmode = "add"
        self.copy = False
        self.sel = None
        self.sel_hist = []
        self._sel_img = None
        self._sel_keep = None
        self.crop = None                    # [x0, y0, x1, y1] image pixels
        self._pts = []
        self._act = None                    # current mouse action
        self._hover = None
        self._patch = None
        self._off = (0.0, 0.0)
        self._pan0 = None
        self._set_cursor()

    # ------------------------------------------------------------ settings from the window
    def set_tool(self, t):
        if t != self.tool:
            self._act = None
            self._pts = []
            if t != "select":
                self.clear_selection()
            self.crop = None
        self.tool = t
        if t == "crop":
            self.start_crop()
        self._set_cursor()
        self.update()

    def set_shape(self, s):
        self.shape = s
        self.update()

    def set_brush(self, v):
        self.brush = float(v)
        self.update()

    def set_selmode(self, m):
        self.selmode = m

    def set_copy(self, c):
        self.copy = bool(c)

    def set_show_orig(self, on):
        self.show_orig = on
        self._rebuild()
        self.update()

    def _set_cursor(self):
        c = {"hand": Qt.CursorShape.OpenHandCursor, "crop": Qt.CursorShape.CrossCursor,
             "erase": Qt.CursorShape.CrossCursor, "select": Qt.CursorShape.CrossCursor}.get(self.tool)
        if self.tool == "pick":
            self.setCursor(dropper_cursor())
        else:
            self.setCursor(c or Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------ picture
    def set_image(self, arr, orig, new_picture=False):
        same = self.arr is not None and arr is not None and self.arr.shape == arr.shape and not new_picture
        self.arr, self.orig = arr, orig
        self._rebuild()
        if new_picture:
            self.clear_selection()
            self.crop = None
            if self.tool == "crop":
                self.start_crop()
        if arr is not None and not same:
            self.fit()
        self.update()

    def _rebuild(self):
        a = self.orig if (self.show_orig and self.orig is not None) else self.arr
        if a is None:
            self._img = None
            return
        self._img, self._keep = rgb_qimage(a)

    def fit(self):
        if self.arr is None:
            return
        h, w = self.arr.shape[:2]
        self.z = min(self.width() / float(w), self.height() / float(h)) * 0.97
        self.ox = (self.width() - w * self.z) / 2.0
        self.oy = (self.height() - h * self.z) / 2.0
        self.update()

    def resizeEvent(self, e):
        if self.arr is not None:
            self.fit()
        super().resizeEvent(e)

    def to_img(self, pos):
        return (pos.x() - self.ox) / self.z, (pos.y() - self.oy) / self.z

    def to_scr(self, x, y):
        return x * self.z + self.ox, y * self.z + self.oy

    # ------------------------------------------------------------ selection (add / subtract / undo)
    def has_selection(self):
        return self.sel is not None

    def clear_selection(self):
        had = self.sel is not None
        self.sel, self.sel_hist = None, []
        self._sel_img = None
        if had:
            self.selChanged.emit(False)
        self.update()

    def undo_selection(self):
        if not self.sel_hist:
            return False
        self.sel = self.sel_hist.pop()
        self._make_sel_img()
        self.selChanged.emit(self.sel is not None)
        self.update()
        return True

    def select_all(self):
        if self.arr is None:
            return
        self.sel_hist.append(self.sel)
        self.sel = np.ones(self.arr.shape[:2], np.float32)
        self._make_sel_img()
        self.selChanged.emit(True)
        self.update()

    def _make_sel_img(self):
        m = self.sel
        if m is None:
            self._sel_img = None
            return
        h, w = m.shape
        hard = m > 0.5
        er = hard.copy()
        er[1:-1, 1:-1] &= hard[:-2, 1:-1] & hard[2:, 1:-1] & hard[1:-1, :-2] & hard[1:-1, 2:]
        er[0, :] = er[-1, :] = False
        er[:, 0] = er[:, -1] = False
        edge = hard & ~er
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 99, 102, 241
        rgba[..., 3] = (m * 95).astype(np.uint8)
        rgba[edge] = (255, 255, 255, 255)
        self._sel_img, self._sel_keep = rgba_qimage(rgba)

    def _inside_sel(self, x, y):
        if self.sel is None:
            return False
        h, w = self.sel.shape
        xi, yi = int(x), int(y)
        return 0 <= xi < w and 0 <= yi < h and self.sel[yi, xi] > 0.5

    def _add_shape_to_selection(self, shape, data, mode):
        h, w = self.arr.shape[:2]
        new = core.shape_mask((w, h), shape, data, self.brush)
        self.sel_hist.append(self.sel)
        del self.sel_hist[:-30]
        self.sel = core.sel_combine(self.sel, new, mode)
        self._make_sel_img()
        self.selChanged.emit(self.sel is not None)

    def _make_patch(self):
        rgb = self.arr
        rgba = np.dstack([rgb, (self.sel * 255).astype(np.uint8)])
        ys, xs = np.nonzero(self.sel > 0.02)
        if ys.size == 0:
            return None
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        img, keep = rgba_qimage(rgba[y0:y1, x0:x1].copy())
        return img, keep, (int(x0), int(y0))

    # ------------------------------------------------------------ crop
    def start_crop(self):
        if self.arr is None:
            return
        h, w = self.arr.shape[:2]
        self.crop = [0.05 * w, 0.05 * h, 0.95 * w, 0.95 * h]
        self.update()

    def crop_rect(self):
        return None if self.crop is None else tuple(self.crop)

    def _crop_hit(self, pos):
        x0, y0 = self.to_scr(self.crop[0], self.crop[1])
        x1, y1 = self.to_scr(self.crop[2], self.crop[3])
        x, y = pos.x(), pos.y()
        near = lambda a, b: abs(a - b) <= HANDLE
        l, r, t, b = near(x, x0), near(x, x1), near(y, y0), near(y, y1)
        in_x, in_y = x0 - HANDLE <= x <= x1 + HANDLE, y0 - HANDLE <= y <= y1 + HANDLE
        if l and t:
            return "tl"
        if r and t:
            return "tr"
        if l and b:
            return "bl"
        if r and b:
            return "br"
        if l and in_y:
            return "l"
        if r and in_y:
            return "r"
        if t and in_x:
            return "t"
        if b and in_x:
            return "b"
        if x0 < x < x1 and y0 < y < y1:
            return "move"
        return None

    # ------------------------------------------------------------ mouse
    def wheelEvent(self, e):
        if self.arr is None:
            return
        d = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.slide.emit(-1 if d > 0 else 1)
            return
        f = 1.15 if d > 0 else 1 / 1.15
        pos = e.position()
        nz = max(0.03, min(40.0, self.z * f))
        f = nz / self.z
        self.ox = pos.x() - (pos.x() - self.ox) * f
        self.oy = pos.y() - (pos.y() - self.oy) * f
        self.z = nz
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def mousePressEvent(self, e):
        if self.arr is None:
            return
        self.setFocus()
        pos = e.position()
        btn = e.button()
        if btn in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or \
                (btn == Qt.MouseButton.LeftButton and self.tool == "hand"):
            self._act = "pan"
            self._pan0 = (pos.x(), pos.y(), self.ox, self.oy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if btn != Qt.MouseButton.LeftButton:
            return
        x, y = self.to_img(pos)
        h, w = self.arr.shape[:2]
        if self.tool == "pick":
            if 0 <= x < w and 0 <= y < h:
                self.picked.emit(int(x), int(y))
            return
        if self.tool == "crop" and self.crop is not None:
            hit = self._crop_hit(pos)
            if hit:
                self._act = ("crop", hit, x, y, list(self.crop))
            return
        mods = e.modifiers()
        sub = bool(mods & Qt.KeyboardModifier.AltModifier) or self.selmode == "sub"
        if self.tool == "select":
            force_add = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            if self._inside_sel(x, y) and not sub and not force_add:
                got = self._make_patch()
                if got:
                    self._patch = got
                    self._off = (0.0, 0.0)
                    self._act = ("move", x, y)
                    self.update()
                    return
            self._act = ("draw", "sub" if sub else "add")
        elif self.tool == "erase":
            self._act = ("draw", "erase")
        else:
            return
        self._pts = [(x, y)]
        self._start = (x, y)
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        self._hover = pos
        a = self._act
        if a == "pan":
            x0, y0, ox, oy = self._pan0
            self.ox, self.oy = ox + pos.x() - x0, oy + pos.y() - y0
        elif a and a[0] == "move":
            x, y = self.to_img(pos)
            self._off = (x - a[1], y - a[2])
        elif a and a[0] == "draw":
            x, y = self.to_img(pos)
            if self.shape == "rect":
                self._pts = [self._start, (x, y)]
            else:
                self._pts.append((x, y))
        elif a and a[0] == "crop":
            self._drag_crop(pos)
        elif self.tool == "crop" and self.crop is not None:
            hit = self._crop_hit(pos)
            self.setCursor({"tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
                            "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
                            "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
                            "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
                            "move": Qt.CursorShape.SizeAllCursor}.get(hit, Qt.CursorShape.CrossCursor))
        self.update()

    def _drag_crop(self, pos):
        _, hit, x0, y0, box = self._act
        x, y = self.to_img(pos)
        dx, dy = x - x0, y - y0
        h, w = self.arr.shape[:2]
        l, t, r, b = box
        if hit == "move":
            dx = max(-l, min(w - r, dx))
            dy = max(-t, min(h - b, dy))
            l, r, t, b = l + dx, r + dx, t + dy, b + dy
        else:
            if "l" in hit:
                l = max(0, min(r - 8, l + dx))
            if "r" in hit:
                r = min(w, max(l + 8, r + dx))
            if "t" in hit:
                t = max(0, min(b - 8, t + dy))
            if "b" in hit:
                b = min(h, max(t + 8, b + dy))
        self.crop = [l, t, r, b]

    def mouseReleaseEvent(self, e):
        a, self._act = self._act, None
        if a == "pan":
            self._set_cursor()
            return
        if a is None:
            return
        if a[0] == "move":
            dx, dy = self._off
            patch, self._patch, self._off = self._patch, None, (0.0, 0.0)
            if abs(dx) + abs(dy) >= 3:
                self.moveDone.emit(int(round(dx)), int(round(dy)), self.copy)
            self.update()
            return
        if a[0] == "draw":
            pts, self._pts = self._pts, []
            if self.shape == "rect":
                if len(pts) < 2 or abs(pts[0][0] - pts[1][0]) < 3 or abs(pts[0][1] - pts[1][1]) < 3:
                    self.update()
                    return
                data = (pts[0][0], pts[0][1], pts[1][0], pts[1][1])
                shape = "rect"
            elif self.shape == "lasso":
                if len(pts) < 4:
                    self.update()
                    return
                data, shape = pts, "lasso"
            else:
                data, shape = pts, "brush"
            if a[1] == "erase":
                self.eraseDone.emit(shape, data, self.brush)
            else:
                self._add_shape_to_selection(shape, data, a[1])
            self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    # ------------------------------------------------------------ painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0d12"))
        if self._img is None:
            return
        h, w = self.arr.shape[:2]
        p.save()
        p.translate(self.ox, self.oy)
        p.scale(self.z, self.z)
        # soft card shadow
        for i in range(6, 0, -1):
            p.fillRect(QRectF(-i * 1.5 / self.z, -i * 1.5 / self.z + 3 / self.z, w + i * 3 / self.z,
                              h + i * 3 / self.z), QColor(0, 0, 0, 14))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.z < 2.5)
        p.drawImage(0, 0, self._img)
        if self._sel_img is not None:
            p.setOpacity(0.35 if (self._act and self._act[0] == "move") else 1.0)
            p.drawImage(0, 0, self._sel_img)
            p.setOpacity(1.0)
        if self._patch is not None:
            img, _, (px, py) = self._patch
            p.setOpacity(0.92)
            p.drawImage(px + self._off[0], py + self._off[1], img)
            p.setOpacity(1.0)
        self._paint_shape(p)
        p.restore()
        if self.crop is not None and self.tool == "crop":
            self._paint_crop(p)
        elif self.tool in ("erase", "select") and self.shape == "brush" and self._hover is not None and not self._act:
            r = self.brush / 2.0 * self.z
            p.setPen(QPen(QColor("white"), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._hover, r, r)
            p.setPen(QPen(QColor("#1b2030"), 1))
            p.drawEllipse(self._hover, r + 1.5, r + 1.5)
        p.end()

    def _paint_shape(self, p):
        if not self._pts or not (self._act and self._act[0] == "draw"):
            return
        erase = self._act[1] == "erase"
        sub = self._act[1] == "sub"
        col = QColor(239, 68, 68) if (erase or sub) else QColor(99, 102, 241)
        fill = QColor(col.red(), col.green(), col.blue(), 70)
        if self.shape == "brush":
            pen = QPen(fill, self.brush, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            if len(self._pts) == 1:
                x, y = self._pts[0]
                p.drawPoint(QPointF(x, y))
            else:
                p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in self._pts]))
            return
        pen = QPen(col, 2.0 / self.z, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(fill)
        if self.shape == "rect" and len(self._pts) == 2:
            (x0, y0), (x1, y1) = self._pts
            p.drawRect(QRectF(QPointF(min(x0, x1), min(y0, y1)), QPointF(max(x0, x1), max(y0, y1))))
        elif self.shape == "lasso":
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in self._pts]))

    def _paint_crop(self, p):
        x0, y0 = self.to_scr(self.crop[0], self.crop[1])
        x1, y1 = self.to_scr(self.crop[2], self.crop[3])
        dim = QColor(0, 0, 0, 150)
        W, H = self.width(), self.height()
        p.fillRect(QRectF(0, 0, W, y0), dim)
        p.fillRect(QRectF(0, y1, W, H - y1), dim)
        p.fillRect(QRectF(0, y0, x0, y1 - y0), dim)
        p.fillRect(QRectF(x1, y0, W - x1, y1 - y0), dim)
        p.setPen(QPen(QColor("white"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        p.setBrush(QColor("#6366f1"))
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1),
                       (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)):
            p.drawEllipse(QPointF(cx, cy), 6, 6)
