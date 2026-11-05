"""Main window: one big picture (card), the few controls that matter, thumbnails below."""
import os
import tempfile
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import QItemSelectionModel, QMimeData, Qt, QSize, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QDrag, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QColorDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog,
                               QLabel, QListView, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QStackedWidget, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

import core
import fixer
import jobs
import scanner
import winutil
from canvas import Canvas, rgb_qimage
from phonedlg import PhoneDialog
from theme import make_icon

HINTS = {
    "hand": "Drag = move the picture.  Mouse wheel = zoom.  Double-click = fit.  Shift + wheel = next picture.",
    "pick": "Click on a text colour in the picture. It gets its own slider on the right.",
    "erase": "Paint / draw over what you want to remove. It is gone the moment you let go (Ctrl+Z brings it back).",
    "select": "Draw an area, then drag it to move it. Wrong part? Use Subtract, or Ctrl+Z to undo the last step.",
    "crop": "Drag the box or its edges, then press Apply (Enter).",
}


class NoWheelSlider(QSlider):
    def wheelEvent(self, e):
        e.ignore()


class SRow(QWidget):
    """Title + value + slider. Double-click the slider to reset it."""
    changed = Signal(int)

    def __init__(self, title, lo, hi, val, fmt=lambda v: "%d%%" % v, tip=""):
        super().__init__()
        self.fmt, self.default = fmt, val
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)
        top = QHBoxLayout()
        self.lbl = QLabel(title)
        self.val = QLabel()
        self.val.setObjectName("muted")
        top.addWidget(self.lbl)
        top.addStretch(1)
        top.addWidget(self.val)
        self.s = NoWheelSlider(Qt.Orientation.Horizontal)
        self.s.setRange(lo, hi)
        self.s.setValue(val)
        self.s.valueChanged.connect(self._ch)
        self.s.mouseDoubleClickEvent = lambda e: self.s.setValue(self.default)
        if tip:
            self.setToolTip(tip)
        v.addLayout(top)
        v.addWidget(self.s)
        self.val.setText(fmt(val))

    def _ch(self, v):
        self.val.setText(self.fmt(v))
        self.changed.emit(v)

    def set(self, v):
        self.s.blockSignals(True)
        self.s.setValue(v)
        self.s.blockSignals(False)
        self.val.setText(self.fmt(v))


class ThumbList(QListWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setObjectName("strip")
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setIconSize(QSize(110, 84))
        self.setGridSize(QSize(128, 112))
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self.setFixedHeight(128)
        self.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setToolTip("Select several (Ctrl / Shift + click), then drag them out to Explorer, a print program, "
                        "WhatsApp Web ...")

    def enterEvent(self, e):
        self.win.prepare_exports()
        super().enterEvent(e)

    def startDrag(self, actions):
        self.win.start_drag(self, self.win.selected_items())


class DragButton(QPushButton):
    def __init__(self, win):
        super().__init__("Drag out  \u283f")
        self.win = win
        self._press = None
        self.setToolTip("Press and drag from here: the edited pictures (the selected ones) go to Explorer, a print "
                        "program, WhatsApp Web ...")
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def enterEvent(self, e):
        self.win.prepare_exports()
        super().enterEvent(e)

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None and (e.buttons() & Qt.MouseButton.LeftButton) and \
                (e.position().toPoint() - self._press).manhattanLength() >= QApplication.startDragDistance():
            self._press = None
            self.win.start_drag(self, self.win.selected_items())
            self.setDown(False)
            return
        super().mouseMoveEvent(e)


def _btn(text, tip="", checkable=False, **props):
    b = QPushButton(text)
    b.setCheckable(checkable)
    if tip:
        b.setToolTip(tip)
    for k, v in props.items():
        b.setProperty(k, v)
    return b


class MainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.items = []
        self.cur = None
        self.phone = None
        self.phone_dlg = None
        self.dirty = []
        self.busy = False
        self._quit = False
        self.shapes = {"erase": "brush", "select": "rect"}
        self.tool = "hand"
        self.jobs = jobs.JobRunner(self)
        self.jobs.changed.connect(self._busy)
        self.setWindowTitle("Image Fixer")
        self.setWindowIcon(make_icon())
        self.resize(1360, 860)
        self.setAcceptDrops(True)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(130)
        self.timer.timeout.connect(self._kick)
        self._build()
        self._shortcuts()
        self._make_tray()
        self._tool("hand")

    # ================================================================== layout
    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        col = QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._header())
        body = QHBoxLayout()
        body.setSpacing(0)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._empty_page())
        self.stack.addWidget(self._viewer_page())
        body.addWidget(self.stack, 1)
        body.addWidget(self._panel())
        col.addLayout(body, 1)
        col.addWidget(self._bottom())

    def _header(self):
        h = QWidget()
        h.setObjectName("header")
        r = QHBoxLayout(h)
        r.setContentsMargins(16, 10, 16, 10)
        r.setSpacing(8)
        logo = QLabel()
        from theme import logo_pixmap
        logo.setPixmap(logo_pixmap(28))
        brand = QLabel("Image Fixer")
        brand.setObjectName("brand")
        self.btnAdd = _btn("Add pictures", "Choose files (or drag them into the window, or Ctrl+V)", accent="true")
        self.btnPhone = _btn("Add from phone", "Take pictures / scan pages with your phone camera")
        self.btnAdd.clicked.connect(self.choose_files)
        self.btnPhone.clicked.connect(self.open_phone)
        r.addWidget(logo)
        r.addWidget(brand)
        r.addSpacing(14)
        r.addWidget(self.btnAdd)
        r.addWidget(self.btnPhone)
        r.addStretch(1)
        self.btnSaveSel = _btn("Save selected  \u25be")
        self.btnSaveAll = _btn("Save all  \u25be", accent="true")
        for b, which in ((self.btnSaveSel, "sel"), (self.btnSaveAll, "all")):
            m = QMenu(b)
            m.addAction("As pictures", lambda w=which: self.save(w, "pics"))
            m.addAction("As one PDF (A4 pages)", lambda w=which: self.save(w, "a4"))
            m.addAction("As one PDF (page = picture shape)", lambda w=which: self.save(w, "fit"))
            b.setMenu(m)
            r.addWidget(b)
        more = _btn("\u22ef", "Settings", flat="true")
        m = QMenu(more)
        self.actAuto = m.addAction("Start with Windows")
        self.actAuto.setCheckable(True)
        self.actAuto.setChecked(winutil.is_autostart())
        self.actAuto.toggled.connect(winutil.set_autostart)
        self.actSend = m.addAction("Show in right-click > Send to")
        self.actSend.setCheckable(True)
        self.actSend.setChecked(self.settings.value("sendto", True, type=bool))
        self.actSend.toggled.connect(self._sendto)
        m.addAction("Choose the save folder ...", self.choose_folder)
        m.addAction("Phone connection / port ...", self.open_phone)
        more.setMenu(m)
        r.addWidget(more)
        return h

    def _empty_page(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFrame()
        f.setObjectName("dropzone")
        f.setFixedSize(520, 300)
        fv = QVBoxLayout(f)
        fv.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fv.setSpacing(12)
        t = QLabel("Drop pictures here")
        t.setObjectName("big")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s = QLabel("Then press  \u2728 Auto clean  on the right.\nShadows and grey / green paper turn white, text gets clear.")
        s.setObjectName("muted")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b = _btn("Choose pictures", accent="true")
        b.clicked.connect(self.choose_files)
        for x in (t, s, b):
            fv.addWidget(x, 0, Qt.AlignmentFlag.AlignCenter)
        v.addWidget(f)
        return w

    def _viewer_page(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 8, 0, 0)
        v.setSpacing(6)
        # ---- tools
        tb = QHBoxLayout()
        tb.setSpacing(4)
        self.toolBtns = {}
        grp = QButtonGroup(self)
        for key, text, tip in (("hand", "\u270b Move view", "Look around, zoom"),
                               ("pick", "\U0001f4a7 Pick color", "Click a text colour: change it or make it stronger"),
                               ("erase", "\U0001f9fd Erase", "Remove part of the picture"),
                               ("select", "\u2b1a Select / Move", "Select an area, then drag it elsewhere or copy it"),
                               ("crop", "\u2702 Crop", "Cut away the edges")):
            b = _btn(text, tip, True)
            grp.addButton(b)
            b.clicked.connect(lambda _=False, k=key: self._tool(k))
            self.toolBtns[key] = b
            tb.addWidget(b)
        tb.addSpacing(10)
        for text, tip, fn in (("\u27f2", "Rotate left", lambda: self.rotate(90)),
                              ("\u27f3", "Rotate right", lambda: self.rotate(270)),
                              ("Angle ...", "Rotate by any angle", self.rotate_any),
                              ("Straighten page", "Find the paper in the photo and make it straight", self.straighten),
                              ("Undo", "Ctrl+Z", self.undo), ("Fit", "Ctrl+0", lambda: self.canvas.fit())):
            b = _btn(text, tip)
            b.clicked.connect(fn)
            tb.addWidget(b)
        self.btnCmp = _btn("Hold: original", "Press and hold to see the picture before editing")
        self.btnCmp.pressed.connect(lambda: self.canvas.set_show_orig(True))
        self.btnCmp.released.connect(lambda: self.canvas.set_show_orig(False))
        tb.addWidget(self.btnCmp)
        tb.addStretch(1)
        self.lblCount = QLabel("")
        self.lblCount.setObjectName("sub")
        tb.addWidget(self.lblCount)
        tb.addSpacing(10)
        v.addLayout(tb)
        # ---- options of the tool
        v.addWidget(self._options())
        # ---- the card
        row = QHBoxLayout()
        row.setSpacing(0)
        self.btnPrev = _btn("\u2039", "Previous picture (Left key)")
        self.btnNext = _btn("\u203a", "Next picture (Right key)")
        for b in (self.btnPrev, self.btnNext):
            b.setObjectName("nav")
            b.setFixedWidth(44)
        self.btnPrev.clicked.connect(lambda: self.go(-1))
        self.btnNext.clicked.connect(lambda: self.go(1))
        self.canvas = Canvas()
        self.canvas.picked.connect(self.picked)
        self.canvas.eraseDone.connect(self.erase_drawn)
        self.canvas.moveDone.connect(self.move_done)
        self.canvas.slide.connect(self.go)
        self.canvas.selChanged.connect(lambda on: self.btnEraseSel.setEnabled(on))
        row.addWidget(self.btnPrev)
        row.addWidget(self.canvas, 1)
        row.addWidget(self.btnNext)
        v.addLayout(row, 1)
        return w

    def _options(self):
        bar = QWidget()
        r = QHBoxLayout(bar)
        r.setContentsMargins(0, 0, 10, 0)
        r.setSpacing(6)
        # shape (erase + select)
        self.optShape = QWidget()
        sr = QHBoxLayout(self.optShape)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(4)
        self.shapeBtns = {}
        sg = QButtonGroup(self)
        for key, text, tip in (("brush", "Brush", "Paint over it"), ("rect", "Box", "Drag a rectangle"),
                               ("lasso", "Free shape", "Draw around it")):
            b = _btn(text, tip, True)
            sg.addButton(b)
            b.clicked.connect(lambda _=False, k=key: self._shape(k))
            self.shapeBtns[key] = b
            sr.addWidget(b)
        self.brushRow = QWidget()
        br = QHBoxLayout(self.brushRow)
        br.setContentsMargins(0, 0, 0, 0)
        br.addWidget(QLabel("Size"))
        self.brushSlider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.brushSlider.setRange(6, 160)
        self.brushSlider.setValue(30)
        self.brushSlider.setFixedWidth(120)
        self.brushSlider.valueChanged.connect(lambda v: self.canvas.set_brush(v))
        br.addWidget(self.brushSlider)
        sr.addWidget(self.brushRow)
        r.addWidget(self.optShape)
        # select only
        self.optSelect = QWidget()
        s = QHBoxLayout(self.optSelect)
        s.setContentsMargins(0, 0, 0, 0)
        s.setSpacing(4)
        g = QButtonGroup(self)
        self.btnAddSel = _btn("+ Add", "New areas are added to the selection (Shift)", True)
        self.btnSubSel = _btn("\u2212 Subtract", "Painting takes that part out of the selection (Alt)", True)
        self.btnAddSel.setChecked(True)
        g.addButton(self.btnAddSel)
        g.addButton(self.btnSubSel)
        self.btnAddSel.clicked.connect(lambda: self.canvas.set_selmode("add"))
        self.btnSubSel.clicked.connect(lambda: self.canvas.set_selmode("sub"))
        self.btnCopy = _btn("Copy (keep the original)", "Off = the area moves, the old place is filled like paper.", True)
        self.btnCopy.toggled.connect(self.canvas_copy)
        self.btnEraseSel = _btn("Erase selected", "Delete key")
        self.btnEraseSel.setEnabled(False)
        self.btnEraseSel.clicked.connect(self.erase_selected)
        bAll = _btn("Select all", "Ctrl+A")
        bAll.clicked.connect(lambda: self.canvas.select_all())
        bClr = _btn("Clear selection", "Ctrl+D / Esc")
        bClr.clicked.connect(lambda: self.canvas.clear_selection())
        for b in (self.btnAddSel, self.btnSubSel, self.btnCopy, self.btnEraseSel, bAll, bClr):
            s.addWidget(b)
        r.addWidget(self.optSelect)
        # crop only
        self.optCrop = QWidget()
        c = QHBoxLayout(self.optCrop)
        c.setContentsMargins(0, 0, 0, 0)
        ok = _btn("Apply crop (Enter)", accent="true")
        ok.clicked.connect(self.crop_apply)
        no = _btn("Cancel (Esc)")
        no.clicked.connect(lambda: self.toolBtns["hand"].click())
        c.addWidget(ok)
        c.addWidget(no)
        r.addWidget(self.optCrop)
        self.hint = QLabel("")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        r.addWidget(self.hint, 1)
        return bar

    def _panel(self):
        outer = QWidget()
        outer.setObjectName("panel")
        outer.setFixedWidth(340)
        ov = QVBoxLayout(outer)
        ov.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        ov.addWidget(sc)
        inner = QWidget()
        sc.setWidget(inner)
        v = QVBoxLayout(inner)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)
        t = QLabel("Make it print-ready")
        t.setObjectName("title")
        v.addWidget(t)
        self.btnAuto = _btn("\u2728  Auto clean this picture", "Shadow, grey and green tint gone, white paper, clear text",
                            accent="true")
        self.btnAutoAll = _btn("\u2728  Auto clean ALL pictures")
        self.btnAuto.clicked.connect(lambda: self.auto_clean(False))
        self.btnAutoAll.clicked.connect(lambda: self.auto_clean(True))
        v.addWidget(self.btnAuto)
        v.addWidget(self.btnAutoAll)
        self.rows = {}
        pct = lambda x: "%d%%" % x
        spec = (("clean", "Remove shadow and grey / green tint", 0, 100, 0, pct,
                 "Evens out the light on the paper. Text, stamps and shapes stay."),
                ("white", "Whiter paper", 0, 100, 0, pct, "Light grey paper becomes pure white (saves ink)."),
                ("text", "Sharper text and lines", 0, 100, 0, pct, "Clearer thin text, lines and shapes."),
                ("dark", "Darker, bolder text", 0, 100, 0, pct, "Faint writing gets stronger. Paper stays white."),
                ("satv", "Colour strength", 0, 100, 0, pct, "More saturated colours. Black and white are not changed."))
        for key, title, lo, hi, val, fmt, tip in spec:
            r = SRow(title, lo, hi, val, fmt, tip + "\nDouble-click the slider to reset it.")
            r.changed.connect(lambda x, k=key: self._param(k, x))
            self.rows[key] = r
            v.addWidget(r)
        two = QHBoxLayout()
        bReset = _btn("Reset picture", "Back to the original picture, all edits off")
        bReset.clicked.connect(self.reset_item)
        bAll = _btn("Use for all pictures", "Give every picture these same settings")
        bAll.clicked.connect(self.use_for_all)
        two.addWidget(bReset)
        two.addWidget(bAll)
        v.addLayout(two)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#262c3b")
        v.addWidget(line)
        h = QLabel("Text colours")
        h.setObjectName("sub")
        v.addWidget(h)
        note = QLabel("Pick a colour in the picture: make it stronger (slider to the right), fade it away (to the "
                      "left) or change it into another colour.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        v.addWidget(note)
        self.btnPickPanel = _btn("\U0001f4a7  Pick a colour from the picture", accent="true")
        self.btnPickPanel.clicked.connect(lambda: self.toolBtns["pick"].click())
        v.addWidget(self.btnPickPanel)
        self.adjBox = QVBoxLayout()
        self.adjBox.setSpacing(8)
        v.addLayout(self.adjBox)
        v.addStretch(1)
        return outer

    def _bottom(self):
        b = QWidget()
        b.setObjectName("bottom")
        v = QVBoxLayout(b)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(4)
        self.list = ThumbList(self)
        self.list.currentRowChanged.connect(self._row_changed)
        v.addWidget(self.list)
        r = QHBoxLayout()
        r.setSpacing(6)
        bl = _btn("\u25c0 Earlier", "Move this picture one place to the left (PDF page order)")
        br = _btn("Later \u25b6", "Move this picture one place to the right")
        rm = _btn("Remove from list", "Takes the picture out of the program (the file is not deleted)", danger="true")
        bl.clicked.connect(lambda: self.shift(-1))
        br.clicked.connect(lambda: self.shift(1))
        rm.clicked.connect(self.remove_selected)
        self.btnDrag = DragButton(self)
        for x in (bl, br, rm, self.btnDrag):
            r.addWidget(x)
        r.addStretch(1)
        self.lblStatus = QLabel("")
        self.lblStatus.setObjectName("muted")
        self.bar = QProgressBar()
        self.bar.setFixedWidth(220)
        self.bar.setTextVisible(False)
        self.bar.hide()
        r.addWidget(self.lblStatus)
        r.addWidget(self.bar)
        v.addLayout(r)
        return b

    def _shortcuts(self):
        def sc(keys, fn):
            s = QShortcut(QKeySequence(keys), self)
            s.activated.connect(fn)
        sc("Left", lambda: self.go(-1))
        sc("Right", lambda: self.go(1))
        sc("Ctrl+Z", self.undo)
        sc("Ctrl+V", self.paste)
        sc("Ctrl+S", lambda: self.save("sel", "pics"))
        sc("Ctrl+Shift+S", lambda: self.save("all", "pics"))
        sc("Ctrl+0", lambda: self.canvas.fit())
        sc("Ctrl+A", lambda: self.canvas.select_all() if self.tool == "select" else None)
        sc("Ctrl+D", lambda: self.canvas.clear_selection())
        sc("Return", self.crop_apply)
        sc("Enter", self.crop_apply)
        sc("Escape", self.escape)
        sc("Delete", self.erase_selected)

    def _make_tray(self):
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(make_icon(), self)
        self.tray.setToolTip("Image Fixer")
        m = QMenu()
        m.addAction("Open", self.show_front)
        m.addAction("Exit", self.quit_now)
        self.tray.setContextMenu(m)
        self.tray.activated.connect(lambda r: self.show_front() if r == QSystemTrayIcon.ActivationReason.Trigger
                                    else None)
        self.tray.show()

    # ================================================================== small helpers
    def say(self, text):
        self.lblStatus.setText(text)

    def selected_items(self):
        rows = sorted({i.row() for i in self.list.selectedIndexes()})
        got = [self.items[r] for r in rows if r < len(self.items)]
        return got or ([self.cur] if self.cur else [])

    def _busy(self, busy, frac, text):
        self.bar.setVisible(busy)
        if busy:
            if frac < 0:
                self.bar.setRange(0, 0)
            else:
                self.bar.setRange(0, 1000)
                self.bar.setValue(int(frac * 1000))
            self.say(text)
        else:
            self.bar.setRange(0, 1)

    def show_front(self):
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def quit_now(self):
        self._quit = True
        self.close()

    def closeEvent(self, e):
        self.settings.setValue("geometry", self.saveGeometry())
        if self.tray is not None and not self._quit:
            self.hide()
            e.ignore()
            return
        if self.phone is not None:
            self.phone.stop()
        e.accept()
        QApplication.quit()

    # ================================================================== adding pictures
    def choose_files(self):
        exts = " ".join("*" + e for e in sorted(core.IMG_EXT))
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose pictures", self.settings.value("open_dir", "", type=str),
                                                "Pictures (%s)" % exts)
        if paths:
            self.settings.setValue("open_dir", os.path.dirname(paths[0]))
            self.add_files(paths)

    def add_files(self, paths):
        files = []
        for p in paths:
            if os.path.isdir(p):
                files += [os.path.join(p, n) for n in sorted(os.listdir(p))]
            else:
                files.append(p)
        files = [f for f in files if os.path.splitext(f)[1].lower() in core.IMG_EXT and os.path.isfile(f)]
        if not files:
            return

        def work(progress):
            got, bad = [], []
            for i, f in enumerate(files):
                progress(i / float(len(files)), "%d / %d" % (i + 1, len(files)))
                try:
                    got.append(fixer.Item(f, core.load_image(f)))
                except Exception:
                    bad.append(os.path.basename(f))
            return got, bad
        self.jobs.run("Opening pictures", work, self._added)

    def _added(self, res):
        got, bad = res
        first = len(self.items)
        self.items += got
        self._rebuild_list()
        if got:
            self.list.setCurrentRow(first)
        self.say("%d picture(s) added." % len(got) + ("  Could not open: %s" % ", ".join(bad) if bad else ""))

    def paste(self):
        md = QApplication.clipboard().mimeData()
        if md.hasUrls():
            self.add_files([u.toLocalFile() for u in md.urls() if u.isLocalFile()])
        elif md.hasImage():
            self._add_qimage(QApplication.clipboard().image())

    def _add_qimage(self, qi):
        if qi.isNull():
            return
        d = os.path.join(tempfile.gettempdir(), "imagefixer_paste")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "Pasted_%s.png" % time.strftime("%H%M%S"))
        qi.save(p, "PNG")
        self.add_files([p])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            self.add_files([u.toLocalFile() for u in md.urls() if u.isLocalFile()])
        elif md.hasImage():
            self._add_qimage(md.imageData())
        e.acceptProposedAction()
        self.show_front()

    def phone_received(self, paths):
        self.show_front()
        self.add_files(paths)
        self.say("Received %d picture(s) from the phone." % len(paths))

    def open_phone(self):
        if self.phone_dlg is None:
            self.phone_dlg = PhoneDialog(self)
        self.phone_dlg.show_front()

    # ================================================================== the list / current picture
    def _icon(self, it):
        qi, keep = rgb_qimage(it.view)
        return QIcon(QPixmap.fromImage(qi.scaled(110, 84, Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)))

    def _rebuild_list(self):
        sel = [self.items.index(i) for i in self.selected_items() if i in self.items]
        cur = self.items.index(self.cur) if self.cur in self.items else -1
        self.list.blockSignals(True)
        self.list.clear()
        for n, it in enumerate(self.items):
            li = QListWidgetItem(self._icon(it), "%d  %s" % (n + 1, it.name[:16]))
            li.setSizeHint(QSize(128, 112))
            self.list.addItem(li)
        for r in sel:
            self.list.item(r).setSelected(True)
        if cur >= 0:
            self.list.setCurrentRow(cur, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.list.blockSignals(False)
        self.stack.setCurrentIndex(1 if self.items else 0)
        self._count()
        if self.items and self.cur not in self.items:
            self.list.setCurrentRow(0)

    def _count(self):
        n = len(self.items)
        i = self.items.index(self.cur) + 1 if self.cur in self.items else 0
        self.lblCount.setText("%d / %d" % (i, n) if n else "")

    def _row_changed(self, row):
        if 0 <= row < len(self.items):
            self.set_cur(self.items[row])

    def go(self, d):
        if not self.items:
            return
        r = max(0, min(len(self.items) - 1, self.list.currentRow() + d))
        self.list.setCurrentRow(r)

    def set_cur(self, it):
        self.cur = it
        self.canvas.set_image(it.view, it.prev, new_picture=True)
        self._sync_panel()
        self._count()
        if it.view is it.prev and not fixer.is_plain(it.p):
            self.request(it)

    def shift(self, d):
        if self.cur not in self.items:
            return
        i = self.items.index(self.cur)
        j = i + d
        if 0 <= j < len(self.items):
            self.items[i], self.items[j] = self.items[j], self.items[i]
            self._rebuild_list()
            self.list.setCurrentRow(j)

    def remove_selected(self):
        gone = self.selected_items()
        if not gone:
            return
        row = self.list.currentRow()
        self.items = [i for i in self.items if i not in gone]
        self.dirty = [i for i in self.dirty if i not in gone]
        self.cur = None
        self._rebuild_list()
        if self.items:
            self.list.setCurrentRow(min(row, len(self.items) - 1))
        else:
            self.canvas.set_image(None, None)
            self._clear_rows()

    # ================================================================== settings sliders
    def _sync_panel(self):
        p = self.cur.p
        for k in ("clean", "white", "text", "dark"):
            self.rows[k].set(p[k])
        self.rows["satv"].set(int(round((p["sat"] - 100) / 1.5)))
        self._rebuild_rows()

    def _param(self, key, val):
        if self.cur is None:
            return
        if key == "satv":
            self.cur.p["sat"] = 100 + int(round(val * 1.5))
        else:
            self.cur.p[key] = val
        self._changed(self.cur)

    def _changed(self, it):
        it.version += 1
        it.export = None
        self.request(it)

    def auto_clean(self, everything):
        targets = self.items if everything else ([self.cur] if self.cur else [])
        for it in targets:
            it.p.update(fixer.PRESET)
            self._changed(it)
        if self.cur:
            self._sync_panel()
        if targets:
            self.say("Cleaning %d picture(s) ..." % len(targets))

    def use_for_all(self):
        if self.cur is None:
            return
        for it in self.items:
            if it is not self.cur:
                it.p = fixer.copy_params(self.cur.p)
                self._changed(it)
        self.say("The same settings are used for all %d pictures." % len(self.items))

    def reset_item(self):
        if self.cur is None:
            return
        self.cur.reset()
        self.cur.export = None
        self.canvas.set_image(self.cur.view, self.cur.prev, new_picture=True)
        self._sync_panel()
        self._refresh_icon(self.cur)

    # ================================================================== live preview (background, one at a time)
    def request(self, it):
        if it in self.items and it not in self.dirty:
            self.dirty.append(it)
        self.timer.start()

    def _kick(self):
        if self.busy or not self.dirty:
            return
        it = self.cur if self.cur in self.dirty else self.dirty[0]
        self.dirty.remove(it)
        self.busy = True
        ver = it.version
        jobs.run_background(lambda progress: fixer.make_view(it),
                            done=lambda res: self._viewed(it, ver, res),
                            fail=lambda msg: self._view_failed(it, msg))

    def _viewed(self, it, ver, res):
        self.busy = False
        if it in self.items:
            if ver == it.version:
                it.view = res[0]
                self._refresh_icon(it)
                if it is self.cur:
                    self.canvas.set_image(it.view, it.prev)
            else:
                self.request(it)
        self._kick()

    def _view_failed(self, it, msg):
        self.busy = False
        self.say("Something went wrong while updating the preview (see error.log).")
        self._kick()

    def _refresh_icon(self, it):
        if it in self.items:
            self.list.item(self.items.index(it)).setIcon(self._icon(it))

    # ================================================================== colour rows
    def _clear_rows(self):
        while self.adjBox.count():
            w = self.adjBox.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _rebuild_rows(self):
        self._clear_rows()
        if self.cur is None:
            return
        for a in self.cur.p["adjs"]:
            self.adjBox.addWidget(self._adj_row(a))

    def _swatch(self, rgb):
        s = QLabel()
        s.setFixedSize(22, 22)
        s.setStyleSheet("background:%s;border:1px solid #5a6178;border-radius:6px" % core.hex_of(rgb))
        return s

    def _adj_row(self, a):
        f = QFrame()
        f.setObjectName("adjRow")
        v = QVBoxLayout(f)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(self._swatch(a["rgb"]))
        if a.get("to"):
            top.addWidget(QLabel("\u2192"))
            top.addWidget(self._swatch(a["to"]))
        top.addWidget(QLabel(a["label"]))
        top.addStretch(1)
        chg = _btn("Change colour", "Turn this colour into another colour", flat="true")
        chg.clicked.connect(lambda: self._change_color(a))
        rm = _btn("\u2715", "Remove this row", flat="true")
        rm.clicked.connect(lambda: self._remove_adj(a))
        top.addWidget(chg)
        top.addWidget(rm)
        v.addLayout(top)
        fmt = lambda x: "same" if x == 100 else ("gone" if x == 0 else ("%d%%" % x))
        s = SRow("Strength", 0, 200, int(a["k"]), fmt, "Left = fades into the paper, right = stronger / darker.")
        s.changed.connect(lambda x: self._adj_k(a, x))
        v.addWidget(s)
        return f

    def _adj_k(self, a, k):
        a["k"] = k
        if self.cur:
            self._changed(self.cur)

    def _change_color(self, a):
        start = QColor(*(a.get("to") or a["rgb"]))
        c = QColorDialog.getColor(start, self, "New colour for this text")
        if c.isValid():
            a["to"] = [c.red(), c.green(), c.blue()]
            self._rebuild_rows()
            self._changed(self.cur)

    def _remove_adj(self, a):
        self.cur.p["adjs"] = [x for x in self.cur.p["adjs"] if x is not a]
        self._rebuild_rows()
        self._changed(self.cur)

    def picked(self, x, y):
        it = self.cur
        if it is None:
            return
        arr = it.cache.get("arr")
        if arr is None or arr.shape != it.prev.shape:
            arr = it.view
        patch = arr[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)
        rgb = [int(v) for v in np.median(patch, axis=0)]
        lab = core.rgb_to_lab(np.array([[rgb]], np.uint8))[0, 0]
        if np.hypot(float(lab[1]), float(lab[2])) < core.NEUTRAL_C:
            L = float(lab[0])
            a = core.new_band_adj(L, max(0.0, L - 11.0), min(100.0, L + 11.0))
        else:
            a = core.new_color_adj(rgb, lab, 34.0)
        keys = {core.adj_key(b) for b in it.p["adjs"]}
        if core.adj_key(a) in keys:
            self.say("That colour is already in the list.")
            return
        it.p["adjs"].append(a)
        self._rebuild_rows()
        self._changed(it)
        self.say("Added: %s. Use its slider on the right." % a["label"])

    # ================================================================== tools
    def _tool(self, name):
        self.tool = name
        self.toolBtns[name].setChecked(True)
        self.canvas.set_tool(name)
        if name in self.shapes:
            self._shape(self.shapes[name])
        self.optShape.setVisible(name in ("erase", "select"))
        self.optSelect.setVisible(name == "select")
        self.optCrop.setVisible(name == "crop")
        self.hint.setText(HINTS[name])

    def _shape(self, s):
        if self.tool in self.shapes:
            self.shapes[self.tool] = s
        self.canvas.set_shape(s)
        self.shapeBtns[s].setChecked(True)
        self.brushRow.setVisible(s == "brush")

    def canvas_copy(self, on):
        self.canvas.set_copy(on)

    def escape(self):
        if self.tool == "crop":
            self.toolBtns["hand"].click()
        else:
            self.canvas.clear_selection()

    def undo(self):
        if self.cur is None:
            return
        if self.tool == "select" and self.canvas.undo_selection():
            return
        if self.cur.do_undo():
            self.cur.export = None
            self.canvas.set_image(self.cur.view, self.cur.prev, new_picture=True)
            self.request(self.cur)
            self.say("Undone.")
        else:
            self.say("Nothing to undo.")

    # ---- changes of the real picture (always on the full-size copy, in the background)
    def _edit(self, label, fn, after=None):
        it = self.cur
        if it is None:
            return

        def done(res):
            if it not in self.items:
                return
            it.push_undo()
            it.set_base(res)
            it.export = None
            self.request(it)
            self._refresh_icon(it)
            if it is self.cur:
                self.canvas.set_image(it.view, it.prev, new_picture=True)
            if after:
                after()
        self.jobs.run(label, lambda progress: fn(it.base), done,
                      lambda msg: self.say("That did not work: %s" % msg.splitlines()[0]))

    def rotate(self, deg):
        self._edit("Rotating", lambda b: core.rotate_image(b, deg))

    def rotate_any(self):
        v, ok = QInputDialog.getDouble(self, "Rotate", "Degrees (counter-clockwise; use a minus sign for clockwise):",
                                       0.0, -360.0, 360.0, 1)
        if ok and v:
            self._edit("Rotating", lambda b: core.rotate_image(b, v))

    def straighten(self):
        def fn(base):
            quad, found = scanner.detect_document(base)
            if not found:
                raise ValueError("The paper edges were not found. Use Crop instead.")
            return scanner.straighten(base, quad)
        self._edit("Straightening the page", fn)

    def erase_drawn(self, shape, data, brush):
        it = self.cur
        s = it.scale
        d = tuple(v / s for v in data) if shape == "rect" else [(x / s, y / s) for x, y in data]
        b = brush / s
        self._edit("Erasing", lambda base: core.erase_region(base, core.make_mask(base.size, shape, d, b)))

    def erase_selected(self):
        if self.cur is None or self.tool != "select" or not self.canvas.has_selection():
            return
        sel = self.canvas.sel
        self.canvas.clear_selection()
        self._edit("Erasing", lambda base: core.erase_region(
            base, Image.fromarray(np.clip(core.resize_mask(sel, base.size) * 255 + 0.5, 0, 255).astype(np.uint8))))

    def move_done(self, dx, dy, copy):
        if self.cur is None or not self.canvas.has_selection():
            return
        s = self.cur.scale
        sel = self.canvas.sel
        self.canvas.clear_selection()
        self._edit("Copying" if copy else "Moving", lambda base: core.move_region(
            base, Image.fromarray(np.clip(core.resize_mask(sel, base.size) * 255 + 0.5, 0, 255).astype(np.uint8)),
            dx / s, dy / s, copy))

    def crop_apply(self):
        if self.tool != "crop" or self.cur is None:
            return
        box = self.canvas.crop_rect()
        if box is None:
            return
        s = self.cur.scale
        b = tuple(v / s for v in box)
        self._edit("Cropping", lambda base: core.crop_box(base, b), after=lambda: self.toolBtns["hand"].click())

    # ================================================================== saving / dragging out
    def _snapshot(self, items):
        return [(it.name, it.base, fixer.copy_params(it.p), it.path) for it in items]

    def _which(self, which):
        return list(self.items) if which == "all" else self.selected_items()

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for saved pictures", self.settings.value("save_dir", "", type=str))
        if d:
            self.settings.setValue("save_dir", d)
            self.say("Save folder: %s" % d)
        return d

    def save(self, which, kind):
        items = self._which(which)
        if not items:
            self.say("No pictures.")
            return
        snap = self._snapshot(items)
        folder = self.settings.value("save_dir", "", type=str) or os.path.dirname(items[0].path)
        if kind in ("a4", "fit"):
            path, _ = QFileDialog.getSaveFileName(self, "Save PDF", os.path.join(folder, "Pictures.pdf"), "PDF (*.pdf)")
            if not path:
                return
            self.settings.setValue("save_dir", os.path.dirname(path))
            sources = [(lambda b=b, p=p: fixer.full_image(b, p)) for _, b, p, _ in snap]
            self.jobs.run("Saving PDF", lambda progress: core.save_pdf(sources, path, kind, 95, progress),
                          lambda r: self.saved(path, "PDF with %d page(s)" % len(snap)), self.save_failed)
            return
        if len(snap) == 1:
            name, base, p, src = snap[0]
            ext = core.out_ext(src)
            path, _ = QFileDialog.getSaveFileName(
                self, "Save picture", core.unique_path(folder, name + "_fixed", ext),
                "Pictures (*.jpg *.png *.webp *.bmp *.tif *.pdf)")
            if not path:
                return
            self.settings.setValue("save_dir", os.path.dirname(path))
            self.jobs.run("Saving", lambda progress: core.save_image(fixer.full_image(base, p), path),
                          lambda r: self.saved(path, "Saved"), self.save_failed)
            return
        d = QFileDialog.getExistingDirectory(self, "Save %d pictures into this folder" % len(snap), folder)
        if not d:
            return
        self.settings.setValue("save_dir", d)

        def work(progress):
            for i, (name, base, p, src) in enumerate(snap):
                progress(i / float(len(snap)), "%d / %d" % (i + 1, len(snap)))
                ext = core.out_ext(src)
                core.save_image(fixer.full_image(base, p), core.unique_path(d, name + "_fixed", ext))
            progress(1.0, "")
        self.jobs.run("Saving", work, lambda r: self.saved(d, "%d pictures saved" % len(snap)), self.save_failed)

    def saved(self, path, text):
        self.say("%s: %s" % (text, path))
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def save_failed(self, msg):
        self.say("Saving failed: %s" % msg.splitlines()[0])
        QMessageBox.warning(self, "Image Fixer", "Saving failed:\n\n%s" % msg.splitlines()[0])

    def export_file(self, it):
        """The edited picture as a real file (for dragging). Reused while nothing changed."""
        if it.export and it.export[0] == it.version and os.path.exists(it.export[1]):
            return it.export[1]
        ver = it.version
        ext = core.out_ext(it.path)
        if ext == ".pdf":
            ext = ".png"
        folder = os.path.join(tempfile.gettempdir(), "imagefixer_out", "%x" % id(it))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "%s_fixed%s" % (it.name, ext))
        core.save_image(fixer.full_image(it.base, fixer.copy_params(it.p)), path)
        it.export = (ver, path)
        return path

    def prepare_exports(self):
        todo = [i for i in self.selected_items() if not (i.export and i.export[0] == i.version)]
        if todo:
            jobs.run_background(lambda progress: [self.export_file(i) for i in todo])

    def start_drag(self, source, items):
        if not items:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            urls = [QUrl.fromLocalFile(self.export_file(i)) for i in items]
        finally:
            QApplication.restoreOverrideCursor()
        md = QMimeData()
        md.setUrls(urls)
        d = QDrag(source)
        d.setMimeData(md)
        qi, keep = rgb_qimage(items[0].view)
        d.setPixmap(QPixmap.fromImage(qi.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)))
        d.exec(Qt.DropAction.CopyAction)

    def _sendto(self, on):
        self.settings.setValue("sendto", on)
        winutil.set_sendto(on)
