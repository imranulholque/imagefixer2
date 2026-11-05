"""Self test used by the build: opens the real window invisibly and uses the main features.
Run by hand with:  python smoke_test.py   (a failing line is printed with the exact error)"""
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

failed = []


def step(name, fn):
    try:
        fn()
        print("PASS", name, flush=True)
    except Exception:
        failed.append(name)
        print("FAIL", name, flush=True)
        traceback.print_exc()
        sys.stdout.flush()


def make_pages(folder):
    out = []
    font = ImageFont.load_default(size=26)
    yy, xx = np.mgrid[0:600, 0:900]
    shade = (0.6 + 0.4 * xx / 900.0)[..., None]
    for i in range(3):
        paper = np.clip(np.array([205, 225, 205], np.float32) * shade, 0, 255).astype(np.uint8)
        im = Image.fromarray(paper)
        d = ImageDraw.Draw(im)
        for r in range(8):
            col = [(10, 10, 10), (90, 90, 90), (190, 30, 30)][r % 3]
            d.text((40, 30 + r * 60), "Sample text line %d of page %d" % (r, i), fill=col, font=font)
        p = os.path.join(folder, "page%d.jpg" % i)
        im.save(p, quality=90)
        out.append(p)
    return out


def main():
    app = QApplication(sys.argv)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    import core
    import fixer
    import ui
    from theme import QSS
    app.setStyleSheet(QSS)
    tmp = tempfile.mkdtemp()
    pages = make_pages(tmp)
    settings = QSettings(os.path.join(tmp, "t.ini"), QSettings.Format.IniFormat)
    settings.setValue("save_dir", tmp)
    win = ui.MainWindow(settings)
    win.show()

    def settle():
        win.jobs.wait_idle(60)
        end = time.time() + 30
        while (win.busy or win.dirty or win.timer.isActive()) and time.time() < end:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    step("add pictures", lambda: (win.add_files(pages), settle(), _eq(len(win.items), 3)))
    step("auto clean all", lambda: (win.auto_clean(True), settle(), _white(win.items[0].view)))
    step("slide to next", lambda: (win.go(1), _eq(win.cur is win.items[1], True)))
    step("pick a colour", lambda: (win.toolBtns["pick"].click(), win.picked(60, 40), settle(),
                                   _eq(len(win.cur.p["adjs"]) >= 1, True)))
    step("change colour", lambda: (_recolor(win), settle()))
    step("rotate", lambda: (win.rotate(90), settle(), _eq(win.cur.base.size, (600, 900))))
    step("undo", lambda: (win.undo(), settle(), _eq(win.cur.base.size, (900, 600))))
    step("erase", lambda: (win.erase_drawn("rect", (20, 20, 200, 80), 20.0), settle()))
    step("select + move", lambda: _select_move(win, settle))
    step("crop", lambda: (win.toolBtns["crop"].click(), win.crop_apply(), settle()))
    step("save all", lambda: (win.save("all", "pics") if _auto_dir(win, tmp) else None, settle(), _saved(tmp)))
    step("pdf", lambda: _pdf(win, tmp, settle))
    step("export for drag", lambda: _eq(os.path.exists(win.export_file(win.items[0])), True))
    step("remove", lambda: (win.remove_selected(), _eq(len(win.items), 2)))
    print("FAILED: %s" % failed if failed else "ALL PASSED")
    return 1 if failed else 0


def _eq(a, b):
    assert a == b, "%r != %r" % (a, b)


def _white(arr):
    assert arr[5, 5].min() >= 250, "paper is not white: %s" % arr[5, 5]


def _recolor(win):
    a = win.cur.p["adjs"][0]
    a["to"] = [20, 60, 200]
    win._changed(win.cur)


def _select_move(win, settle):
    c = win.canvas
    win.toolBtns["select"].click()
    c.select_all()
    assert c.has_selection()
    assert c.undo_selection() and not c.has_selection()
    import numpy as np
    c.sel = np.zeros(c.arr.shape[:2], np.float32)
    c.sel[30:90, 30:300] = 1.0
    c.selChanged.emit(True)
    win.move_done(60, 60, True)
    settle()


def _auto_dir(win, tmp):
    import ui
    out = os.path.join(tmp, "out")
    os.makedirs(out, exist_ok=True)
    ui.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: out)      # no real dialogs in a test
    ui.QDesktopServices = type("D", (), {"openUrl": staticmethod(lambda u: True)})
    return True


def _saved(tmp):
    n = [f for f in os.listdir(os.path.join(tmp, "out")) if "_fixed" in f]
    assert len(n) == 3, n


def _pdf(win, tmp, settle):
    import ui
    path = os.path.join(tmp, "all.pdf")
    ui.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (path, ""))
    win.save("all", "a4")
    settle()
    assert os.path.getsize(path) > 1000


if __name__ == "__main__":
    sys.exit(main())
