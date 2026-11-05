"""Background work: slow operations run in a worker thread, the window stays alive and shows a progress bar."""
import traceback
from collections import deque

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


def _log(text):
    import os
    try:
        d = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "ImageFixer")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "error.log"), "a", encoding="utf-8") as f:
            f.write(text + "\n" + "-" * 60 + "\n")
    except Exception:
        pass


class _Signals(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)


class _Job(QRunnable):
    def __init__(self, fn, signals):
        super().__init__()
        self.fn, self.sig = fn, signals
        self.setAutoDelete(True)

    def run(self):
        try:
            res = self.fn(lambda frac=-1.0, text="": self.sig.progress.emit(float(frac), text))
        except Exception as ex:                      # shown to the user, details in the log
            self.sig.failed.emit("%s\n%s" % (ex, traceback.format_exc()))
            return
        self.sig.finished.emit(res)


class JobRunner(QObject):
    """One job at a time (they are queued). run(label, fn, done, fail):
         fn(progress) runs in a worker thread; progress(fraction 0..1 or -1 = unknown, text)
         done(result) / fail(message) are called in the window thread."""
    changed = Signal(bool, float, str)           # busy, fraction (-1 = unknown), text

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q = deque()
        self._cur = None
        self._label = ""
        self.busy = False

    def run(self, label, fn, done=None, fail=None):
        self._q.append((label, fn, done, fail))
        if self._cur is None:
            self._next()

    def _next(self):
        if not self._q:
            self._cur = None
            self.busy = False
            self.changed.emit(False, 0.0, "")
            return
        label, fn, done, fail = self._q.popleft()
        sig = _Signals()
        self._cur = (sig, done, fail)               # keep alive until finished
        self._label = label
        self.busy = True
        sig.progress.connect(self._progress)
        sig.finished.connect(self._finished)
        sig.failed.connect(self._failed)
        self.changed.emit(True, -1.0, label)
        QThreadPool.globalInstance().start(_Job(fn, sig))

    def _progress(self, frac, text):
        self.changed.emit(True, frac, "%s  %s" % (self._label, text) if text else self._label)

    def _finished(self, res):
        done = self._cur[1]
        try:
            if done is not None:
                done(res)
        finally:
            self._next()

    def _failed(self, msg):
        fail = self._cur[2]
        try:
            if fail is not None:
                fail(msg)
            else:
                _log(msg)
        finally:
            self._next()

    def wait_idle(self, timeout_s=60.0):
        """Used by the self test: process events until all jobs are finished."""
        import time
        from PySide6.QtWidgets import QApplication
        end = time.time() + timeout_s
        while (self.busy or self._q) and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()
        return not self.busy


# ---- background work that must NOT lock the window (live preview, preparing a file to drag out)
_live = set()


class _Bg(QObject):
    def __init__(self, fn, done, fail):
        super().__init__()
        self._done, self._fail = done, fail
        self.sig = _Signals()
        self.sig.finished.connect(self._fin)
        self.sig.failed.connect(self._err)
        _live.add(self)
        QThreadPool.globalInstance().start(_Job(lambda progress: fn(progress), self.sig))

    def _fin(self, res):
        _live.discard(self)
        if self._done is not None:
            self._done(res)

    def _err(self, msg):
        _live.discard(self)
        _log(msg)
        if self._fail is not None:
            self._fail(msg)


def run_background(fn, done=None, fail=None):
    """fn(progress) in a worker thread; done(result) / fail(message) in the window thread. Nothing is locked."""
    return _Bg(fn, done, fail)
