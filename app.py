"""Image Fixer - entry point.
One copy only: starting it again (or Send to) hands the pictures to the copy that is already running."""
import json
import os
import sys
import traceback

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox

LOG_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "ImageFixer")
_shown = {"n": 0}


def log_error(text):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "error.log"), "a", encoding="utf-8") as f:
            f.write(text + "\n" + "-" * 60 + "\n")
    except Exception:
        pass


def hook(kind, value, tb):
    text = "".join(traceback.format_exception(kind, value, tb))
    log_error(text)
    if _shown["n"] < 2:
        _shown["n"] += 1
        try:
            QMessageBox.critical(None, "Image Fixer", "Something went wrong:\n\n%s\n\nDetails: %s"
                                 % (value, os.path.join(LOG_DIR, "error.log")))
        except Exception:
            pass


def forward(name, paths):
    """Send the file list to an already running copy. True = it took them, we can quit."""
    s = QLocalSocket()
    s.connectToServer(name)
    if not s.waitForConnected(400):
        return False
    s.write(json.dumps({"paths": paths}).encode("utf-8"))
    s.flush()
    s.waitForBytesWritten(3000)
    s.disconnectFromServer()
    if s.state() != QLocalSocket.LocalSocketState.UnconnectedState:
        s.waitForDisconnected(1000)
    return True


class Listener(QObject):
    received = Signal(list)

    def __init__(self, name, parent=None):
        super().__init__(parent)
        QLocalServer.removeServer(name)
        self.server = QLocalServer(self)
        self.ok = self.server.listen(name)
        self.server.newConnection.connect(self._new)

    def _new(self):
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            buf = bytearray()
            sock.readyRead.connect(lambda s=sock, b=buf: b.extend(bytes(s.readAll())))
            sock.disconnected.connect(lambda s=sock, b=buf: self._done(s, b))

    def _done(self, sock, buf):
        buf.extend(bytes(sock.readAll()))
        try:
            paths = json.loads(buf.decode("utf-8")).get("paths", [])
        except Exception:
            paths = []
        sock.deleteLater()
        self.received.emit(paths)


def main():
    args = sys.argv[1:]
    paths = [os.path.abspath(a) for a in args if not a.startswith("--")]
    start_hidden = "--tray" in args
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ImageFixer.App")
        except Exception:
            pass
    sys.excepthook = hook
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("Image Fixer")
    app.setOrganizationName("ImageFixer")
    name = "ImageFixer-" + (os.environ.get("USERNAME") or os.environ.get("USER") or "user")
    if forward(name, paths):
        return 0

    # not running yet: load the heavy parts now (numpy, Pillow, the window)
    import ui
    import winutil
    from theme import QSS, make_icon
    app.setStyleSheet(QSS)
    app.setWindowIcon(make_icon())
    settings = QSettings("ImageFixer", "ImageFixer")
    win = ui.MainWindow(settings)
    listener = Listener(name, app)

    def on_files(p):
        win.show_front()
        if p:
            win.add_files(p)
    listener.received.connect(on_files)

    # keep the shortcut / start-up entry pointing at this exe, even if the folder was moved
    if settings.value("sendto", True, type=bool):
        winutil.set_sendto(True)
    if winutil.is_autostart():
        winutil.set_autostart(True)

    geo = settings.value("geometry")
    if geo is not None:
        win.restoreGeometry(geo)
    else:
        win.setWindowState(Qt.WindowState.WindowMaximized)
    if paths:
        win.add_files(paths)
    if not start_hidden or paths or win.tray is None:
        win.show_front()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
