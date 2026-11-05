"""'Add from phone' window: QR code + connection status."""
import os
import tempfile

import segno
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout

import phone


def qr_pixmap(text, px=300):
    qr = segno.make(text, error="m")
    rows = [list(r) for r in qr.matrix]
    n = len(rows)
    border = 3
    cell = max(1, px // (n + 2 * border))
    size = cell * (n + 2 * border)
    pm = QPixmap(size, size)
    pm.fill(QColor("white"))
    p = QPainter(pm)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#0b0e13"))
    for y, row in enumerate(rows):
        for x, v in enumerate(row):
            if v:
                p.drawRect((x + border) * cell, (y + border) * cell, cell, cell)
    p.end()
    return pm


class PhoneDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Add from phone")
        self.setModal(False)
        self.setMinimumWidth(420)
        self._url = None
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(12)
        t = QLabel("Scan this code with your phone")
        t.setObjectName("title")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Phone and computer must be on the same Wi-Fi. The phone's camera opens a web page - "
                     "no app is needed.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr = QLabel()
        self.qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr.setMinimumHeight(310)
        self.lblUrl = QLabel("")
        self.lblUrl.setObjectName("muted")
        self.lblUrl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lblUrl.setWordWrap(True)
        self.lblUrl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lblState = QLabel("")
        self.lblState.setObjectName("sub")
        self.lblState.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lblState.setWordWrap(True)
        row = QHBoxLayout()
        self.btnCopy = QPushButton("Copy link")
        self.btnNew = QPushButton("New code")
        self.btnNew.setToolTip("Makes a new secret link. The old QR code stops working.")
        self.btnStop = QPushButton("Disconnect")
        for b in (self.btnCopy, self.btnNew, self.btnStop):
            row.addWidget(b)
        for w in (t, sub, self.qr, self.lblUrl, self.lblState):
            v.addWidget(w)
        v.addLayout(row)
        prow = QHBoxLayout()
        prow.addStretch(1)
        prow.addWidget(QLabel("Port:"))
        self.spPort = QSpinBox()
        self.spPort.setRange(0, 65535)
        self.spPort.setSpecialValueText("automatic")
        self.spPort.setValue(win.settings.value("phone_port", 0, type=int))
        self.spPort.setToolTip("0 / automatic = the program picks a free port. Type a number (for example 8765) "
                               "if you want it to stay the same, or to open it in the firewall.")
        self.btnPort = QPushButton("Use this port")
        self.btnPort.clicked.connect(self._use_port)
        prow.addWidget(self.spPort)
        prow.addWidget(self.btnPort)
        prow.addStretch(1)
        v.addLayout(prow)
        self.btnCopy.clicked.connect(self._copy)
        self.btnNew.clicked.connect(lambda: self._start(new_token=True))
        self.btnStop.clicked.connect(self._stop)
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self._tick)

    # ---------------------------------------------------------------
    def show_front(self):
        self._start()
        self.show()
        self.raise_()
        self.activateWindow()
        self.timer.start()

    def _server(self):
        return self.win.phone

    def _start(self, new_token=False):
        if new_token and self.win.phone is not None:
            self.win.phone.stop()
            self.win.phone = None
        if self.win.phone is None:
            inbox = os.path.join(tempfile.gettempdir(), "imagefixer_phone")
            srv = phone.PhoneServer(inbox, self.win)
            srv.received.connect(self.win.phone_received)
            self.win.phone = srv
        srv = self.win.phone
        if not srv.running():
            port = self.win.settings.value("phone_port", 0, type=int)
            if not srv.start(port):
                self.lblState.setText("The connection could not be started (a firewall or another program may block it).")
                return
            self.win.settings.setValue("phone_port", srv.port)
        self._url = None
        self._tick()

    def _use_port(self):
        want = self.spPort.value()
        self.win.settings.setValue("phone_port", want)
        if self.win.phone is not None:
            self.win.phone.stop()
            self.win.phone = None
        self._start()
        srv = self.win.phone
        if srv is not None and srv.running():
            self.spPort.setValue(srv.port)               # the port really in use
            if want and srv.port != want:
                self.lblState.setText("Port %d is busy, so port %d is used instead." % (want, srv.port))

    def _stop(self):
        if self.win.phone is not None:
            self.win.phone.stop()
            self.win.phone = None
        self._url = None
        self.qr.clear()
        self.lblUrl.setText("")
        self.lblState.setText("Disconnected. Close this window or press New code to connect again.")
        self.timer.stop()

    def _copy(self):
        if self._url:
            QGuiApplication.clipboard().setText(self._url)

    def _tick(self):
        """Every second or so: the address may have changed (Wi-Fi switched, PC woke up) -> new QR code."""
        srv = self.win.phone
        if srv is None or not srv.running():
            return
        url = srv.url()
        if url != self._url:
            self._url = url
            if url is None:
                self.qr.clear()
                self.lblUrl.setText("")
            else:
                self.qr.setPixmap(qr_pixmap(url))
                self.lblUrl.setText(url)
        if url is None:
            self.lblState.setText("No Wi-Fi / network found. Connect this computer to the same Wi-Fi as the phone.")
        elif srv.phone_online():
            self.lblState.setText("✓ Phone connected   -   pictures received: %d" % srv.total)
        else:
            self.lblState.setText("Waiting for the phone ...   (pictures received: %d)" % srv.total)

    def closeEvent(self, e):
        self.timer.stop()          # the connection stays open: pictures can still arrive
        super().closeEvent(e)

    def hideEvent(self, e):
        self.timer.stop()
        super().hideEvent(e)
