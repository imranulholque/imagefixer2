"""Look and feel: colours, style sheet, app icon, eyedropper cursor."""
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (QBrush, QColor, QCursor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPixmap, QPolygonF)

ACCENT = "#6366f1"


def logo_pixmap(size):
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    g = QLinearGradient(0, 0, size, size)
    g.setColorAt(0.0, QColor("#6366f1"))
    g.setColorAt(0.55, QColor("#a855f7"))
    g.setColorAt(1.0, QColor("#ec4899"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(g))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    s = size / 64.0
    # picture frame
    p.setPen(QPen(QColor("white"), max(1.5, 4 * s), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(14 * s, 16 * s, 36 * s, 32 * s), 5 * s, 5 * s)
    # sun + hills
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("white")))
    p.drawEllipse(QPointF(26 * s, 28 * s), 4 * s, 4 * s)
    hills = QPolygonF([QPointF(18 * s, 44 * s), QPointF(29 * s, 33 * s), QPointF(36 * s, 40 * s),
                       QPointF(41 * s, 35 * s), QPointF(46 * s, 44 * s)])
    p.drawPolygon(hills)
    p.end()
    return pm


def make_icon():
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(logo_pixmap(s))
    return ic


def dropper_cursor():
    """Little eyedropper; the hot spot is the tip (bottom left)."""
    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for width, color in ((7.0, "#ffffff"), (4.0, "#1b2030")):
        pen = QPen(QColor(color), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(24, 8), QPointF(10, 22))
    p.setPen(QPen(QColor("#ffffff"), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(10, 22), QPointF(4, 28))
    p.setPen(QPen(QColor("#1b2030"), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(10, 22), QPointF(4, 28))
    p.setPen(QPen(QColor("#ffffff"), 1.5))
    p.setBrush(QBrush(QColor("#1b2030")))
    p.drawEllipse(QPointF(26, 6), 4.5, 4.5)
    p.end()
    return QCursor(pm, 4, 28)


QSS = """
* { font-family: "Segoe UI"; font-size: 13px; color: #e8eaed; }
QMainWindow, QDialog, QMessageBox { background: #0e1116; }
QWidget#root { background: #0e1116; }
QWidget#header { background: #0a0c10; border-bottom: 1px solid #1c2130; }
QWidget#bottom { background: #0a0c10; border-top: 1px solid #1c2130; }
QWidget#panel { background: #12161e; border-left: 1px solid #1f2535; }
QLabel { background: transparent; }
QLabel#title { font-size: 17px; font-weight: 700; }
QLabel#brand { font-size: 18px; font-weight: 700; }
QLabel#sub { font-size: 14px; font-weight: 600; }
QLabel#muted, QLabel#hint { color: #8d95a8; }
QLabel#big { font-size: 26px; font-weight: 700; }
QPushButton { background: #222836; border: 1px solid #30374a; border-radius: 9px; padding: 7px 13px; }
QPushButton:hover { background: #2b3246; }
QPushButton:pressed { background: #1b2030; }
QPushButton:disabled { color: #5d6475; background: #181c26; border-color: #232838; }
QPushButton:checked { background: #4f46e5; border-color: #6366f1; color: white; }
QPushButton[accent="true"] { border: none; font-weight: 600; color: white;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6366f1, stop:1 #a855f7); }
QPushButton[accent="true"]:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7578ff, stop:1 #b66dff); }
QPushButton[accent="true"]:disabled { background: #2a2f3f; color: #6b7388; }
QPushButton[danger="true"] { background: #3a1b22; border-color: #5a2630; color: #ffb4bd; }
QPushButton[danger="true"]:hover { background: #4a222b; }
QPushButton[tab="true"] { background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0;
  padding: 7px 16px; font-weight: 600; color: #9aa3b8; }
QPushButton[tab="true"]:hover { color: white; background: #171c28; }
QPushButton[tab="true"]:checked { background: transparent; color: white; border-bottom: 2px solid #818cf8; }
QProgressBar { background: #222836; border: none; border-radius: 4px; height: 8px; text-align: center; }
QProgressBar::chunk { background: #6366f1; border-radius: 4px; }
QDoubleSpinBox, QComboBox { background: #1a1f2b; border: 1px solid #30374a; border-radius: 8px; padding: 5px 8px; }
QPushButton[flat="true"] { background: transparent; border: none; padding: 4px 8px; color: #9aa3b8; }
QPushButton[flat="true"]:hover { color: white; background: #1f2535; }
QPushButton#nav { font-size: 20px; padding: 2px 12px; }
QSlider::groove:horizontal { height: 5px; background: #2a3042; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #6366f1; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: #c7d2fe; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #2f3649; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #2f3649; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QFrame#adjRow { background: #1a1f2b; border: 1px solid #262c3b; border-radius: 12px; }
QFrame#adjRow[hl="true"] { border: 1px solid #818cf8; background: #1e2335; }
QFrame#group { background: #151924; border: 1px solid #262c3b; border-radius: 14px; }
QFrame#dropzone { border: 2px dashed #3a4363; border-radius: 26px; background: #11151c; }
QListWidget#strip { background: #0a0c10; border: none; outline: 0; }
QListWidget#strip::item { color: #8d95a8; border: 1px solid transparent; border-radius: 10px; padding: 3px; }
QListWidget#strip::item:hover { background: #171c28; }
QListWidget#strip::item:selected { background: #1f2640; border: 1px solid #6366f1; color: white; }
QMenu { background: #171b24; border: 1px solid #30374a; padding: 6px; border-radius: 10px; }
QMenu::item { padding: 8px 24px 8px 14px; border-radius: 7px; }
QMenu::item:selected { background: #2b3246; }
QMenu::separator { height: 1px; background: #2a3042; margin: 5px 8px; }
QCheckBox { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 5px; border: 1px solid #4a5268; background: #1a1f2b; }
QCheckBox::indicator:hover { border-color: #818cf8; }
QCheckBox::indicator:checked { background: #6366f1; border-color: #a5b4fc; }
QToolTip { background: #222836; color: #e8eaed; border: 1px solid #3a4260; padding: 5px 7px; }
"""
