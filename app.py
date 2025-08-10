# -*- coding: utf-8 -*-
"""
Updated on Sat Aug  9 2025


"""
import os
import sys
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)
os.environ["RAPIDOCR_HOME"] = os.path.join(base_path, "models", "rapidocr")
import mss
from rapidocr_onnxruntime import RapidOCR
import pandas as pd
import numpy as np
import re
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QDialog, QHeaderView, QLineEdit, QTextEdit
)
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QFont

import cv2

ocr = RapidOCR()

# ---- Hardcoded font sizes ----
STATUS_FONT_PT = 12
PROGRAM_FONT_PT = 12

# ---- Allowed numeric charset ----
ALLOWED_CHARS = set("0123456789-+.")

class ZoneEditor(QDialog):
    def __init__(self, zones, parent_app):
        super().__init__()
        self.setWindowTitle("Edit OCR Zones")
        self.resize(500, 300)
        self.parent_app = parent_app
        self.table = QTableWidget(len(zones), 4)
        self.table.setHorizontalHeaderLabels(["X1", "Y1", "X2", "Y2"])
        for i, (x1, y1, x2, y2) in enumerate(zones):
            self.table.setItem(i, 0, QTableWidgetItem(str(x1)))
            self.table.setItem(i, 1, QTableWidgetItem(str(y1)))
            self.table.setItem(i, 2, QTableWidgetItem(str(x2)))
            self.table.setItem(i, 3, QTableWidgetItem(str(y2)))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_zones)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.apply_btn)
        self.setLayout(layout)

    def apply_zones(self):
        new_zones = []
        for row in range(self.table.rowCount()):
            try:
                x1 = float(self.table.item(row, 0).text())
                y1 = float(self.table.item(row, 1).text())
                x2 = float(self.table.item(row, 2).text())
                y2 = float(self.table.item(row, 3).text())
                new_zones.append((x1, y1, x2, y2))
            except:
                continue
        self.parent_app.ocr_zone_norms = new_zones
        self.accept()


class ScreenshotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIL_addon for profile welding - by MJN")
        self.setFixedSize(600, 150)
        title_label = QLabel()
        title_label.setText('SIL addon for profile welding - <i>by MJN</i>')
        title_label.setAlignment(Qt.AlignCenter)

        # Buttons and fields
        self.define_region_btn = QPushButton("Select region for coordinates")
        self.capture_btn = QPushButton("Record the coordinates")
        self.get_program_btn = QPushButton("Get Program")
        self.view_btn = QPushButton("View All Captures")
        self.view_zones_btn = QPushButton("View OCR Zones")
        self.view_table_btn = QPushButton("View OCR Table")
        self.defocus_input = QLineEdit("0")
        self.defocus_input.setFixedWidth(60)
        self.status = QLabel("Select the region for coordinates")
        self.status.setAlignment(Qt.AlignCenter)

        # Apply hardcoded status font
        status_font = QFont()
        status_font.setPointSize(STATUS_FONT_PT)
        self.status.setFont(status_font)

        # Layouts
        row1 = QHBoxLayout()
        row1.addWidget(self.define_region_btn)
        row1.addWidget(self.capture_btn)
        row1.addWidget(self.get_program_btn)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Defocus:"))
        row2.addWidget(self.defocus_input)
        row2.addSpacing(12)
        row2.addWidget(self.view_btn)
        row2.addWidget(self.view_zones_btn)
        row2.addWidget(self.view_table_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addWidget(self.status)
        self.setLayout(main_layout)

        # State variables
        self.capture_btn.setEnabled(False)
        self.get_program_btn.setEnabled(False)
        self.view_table_btn.setEnabled(False)
        self.capture_count = 0
        self.max_captures = 6
        self.region_defined = False
        self.region = None

        # Connections
        self.define_region_btn.clicked.connect(self.select_region)
        self.capture_btn.clicked.connect(self.capture)
        self.get_program_btn.clicked.connect(self.handle_get_program)
        self.view_btn.clicked.connect(self.view_all_images)
        self.view_zones_btn.clicked.connect(self.view_ocr_zones)
        self.view_table_btn.clicked.connect(self.show_ocr_table)

        # OCR zones (relative) - unchanged by default
        self.ocr_zone_norms = [
            (0.46, 0.05, 0.74, 0.23),
            (0.46, 0.30, 0.74, 0.48),
            (0.46, 0.55, 0.74, 0.73),
            (0.46, 0.80, 0.74, 0.98),
        ]

        # SERIAL-LABELED capture names
        self.capture_labels = [
            "Y_back",
            "Y_front",
            "X_left",
            "X_right",
            "X_left after 180° rotation",
            "Y_back after 180° rotation",
        ]

        # Status update messages using serial labels
        self.status_messages = [
            f"1) Locate point for {self.capture_labels[0]}",
            f"2) Locate point for {self.capture_labels[1]}",
            f"3) Locate point for {self.capture_labels[2]}",
            f"4) Locate point for {self.capture_labels[3]}",
            f"5) Locate point for {self.capture_labels[4]}",
            f"6) Locate point for {self.capture_labels[5]}",
            "All six points recorded. Define defocus and get program.",
        ]

    # ---- Region selection ----
    def select_region(self):
        QApplication.processEvents()
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            self.full_image = QImage(screenshot.rgb, screenshot.width, screenshot.height, QImage.Format_RGB888)
            self.selector = RegionOverlay(self, self.full_image)
            self.selector.showFullScreen()

    def region_selected(self, region_rect):
        self.region = region_rect
        self.region_defined = True
        # If user redefines region after exactly one capture, discard the old first image and reset
        if self.capture_count == 1:
            try:
                if os.path.exists('capture_1.png'):
                    os.remove('capture_1.png')
            except Exception:
                pass
            self.capture_count = 0
            self.capture_btn.setEnabled(True)
            self.status.setText(self.status_messages[0] + ' (region changed — please recapture Y_back)')
        else:
            self.capture_btn.setEnabled(True)
            self.status.setText(self.status_messages[0])

    # ---- Capture flow ----
    def capture(self):
        if not self.region_defined or self.capture_count >= self.max_captures:
            return

        with mss.mss() as sct:
            monitor = {
                "top": self.region.y(),
                "left": self.region.x(),
                "width": self.region.width(),
                "height": self.region.height(),
            }
            sct_img = sct.grab(monitor)
            filename = f"capture_{self.capture_count + 1}.png"
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=filename)
            self.capture_count += 1

        if self.capture_count == self.max_captures:
            self.capture_btn.setEnabled(False)
            # Show “points captured” status first
            self.status.setText("Points captured. Wait while processing")
            QApplication.processEvents()  # force UI to update before OCR starts
            self.run_ocr()
        else:
            self.status.setText(self.status_messages[self.capture_count])


    def get_absolute_zones(self, w, h):
        return [
            QRect(int(x1 * w), int(y1 * h), int((x2 - x1) * w), int((y2 - y1) * h))
            for (x1, y1, x2, y2) in self.ocr_zone_norms
        ]

    # ---- OCR helpers ----
    def _prep_for_ocr(self, arr_rgb: np.ndarray):
        """Return two thresholded variants (black-on-white and inverted)."""
        g = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        pad = max(2, int(min(w, h) * 0.01))
        g = g[pad:h - pad, pad:w - pad]

        # 2× upscale fattens thin strokes
        g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        g = cv2.GaussianBlur(g, (3, 3), 0)
        th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = 255 - th
        return [th, inv]

    def _segment_by_vproj(self, bw_img: np.ndarray):
        """Segment line into slices by vertical gaps. Expects 0/255 single-channel (digits ~ black=0)."""
        # Convert to 0/1 for convenience
        bin01 = (bw_img == 0).astype(np.uint8)
        col_sum = bin01.sum(axis=0)
        # A column is "gap" if very few black pixels
        gap_thr = max(1, int(bw_img.shape[0] * 0.01))
        gaps = (col_sum <= gap_thr).astype(np.uint8)

        slices = []
        in_run = False
        start = 0
        for x, g in enumerate(gaps):
            if not in_run and g == 0:
                in_run = True
                start = x
            elif in_run and g == 1:
                end = x
                if end - start > 2:  # ignore tiny runs
                    slices.append((start, end))
                in_run = False
        if in_run:
            end = len(gaps) - 1
            if end - start > 2:
                slices.append((start, end))

        # Merge overly small gaps between adjacent slices
        merged = []
        if slices:
            cur_s = list(slices[0])
            for s, e in slices[1:]:
                if s - cur_s[1] < 3:
                    cur_s[1] = e
                else:
                    merged.append(tuple(cur_s))
                    cur_s = [s, e]
            merged.append(tuple(cur_s))
        return merged

    def _standardize_slice(self, bw_img: np.ndarray, x1: int, x2: int, target_h: int = 48, pad_w: int = 8):
        """Crop slice x1:x2, resize to target height keeping aspect, then pad to a tidy canvas."""
        crop = bw_img[:, max(0, x1):min(bw_img.shape[1], x2)]
        if crop.size == 0:
            return None
        # Ensure digit is black on white for visualization; keep as grayscale
        h, w = crop.shape
        if h == 0 or w == 0:
            return None
        # Resize to target height
        new_w = max(1, int(w * (target_h / float(h))))
        resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        # Pad to give some margins
        canvas = 255 * np.ones((target_h + 2 * pad_w, new_w + 2 * pad_w), dtype=np.uint8)
        canvas[pad_w:pad_w + target_h, pad_w:pad_w + new_w] = resized
        # Convert to 3-channel RGB for RapidOCR
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    def _ocr_line_or_slices(self, arr_rgb: np.ndarray):
        """Try line OCR first; if empty/invalid, segment into slices and OCR each, then join."""
        variants = self._prep_for_ocr(arr_rgb)

        best_text = ""
        best_score = -1.0
        best_bw_for_slices = None

        # 1) Try full-line recognition on both polarities
        for v in variants:
            rgb = cv2.cvtColor(v, cv2.COLOR_GRAY2RGB)
            res, _ = ocr(rgb)
            if res:
                score = sum(r[2] for r in res if len(r) >= 3)
                txt = " ".join(r[1] for r in res)
                # Filter allowed chars
                txt = "".join(ch if ch in ALLOWED_CHARS or ch == " " else " " for ch in txt)
                if score > best_score:
                    best_score, best_text = score, txt
                    best_bw_for_slices = v

        # If the line OCR produced a usable number, return it
        m = re.search(r"[-+]?\d*\.?\d+", best_text or "")
        if m and m.group(0):
            return m.group(0)

        # 2) Fallback: segment into slices and OCR per-slice
        # Choose the variant with more black pixels (heuristic)
        if best_bw_for_slices is None:
            best_bw_for_slices = variants[0]
        # Ensure black digits on white background for segmentation
        # We'll assume digits are darker; choose inversion that yields more black pixels
        if (best_bw_for_slices == 255).sum() < (best_bw_for_slices == 0).sum():
            bw = best_bw_for_slices
        else:
            bw = 255 - best_bw_for_slices

        slices = self._segment_by_vproj(bw)
        pieces = []
        for (x1, x2) in slices:
            std_img = self._standardize_slice(bw, x1, x2, target_h=48, pad_w=6)
            if std_img is None:
                continue
            res, _ = ocr(std_img)
            if not res:
                continue
            txt = " ".join(r[1] for r in res)
            txt = "".join(ch for ch in txt if ch in ALLOWED_CHARS)
            if txt:
                pieces.append(txt)

        joined = "".join(pieces)
        m2 = re.search(r"[-+]?\d*\.?\d+", joined)
        if m2 and m2.group(0):
            return m2.group(0)
        return ""

    # ---- OCR ----
    def run_ocr(self):
        labels = ["X", "Y", "Z", "R"]
        data = {label: [] for label in labels}
        for i in range(1, 7):
            img = Image.open(f"capture_{i}.png")
            w, h = img.size
            zones = self.get_absolute_zones(w, h)
            for j, label in enumerate(labels):
                try:
                    crop = img.crop(
                        (
                            zones[j].x(),
                            zones[j].y(),
                            zones[j].x() + zones[j].width(),
                            zones[j].y() + zones[j].height(),
                        )
                    )
                    arr = np.array(crop.convert("RGB"))
                    val_txt = self._ocr_line_or_slices(arr)
                    val = float(val_txt) if val_txt else None
                    data[label].append(val)
                except Exception:
                    data[label].append(None)

        self.ocr_df = pd.DataFrame(data, index=[f"Capture_{i}" for i in range(1, 7)]).T
        self.ocr_df.to_csv("third_column_ocr_result.csv")
        self.status.setText("OCR complete. Table saved.")
        self.view_table_btn.setEnabled(True)
        self.get_program_btn.setEnabled(True)

    # ---- Program generation ----
    def handle_get_program(self):
        try:
            df = self.ocr_df
            Y_back = df.loc["Y", "Capture_1"]
            Y_front = df.loc["Y", "Capture_2"]
            X_left = df.loc["X", "Capture_3"]
            X_right = df.loc["X", "Capture_4"]
            X_left_aft_rot = df.loc["X", "Capture_5"]
            Y_back_aft_rot = df.loc["Y", "Capture_6"]
            Z_back = df.loc["Z", "Capture_1"]
            Z_front = df.loc["Z", "Capture_2"]
            Z_right = df.loc["Z", "Capture_4"]  # with X_right capture
            Z_left = df.loc["Z", "Capture_3"]  # with X_left capture

            defocus = float(self.defocus_input.text())
            code = self.get_program(
                Y_back,
                Y_front,
                X_right,
                X_left,
                X_left_aft_rot,
                Y_back_aft_rot,
                Z_back,
                Z_front,
                Z_right,
                Z_left,
                defocus,
            )
            self.show_code_window(code)
        except Exception as e:
            self.status.setText(f"Error: {e}")

    def get_program(
        self,
        Y_back,
        Y_front,
        X_right,
        X_left,
        X_left_aft_rot,
        Y_back_aft_rot,
        Z_back,
        Z_front,
        Z_right,
        Z_left,
        defocus,
    ):
        Z0 = Z_front
        Z1 = Z_front + (Z_right - Z_left) / 128 * 40
        Z2 = Z1 + (Z_back - Z_front)
        Z3 = Z_back + (Z_left - Z_right) / 128 * 40
        Z4 = Z3 + (Z_front - Z_back)
        Rad = (Y_back - Y_front) / 2
        X_lin = (X_right - X_left) - 2 * Rad
        Job_off_x = (X_left - X_left_aft_rot) / 2
        Job_off_y = (Y_back - Y_back_aft_rot) / 2
        X1 = X_lin / 2
        Y1 = -Rad
        X2 = X1 + 2 * Job_off_x
        Y2 = Y1 - 2 * Job_off_y
        X3 = X1 - 2 * Job_off_x
        Y3 = Y1
        I1 = -(X_lin / 2 - Job_off_x)
        J1 = Job_off_y
        I2 = -(X_lin / 2 + Job_off_x)
        J2 = Job_off_y

        return f"""
G90
G54
M68  R{Rad:.3f}  X{abs(I1):.3f}
G00  X0  Y{Y1:.3f}  Z{(Z0 + defocus):.3f}  C360  
G04  P2
G01  X{X1:.3f}  Y{Y1:.3f}  Z{(Z1 + defocus):.3f}
G02  X{-X2:.3f}  Y{Y2:.3f}  Z{(Z2 + defocus):.3f}  C180  I{I1:.3f}  J{J1:.3f}
G01  X{X3:.3f}  Y{Y2:.3f}  Z{(Z3 + defocus):.3f}
G02  X{-X1:.3f}  Y{Y1:.3f}  Z{(Z4 + defocus):.3f}  C0  I{I2:.3f}  J{J2:.3f}
G01  X3  Y{Y3:.3f}   Z{(Z0 + defocus):.3f}
G01  X5  Z{(Z0 + defocus + 1.5):.3f}
"""

    def show_code_window(self, code):
        dialog = QDialog(self)
        dialog.setWindowTitle("Generated Program")
        layout = QVBoxLayout()
        box = QTextEdit()

        # Apply hardcoded program font size
        f = QFont()
        f.setPointSize(PROGRAM_FONT_PT)
        box.setFont(f)

        box.setPlainText(code)
        box.setReadOnly(True)
        layout.addWidget(box)
        dialog.setLayout(layout)
        dialog.resize(700, 400)
        dialog.exec_()

    # ---- Viewers ----
    def view_all_images(self):
        if self.capture_count < self.max_captures:
            previous = self.status.text()
            self.status.setText("Capture all 6 screenshots first.")
            QTimer.singleShot(
                3000,
                lambda: self.status.setText(
                    previous
                    if self.capture_count < self.max_captures
                    else self.status_messages[6]
                ),
            )
            return

        viewer = QWidget()
        viewer.setWindowTitle("All Captures (2x3 Grid)")
        layout = QGridLayout()

        for i in range(6):
            file = f"capture_{i + 1}.png"
            if not os.path.exists(file):
                continue

            pix = QPixmap(file).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label = QLabel()
            label.setPixmap(pix)
            label.setAlignment(Qt.AlignCenter)

            # Use serial-labeled titles
            title = QLabel(self.capture_labels[i])
            title.setAlignment(Qt.AlignCenter)

            box = QVBoxLayout()
            box.addWidget(title)
            box.addWidget(label)

            container = QGroupBox()
            container.setLayout(box)

            layout.addWidget(container, i // 3, i % 3)

        viewer.setLayout(layout)
        viewer.resize(1000, 600)
        viewer.show()
        self.status.setText("All screenshots displayed.")
        self.viewer_window = viewer  # keep a reference to avoid GC

    def view_ocr_zones(self):
        if not os.path.exists("capture_1.png"):
            self.status.setText("capture_1.png not found.")
            return

        self.zone_viewer = QWidget()
        self.zone_viewer.setWindowTitle("OCR Zones Preview")
        layout = QVBoxLayout()
        self.zone_layout = layout
        self.zone_label = QLabel()
        layout.addWidget(self.zone_label)

        # Generate preview image with rectangles
        base = QPixmap("capture_1.png")
        width, height = base.width(), base.height()
        overlay = QPixmap(base.size())
        overlay.fill(Qt.transparent)

        painter = QPainter(overlay)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        for rect in self.get_absolute_zones(width, height):
            painter.drawRect(rect)
        painter.end()

        final = QPixmap(base)
        painter = QPainter(final)
        painter.drawPixmap(0, 0, overlay)
        painter.end()

        self.zone_label.setPixmap(final.scaled(600, 400, Qt.KeepAspectRatio))

        # Add the adjust button
        adjust_btn = QPushButton("Adjust OCR Zones")
        adjust_btn.clicked.connect(self.open_zone_editor)
        layout.addWidget(adjust_btn)

        self.zone_viewer.setLayout(layout)
        self.zone_viewer.resize(650, 500)
        self.zone_viewer.show()

    def update_zone_preview(self):
        base = QPixmap("capture_1.png")
        width = base.width()
        height = base.height()
        zones = self.get_absolute_zones(width, height)

        overlay = QPixmap(base.size())
        overlay.fill(Qt.transparent)

        painter = QPainter(overlay)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)

        for rect in zones:
            painter.drawRect(rect)
        painter.end()

        combined = QPixmap(base)
        painter = QPainter(combined)
        painter.drawPixmap(0, 0, overlay)
        painter.end()

        self.zone_label.setPixmap(combined.scaled(600, 400, Qt.KeepAspectRatio))
        if self.zone_layout.indexOf(self.zone_label) == -1:
            self.zone_layout.insertWidget(0, self.zone_label)

    def open_zone_editor(self):
        editor = ZoneEditor(self.ocr_zone_norms, self)
        editor.exec_()
        self.update_zone_preview()

    def show_ocr_table(self):
        if not hasattr(self, "ocr_df"):
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("OCR Result Table")
        layout = QVBoxLayout()

        table = QTableWidget()
        df = self.ocr_df
        table.setRowCount(len(df.index))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        table.setVerticalHeaderLabels(df.index.tolist())

        for i, row_label in enumerate(df.index):
            for j, col_label in enumerate(df.columns):
                value = df.loc[row_label, col_label]
                item = QTableWidgetItem("" if pd.isna(value) else str(value))
                table.setItem(i, j, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(table)
        dialog.setLayout(layout)
        dialog.resize(500, 300)
        dialog.exec_()


class RegionOverlay(QWidget):
    def __init__(self, parent_app, background_img):
        super().__init__()
        self.parent_app = parent_app
        self.bg = background_img
        self.begin = QPoint()
        self.end = QPoint()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        rect = QRect(self.begin, self.end).normalized()
        self.parent_app.region_selected(rect)
        self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.bg)
        if not self.begin.isNull() and not self.end.isNull():
            rect = QRect(self.begin, self.end)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(rect)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ScreenshotApp()
    win.show()
    sys.exit(app.exec_())
