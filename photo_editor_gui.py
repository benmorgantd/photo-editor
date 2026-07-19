"""PyQt6 desktop wrapper module running heavy multi-core image processing pipelines.

Provides standard dialog windows, automated workspace batch queues, active monitor 
histograms, and synchronized local execution logging routines.
"""

import traceback
import sys
import os
import json
import time
import datetime
import tempfile
import configparser
import logging
from typing import Any, List
import copy

import cv2
import numpy as np

try:
    import rawpy
    HAS_RAWPY_GUI = True
except ImportError:
    HAS_RAWPY_GUI = False

from PIL import Image, ImageOps

from PyQt6.QtCore import Qt, QDir, QThread, pyqtSignal, QTimer, QModelIndex, QObject, QRunnable, QThreadPool, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QFileSystemModel, QPainter, QKeyEvent, QFont, QIcon, QColor, QBrush, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTreeView, QGraphicsView, QGraphicsScene, QPushButton,
    QSlider, QCheckBox, QComboBox, QLabel, QGroupBox, QScrollArea, QFileDialog,
    QFrame, QLineEdit, QMenu, QProgressDialog, QMessageBox, QGraphicsPixmapItem, QRadioButton, QButtonGroup, QDoubleSpinBox, QSizePolicy
)

from photo_editor_core import PhotoEditor, export_photo, SUPPORTED_EXTENSIONS, RAW_EXTENSIONS, FILM_PROFILES, PipelineState

# Setup persistent file logging locations locally inside specific subfolder structures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"session_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PhotoEditor.GUI")
logger.info(f"Initialized application logging workspace output layout target: {log_file_path}")

def global_exception_handler(exc_type, exc_value, exc_traceback):
    # Allow standard Ctrl+C keyboard interrupts to kill the process normally
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the crash with the full traceback
    logger.critical("Uncaught exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))

# Bind the custom handler to Python's global exception hook
sys.excepthook = global_exception_handler

if not HAS_RAWPY_GUI:
    logger.warning("GUI running without local environment rawpy binding confirmation context.")

class ExportWorker(QThread):
    """Background worker thread executing heavy matrix transformations and disk I/O.

    Attributes:
        export_queue (List[tuple]): Collection of target export file tuples.
        preset_map (Dict[str, Any]): Main fallback configuration preset context mappings.
    """
    
    progress_signal = pyqtSignal(str)
    file_done_signal = pyqtSignal(int, int)  # processed, total
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, export_queue: list, preset_map: dict):
        """Initializes the background thread pool queue asset parameters."""
        super().__init__()
        self.export_queue = export_queue
        self.preset_map = preset_map
        
        # Thread-safe flag for cancellation
        self._is_cancelled = False

    def cancel(self):
        """Requests the thread to halt execution at the next safe point."""
        self._is_cancelled = True

    def run(self):
        """Executes background asset encoding operations looping through the job parameters."""
        total_files = len(self.export_queue)
        logger.info(f"Background Export Worker Thread triggered for total batch volume size: {total_files}")
        
        try:
            for idx, item in enumerate(self.export_queue):
                # Check for cancellation at the start of each file
                if self._is_cancelled:
                    logger.info("User cancelled export, exiting...")
                    break

                img_path, out_path, item_preset = item
                active_crop_variant = item_preset['active_crop_variant']
                crop_data = item_preset['crop_variants'][active_crop_variant]
                base_name = os.path.basename(img_path)
                
                logger.info(f"Starting batch task segment processing pass ({idx+1}/{total_files}): {base_name}")
                self.progress_signal.emit(f"[{idx+1}/{total_files}] Loading high-fidelity asset matrix for: {base_name}...")
                
                # TODO: start an instance of PhotoEditor
                full_res = PhotoEditor.load_image_matrix(img_path, preview=False)

                # Check again before the next heavy operation
                if self._is_cancelled:
                    break

                self.progress_signal.emit(f"[{idx+1}/{total_files}] Applying high-res crop dimensions for: {base_name}...")
                cropped_full = PhotoEditor.apply_crop(full_res, crop_data)

                if self._is_cancelled:
                    break

                self.progress_signal.emit(f"[{idx+1}/{total_files}] Computing multi-core tile pixel transformations...")
                photo_editor = PhotoEditor(img_path)
                processed_full = photo_editor.run_parallel_pipeline(cropped_full, item_preset, None)

                if self._is_cancelled:
                    break

                self.progress_signal.emit(f"[{idx+1}/{total_files}] Writing compressed target output JPG to disk...")
                export_photo(processed_full, out_path, item_preset)

                logger.info(f"Successfully processed image output save task configuration: {out_path}")
                self.file_done_signal.emit(idx + 1, total_files)

            # Determine how the loop ended
            if self._is_cancelled:
                logger.info("Batch execution aborted by user.")
                self.finished_signal.emit(False, "Export cancelled by user.")
            else:
                self.finished_signal.emit(True, f"Successfully exported {total_files} asset layers safely.")
                
        except Exception as e:
            logger.error(f"Batch execution context worker thread encountered fatal termination: {e}")
            self.finished_signal.emit(False, f"Export operation stopped due to exception: {str(e)}")


class ImageLoaderSignals(QObject):
    """Defines the signals available from a running image loading worker thread."""
    finished = pyqtSignal(str, object)  # Emits (file_path, full_res_matrix)
    error = pyqtSignal(str, str)        # Emits (file_path, error_message)


class FullResLoaderWorker(QRunnable):
    """Worker task that decodes full-resolution matrices in a background thread pool."""
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.signals = ImageLoaderSignals()

    def run(self):
        try:
            # This heavy I/O and matrix decoding now runs off the main event loop
            matrix = PhotoEditor.load_image_matrix(self.file_path, preview=True, max_width=720)
            self.signals.finished.emit(self.file_path, matrix)
        except Exception as e:
            err_msg = f"{e}\n{traceback.format_exc()}"
            self.signals.error.emit(self.file_path, err_msg)



class FileEditHistory:
    def __init__(self, default_preset: dict):
        # Initialize the stack with a deep copy of the default state
        self.states = [copy.deepcopy(default_preset)]
        self.current_index = 0

    def push_state(self, new_preset: dict):
        # If we undid and then made a new edit, truncate the redo future
        self.states = self.states[:self.current_index + 1]
        
        # Append the new state and advance the pointer
        self.states.append(copy.deepcopy(new_preset))
        self.current_index += 1

    def undo(self) -> dict:
        if self.can_undo():
            self.current_index -= 1
        return self.get_current_state()

    def redo(self) -> dict:
        if self.can_redo():
            self.current_index += 1
        return self.get_current_state()

    def can_undo(self) -> bool:
        return self.current_index > 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.states) - 1

    def get_current_state(self) -> dict:
        return copy.deepcopy(self.states[self.current_index])
    

class CustomFileSystemModel(QFileSystemModel):
    """Extended file model inserting contextual status markers dynamically inside listings.

    Attributes:
        main_window (QMainWindow): Reference hook linking parent application structures.
    """
    
    def __init__(self, main_window, parent=None):
        """Initializes custom proxy model structures.

        Args:
            main_window (QMainWindow): Main reference mapping interface window context layer.
            parent (QWidget, optional): Underlying widget parent framework hook references.
        """
        super().__init__(parent)
        self.main_window = main_window

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        """Overrides file naming models to append priority badge labels in active tree slots.

        Args:
            index (QModelIndex): Active coordinate identifier matching requested table structures.
            role (int): Render layout identifier type description.

        Returns:
            Any: Formatted model item properties data payload strings or fonts.
        """
        if role == Qt.ItemDataRole.DisplayRole and index.column() == 0:
            path = self.filePath(index)
            base_string = super().data(index, role)
            if not base_string:
                return base_string

            prefix = ""
            suffix = ""
            if path in self.main_window.starred_files:
                prefix += "⭐ "
            if path in self.main_window.exported_files:
                suffix += " 💾"
            if path in self.main_window.marked_files:
                prefix += "✅ "

            return f"{prefix}{base_string}{suffix}"
            
        if role == Qt.ItemDataRole.FontRole:
            path = self.filePath(index)
            if path in self.main_window.edited_files:
                font = super().data(index, role)
                if not isinstance(font, QFont):
                    font = QFont()
                font.setItalic(True)
                return font
        
        if role == Qt.ItemDataRole.ForegroundRole:
            path = self.filePath(index)
            if path not in self.main_window.edited_files:
                return QBrush(QColor(Qt.GlobalColor.gray))
                
        return super().data(index, role)


class HistogramWidget(QFrame):
    """Custom painting widget to extract and graph floating-point luminance distributions.

    Attributes:
        hist_data (np.ndarray): Gathered density calculation arrays representing active data profiles.
        clipping_left (bool): Out-of-bounds crushed exposure confirmation indicator flags.
        clipping_right (bool): Highlight burn confirmation indicator flags.
    """

    def __init__(self, parent=None):
        """Initializes standardized style constraints for the embedded paint field.

        Args:
            parent (QWidget, optional): Parent element layer targets reference.
        """
        super().__init__(parent)
        self.setFixedHeight(115)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.hist_data = None
        self.clipping_left = False
        self.clipping_right = False

    def render_histogram(self, rgb_matrix: np.ndarray):
        """Transforms multi-channel floats into standard brightness distribution ranges.

        Args:
            rgb_matrix (np.ndarray): Rendered raster data matrix containing image details.
        """
        if rgb_matrix is None:
            self.hist_data = None
            self.update()
            return

        gray = np.float32(0.299) * rgb_matrix[:, :, 0] + np.float32(0.587) * rgb_matrix[:, :, 1] + np.float32(0.114) * rgb_matrix[:, :, 2]
        gray_uint8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)

        hist = cv2.calcHist([gray_uint8], [0], None, [256], [0, 256])
        self.hist_data = hist

        pixel_ceiling = gray_uint8.size * 0.003
        self.clipping_left = hist[0][0] > pixel_ceiling
        self.clipping_right = hist[255][0] > pixel_ceiling
        self.update()

    def paintEvent(self, event):
        """Executes targeted vector draws on localized graphics canvas updates.

        Args:
            event (QPaintEvent): System draw update parameters structure wrapper hook.
        """
        painter = QPainter(self)
        rect = self.contentsRect()
        painter.fillRect(rect, Qt.GlobalColor.black)

        if self.hist_data is None:
            return

        max_val = np.max(self.hist_data[1:-1])
        if max_val == 0: max_val = 1

        w, h = rect.width(), rect.height()
        pen = painter.pen()
        pen.setColor(Qt.GlobalColor.darkGray)
        painter.setPen(pen)

        for i in range(256):
            x = int((i / 256.0) * w)
            bin_value = self.hist_data[i][0]
            bar_height = int((bin_value / max_val) * (h - 15))
            bar_height = min(bar_height, h - 15)
            painter.drawLine(x, h, x, h - bar_height)

        if self.clipping_left:
            painter.setPen(Qt.GlobalColor.red)
            painter.drawText(8, 18, "CRUSHED BLACKS")
        if self.clipping_right:
            painter.setPen(Qt.GlobalColor.red)
            painter.drawText(w - 110, 18, "CLIPPED WHITES")


class CollapsibleGroupBox(QGroupBox):
    """Custom container element introducing uniform drawer visibility states.

    Attributes:
        toggle_btn (QPushButton): Panel drawer state management button.
        content_widget (QWidget): Nested child widget collection array layer.
    """

    def __init__(self, title: str, parent=None):
        """Constructs layout panels mapped inside sliding drawer bounds.

        Args:
            title (str): Group descriptive display text.
            parent (QWidget, optional): Target alignment hierarchy reference.
        """
        super().__init__(title, parent)
        self.box_layout = QVBoxLayout(self)
        self.box_layout.setContentsMargins(6, 8, 6, 6)
        self.box_layout.setSpacing(4)

        self.toggle_btn = QPushButton("▼ Collapse Panel Layer")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)
        self.toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; padding: 4px; "
            "border: 1px solid #444; background-color: #353535; color: #eee; border-radius: 3px; }"
            "QPushButton:checked { background-color: #252525; color: #999; }"
        )
        self.toggle_btn.toggled.connect(self._on_toggle_triggered)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 4, 0, 0)
        self.content_layout.setSpacing(4)

        self.box_layout.addWidget(self.toggle_btn)
        self.box_layout.addWidget(self.content_widget)

    def _on_toggle_triggered(self, checked: bool):
        """Updates internal visualization configurations based on state parameters.

        Args:
            checked (bool): Drawer display state validation flag.
        """
        if checked:
            self.content_widget.hide()
            self.toggle_btn.setText("▶ Expand Panel Layer")
        else:
            self.content_widget.show()
            self.toggle_btn.setText("▼ Collapse Panel Layer")

# TODO: fit frame to view method and alt+rc zooming
class ZoomableGraphicsView(QGraphicsView):
    """Viewport container supporting mouse wheel zooming and click-and-drag panning."""

    ZOOM_SENSITIVITY = 0.05

    def __init__(self, parent=None):
        """Constructs viewport parameters matching interactive scene controls.

        Args:
            parent (QWidget, optional): Main container layout context.
        """
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QApplication.palette().dark())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event):
        # Calculate zoom factor
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        
        # Save the scene position before scaling
        pos_before = self.mapToScene(event.position().toPoint())
        
        # Scale the view
        self.scale(factor, factor)
        
        # Map the mouse position to scene again and adjust center 
        # to keep the mouse anchored to the same spot
        pos_after = self.mapToScene(event.position().toPoint())
        delta = pos_after - pos_before
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
    
    def zoom_to_fit(self):
        scene = self.scene()
        if not scene or scene.itemsBoundingRect().isNull():
            return

        # 1. Disable auto-alignment so it doesn't fight our manual scroll calculation
        self.setAlignment(Qt.AlignmentFlag(0)) 
        
        # 2. Reset transform to get a clean baseline
        self.resetTransform()
        
        # 3. Calculate the scale factor
        target_rect = scene.itemsBoundingRect()
        view_rect = self.viewport().rect()
        
        scale_factor = min(
            view_rect.width() / target_rect.width(),
            view_rect.height() / target_rect.height()
        ) * 0.95
        
        # 4. Apply scale
        self.scale(scale_factor, scale_factor)
        
        # 5. Force center using the scene center point
        # centerOn() works best when the view doesn't have an Alignment set
        self.centerOn(target_rect.center())

# ---------------------------------------------------------
# Native 1D Natural Cubic Spline Interpolation Solver
# ---------------------------------------------------------
def solve_natural_cubic_spline(x, y):
    """
    Computes the coefficients for a natural cubic spline.
    x and y must be 1D arrays/lists of length N.
    Returns a function that interpolates new values.
    """
    n = len(x)
    h = np.diff(x)
    
    # Set up the tridiagonal system for the second derivatives (M)
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # Natural boundary conditions: M_0 = M_{n-1} = 0
    A[0, 0] = 1.0
    A[n-1, n-1] = 1.0
    
    for i in range(1, n - 1):
        A[i, i-1] = h[i-1] / 6.0
        A[i, i] = (h[i-1] + h[i]) / 3.0
        A[i, i+1] = h[i] / 6.0
        B[i] = (y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]
        
    M = np.linalg.solve(A, B)
    
    def interpolate(x_new):
        x_new = np.atleast_1d(x_new)
        y_new = np.zeros_like(x_new, dtype=float)
        
        for idx, val in enumerate(x_new):
            # Clamp value to range
            val = np.clip(val, x[0], x[-1])
            # Find interval
            i = np.searchsorted(x, val) - 1
            i = max(0, min(i, n - 2))
            
            hi = h[i]
            a = (x[i+1] - val) / hi
            b = (val - x[i]) / hi
            
            y_val = (a * y[i] + b * y[i+1] + 
                     ((a**3 - a) * M[i] + (b**3 - b) * M[i+1]) * (hi**2) / 6.0)
            y_new[idx] = np.clip(y_val, 0.0, 1.0) # Keep within 0.0 - 1.0 boundaries
        return y_new
        
    return interpolate


# ---------------------------------------------------------
# Custom Interactive Curves Widget
# ---------------------------------------------------------
class CurveCanvas(QWidget):
    """Internal canvas widget dedicated solely to rendering and mouse interaction."""
    pointDragged = pyqtSignal(str, int, float)
    pointReleased = pyqtSignal(str, int, float)
    selectionChanged = pyqtSignal(object)  # Emits int index or None
    pointMovedLive = pyqtSignal(float)     # Emits Y value during drag for UI sync

    DEFAULT_CURVE_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.x_points = self.DEFAULT_CURVE_VALS.copy()
        self.curves = {
            'r': self.DEFAULT_CURVE_VALS.copy(),
            'g': self.DEFAULT_CURVE_VALS.copy(),
            'b': self.DEFAULT_CURVE_VALS.copy()
        }
        
        self.current_channel = 'r'
        self.selected_idx = None
        self.hover_point_idx = None
        self.is_dragging = False
        
        # Throttling configuration
        self.min_val_delta = 0.005      # Minimum Y value change required to emit
        self.min_time_delta = 0.033     # Minimum time (seconds) between emits (~60Hz)
        self._last_emit_time = 0.0
        self._last_emitted_val = None
        
        self.margin = 30
        self.point_radius = 6

    def set_channel(self, channel):
        if channel in ['r', 'g', 'b']:
            self.current_channel = channel
            self.hover_point_idx = None
            self.update()

    def get_curves_dict(self):
        return {ch: list(self.curves[ch]) for ch in ['r', 'g', 'b']}

    def set_curve_values(self, r_vals, g_vals, b_vals):
        for channel, vals in [('r', r_vals), ('g', g_vals), ('b', b_vals)]:
            if len(vals) != 5:
                raise ValueError(f"Each channel must have exactly 5 points. Got {len(vals)} for '{channel}'.")
            self.curves[channel] = [max(0.0, min(1.0, float(v))) for v in vals]
        
        self.selected_idx = None
        self.hover_point_idx = None
        self.selectionChanged.emit(None)
        self.update()

    def set_selected_point_y(self, val):
        """Called when the user edits the Y spinbox manually."""
        if self.selected_idx is not None:
            clamped_val = max(0.0, min(1.0, float(val)))
            self.curves[self.current_channel][self.selected_idx] = clamped_val
            self.pointDragged.emit(self.current_channel, self.selected_idx, clamped_val)
            self.pointReleased.emit(self.current_channel, self.selected_idx, clamped_val)
            self.update()

    # --- Geometry Mapping Helpers ---
    def _canvas_rect(self):
        w, h = self.width(), self.height()
        size = min(w, h) - (self.margin * 2)
        x_offset = (w - size) / 2
        y_offset = (h - size) / 2
        return QRectF(x_offset, y_offset, size, size)

    def _to_pixels(self, norm_x, norm_y, rect):
        px = rect.left() + norm_x * rect.width()
        py = rect.bottom() - norm_y * rect.height()
        return QPointF(px, py)

    def _to_normalized_y(self, pixel_y, rect):
        norm_y = (rect.bottom() - pixel_y) / rect.height()
        return max(0.0, min(1.0, norm_y))

    # --- Mouse Events ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            rect = self._canvas_rect()
            mouse_pos = event.position()
            y_vals = self.curves[self.current_channel]
            
            hit_idx = None
            for i in range(5):
                pt_pixels = self._to_pixels(self.x_points[i], y_vals[i], rect)
                if (mouse_pos - pt_pixels).manhattanLength() < 15:
                    hit_idx = i
                    break
            
            self.selected_idx = hit_idx
            self.selectionChanged.emit(hit_idx)
            
            if hit_idx is not None:
                self.is_dragging = True
                self._last_emitted_val = y_vals[hit_idx]
                self._last_emit_time = time.monotonic()
            
            self.update()

    def mouseMoveEvent(self, event):
        rect = self._canvas_rect()
        mouse_pos = event.position()
        y_vals = self.curves[self.current_channel]
        
        if self.is_dragging and self.selected_idx is not None:
            norm_y = self._to_normalized_y(mouse_pos.y(), rect)
            self.curves[self.current_channel][self.selected_idx] = norm_y
            
            # Always sync UI spinbox during drag
            self.pointMovedLive.emit(norm_y)
            
            # Evaluate throttle gates
            current_time = time.monotonic()
            time_delta = current_time - self._last_emit_time
            val_delta = abs(norm_y - (self._last_emitted_val if self._last_emitted_val is not None else 0.0))
            
            if val_delta >= self.min_val_delta and time_delta >= self.min_time_delta:
                self.pointDragged.emit(self.current_channel, self.selected_idx, norm_y)
                self._last_emit_time = current_time
                self._last_emitted_val = norm_y
                
            self.update()
            return
            
        # Hover logic when not dragging
        old_hover = self.hover_point_idx
        self.hover_point_idx = None
        for i in range(5):
            pt_pixels = self._to_pixels(self.x_points[i], y_vals[i], rect)
            if (mouse_pos - pt_pixels).manhattanLength() < 12:
                self.hover_point_idx = i
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                break
                
        if self.hover_point_idx is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
        if old_hover != self.hover_point_idx:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            if self.selected_idx is not None:
                final_val = self.curves[self.current_channel][self.selected_idx]
                
                # Guarantee downstream sync if throttled out on the last micro-movement
                if final_val != self._last_emitted_val:
                    self.pointDragged.emit(self.current_channel, self.selected_idx, final_val)
                
                self.pointReleased.emit(self.current_channel, self.selected_idx, final_val)
            self.update()

    # --- Drawing Logic ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._canvas_rect()
        
        painter.fillRect(rect, QColor(25, 25, 25))
        
        grid_pen = QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for val in [0.25, 0.5, 0.75]:
            painter.drawLine(self._to_pixels(val, 1.0, rect), self._to_pixels(val, 0.0, rect))
            painter.drawLine(self._to_pixels(0.0, val, rect), self._to_pixels(1.0, val, rect))

        border_pen = QPen(QColor(90, 90, 90), 1, Qt.PenStyle.SolidLine)
        painter.setPen(border_pen)
        painter.drawRect(rect)
        
        guide_pen = QPen(QColor(50, 50, 50), 1, Qt.PenStyle.SolidLine)
        painter.setPen(guide_pen)
        painter.drawLine(self._to_pixels(0.0, 0.0, rect), self._to_pixels(1.0, 1.0, rect))

        channel_colors = {
            'r': QColor(235, 75, 75),
            'g': QColor(75, 215, 75),
            'b': QColor(75, 140, 245)
        }
        line_color = channel_colors[self.current_channel]

        y_vals = self.curves[self.current_channel]
        spline_eval = solve_natural_cubic_spline(self.x_points, y_vals)
        
        painter.setPen(QPen(line_color, 2, Qt.PenStyle.SolidLine))
        resolution = 100
        x_samples = np.linspace(0.0, 1.0, resolution)
        y_samples = spline_eval(x_samples)
        
        for idx in range(resolution - 1):
            p1 = self._to_pixels(x_samples[idx], y_samples[idx], rect)
            p2 = self._to_pixels(x_samples[idx+1], y_samples[idx+1], rect)
            painter.drawLine(p1, p2)

        for i in range(5):
            pt = self._to_pixels(self.x_points[i], y_vals[i], rect)
            
            # Distinct visual priority: Selected > Hover > Standard
            if i == self.selected_idx:
                painter.setBrush(QBrush(line_color))
                painter.setPen(QPen(QColor(255, 255, 255), 2.5))
                r = self.point_radius + 3
            elif i == self.hover_point_idx:
                painter.setBrush(QBrush(line_color))
                painter.setPen(QPen(QColor(220, 220, 220), 1.5))
                r = self.point_radius + 1.5
            else:
                painter.setBrush(QBrush(QColor(30, 30, 30)))
                painter.setPen(QPen(line_color, 2))
                r = self.point_radius
                
            painter.drawEllipse(pt, r, r)


# ---------------------------------------------------------
# Main Wrapper Widget (Exposes Original API + Control Bar)
# ---------------------------------------------------------
class CurveEditorWidget(QWidget):
    pointDragged = pyqtSignal(str, int, float)
    pointReleased = pyqtSignal(str, int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 380)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # --- Top Control Bar ---
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 5, 10, 0)
        
        self.x_label = QLabel("X:")
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(0.0, 1.0)
        self.x_spin.setDecimals(3)
        self.x_spin.setEnabled(False) # X is permanently locked
        
        self.y_label = QLabel("Y:")
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0.0, 1.0)
        self.y_spin.setDecimals(3)
        self.y_spin.setSingleStep(0.01)
        self.y_spin.setEnabled(False) # Enabled only when point selected
        
        control_layout.addWidget(self.x_label)
        control_layout.addWidget(self.x_spin)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.y_label)
        control_layout.addWidget(self.y_spin)
        control_layout.addStretch()
        
        # --- Canvas ---
        self.canvas = CurveCanvas(self)
        
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas)
        
        # --- Signal Routing ---
        self.canvas.pointDragged.connect(self.pointDragged.emit)
        self.canvas.pointReleased.connect(self.pointReleased.emit)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        self.canvas.pointMovedLive.connect(self._update_y_spinbox_silent)
        self.y_spin.valueChanged.connect(self.canvas.set_selected_point_y)

    def _on_selection_changed(self, idx):
        if idx is None:
            self.x_spin.setValue(0.0)
            self._update_y_spinbox_silent(0.0)
            self.y_spin.setEnabled(False)
        else:
            self.x_spin.setValue(self.canvas.x_points[idx])
            current_y = self.canvas.curves[self.canvas.current_channel][idx]
            self._update_y_spinbox_silent(current_y)
            self.y_spin.setEnabled(True)

    def _update_y_spinbox_silent(self, val):
        """Updates spinbox text without triggering valueChanged signal back to canvas."""
        self.y_spin.blockSignals(True)
        self.y_spin.setValue(val)
        self.y_spin.blockSignals(False)

    # --- Proxy Public API Methods ---
    def set_channel(self, channel):
        self.canvas.set_channel(channel)
        # Refresh spinbox if a point was already selected when switching channels
        if self.canvas.selected_idx is not None:
            new_y = self.canvas.curves[self.canvas.current_channel][self.canvas.selected_idx]
            self._update_y_spinbox_silent(new_y)

    def get_curves_dict(self):
        return self.canvas.get_curves_dict()

    def set_curve_values(self, r_vals, g_vals, b_vals):
        self.canvas.set_curve_values(r_vals, g_vals, b_vals)

class MainWindow(QMainWindow):
    """Main window object orchestrates user interface layout structures.

    Attributes:
        current_file_path (str): File destination address mapping link string.
        preview_matrix (np.ndarray): Fast downsampled source array reference.
        preset (Dict[str, Any]): Working property parameter data structures.
        manifest_json_path (str): Consolidated database location path tracking metadata presets.
    """

    def __init__(self):
        """Prepares internal state parameters and initiates interface construction routines."""
        super().__init__()
        # Render the camera emoji to a Pixmap since QIcon needs an image
        icon_pixmap = QPixmap(64, 64)
        icon_pixmap.fill(Qt.GlobalColor.transparent) # Ensure background is clear
        
        painter = QPainter(icon_pixmap)
        painter.setFont(QFont("Segoe UI Emoji", 40)) # Use Windows emoji font
        painter.drawText(icon_pixmap.rect(), 0x0004 | 0x0080, "📸") # Centered alignment
        painter.end()
        
        # 3. Set the generated icon
        QApplication.instance().setWindowIcon(QIcon(icon_pixmap))
        self.setWindowTitle("Free Python Desktop Photo Editor by Ben Morgan")
        self.resize(1600, 950)

        self.crop_data_cache = {'cropped_matrix' : None, 'crop_data' : None}

        self.default_curve_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

        self.photo_editor_instance = None

        self.thread_pool = QThreadPool.globalInstance()

        # Create a cache of the image matrices we load. These never change, even after edits.
        self.image_matrix_cache = dict()
        self.pipeline_state_cache = dict()

        self.history_registry = dict()

        self.current_file_path = ""
        self.preview_matrix = None
        self.proxy_matrix = None
        self.show_original_state = False
        self.is_updating_ui = False
        self.export_thread = None
        self.copied_settings_buffer = None
        self.current_selected_folder_path = ""
        self.is_in_browse_mode = False
        self.is_export_all_variants_set = True

        self.cache_directory = os.path.join(tempfile.gettempdir(), "photo_editor_session_cache")
        os.makedirs(self.cache_directory, exist_ok=True)
        self.state_ini_path = os.path.join(self.cache_directory, "session_state.ini")
        self.registry_json_path = os.path.join(self.cache_directory, "file_status_registry.json")
        self.manifest_json_folder = os.path.join(self.cache_directory, 'preset_manifests')
        if not os.path.exists(self.manifest_json_folder):
            os.makedirs(self.manifest_json_folder)
        # self.manifest_json_path = os.path.join(self.cache_directory, "photo_editor_presets_manifest.json")

        self.starred_files = set()
        self.exported_files = set()
        self.marked_files = set()
        self.edited_files = set()
        self._load_file_status_registry()

        self.default_preset = PhotoEditor.DEFAULT_PRESET
        self.preset = json.loads(json.dumps(self.default_preset))
        self.sliders_map = {}

        self._init_menu_bar()
        self._init_ui_layout()
        self._populate_local_presets()
        
        self.preview_render_timer = QTimer(self)
        self.preview_render_timer.setSingleShot(True)
        self.preview_render_timer.timeout.connect(self._render_stationary_pane)
        self.preview_target_path = ""

        self.is_scrubbing = False

        self.showMaximized()
        logger.info("Main photo editor workspace frame initialization lifecycle complete.")

    @property
    def active_crop_data(self):
        crop_variants = self.preset.get('crop_variants', dict())
        return crop_variants.get(self.active_crop_variant, PhotoEditor.DEFAULT_PRESET['crop_variants']['default'])
    
    @property
    def active_crop_variant(self):
        return self.preset.get('active_crop_variant', 'default')
    
    def _get_preset_manifest_filepath_for_image(self, image_filepath: str) -> str:
        # combine the folder it's from and the file's name for a more unique name
        folder_name = image_filepath.split('/')[-2].replace(' ', '_')
        file_name = image_filepath.split('/')[-1]
        # ex: yellowstone_2026_d2577.dng.json
        manifest_file_name = f'{folder_name}_{file_name}.json'
        manifest_file_path = os.path.join(self.manifest_json_folder, manifest_file_name)

        return manifest_file_path

    def _read_preset_from_manifest(self, file_path: str) -> dict:
        """Queries the consolidated database file using absolute path strings as keys.

        Args:
            file_path (str): Unique path destination targeting an image configuration.

        Returns:
            dict: Preset mapping configuration properties.
        """

        manifest_file_path = self._get_preset_manifest_filepath_for_image(file_path)
        logger.info(f'Using manifest filepath {manifest_file_path}')

        if os.path.exists(manifest_file_path):
            try:
                with open(manifest_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                logger.error(f'Failed to get preset manifest data for image {file_path}: {e}')
        
        logger.info(f"Edits do not exist yet for {file_path}, using default.")
        result = self.default_preset.copy()
        return result

    def _write_preset_to_manifest(self, file_path: str, preset_data: dict):
        """Persists the updated metadata preset block inside the single manifest archive file.

        Args:
            file_path (str): Direct image location key value.
            preset_data (dict): Snapshotted parameters data metrics layout maps.
        """

        manifest_file_path = self._get_preset_manifest_filepath_for_image(file_path)

        # Just overwrite the file
        try:
            with open(manifest_file_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f'Failed to write manifest json data for file {manifest_file_path}: {e}')

    def _init_menu_bar(self):
        """Assembles parent layout level drop-down action lists on the main toolbar shell."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        set_root_action = QAction("&Set Root Folder...", self)
        set_root_action.setShortcut(QKeySequence("Ctrl+O"))
        set_root_action.triggered.connect(self._change_root_folder)
        file_menu.addAction(set_root_action)

        file_menu.addSeparator()

        open_preset_action = QAction("&Open Preset...", self)
        open_preset_action.setShortcut(QKeySequence("Ctrl+P"))
        open_preset_action.triggered.connect(self._open_preset_file)
        file_menu.addAction(open_preset_action)

        save_preset_action = QAction("&Save Edits to Preset...", self)
        save_preset_action.setShortcut(QKeySequence("Ctrl+S"))
        save_preset_action.triggered.connect(self._save_preset_file)
        file_menu.addAction(save_preset_action)

        file_menu.addSeparator()

        self.export_action = QAction("&Export Selected...", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(self._trigger_file_export)
        file_menu.addAction(self.export_action)

        export_all_starred_action = QAction("Export All &Starred Photos...", self)
        export_all_starred_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_all_starred_action.triggered.connect(self._trigger_batch_starred_export)
        file_menu.addAction(export_all_starred_action)

        edit_menu = menu_bar.addMenu('&Edit')

        undo_action = QAction('&Undo', self)
        undo_action.setShortcut(QKeySequence('Ctrl+Z'))
        undo_action.triggered.connect(self._undo)
        edit_menu.addAction(undo_action)

        reset_edits_action = QAction('&Reset Edits', self)
        reset_edits_action.setShortcut(QKeySequence('Ctrl+R'))
        reset_edits_action.triggered.connect(self._reset_edits)
        edit_menu.addAction(reset_edits_action)
        
        copy_settings_action = QAction("&Copy Settings", self)
        copy_settings_action.setShortcut(QKeySequence("Ctrl+C"))
        copy_settings_action.triggered.connect(self._copy_edit_settings)
        edit_menu.addAction(copy_settings_action)

        paste_settings_action = QAction("&Paste Settings", self)
        paste_settings_action.setShortcut(QKeySequence("Ctrl+V"))
        paste_settings_action.triggered.connect(self._paste_edit_settings)
        edit_menu.addAction(paste_settings_action)

        apply_profiles_menu = edit_menu.addMenu('&Apply Film Profile')

        for profile_name in FILM_PROFILES:
            profile_action = QAction(f'&{profile_name}', self)
            profile_action.triggered.connect(lambda _, name=profile_name: self._apply_film_profile(name))
            apply_profiles_menu.addAction(profile_action)

        tools_menu = menu_bar.addMenu("&Tools")

        self.highlight_clipped_action = QAction("&Highlight Clipped Pixels", self)
        self.highlight_clipped_action.setCheckable(True)
        self.highlight_clipped_action.triggered.connect(self._refresh_viewport)
        tools_menu.addAction(self.highlight_clipped_action)

        tools_menu.addSeparator()
        self.toggle_browse_mode_action = QAction("&Toggle Browse Mode", self)
        self.toggle_browse_mode_action.setCheckable(True)
        self.toggle_browse_mode_action.triggered.connect(self._toggle_browse_mode)
        tools_menu.addAction(self.toggle_browse_mode_action)

        tools_menu.addSeparator()
        self.export_variants_action = QAction("&Export All Variants", self)
        self.export_variants_action.setCheckable(True)
        self.export_variants_action.setChecked(True)
        self.export_variants_action.triggered.connect(self._toggle_export_variants)
        tools_menu.addAction(self.export_variants_action)

        tools_menu.addSeparator()
        self.apply_auto_edit_action = QAction('&Apply Auto Edits', self)
        self.apply_auto_edit_action.triggered.connect(self._apply_auto_edits)
        self.apply_auto_edit_action.setShortcut(QKeySequence("1"))
        tools_menu.addAction(self.apply_auto_edit_action)

        self.apply_auto_edit_to_folder_action = QAction('&Apply Auto Edits to Folder', self)
        self.apply_auto_edit_to_folder_action.triggered.connect(self._apply_auto_edits_to_folder)
        tools_menu.addAction(self.apply_auto_edit_to_folder_action)

        debug_menu = menu_bar.addMenu("&Debug")
        
        open_cache_action = QAction("Open Cache Folder Location Pass", self)
        open_cache_action.triggered.connect(self._debug_open_cache_folder)
        debug_menu.addAction(open_cache_action)

        calc_size_action = QAction("Calculate Cache Disk Size Footprint", self)
        calc_size_action.triggered.connect(self._debug_calculate_cache_size)
        debug_menu.addAction(calc_size_action)

        purge_cache_action = QAction("Purge Session Cache Files Completely", self)
        purge_cache_action.triggered.connect(self._debug_purge_cache_files)
        debug_menu.addAction(purge_cache_action)

        view_menu = menu_bar.addMenu('&View')
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut(QKeySequence("F"))
        reset_view_action.triggered.connect(self._zoom_to_fit)
        view_menu.addAction(reset_view_action)
    
    def _undo(self):
        logger.info(f'Undoing one step for {self.current_file_path}')
        state_to_render = self.history_registry[self.current_file_path].undo()
        self.preset = state_to_render  # It's already copied
        self._apply_preset_to_ui()
        self._save_current_edits_to_session_cache()
    
    def _reset_edits(self):
        logger.info(f'Reseting edits for {self.current_file_path}')
        self.preset = self.default_preset.copy()
        self._apply_preset_to_ui()
        self._save_current_edits_to_session_cache()
        self.statusBar().showMessage("Reset edits to default")


    def _copy_edit_settings(self):
        """Copies the current preset settings to an internal clipboard buffer or triggers native text copy."""
        focused = self.focusWidget()
        if isinstance(focused, QLineEdit):
            focused.copy()
            return
            
        self.copied_settings_buffer = json.loads(json.dumps(self.preset))
        self.statusBar().showMessage("Preset configurations copied to memory cache buffer.")

    def _paste_edit_settings(self):
        """Pastes the preset settings from the internal clipboard buffer to the active image workspace."""
        focused = self.focusWidget()
        if isinstance(focused, QLineEdit):
            focused.paste()
            return
            
        if self.copied_settings_buffer is None:
            self.statusBar().showMessage("Paste blocked: Memory configuration clipboard is completely empty.")
            return
            
        self.preset = json.loads(json.dumps(self.copied_settings_buffer))
        self._apply_preset_to_ui()
        self._save_current_edits_to_session_cache()
        self.statusBar().showMessage("Preset configurations injected onto active workspace matrix.")
    
    def _apply_film_profile(self, profile_name):
        if profile_name is None:
            self.preset['rgb_curves'] = PhotoEditor.DEFAULT_PRESET['rgb_curves']
            self.preset.setdefault('color_matrix', PhotoEditor.DEFAULT_PRESET['color_matrix'])
        else:
            self.preset.setdefault('rgb_curves', PhotoEditor.DEFAULT_PRESET['rgb_curves'])
            for color in 'rgb':
                curve_values = FILM_PROFILES[profile_name][color]
                self.preset['rgb_curves'][color][1] = curve_values
            self.preset.setdefault('color_matrix', PhotoEditor.DEFAULT_PRESET['color_matrix'])
            self.preset['color_matrix'] = FILM_PROFILES[profile_name]['color_matrix']
        
        self._apply_preset_to_ui()
        self._save_current_edits_to_session_cache()

    def _load_file_status_registry(self):
        """Parses local tracking data configurations to update view states."""
        if os.path.exists(self.registry_json_path):
            logger.info(f"Loading file status registry from {self.registry_json_path}")
            try:
                with open(self.registry_json_path, "r") as f:
                    data = json.load(f)
                    self.starred_files = set(data.get("starred_paths", []))
                    self.exported_files = set(data.get("exported_paths", []))
                    self.marked_files = set(data.get("marked_paths", []))
            except Exception:
                pass

        # if os.path.exists(self.manifest_json_folder):
        #     try:
        #         # TODO: now that we don't have the full path of the edited file stored, we have to edit this
        #         # We can store the filepath in the json data, but then we'd have to open each
        #         for file_path in os.listdir(self.manifest_json_folder):
        #         with open(self.manifest_json_path, "r", encoding="utf-8") as f:
        #             data = json.load(f)
        #             for recorded_path in data.keys():
        #                 if os.path.exists(recorded_path):
        #                     self.edited_files.add(recorded_path)
        #     except Exception:
        #         pass

    def _save_file_status_registry(self):
        """Serializes current priority list indices directly out to local disk paths."""
        try:
            with open(self.registry_json_path, "w") as f:
                json.dump({
                    "starred_paths": list(self.starred_files),
                    "exported_paths": list(self.exported_files),
                    "marked_paths": list(self.marked_files)
                }, f, indent=2)
        except Exception:
            pass

    def _get_initial_browsing_directory(self) -> str:
        """Retrieves path history variables to map initial system browsing locations.

        Returns:
            str: Absolute folder location string directory path.
        """
        config = configparser.ConfigParser()
        if os.path.exists(self.state_ini_path):
            try:
                config.read(self.state_ini_path)
                saved_path = config.get("Session", "last_root_path", fallback="")
                if saved_path and os.path.exists(saved_path):
                    self.current_selected_folder_path = saved_path
                    return saved_path
            except Exception:
                pass
        fallback = QDir.homePath() + "/Pictures"
        self.current_selected_folder_path = fallback
        return fallback

    def _save_browsing_directory_state(self, target_path: str):
        """Updates internal folder path reference hooks inside configuration files.

        Args:
            target_path (str): Valid system directory folder address location.
        """
        self.current_selected_folder_path = target_path
        config = configparser.ConfigParser()
        config["Session"] = {"last_root_path": target_path}
        try:
            with open(self.state_ini_path, "w") as f:
                config.write(f)
        except Exception:
            pass

    def _init_ui_layout(self):
        """Constructs interface columns, sub-panel trees, views, and control sliders."""
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        left_splitter = QSplitter(Qt.Orientation.Vertical)

        self.file_model = CustomFileSystemModel(self)
        self.file_model.setRootPath(QDir.rootPath())
        self.file_model.setNameFilters([f"*{ext}" for ext in SUPPORTED_EXTENSIONS])
        self.file_model.setNameFilterDisables(False)

        initial_dir = self._get_initial_browsing_directory()

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(initial_dir))
        self.tree_view.setColumnHidden(1, True)
        self.tree_view.setColumnHidden(2, True)
        self.tree_view.setColumnHidden(3, True)
        self.tree_view.selectionModel().currentChanged.connect(self._on_tree_selection_changed)
        
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._show_tree_context_menu)
        
        left_splitter.addWidget(self.tree_view)

        self.left_preview_box = QGroupBox("Fast View Selection Thumbnail")
        preview_box_layout = QVBoxLayout(self.left_preview_box)
        preview_box_layout.setContentsMargins(4, 4, 4, 4)
        
        self.stationary_preview_lbl = QLabel("No Highlighted Asset Selected")
        self.stationary_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stationary_preview_lbl.setStyleSheet("background-color: #1a1a1a; border: 1px solid #3a3a3a; color: #777;")
        self.stationary_preview_lbl.setMinimumHeight(240)
        preview_box_layout.addWidget(self.stationary_preview_lbl)
        
        left_splitter.addWidget(self.left_preview_box)

        self.info_panel = QGroupBox("Active Metadata Profiles")
        vbox_info = QVBoxLayout(self.info_panel)
        vbox_info.setSpacing(4)
        
        self.lbl_info_name = QLabel("Name: None Loaded")
        self.lbl_info_res = QLabel("Original Resolution: -")
        self.lbl_info_date = QLabel("Date Modified: -")
        self.lbl_info_size = QLabel("File Disk Size: -")
        
        for widget in [self.lbl_info_name, self.lbl_info_res, self.lbl_info_date, self.lbl_info_size]:
            widget.setWordWrap(True)
            vbox_info.addWidget(widget)
        vbox_info.addStretch()
        
        left_splitter.addWidget(self.info_panel)
        left_splitter.setSizes([400, 260, 160])
        main_splitter.addWidget(left_splitter)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        center_layout.addWidget(self.view)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)
        self._current_view_size = (0, 0)

        control_row = QHBoxLayout()
        self.toggle_btn = QPushButton("Toggle View: Original / Edited [Hold '\\' Key]")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._on_toggle_view_clicked)
        control_row.addWidget(self.toggle_btn)
        center_layout.addLayout(control_row)
        main_splitter.addWidget(center_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        right_panel = QWidget()
        self.sliders_layout = QVBoxLayout(right_panel)

        self.histogram_widget = HistogramWidget()
        self.sliders_layout.addWidget(self.histogram_widget)

        self._build_sliders_interface()

        scroll_area.setWidget(right_panel)
        main_splitter.addWidget(scroll_area)
        main_splitter.setSizes([300, 800, 380])

    def _show_tree_context_menu(self, position):
        """Displays a context menu for starring photos within the active tree view."""
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return
            
        path = self.file_model.filePath(index)
        if os.path.isdir(path):
            return

        context_menu = QMenu(self)
        is_starred = path in self.starred_files
        
        
        star_action = QAction("Remove Star" if is_starred else "Star Photo ⭐", self)
        star_action.triggered.connect(lambda: self._toggle_file_star(path))
        context_menu.addAction(star_action)

        is_marked = path in self.marked_files
        mark_action = QAction("Remove Mark" if is_marked else "Mark Photo ✅", self)
        mark_action.triggered.connect(lambda: self._toggle_file_marked(path))
        context_menu.addAction(mark_action)
        
        context_menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def _toggle_file_star(self, path: str):
        """Toggles the favorite/star status of a specific file path."""
        if path in self.starred_files:
            self.starred_files.remove(path)
            logger.info(f"Removed asset file star tracker registry marker: {path}")
        else:
            self.starred_files.add(path)
            logger.info(f"Registered star priority flag for active node selection: {path}")
            
        self._save_file_status_registry()
        self.file_model.layoutChanged.emit()

    def _toggle_file_marked(self, path: str):
        if path in self.marked_files:
            self.marked_files.remove(path)
            logger.info(f"Removed asset file marked tracker registry marker: {path}")
        else:
            self.marked_files.add(path)
            logger.info(f"Registered marked priority flag for active node selection: {path}")
            
        self._save_file_status_registry()
        self.file_model.layoutChanged.emit()

    def _toggle_aspect_ratio_flip(self):
        """Inverts the active bounding box aspect ratio horizontally vs vertically."""
        is_flipped = self.active_crop_data.get("crop_aspect_ratio_flipped", False)
        self.active_crop_data["crop_aspect_ratio_flipped"] = not is_flipped
        if not self.is_updating_ui:
            self._update_target_resolution_label()
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()
    
    def _toggle_browse_mode(self):
        """Toggles between a state where the center file loads or not"""
        logger.info(f"Toggling Browse Mode to {int(not self.is_in_browse_mode)}")
        self.is_in_browse_mode = not self.is_in_browse_mode
        self._refresh_viewport()
    
    def _toggle_export_variants(self):
        """Toggles if all variants of the photo will be exported"""
        logger.info(f"Toggling export all variants to {not self.is_export_all_variants_set}")
        self.is_export_all_variants_set = not self.is_export_all_variants_set

    def _apply_auto_edits(self):
        '''Applies the auto edit calculation to the image'''
        logger.info(f'Applying auto edits to {self.current_file_path}')
        auto_edit_results = PhotoEditor.calculate_auto_preset(self.preview_matrix)

        _prev_active_crop_variant = self.preset.get('active_crop_variant', 'default')
        _prev_crop_variants = self.preset.get('crop_variants', PhotoEditor.DEFAULT_PRESET['crop_variants'])

        self.preset = PhotoEditor.DEFAULT_PRESET.copy()

        for key, value in auto_edit_results.items():
            if isinstance(value, dict):
                for _k, _v in value.items():
                    self.preset[key][_k] = _v
            else:
                self.preset[key] = value
        
        # restore previous crop data
        self.preset['active_crop_variant'] = _prev_active_crop_variant
        self.preset['crop_variants'] = _prev_crop_variants

        self._apply_preset_to_ui()
        self._save_current_edits_to_session_cache()
    
    def _apply_auto_edits_to_folder(self):
        folder_path = self.file_model.filePath(self.tree_view.rootIndex())
        folder_name = os.path.basename(folder_path)

        # display a warning
        confirm = QMessageBox.question(self, "Apply Auto Edits to Folder", 
                                       "Are you sure you want to apply auto edits to all files in the folder?\n" \
                                       f"This will overwrite any existing edits to files in \"{folder_name}\"",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if confirm != QMessageBox.StandardButton.Yes:
            return

        logger.info(f'Apply auto edits to all files in {folder_path}')


        files_in_folder = os.listdir(folder_path)
        num_files = len(files_in_folder)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        progress_dialog = QProgressDialog("Apply Auto Edits to Folder", None, 0, num_files, self)
        progress_dialog.setWindowTitle(f"Applying Auto Edits to all files in folder: \"{folder_name}\"")
        # progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumWidth(550)
        progress_dialog.setValue(0)

        try:
            progress_dialog.show()

            for i, file_name in enumerate(files_in_folder):
                QApplication.processEvents()
                progress_dialog.setValue(i)
                progress_dialog.setLabelText(file_name)

                file_path = os.path.join(folder_path, file_name)
                file_path = os.path.normpath(file_path).replace('\\', '/')

                if not os.path.isfile(file_path) or not file_name.endswith(SUPPORTED_EXTENSIONS):
                    continue
                
                logger.info(f'Applying auto edits to {file_name}')

                # auto-edits is going to scale down to 512 anyway.
                if file_path in self.image_matrix_cache:
                    preview_matrix = self.image_matrix_cache[file_path]['preview_matrix']
                else:
                    preview_matrix = PhotoEditor.load_image_matrix(file_path, preview=True, max_width=512)
                    self.image_matrix_cache[file_path]['preview_matrix'] = preview_matrix

                auto_edit_results = PhotoEditor.calculate_auto_preset(preview_matrix)

                preset = PhotoEditor.DEFAULT_PRESET.copy()

                for key, value in auto_edit_results.items():
                    if isinstance(value, dict):
                        for _k, _v in value.items():
                            preset[key][_k] = _v
                    else:
                        preset[key] = value
                
                self._write_preset_to_manifest(file_path, preset)
        finally:
            progress_dialog.setValue(progress_dialog.maximum())
            QApplication.restoreOverrideCursor()
        
    
    def _on_tree_selection_changed(self, current: QModelIndex, previous: QModelIndex):
        """Intercepts selection adjustments in the left side tree view pane.

        Args:
            current (QModelIndex): New data node slot targeted by focus shifts.
            previous (QModelIndex): Old abandoned layout cell coordinate index.
        """
        if not current.isValid():
            return
        path = self.file_model.filePath(current)
        if os.path.isdir(path):
            return
        
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return
        
        self.crop_data_cache = dict()
            
        self.preview_target_path = path
        self.preview_render_timer.start(80)
        self._on_file_selected(current)

    def _render_stationary_pane(self):
        """Extracts and scales image data to fit the lower left-hand column preview thumbnail."""
        # TODO: when we set a folder root, we can calculate all these thumbnails in a background thread.
        # Then when we switch we can just load from a cache.
        # TODO: separate this into a static method for generating the thumbnail, then this checks the cache.
        if not self.preview_target_path or not os.path.exists(self.preview_target_path):
            return
        try:
            ext = os.path.splitext(self.preview_target_path)[1].lower()
            if ext in [e.lower() for e in RAW_EXTENSIONS]:
                if not HAS_RAWPY_GUI:
                    self.stationary_preview_lbl.setText("Fast View Disabled:\nMissing rawpy environment")
                    return
                with rawpy.imread(self.preview_target_path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8)
                    
                    # Convert to QImage mapping array safely matching dimensions
                    h, w = rgb.shape[:2]
                    scale = 240.0 / max(w, h)
                    target_w, target_h = int(w * scale), int(h * scale)
                    img_small = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    img_small = np.ascontiguousarray(img_small)
            else:
                # Pillow cleanly handles EXIF orientations and natively supports HEIC configurations
                with Image.open(self.preview_target_path) as img:
                    img = ImageOps.exif_transpose(img)
                    img.thumbnail((240, 240))
                    img_small = np.ascontiguousarray(img.convert('RGB'))
                
            sh, sw, sc = img_small.shape
            q_img = QImage(img_small.data, sw, sh, sc * sw, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.stationary_preview_lbl.setPixmap(pixmap)
        except Exception as ex:
            self.stationary_preview_lbl.setText(f"Fast View Disabled:\n{os.path.basename(self.preview_target_path)}")
            logger.debug(f"Bypassed lazy stationary preview generation step logic pass: {ex}")

    def _build_sliders_interface(self):
        """Assembles interactive group layers strictly mapped to structural criteria."""
        g_preset = CollapsibleGroupBox("Preset Configuration Library")
        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_combo_changed)
        g_preset.content_layout.addWidget(self.preset_combo)
        
        self.cb_instagram = QCheckBox("Do Instagram Compression (Bypass manual scaling restrictions)")
        self.cb_instagram.setChecked(True)
        self.cb_instagram.toggled.connect(lambda state: self._update_preset_key("do_instagram_compression", state))
        g_preset.content_layout.addWidget(self.cb_instagram)

        self.sliders_map["resolution_percentage"] = self._create_slider_row(
            g_preset.content_layout, "Export Scale (%)", 10, 100, 100, 
            lambda v: self._update_preset_key("resolution_percentage", v)
        )
        self.lbl_live_target_res = QLabel("Target Export Res: -")
        self.lbl_live_target_res.setStyleSheet("font-weight: bold; color: #a2a2a2;")
        g_preset.content_layout.addWidget(self.lbl_live_target_res)
        self.sliders_layout.addWidget(g_preset)

        g_crop = CollapsibleGroupBox("Crop & Border Configuration")
        
        # Crop Variants
        self.crop_variant_combo = QComboBox()
        variant_hlayout = QHBoxLayout()
        self.crop_variant_combo.addItem("default")
        self.crop_variant_combo.currentTextChanged.connect(lambda txt: self._on_crop_variant_changed(txt))
        variant_hlayout.addWidget(QLabel("Crop Variant Selection"))
        variant_hlayout.addWidget(self.crop_variant_combo)

        self.add_crop_variant_btn = QPushButton("+")
        self.add_crop_variant_btn.setMaximumWidth(self.add_crop_variant_btn.sizeHint().height())
        self.add_crop_variant_btn.clicked.connect(self._add_crop_variant)
        self.remove_crop_variant_btn = QPushButton("-")
        self.remove_crop_variant_btn.clicked.connect(self._remove_crop_variant)
        self.remove_crop_variant_btn.setMaximumWidth(self.remove_crop_variant_btn.sizeHint().height())
        variant_hlayout.addWidget(self.add_crop_variant_btn)
        variant_hlayout.addWidget(self.remove_crop_variant_btn)
        g_crop.content_layout.addLayout(variant_hlayout)

        self.crop_ratio_combo = QComboBox()
        self.crop_ratio_combo.addItems(["Free", "Original", "1:1", "4:5", "5:7", "8:10", "16:9"])
        self.crop_ratio_combo.currentTextChanged.connect(lambda txt: self._update_preset_key("crop_aspect_ratio", txt))
        g_crop.content_layout.addWidget(QLabel("Aspect Ratio Lock Selection:"))
        g_crop.content_layout.addWidget(self.crop_ratio_combo)

        self.flip_ratio_btn = QPushButton("🔄 Flip Aspect Ratio Orientation")
        self.flip_ratio_btn.clicked.connect(self._toggle_aspect_ratio_flip)
        g_crop.content_layout.addWidget(self.flip_ratio_btn)

        self.sliders_map["crop_rotation"] = self._create_slider_row(g_crop.content_layout, "Crop Rotation (°)", -45, 45, 0, lambda v: self._update_preset_key("crop_rotation", v))
        self.sliders_map["crop_size"] = self._create_slider_row(g_crop.content_layout, "Crop Box Size (%)", 10, 100, 100, lambda v: self._update_preset_key("crop_size", v))
        self.sliders_map["crop_free_width"] = self._create_slider_row(g_crop.content_layout, "Crop Free Width (%)", 10, 100, 100, lambda v: self._update_preset_key("crop_free_width", v))
        self.sliders_map["crop_free_height"] = self._create_slider_row(g_crop.content_layout, "Crop Free Height (%)", 10, 100, 100, lambda v: self._update_preset_key("crop_free_height", v))
        self.sliders_map["crop_center_x"] = self._create_slider_row(g_crop.content_layout, "Crop Center X (%)", 0, 100, 50, lambda v: self._update_preset_key("crop_center_x", v))
        self.sliders_map["crop_center_y"] = self._create_slider_row(g_crop.content_layout, "Crop Center Y (%)", 0, 100, 50, lambda v: self._update_preset_key("crop_center_y", v))
        
        g_crop.content_layout.addSpacing(10)
        self.cb_white_border = QCheckBox("Apply Clean White Print Frame")
        self.cb_white_border.toggled.connect(lambda state: self._update_preset_key("add_white_border", state))
        g_crop.content_layout.addWidget(self.cb_white_border)
        self.sliders_map["white_border_width_pct"] = self._create_slider_row(g_crop.content_layout, "Border Frame Width (%)", 1, 20, 5, lambda v: self._update_preset_key("white_border_width_pct", v))
        self.sliders_layout.addWidget(g_crop)

        g_wb = CollapsibleGroupBox("White Balance Properties")
        self.cb_apply_temp = QCheckBox("Apply Kelvin Temperature Vector Transformation")
        self.cb_apply_temp.setChecked(True)
        self.cb_apply_temp.toggled.connect(lambda state: self._update_preset_key("apply_temperature_adjustment", state))
        g_wb.content_layout.addWidget(self.cb_apply_temp)
        self.sliders_map["temp_kelvin"] = self._create_slider_row(g_wb.content_layout, "Temperature (Kelvin)", 2000, 12000, 6500, lambda v: self._update_preset_key("temp_kelvin", v))
        self.sliders_map["tint"] = self._create_slider_row(g_wb.content_layout, "Tint", -100, 100, 0, lambda v: self._update_preset_key("tint", v / 100.0))
        self.sliders_layout.addWidget(g_wb)

        g_val = CollapsibleGroupBox("Global Exposure & Values")
        self.sliders_map["values_multiplier"] = self._create_slider_row(g_val.content_layout, "Values Shift Multiplier", 0, 100, 100, lambda v: self._update_preset_key("values_multiplier", v / 100.0))
        self.sliders_map["exposure"] = self._create_slider_row(g_val.content_layout, "Exposure (Stops)", -200, 200, 0, lambda v: self._update_preset_key("exposure", v / 100.0))
        self.sliders_map["contrast"] = self._create_slider_row(g_val.content_layout, "Contrast", -100, 100, 0, lambda v: self._update_preset_key("contrast", v / 100.0))
        self.sliders_map["whites"] = self._create_slider_row(g_val.content_layout, "Whites", -100, 100, 0, lambda v: self._update_preset_key("whites", v / 100.0))
        self.sliders_map["blacks"] = self._create_slider_row(g_val.content_layout, "Blacks", -100, 100, 0, lambda v: self._update_preset_key("blacks", v / 100.0))
        self.sliders_map["highlights"] = self._create_slider_row(g_val.content_layout, "Highlights Recovery", -100, 100, 0, lambda v: self._update_preset_key("highlights", v / 100.0))
        self.sliders_map["shadows"] = self._create_slider_row(g_val.content_layout, "Shadows Optimization", -100, 100, 0, lambda v: self._update_preset_key("shadows", v / 100.0))
        self.sliders_map["hdr_compression"] = self._create_slider_row(g_val.content_layout, "HDR Compression Curve", 0, 100, 0, lambda v: self._update_preset_key("hdr_compression", v / 100.0))
        self.sliders_layout.addWidget(g_val)

        g_col = CollapsibleGroupBox("Global Color Engine")
        self.sliders_map["color_multiplier"] = self._create_slider_row(g_col.content_layout, "Global Color Multiplier", 0, 100, 100, lambda v: self._update_preset_key("color_multiplier", v / 100.0))
        self.sliders_map["vibrance"] = self._create_slider_row(g_col.content_layout, "Vibrance (Muted Weight)", -100, 100, 0, lambda v: self._update_preset_key("vibrance", v / 100.0))
        self.sliders_map["saturation"] = self._create_slider_row(g_col.content_layout, "Saturation (Linear Multiplier)", -100, 100, 0, lambda v: self._update_preset_key("saturation", v / 100.0))
        self.sliders_layout.addWidget(g_col)

        # Channel Selectors
        g_curves = CollapsibleGroupBox("RGB Curves")

        channel_layout = QHBoxLayout()
        self.btn_group = QButtonGroup(self)
        
        channels = [("Red", 'r'), ("Green", 'g'), ("Blue", 'b')]
        for label, code in channels:
            btn = QRadioButton(label)
            if code == 'r':
                btn.setChecked(True)
            channel_layout.addWidget(btn)
            self.btn_group.addButton(btn)
            # Route to changing the active curves channel
            btn.clicked.connect(lambda _, c=code: self.curve_editor.set_channel(c))
        
        g_curves.content_layout.addLayout(channel_layout)
        
        # Interactive Editor Canvas
        self.curve_editor = CurveEditorWidget()
        self.curve_editor.pointDragged.connect(self._update_nested_rgb_curves_preset)
        self.curve_editor.pointReleased.connect(self._update_nested_rgb_curves_preset)
        self.sliders_map['rgb_curves'] = self.curve_editor

        g_curves.content_layout.addWidget(self.curve_editor)

        self.sliders_layout.addWidget(g_curves)

        g_ca = CollapsibleGroupBox("Targeted Chromatic Bands")
        self.sliders_map["color_adjustments_multiplier"] = self._create_slider_row(g_ca.content_layout, "Target Bands Multiplier", 0, 100, 100, lambda v: self._update_preset_key("color_adjustments_multiplier", v / 100.0))
        
        active_color_bands = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
        for band in active_color_bands:
            g_band = QGroupBox(f"Band Channel: {band.upper()}")
            vbox_b = QVBoxLayout(g_band)
            self.sliders_map[f"{band}_hue"] = self._create_slider_row(vbox_b, "Hue Shift Orientation", -100, 100, 0, lambda val, b=band: self._update_nested_color_preset(b, "hue", val / 100.0))
            self.sliders_map[f"{band}_sat"] = self._create_slider_row(vbox_b, "Saturation Intensity", -100, 100, 0, lambda val, b=band: self._update_nested_color_preset(b, "sat", val / 100.0))
            g_ca.content_layout.addWidget(g_band)
        self.sliders_layout.addWidget(g_ca)


        g_freq = CollapsibleGroupBox("Local Frequency Adjustments")
        self.sliders_map["texture"] = self._create_slider_row(g_freq.content_layout, "Texture Definition", -100, 100, 0, lambda v: self._update_preset_key("texture", v / 100.0))
        self.sliders_map["clarity"] = self._create_slider_row(g_freq.content_layout, "Clarity Profile", -100, 100, 0, lambda v: self._update_preset_key("clarity", v / 100.0))
        self.sliders_map["gaussian_blur"] = self._create_slider_row(g_freq.content_layout, "Gaussian Spatial Blur Pass", 0, 50, 0, lambda v: self._update_preset_key("gaussian_blur", v / 10.0))
        self.sliders_layout.addWidget(g_freq)

        g_style = CollapsibleGroupBox("Basic Grain")
        self.sliders_map["grain"] = self._create_slider_row(g_style.content_layout, "Grain Density Strength", 0, 100, 0, lambda v: self._update_preset_key("grain", v / 100.0))
        self.sliders_map["grain_size"] = self._create_slider_row(g_style.content_layout, "Grain Cluster Size Scale", 1, 50, 10, lambda v: self._update_preset_key("grain_size", v / 10.0))
        self.sliders_layout.addWidget(g_style)

        g_film_sim = CollapsibleGroupBox('Film Simulation')
        # ==========================================
        # 2. Optical Bloom
        # ==========================================
        self.chk_bloom = QCheckBox("Enable Optical Bloom")
        self.chk_bloom.setChecked(PhotoEditor.DEFAULT_PRESET['enable_bloom'])
        self.chk_bloom.toggled.connect(lambda checked: self._update_preset_key('enable_bloom', checked))
        g_film_sim.content_layout.addWidget(self.chk_bloom)

        self.sliders_map['bloom_threshold'] = self._create_slider_row(
            g_film_sim.content_layout, 'Bloom Threshold', 0, 100, 
            int(PhotoEditor.DEFAULT_PRESET['bloom_threshold'] * 100), 
            lambda v: self._update_preset_key('bloom_threshold', v / 100.0)
        )

        self.sliders_map['bloom_radius'] = self._create_slider_row(
            g_film_sim.content_layout, 'Bloom Radius (px)', 1, 50, 
            int(PhotoEditor.DEFAULT_PRESET['bloom_radius']), 
            lambda v: self._update_preset_key('bloom_radius', float(v))
        )

        self.sliders_map['bloom_strength'] = self._create_slider_row(
            g_film_sim.content_layout, 'Bloom Strength', 0, 100, 
            int(PhotoEditor.DEFAULT_PRESET['bloom_strength'] * 100), 
            lambda v: self._update_preset_key('bloom_strength', v / 100.0)
        )


        # ==========================================
        # 3. Selective Halation
        # ==========================================
        self.chk_halation = QCheckBox("Enable Selective Halation")
        self.chk_halation.setChecked(PhotoEditor.DEFAULT_PRESET['enable_halation'])
        self.chk_halation.toggled.connect(lambda checked: self._update_preset_key('enable_halation', checked))
        g_film_sim.content_layout.addWidget(self.chk_halation)

        self.sliders_map['halation_threshold'] = self._create_slider_row(
            g_film_sim.content_layout, 'Halation Threshold', 0, 100, 
            int(PhotoEditor.DEFAULT_PRESET['halation_threshold'] * 100), 
            lambda v: self._update_preset_key('halation_threshold', v / 100.0)
        )

        self.sliders_map['halation_radius'] = self._create_slider_row(
            g_film_sim.content_layout, 'Halation Radius (px)', 1, 50, 
            int(PhotoEditor.DEFAULT_PRESET['halation_radius']), 
            lambda v: self._update_preset_key('halation_radius', float(v))
        )

        self.sliders_map['halation_strength'] = self._create_slider_row(
            g_film_sim.content_layout, 'Halation Strength', 0, 100, 
            int(PhotoEditor.DEFAULT_PRESET['halation_strength'] * 100), 
            lambda v: self._update_preset_key('halation_strength', v / 100.0)
        )

        # Using -50 to 50 divided by 10.0 allows sub-pixel offsets from -5.0px to +5.0px
        self.sliders_map['halation_offset_x'] = self._create_slider_row(
            g_film_sim.content_layout, 'Halation Offset X', -50, 50, 
            int(PhotoEditor.DEFAULT_PRESET['halation_offset_x'] * 10), 
            lambda v: self._update_preset_key('halation_offset_x', v / 10.0)
        )

        self.sliders_map['halation_offset_y'] = self._create_slider_row(
            g_film_sim.content_layout, 'Halation Offset Y', -50, 50, 
            int(PhotoEditor.DEFAULT_PRESET['halation_offset_y'] * 10), 
            lambda v: self._update_preset_key('halation_offset_y', v / 10.0)
        )


        # ==========================================
        # 4. Smart Grain
        # ==========================================
        self.chk_grain = QCheckBox("Enable Smart Grain")
        self.chk_grain.setChecked(PhotoEditor.DEFAULT_PRESET['enable_grain'])
        self.chk_grain.toggled.connect(lambda checked: self._update_preset_key('enable_grain', checked))
        g_film_sim.content_layout.addWidget(self.chk_grain)

        self.sliders_map['grain_strength'] = self._create_slider_row(
            g_film_sim.content_layout, 'Grain Strength', 0, 100, 
            int(PhotoEditor.DEFAULT_PRESET['grain_strength'] * 100), 
            lambda v: self._update_preset_key('grain_strength', v / 100.0)
        )

        # Scale 5 to 40 divided by 10.0 maps to crystal sizes from 0.5x to 4.0x
        self.sliders_map['grain_size'] = self._create_slider_row(
            g_film_sim.content_layout, 'Grain Roughness (Size)', 5, 40, 
            int(PhotoEditor.DEFAULT_PRESET['grain_size'] * 10), 
            lambda v: self._update_preset_key('grain_size', v / 10.0)
        )

        self.sliders_map['grain_chroma'] = self._create_slider_row(
            g_film_sim.content_layout, 'Grain Color (Chroma)', 0, 100, 
            int(PhotoEditor.DEFAULT_PRESET['grain_chroma'] * 100), 
            lambda v: self._update_preset_key('grain_chroma', v / 100.0)
        )


        # ==========================================
        # 5. Vignette & Optical Softness
        # ==========================================
        self.chk_vignette = QCheckBox("Enable Vignette & Softness")
        self.chk_vignette.setChecked(PhotoEditor.DEFAULT_PRESET['enable_vignette'])
        self.chk_vignette.toggled.connect(lambda checked: self._update_preset_key('enable_vignette', checked))
        g_film_sim.content_layout.addWidget(self.chk_vignette)

        self.sliders_map['vignette_strength'] = self._create_slider_row(
            g_film_sim.content_layout, 'Vignette Strength', 0, 100, 
            int(PhotoEditor.DEFAULT_PRESET['vignette_strength'] * 100), 
            lambda v: self._update_preset_key('vignette_strength', v / 100.0)
        )

        # Radius up to 150 allows aspect-ratio corrected vignetting to reach the extreme corners (~1.414)
        self.sliders_map['vignette_radius'] = self._create_slider_row(
            g_film_sim.content_layout, 'Vignette Radius', 0, 150, 
            int(PhotoEditor.DEFAULT_PRESET['vignette_radius'] * 100), 
            lambda v: self._update_preset_key('vignette_radius', v / 100.0)
        )

        self.sliders_map['vignette_softness'] = self._create_slider_row(
            g_film_sim.content_layout, 'Vignette Feather (Softness)', 1, 100, 
            int(PhotoEditor.DEFAULT_PRESET['vignette_softness'] * 100), 
            lambda v: self._update_preset_key('vignette_softness', v / 100.0)
        )

        self.sliders_map['corner_blur_radius'] = self._create_slider_row(
            g_film_sim.content_layout, 'Corner Optical Blur (px)', 0, 30, 
            int(PhotoEditor.DEFAULT_PRESET['corner_blur_radius']), 
            lambda v: self._update_preset_key('corner_blur_radius', float(v)))

        self.sliders_layout.addWidget(g_film_sim)

    def _create_slider_row(self, parent_layout, label_text: str, mn: int, mx: int, init: int, callback) -> QSlider:
        """Helper method assembling standard labeled rows containing interactive slider objects.

        Args:
            parent_layout (QVBoxLayout): Target framework layout sheet layer hook.
            label_text (str): Slider descriptive identification title text string.
            mn (int): Minimum logical integer range limit value.
            mx (int): Maximum logical integer range limit value.
            init (int): Initial tracking fallback value setting coordinates.
            callback (function): Triggers code blocks cleanly on slider updates.

        Returns:
            QSlider: Constructed interactive adjustment layout widget element.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 2, 0, 2)

        lbl = QLabel(label_text)
        lbl.setMinimumWidth(130)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(init)
        slider.wheelEvent = lambda event: event.ignore()

        value_edit = QLineEdit(str(init))
        value_edit.setFixedWidth(50)
        value_edit.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # 1. State handlers for mouse press/release on the slider handle
        def on_slider_pressed():
            self.is_scrubbing = True

        def on_slider_released():
            self.is_scrubbing = False
            # Force a final high-resolution evaluation at the final resting value
            if not self.is_updating_ui:
                callback(slider.value())

        def on_slider_changed(val):
            value_edit.setText(str(val))
            if not self.is_updating_ui:
                # When dragging, this executes while self.is_scrubbing == True.
                # When using keyboard arrows or clicking the track, self.is_scrubbing == False,
                # which naturally evaluates at full resolution immediately.
                callback(val)

        def on_text_confirmed():
            if self.is_updating_ui: return
            try:
                val = int(value_edit.text())
                val = max(mn, min(mx, val))
                
                # Text inputs should always evaluate at full resolution
                self.is_scrubbing = False 
                self.is_updating_ui = True
                slider.setValue(val)
                self.is_updating_ui = False
                
                value_edit.setText(str(val))
                callback(val)
            except ValueError:
                value_edit.setText(str(slider.value()))

        # 2. Connect the press and release signals
        slider.sliderPressed.connect(on_slider_pressed)
        slider.sliderReleased.connect(on_slider_released)
        slider.valueChanged.connect(on_slider_changed)
        value_edit.editingFinished.connect(on_text_confirmed)
        
        layout.addWidget(lbl)
        layout.addWidget(slider)
        layout.addWidget(value_edit)
        parent_layout.addWidget(row)
        return slider

    def keyPressEvent(self, event: QKeyEvent):
        """Overrides viewport standard listeners to evaluate direct input overrides.

        Args:
            event (QKeyEvent): System input signal parameter descriptors data packet hooks.
        """
        if event.key() == Qt.Key.Key_Backslash and not event.isAutoRepeat():
            self.show_original_state = True
            self.toggle_btn.setChecked(True)
            self._refresh_viewport()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Restores baseline matrix parameters on explicit keyboard release patterns.

        Args:
            event (QKeyEvent): System key signal code blocks wrapper object layers.
        """
        if event.key() == Qt.Key.Key_Backslash and not event.isAutoRepeat():
            self.show_original_state = False
            self.toggle_btn.setChecked(False)
            self._refresh_viewport()
        super().keyReleaseEvent(event)

    def _populate_local_presets(self):
        """Scans the local execution layout directory to map matching profile configurations."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select a workspace preset...")
        try:
            files = [f for f in os.listdir(SCRIPT_DIR) if f.lower().endswith(".json")]
            for f in sorted(files):
                self.preset_combo.addItem(f)
        except Exception as e:
            logger.error(f"Local preset indexing loop failure alert: {e}")
        self.preset_combo.blockSignals(False)

    def _on_preset_combo_changed(self, index: int):
        """Maps parameter configuration values out on user selection modifications.

        Args:
            index (int): Dropdown combo menu choice data coordinates slot selection integer.
        """
        if index <= 0: return
        filename = self.preset_combo.itemText(index)
        self._load_preset_from_path(os.path.join(SCRIPT_DIR, filename))

    def _load_preset_from_path(self, path: str):
        """Loads and unpacks the target JSON preset file values onto the active grid.

        Args:
            path (str): The absolute layout string path destination location target file.
        """
        try:
            with open(path, "r") as f:
                loaded_preset = json.load(f)

            sanitized_preset = json.loads(json.dumps(self.default_preset))
            for k, v in loaded_preset.items():
                if k == "color_adjustments" and isinstance(v, dict):
                    for band, values in v.items():
                        if band in sanitized_preset["color_adjustments"] and isinstance(values, dict):
                            sanitized_preset["color_adjustments"][band].update(values)
                else:
                    sanitized_preset[k] = v

            self.preset = sanitized_preset
            self._apply_preset_to_ui()
            self._save_current_edits_to_session_cache()
            self.statusBar().showMessage(f"Loaded configuration matrix preset: {os.path.basename(path)}")
        except Exception as e:
            logger.error(f"Failed to cleanly unpack workspace layout properties: {e}")

    def _apply_preset_to_ui(self):
        """Maps parameters sequentially across GUI element items, blocking nested validation loops."""
        # When is_updating_ui is True, we won't apply the preset 100's of times (signals are essentially disconnected)
        self.is_updating_ui = True

        # Add all crop variants
        self.crop_variant_combo.currentTextChanged.disconnect()
        self.crop_variant_combo.clear()
        for crop_variant_name in self.preset.get('crop_variants'):
            self.crop_variant_combo.addItem(crop_variant_name)
        self.crop_variant_combo.setCurrentText(self.active_crop_variant)
        self.crop_variant_combo.currentTextChanged.connect(self._on_crop_variant_changed)

        self.cb_instagram.setChecked(self.active_crop_data.get("do_instagram_compression", True))
        self.cb_apply_temp.setChecked(self.preset.get("apply_temperature_adjustment", True))

        self.crop_ratio_combo.setCurrentText(self.active_crop_data.get("crop_aspect_ratio", "Free"))
        self.cb_white_border.setChecked(self.active_crop_data.get("add_white_border", False))

        self.sliders_map["resolution_percentage"].setValue(int(self.active_crop_data.get("resolution_percentage", 100)))
        self.sliders_map["crop_rotation"].setValue(int(self.active_crop_data.get("crop_rotation", 0)))
        self.sliders_map["crop_size"].setValue(int(self.active_crop_data.get("crop_size", 100)))
        self.sliders_map["crop_free_width"].setValue(int(self.active_crop_data.get("crop_free_width", 100)))
        self.sliders_map["crop_free_height"].setValue(int(self.active_crop_data.get("crop_free_height", 100)))
        self.sliders_map["crop_center_x"].setValue(int(self.active_crop_data.get("crop_center_x", 50)))
        self.sliders_map["crop_center_y"].setValue(int(self.active_crop_data.get("crop_center_y", 50)))
        self.sliders_map["white_border_width_pct"].setValue(int(self.active_crop_data.get("white_border_width_pct", 5)))

        self.sliders_map["values_multiplier"].setValue(int(self.preset.get("values_multiplier", 1.0) * 100))
        self.sliders_map["color_multiplier"].setValue(int(self.preset.get("color_multiplier", 1.0) * 100))
        self.sliders_map["color_adjustments_multiplier"].setValue(int(self.preset.get("color_adjustments_multiplier", 1.0) * 100))

        self.sliders_map["exposure"].setValue(int(self.preset.get("exposure", 0.0) * 100))
        self.sliders_map["contrast"].setValue(int(self.preset.get("contrast", 0.0) * 100))
        self.sliders_map["whites"].setValue(int(self.preset.get("whites", 0.0) * 100))
        self.sliders_map["blacks"].setValue(int(self.preset.get("blacks", 0.0) * 100))
        self.sliders_map["highlights"].setValue(int(self.preset.get("highlights", 0.0) * 100))
        self.sliders_map["shadows"].setValue(int(self.preset.get("shadows", 0.0) * 100))
        self.sliders_map["hdr_compression"].setValue(int(self.preset.get("hdr_compression", 0.0) * 100))

        self.sliders_map["temp_kelvin"].setValue(int(self.preset.get("temp_kelvin", 6500)))
        self.sliders_map["tint"].setValue(int(self.preset.get("tint", 0.0) * 100))

        self.sliders_map["texture"].setValue(int(self.preset.get("texture", 0.0) * 100))
        self.sliders_map["clarity"].setValue(int(self.preset.get("clarity", 0.0) * 100))
        self.sliders_map["gaussian_blur"].setValue(int(self.preset.get("gaussian_blur", 0.0) * 10))
        self.sliders_map["vibrance"].setValue(int(self.preset.get("vibrance", 0.0) * 100))
        self.sliders_map["saturation"].setValue(int(self.preset.get("saturation", 0.0) * 100))

        # rgb curves
        
        rgb_curves = self.preset.get('rgb_curves')
        if rgb_curves:
            self.sliders_map['rgb_curves'].set_curve_values(rgb_curves['r'][1], rgb_curves['g'][1], rgb_curves['b'][1])
        else:
            self.sliders_map['rgb_curves'].set_curve_values(self.default_curve_vals, self.default_curve_vals, self.default_curve_vals)

        color_adj = self.preset.get("color_adjustments", {})
        active_color_bands = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
        for band in active_color_bands:
            band_data = color_adj.get(band, {"hue": 0.0, "sat": 0.0})
            self.sliders_map[f"{band}_hue"].setValue(int(band_data.get("hue", 0.0) * 100))
            self.sliders_map[f"{band}_sat"].setValue(int(band_data.get("sat", 0.0) * 100))

        self.sliders_map["grain"].setValue(int(self.preset.get("grain", 0.0) * 100))
        self.sliders_map["grain_size"].setValue(int(self.preset.get("grain_size", 1.0) * 10))

        self.is_updating_ui = False

        self._update_undo_history_for_file(self.current_file_path)
        self._update_target_resolution_label()
        self._refresh_viewport()
    
    def _update_undo_history_for_file(self, file_path):
        self.history_registry.setdefault(file_path, FileEditHistory(self._read_preset_from_manifest(self.current_file_path)))
        self.history_registry[file_path].push_state(self.preset)

    def _update_target_resolution_label(self):
        """Calculates locked pixel bounds based on current crop selection modes and outputs data metrics."""
        if not self.current_file_path:
            self.lbl_live_target_res.setText("Target Export Res: -")
            return
            
        try:
            ext = os.path.splitext(self.current_file_path)[1].lower()
            if ext in [e.lower() for e in RAW_EXTENSIONS]:
                if not HAS_RAWPY_GUI:
                    self.lbl_live_target_res.setText("Target Export Res: [Missing rawpy]")
                    return
                with rawpy.imread(self.current_file_path) as raw:
                    w, h = raw.sizes.width, raw.sizes.height
            else:
                with Image.open(self.current_file_path) as img:
                    img = ImageOps.exif_transpose(img)
                    w, h = img.size
        except Exception:
            self.lbl_live_target_res.setText("Target Export Res: -")
            return
            
        ratio_mode = self.active_crop_data.get("crop_aspect_ratio", "Free")
        if ratio_mode == "Free":
            box_w = int(max(10, min(100, self.preset.get("crop_free_width", 100))) / 100.0 * w)
            box_h = int(max(10, min(100, self.preset.get("crop_free_height", 100))) / 100.0 * h)
            if self.active_crop_data.get("crop_aspect_ratio_flipped", False):
                box_w, box_h = box_h, box_w
        else:
            if ratio_mode == "Original": target_ratio = w / h
            elif ratio_mode == "1:1": target_ratio = 1.0
            elif ratio_mode == "4:5": target_ratio = 5.0 / 4.0 if w >= h else 4.0 / 5.0
            elif ratio_mode == "5:7": target_ratio = 7.0 / 5.0 if w >= h else 5.0 / 7.0
            elif ratio_mode == "8:10": target_ratio = 10.0 / 8.0 if w >= h else 8.0 / 10.0
            elif ratio_mode == "16:9": target_ratio = 16.0 / 9.0 if w >= h else 9.0 / 16.0
            else: target_ratio = w / h
                
            if self.active_crop_data.get("crop_aspect_ratio_flipped", False):
                target_ratio = 1.0 / target_ratio
                
            if w / h >= target_ratio:
                max_h = h
                max_w = int(h * target_ratio)
            else:
                max_w = w
                max_h = int(w / target_ratio)
                
            size_scale = max(10, min(100, self.preset.get("crop_size", 100))) / 100.0
            box_w = int(max_w * size_scale)
            box_h = int(max_h * size_scale)

        if self.active_crop_data.get("do_instagram_compression", True):
            final_w = 1080
            final_h = int(final_w * (box_h / box_w))
        else:
            pct = self.preset.get("resolution_percentage", 100) / 100.0
            final_w = int(box_w * pct)
            final_h = int(box_h * pct)
            
        self.lbl_live_target_res.setText(f"Target Export Res: {final_w} x {final_h}")

    def _save_current_edits_to_session_cache(self):
        """Saves snapshotted editing parameter dictionaries to the centralized manifest storage registry."""
        if not self.current_file_path: return
        try:
            self._write_preset_to_manifest(self.current_file_path, self.preset)
            if self.current_file_path not in self.edited_files:
                self.edited_files.add(self.current_file_path)
                self.file_model.layoutChanged.emit()
        except Exception:
            pass
    
    # TODO: with crop variants, make way to add \ remove them in the ui and make sure their values save and are editable
    def _update_preset_key(self, key: str, value: Any):
        """Updates targeted tracking parameter metrics inside global configurations.

        Args:
            key (str): Key matching requested modification slots.
            value (Any): Payload setting values configuration attributes.
        """
        # TODO: if the key is part of the crop keys. This is making it more complex
        if key in self.active_crop_data:
            self._update_nested_crop_variant_preset(key, value)
        else:
            self.preset[key] = value
            if not self.is_updating_ui:
                self._update_target_resolution_label()
                self._save_current_edits_to_session_cache()
                self._refresh_viewport()

    def _update_nested_color_preset(self, band: str, prop: str, value: float):
        """Modifies targeted spectral value slots inside isolated sub-dictionaries.

        Args:
            band (str): Specific target frequency channel keyword identifier block.
            prop (str): Parameter type descriptor slot index key (hue / sat).
            value (float): Granular offset floating point scalar value setting.
        """
        self.preset["color_adjustments"][band][prop] = value
        if not self.is_updating_ui:
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()
    
    def _update_nested_crop_variant_preset(self, key: str, value: Any):
        self.preset['crop_variants'][self.active_crop_variant][key] = value
        if not self.is_updating_ui:
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()
    
    def _update_nested_rgb_curves_preset(self, key: str, index: int, value: float):
        self.preset.setdefault('rgb_curves', PhotoEditor.DEFAULT_PRESET['rgb_curves'])
        self.preset['rgb_curves'][key][1][index] = value
        if not self.is_updating_ui:
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()

    def _open_preset_file(self):
        """Launches localized system dialogue layout prompts to parse stored configuration files."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Preset JSON Configuration", "", "JSON Configurations (*.json)")
        if path:
            self._load_preset_from_path(path)
            combo_idx = self.preset_combo.findText(os.path.basename(path))
            self.preset_combo.setCurrentIndex(combo_idx if combo_idx >= 0 else 0)

    def _save_preset_file(self):
        """Launches disk write serialization interfaces to save preset JSON structures."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Current Edits As Preset JSON", "", "JSON Configurations (*.json)")
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.preset, f, indent=2)
                self.statusBar().showMessage(f"Saved configuration: {os.path.basename(path)}")
                self._populate_local_presets()
                combo_idx = self.preset_combo.findText(os.path.basename(path))
                if combo_idx >= 0: self.preset_combo.setCurrentIndex(combo_idx)
            except Exception as e:
                logger.error(f"Preset writing serialization failure: {e}")

    def _change_root_folder(self):
        """Launches standard system browser trees to reposition the project root entry focus."""
        start_dir = self.current_file_path if self.current_file_path else QDir.currentPath()
        if os.path.isfile(start_dir): start_dir = os.path.dirname(start_dir)
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Project Root Directory", start_dir, QFileDialog.Option.ShowDirsOnly)
        if selected_dir:
            self.tree_view.setRootIndex(self.file_model.index(selected_dir))
            self._save_browsing_directory_state(selected_dir)
            self.statusBar().showMessage(f"Tree viewport root path set to: {selected_dir}")
        
        # TODO: load the _proxy_ matrices only for the next 10 files
        # TODO: start a thread loading the thumbnails for all files

    def _on_file_selected(self, index: QModelIndex):
        """Loads data matrices from newly activated tree folder navigation index items.

        Args:
            index (QModelIndex): Target selection tracking index token layout parameters.
        """
        start = time.time()
        path = self.file_model.filePath(index)
        if os.path.isdir(path): return

        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS: return

        self.current_file_path = path
        logger.info(f"Target file workspace selected node swap focus: {path}")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # Invalidate the pipeline state! 
        # TODO: the pipeline state object has to be unique to the proxy or the preview matrix! 
        # We can't have just one! 
        self.pipeline_state_cache['proxy_pipeline_state'] = None
        self.pipeline_state_cache['preview_pipeline_state'] = None

        # init a new instance of PhotoEditor so we can cache edits
        self.photo_editor_instance = PhotoEditor(self.current_file_path)

        try:
            if self.is_in_browse_mode:
                logger.info("Exiting early because we are in browse mode")
                return
            
            # 1. Load ONLY the fast proxy synchronously on the UI thread
            stored_proxy_matrix = self.image_matrix_cache.get(self.current_file_path, dict()).get('proxy_matrix')
            if stored_proxy_matrix is not None:
                self.proxy_matrix = stored_proxy_matrix.copy()
            else:
                self.proxy_matrix = PhotoEditor.load_image_matrix(self.current_file_path, preview=True, max_width=512)
                self.image_matrix_cache.setdefault(self.current_file_path, dict())
                self.image_matrix_cache[self.current_file_path]['proxy_matrix'] = self.proxy_matrix.copy()
            
            # Temporarily point preview_matrix to the proxy so the UI renders immediately
            self.preview_matrix = self.proxy_matrix.copy()
            self.preset = self._read_preset_from_manifest(path)

            # TODO: we could cache this metadata
            # 2. Extract lightweight metadata (fast I/O only, avoid deep decoding)
            if ext in [e.lower() for e in RAW_EXTENSIONS]:
                if not HAS_RAWPY_GUI:
                    raise ImportError("Cannot parse targeted RAW file metadata because rawpy module is not defined.")
                with rawpy.imread(path) as raw:
                    orig_w, orig_h = raw.sizes.width, raw.sizes.height
            else:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    orig_w, orig_h = img.size
            
            self.lbl_info_name.setText(f"Name: {os.path.basename(path)}")
            self.lbl_info_res.setText(f"Original Resolution: {orig_w} x {orig_h}")
            
            mtime = os.path.getmtime(path)
            date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            self.lbl_info_date.setText(f"Date Modified: {date_str}")
            
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self.lbl_info_size.setText(f"File Disk Size: {size_mb:.2f} MB")
            
            # 3. Apply presets and render the proxy immediately
            self._apply_preset_to_ui()  # This is the longest time.
            self._zoom_to_fit()
            self._save_browsing_directory_state(os.path.dirname(path))

            # 4. Dispatch the background task for the full-resolution matrix
            stored_preview_matrix = self.image_matrix_cache.get(self.current_file_path, dict()).get('preview_matrix')
            if stored_preview_matrix is not None:
                print('using stored preview matrix')
                self.preview_matrix = stored_preview_matrix.copy()
                self._on_fullres_load_complete(self.current_file_path, self.preview_matrix)
            else:
                print('starting fullres background load of preview matrix')
                self._start_background_fullres_load(path)

            # TODO: start a thread loading the proxy matrices for the surrounding 10 files in the tree

        except Exception as e:
            logger.error(f"Critical load validation exception encountered: {e}")
            self.statusBar().showMessage(f"Load Failure: {os.path.basename(path)}")
            self.scene.clear()
            self.preview_matrix = None
            self.proxy_matrix = None
        finally:
            QApplication.restoreOverrideCursor()
            print(f'On file selected time: {time.time() - start}')

    def _start_background_fullres_load(self, path: str):
        """Spawns a background thread to load the full-res matrix without blocking the UI."""
        worker = FullResLoaderWorker(path)
        worker.signals.finished.connect(self._on_fullres_load_complete)
        worker.signals.error.connect(self._on_fullres_load_error)
        self.thread_pool.start(worker)

    def _on_fullres_load_complete(self, loaded_path: str, full_matrix: object):
        """Receives the full-resolution matrix from the background thread and updates the viewport."""
        # Race Condition Guard: If the user clicked another file while this was loading, discard it
        if self.current_file_path != loaded_path:
            logger.info(f"Discarding stale background matrix for: {loaded_path}")
            return

        logger.info(f"Background full-res load complete for: {loaded_path}")
        self.preview_matrix = full_matrix.copy()
        self.image_matrix_cache.setdefault(self.current_file_path, dict())
        self.image_matrix_cache[self.current_file_path]['preview_matrix'] = self.preview_matrix.copy()
        
        # Trigger a silent refresh of your viewport with the high-resolution data
        self._refresh_viewport()

    def _on_fullres_load_error(self, failed_path: str, error_msg: str):
        """Handles background loading exceptions."""
        if self.current_file_path != failed_path:
            return
        logger.error(f"Failed to load full-res matrix in background: {error_msg}")
        self.statusBar().showMessage(f"High-res preview failed to load: {os.path.basename(failed_path)}")

    def _on_toggle_view_clicked(self, checked: bool):
        """Swaps standard target calculation tracks to show comparison modes on button triggers.

        Args:
            checked (bool): Toggle button value confirmation parameter flag.
        """
        self.show_original_state = checked
        self._refresh_viewport()

    def _refresh_viewport(self):
        """
        Re-computes active filters to display raster data on the canvas.
        """
        if self.preview_matrix is None:
            return

        # Use a lower-res proxy matrix during active slider dragging
        if self.is_scrubbing:
            source_matrix = self.proxy_matrix
            pipeline_state = self.pipeline_state_cache.get('proxy_pipeline_state', PipelineState())
        else:
            source_matrix = self.preview_matrix
            pipeline_state = self.pipeline_state_cache.get('preview_pipeline_state', PipelineState())

        if self.show_original_state:
            render_array = source_matrix
        else:
            # TODO: we could probably cache the cropped preview if crop data hasn't changed.
            # TODO: we need to invalidate active crop data when we switch images
            # cropped_preview = None
            # if self.active_crop_data == self.crop_data_cache.get('crop_data'):
            #     if self.is_scrubbing:
            #         cropped_preview = self.crop_data_cache.get('cropped_proxy_matrix')
            #     else:
            #         cropped_preview = self.crop_data_cache.get('cropped_preview_matrix')
            
            # if cropped_preview is None:
            #     cropped_preview = PhotoEditor.apply_crop(source_matrix, self.active_crop_data)
            #     self.crop_data_cache['crop_data'] = self.active_crop_data.copy()

            #     if self.is_scrubbing:
            #         self.crop_data_cache['cropped_proxy_matrix'] = cropped_preview.copy()
            #     else:
            #         self.crop_data_cache['cropped_preview_matrix'] = cropped_preview.copy()
            cropped_preview = PhotoEditor.apply_crop(source_matrix, self.active_crop_data)

            render_array = self.photo_editor_instance.run_parallel_pipeline(cropped_preview, self.preset, pipeline_state, fast_preview=self.is_scrubbing)  # This takes 4 seconds

        # Trigger histogram asynchronously or on downsampled data to prevent UI blocks
        self.histogram_widget.render_histogram(render_array)

        # 2. Optimized conversion: reduce temporary array allocations
        # Ensure working in float32; use out= parameters if your pipeline allows
        img_uint8 = np.empty_like(render_array, dtype=np.uint8)
        np.clip(render_array * 255.0, 0, 255, out=img_uint8, casting="unsafe")

        # 3. Optimized Clipping Mask (evaluates faster than 6 separate channel arrays)
        if not self.show_original_state and self.highlight_clipped_action.isChecked():
            # Check for pure black (all channels 0) or pure white (all channels 255)
            mask_black = not np.any(img_uint8, axis=-1)
            mask_white = np.all(img_uint8 == 255, axis=-1)
            
            # Apply red highlight in-place
            img_uint8[mask_black | mask_white] = [255, 0, 0]

        if not self.show_original_state:
            img_uint8 = PhotoEditor.apply_white_border(img_uint8, self.preset)

        # 4. Zero-copy QImage creation (ensure img_uint8 is C-contiguous)
        if not img_uint8.flags['C_CONTIGUOUS']:
            img_uint8 = np.ascontiguousarray(img_uint8)
            
        h, w, ch = img_uint8.shape
        bytes_per_line = ch * w
        
        # Note: QImage does not own the memory; keep a reference if not converting immediately
        q_image = QImage(img_uint8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 5. Update persistent item instead of clearing scene
        self.pixmap_item.setPixmap(QPixmap.fromImage(q_image))  # This takes little time

    def _zoom_to_fit(self):
        self.view.zoom_to_fit()

    def _trigger_file_export(self):
        """Prepares single file queue requests to push into the multi-core background worker thread."""
        if not self.current_file_path:
            self.statusBar().showMessage("Export failure: No active file target data mapping available.")
            return
        
        self._save_current_edits_to_session_cache()

        suggested_dir = os.path.join(os.path.dirname(self.current_file_path), "edits")
        os.makedirs(suggested_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
        default_out = os.path.join(suggested_dir, f"{base_name}_edit.jpg")

        out_path, _ = QFileDialog.getSaveFileName(self, "Export Production Clean Photo Matrix Asset", default_out, "JPEG Image Map (*.jpg)")
        if not out_path: return

        queue = [(self.current_file_path, out_path, json.loads(json.dumps(self._read_preset_from_manifest(self.current_file_path))))]

        if self.is_export_all_variants_set:
            queue += self._get_export_variants(self.current_file_path, out_path, self.preset)

        self._execute_progressive_export_queue(queue)
    
    def _get_export_variants(self, current_file_path, out_path, preset):
        """Get the export variant data for the given file so we can add it to the queue"""

        logger.info(f"Exporting all jpeg and framing variants for {current_file_path}")

        result = []

        out_path_dir = os.path.dirname(out_path)
        out_path_orig_name, ext = os.path.splitext(os.path.basename(out_path))

        active_crop_variant = preset['active_crop_variant']

        # No frame and uncompressed
        for crop_variant_name, crop_data in preset['crop_variants'].items():
            preset_copy = json.loads(json.dumps(preset))

            # All we have to do to get it to export this is to change the active variant
            preset_copy['active_crop_variant'] = crop_variant_name
            variant_path = os.path.join(out_path_dir, out_path_orig_name + f'_v_crop_{crop_variant_name}' + ext)

            if crop_variant_name != active_crop_variant:
                # otherwise we'd export duplicates
                result.append((current_file_path, variant_path, preset_copy))
            
            # TODO: if it has a border or compression, also create a borderless version here
        
        return result

    def _trigger_batch_starred_export(self):
        """Assembles folder collection files tracking active star records into batch job blocks."""
        if not self.current_selected_folder_path or not os.path.exists(self.current_selected_folder_path):
            QMessageBox.warning(self, "Batch Blocked", "No valid working folder path context active.")
            return

        logger.info(f"Scanning target location folder for starred references: {self.current_selected_folder_path}")
        starred_in_folder = []
        starred_files = [f.replace('\\', '/') for f in self.starred_files]

        try:
            for item in os.listdir(self.current_selected_folder_path):
                full_item_path = os.path.join(self.current_selected_folder_path, item).replace('\\', '/')
                if os.path.isfile(full_item_path) and full_item_path in starred_files:
                    ext = os.path.splitext(full_item_path)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        starred_in_folder.append(full_item_path)
        except Exception as ex:
            logger.error(f"Failed to scan folder contents: {ex}")
            return

        if not starred_in_folder:
            QMessageBox.information(self, "Zero Starred Images Found", f"No starred photo markers match items inside:\n\"{os.path.basename(self.current_selected_folder_path)}\"")
            return

        output_dir = os.path.join(self.current_selected_folder_path, "starred_edits")
        os.makedirs(output_dir, exist_ok=True)

        queue = []
        for img_path in starred_in_folder:
            item_preset = self._read_preset_from_manifest(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_starred_edit.jpg")
            queue.append((img_path, out_path, item_preset))

            if self.is_export_all_variants_set:
                queue += self._get_export_variants(img_path, out_path, item_preset)

        logger.info(f"Queued batch collection assembly done. Count: {len(queue)}")
        self._execute_progressive_export_queue(queue)

    def _execute_progressive_export_queue(self, queue: List[tuple]):
        """Launches standard dual-level progressive loading panel indicators over background jobs.

        Args:
            queue (List[tuple]): Complete batch array matrix mapping collection parameters layer.
        """
        if self.export_thread and self.export_thread.isRunning():
            self.statusBar().showMessage("An active task execution queue sequence is already processing.")
            return

        self.export_action.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        self.progress_dialog = QProgressDialog("Initializing engine processing layers...", "Cancel Batch Operation", 0, len(queue), self)
        self.progress_dialog.setWindowTitle("Production High-Precision Rendering Pipeline")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumWidth(550)
        self.progress_dialog.setValue(0)
        
        self.export_thread = ExportWorker(queue, self.preset)
        self.progress_dialog.canceled.connect(self.export_thread.cancel)

        self.progress_dialog.show()
        self.export_thread.progress_signal.connect(self.progress_dialog.setLabelText)
        
        def handle_progress_step(done, _):
            self.progress_dialog.setValue(done)
            
        self.export_thread.file_done_signal.connect(handle_progress_step)
        self.export_thread.finished_signal.connect(self._on_progressive_queue_completed)
        self.export_thread.start()

    def _on_progressive_queue_completed(self, success: bool, message: str):
        """Closes dial bars cleanly and synchronizes listings upon operational sequence halts.

        Args:
            success (bool): Execution confirmation validation check.
            message (str): Summary tracking description text strings info logs.
        """
        QApplication.restoreOverrideCursor()
        self.export_action.setEnabled(True)
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            
        QMessageBox.information(self, "Export Session Complete Pass", message)
        self.statusBar().showMessage(message)
        
        if success and self.current_file_path:
            self.exported_files.add(self.current_file_path)
            self._save_file_status_registry()
            self.file_model.layoutChanged.emit()

        if self.export_thread:
            self.export_thread.quit()
            self.export_thread.wait()
            self.export_thread = None

    def _debug_open_cache_folder(self):
        """Resolves system native platform utilities to open the persistent temporary files directory."""
        logger.info(f"Shell execution layer opening persistent temp directory layout: {self.cache_directory}")
        if sys.platform == "win32":
            os.startfile(self.cache_directory)
        elif sys.platform == "darwin":
            os.system(f"open '{self.cache_directory}'")
        else:
            os.system(f"xdg-open '{self.cache_directory}'")

    def _debug_calculate_cache_size(self):
        """Computes total bytes written across tracking records inside the temporary workspace."""
        total_bytes = 0
        file_count = 0
        try:
            for item in os.listdir(self.cache_directory):
                fp = os.path.join(self.cache_directory, item)
                if os.path.isfile(fp):
                    total_bytes += os.path.getsize(fp)
                    file_count += 1
            size_mb = total_bytes / (1024 * 1024)
            QMessageBox.information(self, "Cache Size Assessment Metrics", 
                                    f"Current Temporary Storage Footprint Profile:\n\n"
                                    f"• Core Session Records: {file_count} items\n"
                                    f"• Total Disk Allocation: {size_mb:.3f} MB\n\n"
                                    f"Target: {self.cache_directory}")
        except Exception as ex:
            logger.error(f"Failed to calculate cache size metrics context: {ex}")

    def _debug_purge_cache_files(self):
        """Wipes temporary serialization structures safely to refresh the runtime sandbox layout."""
        confirm = QMessageBox.question(self, "Purge Confirmation", 
                                       "Are you sure you want to delete all cached presets and session manifest parameters?\n\n"
                                       "This resets file trees and clears modifications.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            logger.warning("Triggered total clearing of session temporary caches.")
            try:
                for item in os.listdir(self.cache_directory):
                    fp = os.path.join(self.cache_directory, item)
                    if os.path.isfile(fp):
                        os.remove(fp)
                self.edited_files.clear()
                self.starred_files.clear()
                self.exported_files.clear()
                self._save_file_status_registry()
                self.file_model.layoutChanged.emit()
                self.scene.clear()
                self.preview_matrix = None
                self.proxy_matrix = None
                QMessageBox.information(self, "Purge Complete", "Session cache manifest database cleared.")
            except Exception as ex:
                logger.error(f"Error executing complete storage context purge pass: {ex}")
    
    def _on_crop_variant_changed(self, variant):
        if not os.path.isfile(self.current_file_path):
            return
        # TODO: bail if no picture is loaded
        logger.info(f"Crop variant changed to {variant}")

        self._update_preset_key('active_crop_variant', variant)
        self._apply_preset_to_ui()
    
    def _add_crop_variant(self):
        if not os.path.isfile(self.current_file_path):
            return
        num_existing = self.crop_variant_combo.count()

        if num_existing > 1:
            name = f'variant {num_existing}'
        else:
            name = 'variant 1'

        logger.info(f'Adding crop variant {name}')
        # When we add a crop variant we need to give it the default json data
        self.preset['crop_variants'].setdefault(name, dict())
        self.preset['crop_variants'][name] = PhotoEditor.DEFAULT_PRESET['crop_variants']['default'].copy()

        self.crop_variant_combo.addItem(name)
        self.crop_variant_combo.setCurrentIndex(num_existing)

        # TODO: when we build the ui we need to add all saved crop variants
    
    def _remove_crop_variant(self):
        if not os.path.isfile(self.current_file_path):
            return
        variant_name = self.crop_variant_combo.currentText()
        variant_index = self.crop_variant_combo.currentIndex()

        if variant_name != 'default':
            logger.info(f'Removing crop variant {variant_name}')
            # Remove the item from the dict before removing it from the combo box 
            # This way when the callback triggers the item doesn't exist in the dict
            self.preset['crop_variants'].pop(variant_name)
            self.crop_variant_combo.removeItem(variant_index)
            self.crop_variant_combo.setCurrentIndex(max(variant_index - 1), 0)

def main():

    """Application wrapper baseline launcher method layer entry point configuration loop."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()