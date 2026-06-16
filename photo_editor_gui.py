"""A PyQt6 desktop application wrapper for the high-precision photo editor engine.

Features an expandable vertical left-side splitter containing file tree items and
metadata readouts, coupled with organized, collapsible right-side parameter layouts.
"""

import sys
import os
import json
import datetime
import tempfile
import configparser
import base64
from typing import Dict, Any
import cv2
import numpy as np
import rawpy

from PyQt6.QtCore import Qt, QDir, QThread, pyqtSignal, QTimer, QEvent, QModelIndex
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QFileSystemModel, QPainter, QKeyEvent, QCursor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTreeView, QGraphicsView, QGraphicsScene, QPushButton,
    QSlider, QCheckBox, QComboBox, QLabel, QGroupBox, QScrollArea, QFileDialog,
    QFrame, QLineEdit, QMenu
)

from photo_editor_core import PhotoEditor, export_photo, SUPPORTED_EXTENSIONS, RAW_EXTENSIONS


class ExportWorker(QThread):
    """Background worker thread executing heavy matrix transformations and disk I/O."""
    
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, image_path: str, preset: Dict[str, Any], output_path: str):
        super().__init__()
        self.image_path = image_path
        self.preset = json.loads(json.dumps(preset))
        self.output_path = output_path

    def run(self):
        try:
            self.progress_signal.emit("Loading full resolution raw asset matrix...")
            full_res = PhotoEditor.load_image_matrix(self.image_path, preview=False)

            self.progress_signal.emit("Applying full resolution crop framing bounds...")
            cropped_full = PhotoEditor.apply_crop(full_res, self.preset)

            self.progress_signal.emit("Executing multi-core pipeline calculation pass...")
            processed_full = PhotoEditor.run_parallel_pipeline(cropped_full, self.preset)

            self.progress_signal.emit("Writing compressed photo file to disk layout...")
            export_photo(processed_full, self.output_path, self.preset)

            self.finished_signal.emit(True, f"Successfully exported: {os.path.basename(self.output_path)}")
        except Exception as e:
            self.finished_signal.emit(False, f"Export engine failed: {str(e)}")


class CustomFileSystemModel(QFileSystemModel):
    """Extended file model inserting contextual status markers dynamically inside listings."""
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
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

            return f"{prefix}{base_string}{suffix}"
            
        if role == Qt.ItemDataRole.FontRole:
            path = self.filePath(index)
            if path in self.main_window.edited_files:
                font = super().data(index, role)
                if not isinstance(font, QFont):
                    font = QFont()
                font.setItalic(True)
                return font
                
        return super().data(index, role)


class HistogramWidget(QFrame):
    """Custom painting widget to extract and graph floating-point luminance distributions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(115)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.hist_data = None
        self.clipping_left = False
        self.clipping_right = False

    def render_histogram(self, rgb_matrix: np.ndarray):
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
        painter = QPainter(self)
        rect = self.contentsRect()
        painter.fillRect(rect, Qt.GlobalColor.black)

        if self.hist_data is None:
            return

        max_val = np.max(self.hist_data[1:-1])
        if max_val == 0: 
            max_val = 1

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
    """Custom container element introducing uniform drawer visibility states."""

    def __init__(self, title: str, parent=None):
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
        if checked:
            self.content_widget.hide()
            self.toggle_btn.setText("▶ Expand Panel Layer")
        else:
            self.content_widget.show()
            self.toggle_btn.setText("▼ Collapse Panel Layer")


class ZoomableGraphicsView(QGraphicsView):
    """Viewport container supporting mouse wheel zooming and click-and-drag panning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QApplication.palette().dark())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(zoom_factor, zoom_factor)


class MainWindow(QMainWindow):
    """Main window object orchestrates user interface layout structures."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lightroom Pro Desktop GUI Environment")
        self.resize(1500, 900)

        self.current_file_path = ""
        self.preview_matrix = None
        self.show_original_state = False
        self.is_updating_ui = False
        self.export_thread = None
        self.copied_settings_buffer = None

        self.cache_directory = os.path.join(tempfile.gettempdir(), "photo_editor_session_cache")
        os.makedirs(self.cache_directory, exist_ok=True)
        self.state_ini_path = os.path.join(self.cache_directory, "session_state.ini")
        self.registry_json_path = os.path.join(self.cache_directory, "file_status_registry.json")

        self.starred_files = set()
        self.exported_files = set()
        self.edited_files = set()
        self._load_file_status_registry()

        self.default_preset = {
            "do_instagram_compression": True,
            "resolution_percentage": 100,
            "crop_aspect_ratio": "Free",
            "crop_aspect_ratio_flipped": False,
            "crop_rotation": 0,
            "crop_size": 100,
            "crop_center_x": 50,
            "crop_center_y": 50,
            "crop_free_width": 100,
            "crop_free_height": 100,
            "add_white_border": False,
            "white_border_width_pct": 5,
            "apply_temperature_adjustment": True,
            "values_multiplier": 1.0, "color_multiplier": 1.0, "color_adjustments_multiplier": 1.0,
            "hdr_compression": 0.0, "exposure": 0.0, "contrast": 0.0,
            "whites": 0.0, "blacks": 0.0,
            "highlights": 0.0, "shadows": 0.0, "texture": 0.0, "clarity": 0.0,
            "gaussian_blur": 0.0, "vibrance": 0.0, "saturation": 0.0, "grain": 0.0, "grain_size": 1.0,
            "temp_kelvin": 6500, "tint": 0.0,
            "color_adjustments": {
                "red": {"hue": 0.0, "sat": 0.0}, "orange": {"hue": 0.0, "sat": 0.0},
                "yellow": {"hue": 0.0, "sat": 0.0}, "green": {"hue": 0.0, "sat": 0.0}, "blue": {"hue": 0.0, "sat": 0.0}
            }
        }
        self.preset = json.loads(json.dumps(self.default_preset))
        self.sliders_map = {}

        self._init_menu_bar()
        self._init_ui_layout()
        self._populate_local_presets()
        
        self.hover_preview = QLabel(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.hover_preview.setStyleSheet("border: 1px solid #666; background-color: #1a1a1a;")
        self.hover_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hover_preview.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.hover_preview.hide()

        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self._show_hover_preview)
        self.hover_target_path = ""

        self.showMaximized()

    def _get_cache_filename(self, absolute_path: str) -> str:
        encoded_bytes = base64.urlsafe_b64encode(absolute_path.encode('utf-8'))
        return encoded_bytes.decode('utf-8') + ".json"

    def _init_menu_bar(self):
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

        tools_menu = menu_bar.addMenu("&Tools")
        
        copy_settings_action = QAction("&Copy Settings", self)
        copy_settings_action.setShortcut(QKeySequence("Ctrl+C"))
        copy_settings_action.triggered.connect(self._copy_edit_settings)
        tools_menu.addAction(copy_settings_action)

        paste_settings_action = QAction("&Paste Settings", self)
        paste_settings_action.setShortcut(QKeySequence("Ctrl+V"))
        paste_settings_action.triggered.connect(self._paste_edit_settings)
        tools_menu.addAction(paste_settings_action)

        tools_menu.addSeparator()

        self.highlight_clipped_action = QAction("&Highlight Clipped Pixels", self)
        self.highlight_clipped_action.setCheckable(True)
        self.highlight_clipped_action.triggered.connect(self._refresh_viewport)
        tools_menu.addAction(self.highlight_clipped_action)

    def _load_file_status_registry(self):
        if os.path.exists(self.registry_json_path):
            try:
                with open(self.registry_json_path, "r") as f:
                    data = json.load(f)
                    self.starred_files = set(data.get("starred_paths", []))
                    self.exported_files = set(data.get("exported_paths", []))
            except Exception:
                pass

        for filename in os.listdir(self.cache_directory):
            if filename in ("file_status_registry.json", "session_state.ini") or not filename.endswith(".json"):
                continue
            try:
                b64_part = filename[:-5]
                decoded_path = base64.urlsafe_b64decode(b64_part.encode('utf-8')).decode('utf-8')
                if os.path.exists(decoded_path):
                    self.edited_files.add(decoded_path)
            except Exception:
                pass

    def _save_file_status_registry(self):
        try:
            with open(self.registry_json_path, "w") as f:
                json.dump({
                    "starred_paths": list(self.starred_files),
                    "exported_paths": list(self.exported_files)
                }, f, indent=2)
        except Exception:
            pass

    def _get_initial_browsing_directory(self) -> str:
        config = configparser.ConfigParser()
        if os.path.exists(self.state_ini_path):
            try:
                config.read(self.state_ini_path)
                saved_path = config.get("Session", "last_root_path", fallback="")
                if saved_path and os.path.exists(saved_path):
                    return saved_path
            except Exception:
                pass
        return QDir.homePath() + "/Pictures"

    def _save_browsing_directory_state(self, target_path: str):
        config = configparser.ConfigParser()
        config["Session"] = {"last_root_path": target_path}
        try:
            with open(self.state_ini_path, "w") as f:
                config.write(f)
        except Exception:
            pass

    def _copy_edit_settings(self):
        focused = self.focusWidget()
        if isinstance(focused, QLineEdit):
            focused.copy()
            return
            
        self.copied_settings_buffer = json.loads(json.dumps(self.preset))
        self.statusBar().showMessage("Preset configurations copied to memory cache buffer.")

    def _paste_edit_settings(self):
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

    def _init_ui_layout(self):
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # ------------------ PANEL LEFT: FILE TREE & METADATA ------------------
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
        self.tree_view.clicked.connect(self._on_file_selected)
        
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._show_tree_context_menu)
        
        self.tree_view.setMouseTracking(True)
        self.tree_view.entered.connect(self._on_tree_view_entered)
        self.tree_view.viewport().installEventFilter(self)
        
        left_splitter.addWidget(self.tree_view)

        self.info_panel = QGroupBox("Active Metadata Profiles")
        vbox_info = QVBoxLayout(self.info_panel)
        vbox_info.setSpacing(6)
        
        self.lbl_info_name = QLabel("Name: None Loaded")
        self.lbl_info_res = QLabel("Original Resolution: -")
        self.lbl_info_date = QLabel("Date Modified: -")
        self.lbl_info_size = QLabel("File Disk Size: -")
        
        for widget in [self.lbl_info_name, self.lbl_info_res, self.lbl_info_date, self.lbl_info_size]:
            widget.setWordWrap(True)
            vbox_info.addWidget(widget)
        vbox_info.addStretch()
        
        left_splitter.addWidget(self.info_panel)
        left_splitter.setSizes([600, 300])
        main_splitter.addWidget(left_splitter)

        # ------------------ PANEL CENTER: VIEWPORT ------------------
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        center_layout.addWidget(self.view)

        control_row = QHBoxLayout()
        self.toggle_btn = QPushButton("Toggle View: Original / Edited [Hold '\\' Key]")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._on_toggle_view_clicked)
        control_row.addWidget(self.toggle_btn)
        center_layout.addLayout(control_row)
        main_splitter.addWidget(center_widget)

        # ------------------ PANEL RIGHT: SLIDERS CONTROL PANEL ------------------
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
        main_splitter.setSizes([280, 850, 370])

    def eventFilter(self, source, event):
        if source == self.tree_view.viewport() and event.type() == QEvent.Type.Leave:
            self.hover_timer.stop()
            self.hover_preview.hide()
        return super().eventFilter(source, event)

    def _on_tree_view_entered(self, index):
        self.hover_timer.stop()
        path = self.file_model.filePath(index)
        if os.path.isdir(path) or os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
            self.hover_preview.hide()
            return
            
        self.hover_target_path = path
        self.hover_timer.start(300)

    def _show_hover_preview(self):
        if not self.hover_target_path or not os.path.exists(self.hover_target_path):
            return
            
        try:
            ext = os.path.splitext(self.hover_target_path)[1].lower()
            if ext in RAW_EXTENSIONS:
                with rawpy.imread(self.hover_target_path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8)
                    h, w, _ = rgb.shape
                    scale = 320.0 / w
                    img_small = cv2.resize(rgb, (320, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                with Image.open(self.hover_target_path) as img:
                    img.thumbnail((320, 320))
                    img_small = np.array(img.convert("RGB"))
                    
            sh, sw, sc = img_small.shape
            q_img = QImage(img_small.data, sw, sh, sc * sw, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            self.hover_preview.setPixmap(pixmap)
            self.hover_preview.setFixedSize(sw + 4, sh + 4)
            
            cursor_pos = QCursor.pos()
            self.hover_preview.move(cursor_pos.x() + 25, cursor_pos.y() + 12)
            self.hover_preview.show()
        except Exception:
            self.hover_preview.hide()

    def _show_tree_context_menu(self, position):
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
        
        context_menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def _toggle_file_star(self, path: str):
        if path in self.starred_files:
            self.starred_files.remove(path)
        else:
            self.starred_files.add(path)
            
        self._save_file_status_registry()
        self.file_model.layoutChanged.emit()

    def _toggle_aspect_ratio_flip(self):
        is_flipped = self.preset.get("crop_aspect_ratio_flipped", False)
        self.preset["crop_aspect_ratio_flipped"] = not is_flipped
        if not self.is_updating_ui:
            self._update_target_resolution_label()
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()

    def _build_sliders_interface(self):
        """Assembles interactive group layers strictly mapped to structural criteria."""
        
        # 1. PRESET CONFIGURATION DRAWER PANEL
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

        # 2. CROP & FRAMING GEOMETRY MODULE
        g_crop = CollapsibleGroupBox("Crop & Border Configuration")
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

        # 3. GLOBAL EXPOSURE & VALUES PROFILE
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

        # 4. GLOBAL COLOR CONFIGURATION MATRIX
        g_col = CollapsibleGroupBox("Global Color Engine")
        self.sliders_map["color_multiplier"] = self._create_slider_row(g_col.content_layout, "Global Color Multiplier", 0, 100, 100, lambda v: self._update_preset_key("color_multiplier", v / 100.0))
        self.sliders_map["vibrance"] = self._create_slider_row(g_col.content_layout, "Vibrance (Muted Weight)", -100, 100, 0, lambda v: self._update_preset_key("vibrance", v / 100.0))
        self.sliders_map["saturation"] = self._create_slider_row(g_col.content_layout, "Saturation (Linear Multiplier)", -100, 100, 0, lambda v: self._update_preset_key("saturation", v / 100.0))
        self.sliders_layout.addWidget(g_col)

        # 5. WHITE BALANCE PROPERTIES PANEL
        g_wb = CollapsibleGroupBox("White Balance Properties")
        self.cb_apply_temp = QCheckBox("Apply Kelvin Temperature Vector Transformation")
        self.cb_apply_temp.setChecked(True)
        self.cb_apply_temp.toggled.connect(lambda state: self._update_preset_key("apply_temperature_adjustment", state))
        g_wb.content_layout.addWidget(self.cb_apply_temp)
        self.sliders_map["temp_kelvin"] = self._create_slider_row(g_wb.content_layout, "Temperature (Kelvin)", 2000, 12000, 6500, lambda v: self._update_preset_key("temp_kelvin", v))
        self.sliders_map["tint"] = self._create_slider_row(g_wb.content_layout, "Tint", -100, 100, 0, lambda v: self._update_preset_key("tint", v / 100.0))
        self.sliders_layout.addWidget(g_wb)

        # 6. LOCAL FREQUENCY ADJUSTMENTS
        g_freq = CollapsibleGroupBox("Local Frequency Adjustments")
        self.sliders_map["texture"] = self._create_slider_row(g_freq.content_layout, "Texture Definition", -100, 100, 0, lambda v: self._update_preset_key("texture", v / 100.0))
        self.sliders_map["clarity"] = self._create_slider_row(g_freq.content_layout, "Clarity Profile", -100, 100, 0, lambda v: self._update_preset_key("clarity", v / 100.0))
        self.sliders_map["gaussian_blur"] = self._create_slider_row(g_freq.content_layout, "Gaussian Spatial Blur", 0, 50, 0, lambda v: self._update_preset_key("gaussian_blur", v / 10.0))
        self.sliders_layout.addWidget(g_freq)

        # 7. TARGETED BAND SHIFTING
        g_ca = CollapsibleGroupBox("Targeted Chromatic Bands")
        self.sliders_map["color_adjustments_multiplier"] = self._create_slider_row(g_ca.content_layout, "Target Bands Multiplier", 0, 100, 100, lambda v: self._update_preset_key("color_adjustments_multiplier", v / 100.0))
        for band in ["red", "orange", "yellow", "green", "blue"]:
            g_band = QGroupBox(f"Band Channel: {band.upper()}")
            vbox_b = QVBoxLayout(g_band)
            self.sliders_map[f"{band}_hue"] = self._create_slider_row(vbox_b, "Hue Shift", -100, 100, 0, lambda v, b=band: self._update_nested_color_preset(b, "hue", v / 100.0))
            self.sliders_map[f"{band}_sat"] = self._create_slider_row(vbox_b, "Saturation Intensity", -100, 100, 0, lambda v, b=band: self._update_nested_color_preset(b, "sat", v / 100.0))
            g_ca.content_layout.addWidget(g_band)
        self.sliders_layout.addWidget(g_ca)

        # 8. STYLISTIC FILM NOISE OVERLAYS
        g_style = CollapsibleGroupBox("Film Grain Overlays")
        self.sliders_map["grain"] = self._create_slider_row(g_style.content_layout, "Grain Density Strength", 0, 100, 0, lambda v: self._update_preset_key("grain", v / 100.0))
        self.sliders_map["grain_size"] = self._create_slider_row(g_style.content_layout, "Grain Cluster Size Scale", 1, 50, 10, lambda v: self._update_preset_key("grain_size", v / 10.0))
        self.sliders_layout.addWidget(g_style)

    def _create_slider_row(self, parent_layout, label_text: str, mn: int, mx: int, init: int, callback) -> QSlider:
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

        def on_slider_changed(val):
            value_edit.setText(str(val))
            if not self.is_updating_ui:
                callback(val)

        def on_text_confirmed():
            if self.is_updating_ui: 
                return
            try:
                val = int(value_edit.text())
                val = max(mn, min(mx, val))
                self.is_updating_ui = True
                slider.setValue(val)
                self.is_updating_ui = False
                value_edit.setText(str(val))
                callback(val)
            except ValueError:
                value_edit.setText(str(slider.value()))

        slider.valueChanged.connect(on_slider_changed)
        value_edit.editingFinished.connect(on_text_confirmed)
        
        layout.addWidget(lbl)
        layout.addWidget(slider)
        layout.addWidget(value_edit)
        parent_layout.addWidget(row)
        return slider

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Backslash and not event.isAutoRepeat():
            self.show_original_state = True
            self.toggle_btn.setChecked(True)
            self._refresh_viewport()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Backslash and not event.isAutoRepeat():
            self.show_original_state = False
            self.toggle_btn.setChecked(False)
            self._refresh_viewport()
        super().keyReleaseEvent(event)

    def _populate_local_presets(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select a workspace preset...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
            files = [f for f in os.listdir(script_dir) if f.lower().endswith(".json")]
            for f in sorted(files):
                self.preset_combo.addItem(f)
        except Exception as e:
            self.statusBar().showMessage(f"Local file directory parse failure alert: {e}")
        self.preset_combo.blockSignals(False)

    def _on_preset_combo_changed(self, index: int):
        if index <= 0: 
            return
        filename = self.preset_combo.itemText(index)
        script_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
        self._load_preset_from_path(os.path.join(script_dir, filename))

    def _load_preset_from_path(self, path: str):
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
            self.statusBar().showMessage(f"Successfully loaded file preset metrics: {os.path.basename(path)}")
        except Exception as e:
            self.statusBar().showMessage(f"Failed to cleanly unpack file preset layout properties: {e}")

    def _apply_preset_to_ui(self):
        self.is_updating_ui = True
        self.cb_instagram.setChecked(self.preset.get("do_instagram_compression", True))
        self.cb_apply_temp.setChecked(self.preset.get("apply_temperature_adjustment", True))

        self.crop_ratio_combo.setCurrentText(self.preset.get("crop_aspect_ratio", "Free"))
        self.cb_white_border.setChecked(self.preset.get("add_white_border", False))

        self.sliders_map["resolution_percentage"].setValue(int(self.preset.get("resolution_percentage", 100)))
        self.sliders_map["crop_rotation"].setValue(int(self.preset.get("crop_rotation", 0)))
        self.sliders_map["crop_size"].setValue(int(self.preset.get("crop_size", 100)))
        self.sliders_map["crop_free_width"].setValue(int(self.preset.get("crop_free_width", 100)))
        self.sliders_map["crop_free_height"].setValue(int(self.preset.get("crop_free_height", 100)))
        self.sliders_map["crop_center_x"].setValue(int(self.preset.get("crop_center_x", 50)))
        self.sliders_map["crop_center_y"].setValue(int(self.preset.get("crop_center_y", 50)))
        self.sliders_map["white_border_width_pct"].setValue(int(self.preset.get("white_border_width_pct", 5)))

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

        color_adj = self.preset.get("color_adjustments", {})
        for band in ["red", "orange", "yellow", "green", "blue"]:
            band_data = color_adj.get(band, {"hue": 0.0, "sat": 0.0})
            self.sliders_map[f"{band}_hue"].setValue(int(band_data.get("hue", 0.0) * 100))
            self.sliders_map[f"{band}_sat"].setValue(int(band_data.get("sat", 0.0) * 100))

        self.sliders_map["grain"].setValue(int(self.preset.get("grain", 0.0) * 100))
        self.sliders_map["grain_size"].setValue(int(self.preset.get("grain_size", 1.0) * 10))

        self.is_updating_ui = False
        self._update_target_resolution_label()
        self._refresh_viewport()

    def _update_target_resolution_label(self):
        if not self.current_file_path:
            self.lbl_live_target_res.setText("Target Export Res: -")
            return
            
        try:
            ext = os.path.splitext(self.current_file_path)[1].lower()
            if ext in RAW_EXTENSIONS:
                with rawpy.imread(self.current_file_path) as raw:
                    w, h = raw.sizes.width, raw.sizes.height
            else:
                with Image.open(self.current_file_path) as img:
                    w, h = img.size
        except Exception:
            self.lbl_live_target_res.setText("Target Export Res: -")
            return
            
        ratio_mode = self.preset.get("crop_aspect_ratio", "Free")
        if ratio_mode == "Free":
            box_w = int(max(10, min(100, self.preset.get("crop_free_width", 100))) / 100.0 * w)
            box_h = int(max(10, min(100, self.preset.get("crop_free_height", 100))) / 100.0 * h)
            if self.preset.get("crop_aspect_ratio_flipped", False):
                box_w, box_h = box_h, box_w
        else:
            if ratio_mode == "Original": target_ratio = w / h
            elif ratio_mode == "1:1": target_ratio = 1.0
            elif ratio_mode == "4:5": target_ratio = 5.0 / 4.0 if w >= h else 4.0 / 5.0
            elif ratio_mode == "5:7": target_ratio = 7.0 / 5.0 if w >= h else 5.0 / 7.0
            elif ratio_mode == "8:10": target_ratio = 10.0 / 8.0 if w >= h else 8.0 / 10.0
            elif ratio_mode == "16:9": target_ratio = 16.0 / 9.0 if w >= h else 9.0 / 16.0
            else: target_ratio = w / h
                
            if self.preset.get("crop_aspect_ratio_flipped", False):
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

        # Accurately query the locked pixel dimensions based on crop aspect ratio profiles
        if self.preset.get("do_instagram_compression", True):
            final_w = 1080
            final_h = int(final_w * (box_h / box_w))
        else:
            pct = self.preset.get("resolution_percentage", 100) / 100.0
            final_w = int(box_w * pct)
            final_h = int(box_h * pct)
            
        self.lbl_live_target_res.setText(f"Target Export Res: {final_w} x {final_h}")

    def _save_current_edits_to_session_cache(self):
        if not self.current_file_path: 
            return
            
        cache_filename = self._get_cache_filename(self.current_file_path)
        cache_target_path = os.path.join(self.cache_directory, cache_filename)
        try:
            with open(cache_target_path, "w") as f:
                json.dump(self.preset, f, indent=2)
            
            if self.current_file_path not in self.edited_files:
                self.edited_files.add(self.current_file_path)
                self.file_model.layoutChanged.emit()
        except Exception:
            pass

    def _update_preset_key(self, key: str, value: Any):
        self.preset[key] = value
        if not self.is_updating_ui:
            self._update_target_resolution_label()
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()

    def _update_nested_color_preset(self, band: str, prop: str, value: float):
        self.preset["color_adjustments"][band][prop] = value
        if not self.is_updating_ui:
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()

    def _open_preset_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Preset JSON Configuration", "", "JSON Configurations (*.json)")
        if path:
            self._load_preset_from_path(path)
            combo_idx = self.preset_combo.findText(os.path.basename(path))
            self.preset_combo.setCurrentIndex(combo_idx if combo_idx >= 0 else 0)

    def _save_preset_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Current Edits As Preset JSON", "", "JSON Configurations (*.json)")
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.preset, f, indent=2)
                self.statusBar().showMessage(f"Successfully saved preset configuration out to: {os.path.basename(path)}")
                self._populate_local_presets()
                combo_idx = self.preset_combo.findText(os.path.basename(path))
                if combo_idx >= 0: 
                    self.preset_combo.setCurrentIndex(combo_idx)
            except Exception as e:
                self.statusBar().showMessage(f"Preset writing serialization error failure: {e}")

    def _change_root_folder(self):
        start_dir = self.current_file_path if self.current_file_path else QDir.currentPath()
        if os.path.isfile(start_dir): 
            start_dir = os.path.dirname(start_dir)
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Project Root Directory", start_dir, QFileDialog.Option.ShowDirsOnly)
        if selected_dir:
            self.tree_view.setRootIndex(self.file_model.index(selected_dir))
            self._save_browsing_directory_state(selected_dir)
            self.statusBar().showMessage(f"Tree viewport root path set to: {selected_dir}")

    def _on_file_selected(self, index):
        path = self.file_model.filePath(index)
        if os.path.isdir(path): 
            return

        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS: 
            return

        self.current_file_path = path
        self.statusBar().showMessage(f"Loading image preview asset: {os.path.basename(path)}")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            if ext in RAW_EXTENSIONS:
                with rawpy.imread(path) as raw:
                    orig_w, orig_h = raw.sizes.width, raw.sizes.height
            else:
                with Image.open(path) as img:
                    orig_w, orig_h = img.size

            self.lbl_info_name.setText(f"Name: {os.path.basename(path)}")
            self.lbl_info_res.setText(f"Original Resolution: {orig_w} x {orig_h}")
            
            mtime = os.path.getmtime(path)
            date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            self.lbl_info_date.setText(f"Date Modified: {date_str}")
            
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self.lbl_info_size.setText(f"File Disk Size: {size_mb:.2f} MB")

            self.preview_matrix = PhotoEditor.load_image_matrix(path, preview=True)

            cache_filename = self._get_cache_filename(path)
            cache_target_path = os.path.join(self.cache_directory, cache_filename)
            
            if not os.path.exists(cache_target_path):
                old_filename = os.path.basename(path) + ".json"
                old_cache_path = os.path.join(self.cache_directory, old_filename)
                if os.path.exists(old_cache_path):
                    try:
                        with open(old_cache_path, "r") as f:
                            self.preset = json.load(f)
                        with open(cache_target_path, "w") as f:
                            json.dump(self.preset, f, indent=2)
                    except Exception:
                        self.preset = json.loads(json.dumps(self.default_preset))
                else:
                    self.preset = json.loads(json.dumps(self.default_preset))
            else:
                try:
                    with open(cache_target_path, "r") as f:
                        self.preset = json.load(f)
                except Exception:
                    self.preset = json.loads(json.dumps(self.default_preset))

            self._apply_preset_to_ui()
            self._save_browsing_directory_state(os.path.dirname(path))
            self.statusBar().showMessage(f"Active workspace file: {os.path.basename(path)}")
        except Exception as e:
            self.statusBar().showMessage(f"Critical load validation error exception encountered: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_toggle_view_clicked(self, checked):
        self.show_original_state = checked
        self._refresh_viewport()

    def _refresh_viewport(self):
        if self.preview_matrix is None: 
            return

        if self.show_original_state:
            render_array = self.preview_matrix
            self.histogram_widget.render_histogram(render_array)
            img_uint8 = (np.clip(render_array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            # Crop framing takes precedence over structural resolution steps
            cropped_preview = PhotoEditor.apply_crop(self.preview_matrix, self.preset)
            render_array = PhotoEditor.run_parallel_pipeline(cropped_preview, self.preset)
            
            self.histogram_widget.render_histogram(render_array)
            img_uint8 = (np.clip(render_array, 0.0, 1.0) * 255.0).astype(np.uint8)

            if self.highlight_clipped_action.isChecked():
                mask_black = (img_uint8[:, :, 0] == 0) & (img_uint8[:, :, 1] == 0) & (img_uint8[:, :, 2] == 0)
                mask_white = (img_uint8[:, :, 0] == 255) & (img_uint8[:, :, 1] == 255) & (img_uint8[:, :, 2] == 255)
                img_uint8[mask_black | mask_white] = [255, 0, 0]
                
            img_uint8 = PhotoEditor.apply_white_border(img_uint8, self.preset)

        h, w, ch = img_uint8.shape
        q_image = QImage(img_uint8.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, w, h)

    def _trigger_file_export(self):
        if not self.current_file_path:
            self.statusBar().showMessage("Export failure: No active file target data mapping available.")
            return

        if self.export_thread and self.export_thread.isRunning():
            self.statusBar().showMessage("An export thread is already running in the background.")
            return

        suggested_dir = os.path.join(os.path.dirname(self.current_file_path), "edits")
        os.makedirs(suggested_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
        default_out = os.path.join(suggested_dir, f"{base_name}_edit.jpg")

        out_path, _ = QFileDialog.getSaveFileName(self, "Export Production Clean Photo Matrix Asset", default_out, "JPEG Image Map (*.jpg)")
        if not out_path: 
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.export_action.setEnabled(False)

        self.export_thread = ExportWorker(self.current_file_path, self.preset, out_path)
        self.export_thread.progress_signal.connect(self.statusBar().showMessage)
        self.export_thread.finished_signal.connect(self._on_export_thread_completed)
        self.export_thread.start()

    def _on_export_thread_completed(self, success: bool, message: str):
        QApplication.restoreOverrideCursor()
        self.export_action.setEnabled(True)
        self.statusBar().showMessage(message)
        
        if success and self.current_file_path:
            self.exported_files.add(self.current_file_path)
            self._save_file_status_registry()
            self.file_model.layoutChanged.emit()

        if self.export_thread:
            self.export_thread.quit()
            self.export_thread.wait()
            self.export_thread = None


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()