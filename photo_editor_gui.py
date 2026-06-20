"""PyQt6 desktop wrapper module running heavy multi-core image processing pipelines.

Provides standard dialog windows, automated workspace batch queues, active monitor 
histograms, and synchronized local execution logging routines.
"""

import sys
import os
import json
import datetime
import tempfile
import configparser
import shutil
import logging
from typing import Dict, Any, List

import cv2
import numpy as np

try:
    import rawpy
    HAS_RAWPY_GUI = True
except ImportError:
    HAS_RAWPY_GUI = False

from PIL import Image, ImageOps

from PyQt6.QtCore import Qt, QDir, QThread, pyqtSignal, QTimer, QEvent, QModelIndex
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QFileSystemModel, QPainter, QKeyEvent, QCursor, QFont, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTreeView, QGraphicsView, QGraphicsScene, QPushButton,
    QSlider, QCheckBox, QComboBox, QLabel, QGroupBox, QScrollArea, QFileDialog,
    QFrame, QLineEdit, QMenu, QProgressDialog, QMessageBox
)

from photo_editor_core import PhotoEditor, export_photo, SUPPORTED_EXTENSIONS, RAW_EXTENSIONS

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

    def __init__(self, export_queue: List[tuple], preset_map: Dict[str, Any]):
        """Initializes the background thread pool queue asset parameters.

        Args:
            export_queue (List[tuple]): Task items matching format (src_path, out_path, preset).
            preset_map (Dict[str, Any]): Workspace dictionary properties snapshot references.
        """
        super().__init__()
        self.export_queue = export_queue
        self.preset_map = preset_map

    def run(self):
        """Executes background asset encoding operations looping through the job parameters."""
        total_files = len(self.export_queue)
        logger.info(f"Background Export Worker Thread triggered for total batch volume size: {total_files}")
        
        try:
            for idx, item in enumerate(self.export_queue):
                img_path, out_path, item_preset = item
                base_name = os.path.basename(img_path)
                
                logger.info(f"Starting batch task segment processing pass ({idx+1}/{total_files}): {base_name}")
                self.progress_signal.emit(f"[{idx+1}/{total_files}] Loading high-fidelity asset matrix for: {base_name}...")
                
                full_res = PhotoEditor.load_image_matrix(img_path, preview=False)

                self.progress_signal.emit(f"[{idx+1}/{total_files}] Applying high-res crop dimensions for: {base_name}...")
                cropped_full = PhotoEditor.apply_crop(full_res, item_preset)

                self.progress_signal.emit(f"[{idx+1}/{total_files}] Computing multi-core tile pixel transformations...")
                processed_full = PhotoEditor.run_parallel_pipeline(cropped_full, item_preset)

                self.progress_signal.emit(f"[{idx+1}/{total_files}] Writing compressed target output JPG to disk...")
                export_photo(processed_full, out_path, item_preset)

                logger.info(f"Successfully processed image output save task configuration: {out_path}")
                self.file_done_signal.emit(idx + 1, total_files)

            self.finished_signal.emit(True, f"Successfully exported {total_files} asset layers safely.")
        except Exception as e:
            logger.error(f"Batch execution context worker thread encountered fatal termination: {e}")
            self.finished_signal.emit(False, f"Export operation stopped due to exception: {str(e)}")


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


class ZoomableGraphicsView(QGraphicsView):
    """Viewport container supporting mouse wheel zooming and click-and-drag panning."""

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
        """Applies scalable matrix multipliers to adjust viewport magnification.

        Args:
            event (QWheelEvent): Input location angle parameter wrapper blocks.
        """
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(zoom_factor, zoom_factor)


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
        self.setWindowTitle("Lightroom Pro Desktop GUI Environment")
        self.resize(1600, 950)

        self.current_file_path = ""
        self.preview_matrix = None
        self.show_original_state = False
        self.is_updating_ui = False
        self.export_thread = None
        self.copied_settings_buffer = None
        self.current_selected_folder_path = ""

        self.cache_directory = os.path.join(tempfile.gettempdir(), "photo_editor_session_cache")
        os.makedirs(self.cache_directory, exist_ok=True)
        self.state_ini_path = os.path.join(self.cache_directory, "session_state.ini")
        self.registry_json_path = os.path.join(self.cache_directory, "file_status_registry.json")
        self.manifest_json_path = os.path.join(self.cache_directory, "photo_editor_presets_manifest.json")

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
            "white_border_width_pct": 2,
            "apply_temperature_adjustment": True,
            "values_multiplier": 1.0, "color_multiplier": 1.0, "color_adjustments_multiplier": 1.0,
            "hdr_compression": 0.0, "exposure": 0.0, "contrast": 0.0,
            "whites": 0.0, "blacks": 0.0,
            "highlights": 0.0, "shadows": 0.0, "texture": 0.0, "clarity": 0.0,
            "gaussian_blur": 0.0, "vibrance": 0.0, "saturation": 0.0, "grain": 0.0, "grain_size": 1.0,
            "temp_kelvin": 6500, "tint": 0.0,
            "color_adjustments": {
                "red": {"hue": 0.0, "sat": 0.0}, "orange": {"hue": 0.0, "sat": 0.0},
                "yellow": {"hue": 0.0, "sat": 0.0}, "green": {"hue": 0.0, "sat": 0.0},
                "aqua": {"hue": 0.0, "sat": 0.0}, "blue": {"hue": 0.0, "sat": 0.0},
                "purple": {"hue": 0.0, "sat": 0.0}, "magenta": {"hue": 0.0, "sat": 0.0}
            }
        }
        self.preset = json.loads(json.dumps(self.default_preset))
        self.sliders_map = {}

        self._init_menu_bar()
        self._init_ui_layout()
        self._populate_local_presets()
        
        self.preview_render_timer = QTimer(self)
        self.preview_render_timer.setSingleShot(True)
        self.preview_render_timer.timeout.connect(self._render_stationary_pane)
        self.preview_target_path = ""

        self.showMaximized()
        logger.info("Main photo editor workspace frame initialization lifecycle complete.")

    def _read_preset_from_manifest(self, file_path: str) -> dict:
        """Queries the consolidated database file using absolute path strings as keys.

        Args:
            file_path (str): Unique path destination targeting an image configuration.

        Returns:
            dict: Preset mapping configuration properties.
        """
        if os.path.exists(self.manifest_json_path):
            try:
                with open(self.manifest_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if file_path in data:
                        return data[file_path]
            except Exception as e:
                logger.error(f"Failed to query central manifest database registry layout parameters: {e}")
        return json.loads(json.dumps(self.default_preset))

    def _write_preset_to_manifest(self, file_path: str, preset_data: dict):
        """Persists the updated metadata preset block inside the single manifest archive file.

        Args:
            file_path (str): Direct image location key value.
            preset_data (dict): Snapshotted parameters data metrics layout maps.
        """
        data = {}
        if os.path.exists(self.manifest_json_path):
            try:
                with open(self.manifest_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        
        data[file_path] = preset_data
        try:
            with open(self.manifest_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Manifest database disk serialization write fault exception error: {e}")

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

    def _load_file_status_registry(self):
        """Parses local tracking data configurations to update view states."""
        if os.path.exists(self.registry_json_path):
            try:
                with open(self.registry_json_path, "r") as f:
                    data = json.load(f)
                    self.starred_files = set(data.get("starred_paths", []))
                    self.exported_files = set(data.get("exported_paths", []))
            except Exception:
                pass

        if os.path.exists(self.manifest_json_path):
            try:
                with open(self.manifest_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for recorded_path in data.keys():
                        if os.path.exists(recorded_path):
                            self.edited_files.add(recorded_path)
            except Exception:
                pass

    def _save_file_status_registry(self):
        """Serializes current priority list indices directly out to local disk paths."""
        try:
            with open(self.registry_json_path, "w") as f:
                json.dump({
                    "starred_paths": list(self.starred_files),
                    "exported_paths": list(self.exported_files)
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

    def _toggle_aspect_ratio_flip(self):
        """Inverts the active bounding box aspect ratio horizontally vs vertically."""
        is_flipped = self.preset.get("crop_aspect_ratio_flipped", False)
        self.preset["crop_aspect_ratio_flipped"] = not is_flipped
        if not self.is_updating_ui:
            self._update_target_resolution_label()
            self._save_current_edits_to_session_cache()
            self._refresh_viewport()

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
            
        self.preview_target_path = path
        self.preview_render_timer.start(80)
        self._on_file_selected(current)

    def _render_stationary_pane(self):
        """Extracts and scales image data to fit the lower left-hand column preview thumbnail."""
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

        g_wb = CollapsibleGroupBox("White Balance Properties")
        self.cb_apply_temp = QCheckBox("Apply Kelvin Temperature Vector Transformation")
        self.cb_apply_temp.setChecked(True)
        self.cb_apply_temp.toggled.connect(lambda state: self._update_preset_key("apply_temperature_adjustment", state))
        g_wb.content_layout.addWidget(self.cb_apply_temp)
        self.sliders_map["temp_kelvin"] = self._create_slider_row(g_wb.content_layout, "Temperature (Kelvin)", 2000, 12000, 6500, lambda v: self._update_preset_key("temp_kelvin", v))
        self.sliders_map["tint"] = self._create_slider_row(g_wb.content_layout, "Tint", -100, 100, 0, lambda v: self._update_preset_key("tint", v / 100.0))
        self.sliders_layout.addWidget(g_wb)

        g_freq = CollapsibleGroupBox("Local Frequency Adjustments")
        self.sliders_map["texture"] = self._create_slider_row(g_freq.content_layout, "Texture Definition", -100, 100, 0, lambda v: self._update_preset_key("texture", v / 100.0))
        self.sliders_map["clarity"] = self._create_slider_row(g_freq.content_layout, "Clarity Profile", -100, 100, 0, lambda v: self._update_preset_key("clarity", v / 100.0))
        self.sliders_map["gaussian_blur"] = self._create_slider_row(g_freq.content_layout, "Gaussian Spatial Blur Pass", 0, 50, 0, lambda v: self._update_preset_key("gaussian_blur", v / 10.0))
        self.sliders_layout.addWidget(g_freq)

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

        g_style = CollapsibleGroupBox("Film Grain Overlays")
        self.sliders_map["grain"] = self._create_slider_row(g_style.content_layout, "Grain Density Strength", 0, 100, 0, lambda v: self._update_preset_key("grain", v / 100.0))
        self.sliders_map["grain_size"] = self._create_slider_row(g_style.content_layout, "Grain Cluster Size Scale", 1, 50, 10, lambda v: self._update_preset_key("grain_size", v / 10.0))
        self.sliders_layout.addWidget(g_style)

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

        def on_slider_changed(val):
            value_edit.setText(str(val))
            if not self.is_updating_ui:
                callback(val)

        def on_text_confirmed():
            if self.is_updating_ui: return
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
        active_color_bands = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
        for band in active_color_bands:
            band_data = color_adj.get(band, {"hue": 0.0, "sat": 0.0})
            self.sliders_map[f"{band}_hue"].setValue(int(band_data.get("hue", 0.0) * 100))
            self.sliders_map[f"{band}_sat"].setValue(int(band_data.get("sat", 0.0) * 100))

        self.sliders_map["grain"].setValue(int(self.preset.get("grain", 0.0) * 100))
        self.sliders_map["grain_size"].setValue(int(self.preset.get("grain_size", 1.0) * 10))

        self.is_updating_ui = False
        self._update_target_resolution_label()
        self._refresh_viewport()

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

        if self.preset.get("do_instagram_compression", True):
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

    def _update_preset_key(self, key: str, value: Any):
        """Updates targeted tracking parameter metrics inside global configurations.

        Args:
            key (str): Key matching requested modification slots.
            value (Any): Payload setting values configuration attributes.
        """
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

    def _on_file_selected(self, index: QModelIndex):
        """Loads data matrices from newly activated tree folder navigation index items.

        Args:
            index (QModelIndex): Target selection tracking index token layout parameters.
        """
        path = self.file_model.filePath(index)
        if os.path.isdir(path): return

        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS: return

        self.current_file_path = path
        logger.info(f"Target file workspace selected node swap focus: {path}")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
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

            self.preview_matrix = PhotoEditor.load_image_matrix(path, preview=True)
            self.preset = self._read_preset_from_manifest(path)
            self._apply_preset_to_ui()
            self._save_browsing_directory_state(os.path.dirname(path))
        except Exception as e:
            logger.error(f"Critical load validation exception encountered: {e}")
            self.statusBar().showMessage(f"Load Failure: {os.path.basename(path)}")
            self.scene.clear()
            self.preview_matrix = None
        finally:
            QApplication.restoreOverrideCursor()

    def _on_toggle_view_clicked(self, checked: bool):
        """Swaps standard target calculation tracks to show comparison modes on button triggers.

        Args:
            checked (bool): Toggle button value confirmation parameter flag.
        """
        self.show_original_state = checked
        self._refresh_viewport()

    def _refresh_viewport(self):
        """Re-computes active filters to display raster data on the center view canvas."""
        if self.preview_matrix is None: return

        if self.show_original_state:
            render_array = self.preview_matrix
            self.histogram_widget.render_histogram(render_array)
            img_uint8 = (np.clip(render_array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
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
        """Prepares single file queue requests to push into the multi-core background worker thread."""
        if not self.current_file_path:
            self.statusBar().showMessage("Export failure: No active file target data mapping available.")
            return

        suggested_dir = os.path.join(os.path.dirname(self.current_file_path), "edits")
        os.makedirs(suggested_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
        default_out = os.path.join(suggested_dir, f"{base_name}_edit.jpg")

        out_path, _ = QFileDialog.getSaveFileName(self, "Export Production Clean Photo Matrix Asset", default_out, "JPEG Image Map (*.jpg)")
        if not out_path: return

        queue = [(self.current_file_path, out_path, json.loads(json.dumps(self.preset)))]
        self._execute_progressive_export_queue(queue)

    def _trigger_batch_starred_export(self):
        """Assembles folder collection files tracking active star records into batch job blocks."""
        if not self.current_selected_folder_path or not os.path.exists(self.current_selected_folder_path):
            QMessageBox.warning(self, "Batch Blocked", "No valid working folder path context active.")
            return

        logger.info(f"Scanning target location folder for starred references: {self.current_selected_folder_path}")
        starred_in_folder = []
        
        try:
            for item in os.listdir(self.current_selected_folder_path):
                full_item_path = os.path.join(self.current_selected_folder_path, item)
                if os.path.isfile(full_item_path) and full_item_path in self.starred_files:
                    ext = os.path.splitext(full_item_path)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        starred_in_folder.append(full_item_path)
        except Exception as ex:
            logger.error(f"Failed to scan folder contents: {ex}")
            return

        if not starred_in_folder:
            QMessageBox.information(self, "Zero Starred Images Found", f"No starred photo markers match items inside:\n{os.path.basename(self.current_selected_folder_path)}")
            return

        output_dir = os.path.join(self.current_selected_folder_path, "starred_edits")
        os.makedirs(output_dir, exist_ok=True)

        queue = []
        for img_path in starred_in_folder:
            item_preset = self._read_preset_from_manifest(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_starred_edit.jpg")
            queue.append((img_path, out_path, item_preset))

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
        self.progress_dialog.show()

        self.export_thread = ExportWorker(queue, self.preset)
        self.export_thread.progress_signal.connect(self.progress_dialog.setLabelText)
        
        def handle_progress_step(done, total):
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
                QMessageBox.information(self, "Purge Complete", "Session cache manifest database cleared.")
            except Exception as ex:
                logger.error(f"Error executing complete storage context purge pass: {ex}")


def main():
    """Application wrapper baseline launcher method layer entry point configuration loop."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()