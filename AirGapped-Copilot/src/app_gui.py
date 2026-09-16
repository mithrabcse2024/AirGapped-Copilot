"""
AirGapped Copilot — Desktop GUI
==================================
Premium dark-themed PyQt6 native desktop interface for the
100% offline RAG application running on Qualcomm Snapdragon NPU.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Optional, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QProgressBar, QFileDialog, QScrollArea,
    QSizePolicy, QGroupBox, QGridLayout, QTextBrowser,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QMimeData, QSize, QUrl,
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QDragEnterEvent, QDropEvent, QIcon,
    QTextCursor,
)

log = logging.getLogger("app_gui")

# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg_primary": "#0a0a0f",
    "bg_surface": "#12121a",
    "bg_card": "#1a1a2e",
    "bg_card_hover": "#1f1f38",
    "bg_input": "#0f0f1a",
    "accent": "#00d4aa",
    "accent_dim": "#00a88a",
    "accent_glow": "rgba(0, 212, 170, 0.15)",
    "secondary": "#6c63ff",
    "secondary_dim": "#5a52d5",
    "success": "#00e676",
    "warning": "#ffab40",
    "error": "#ff5252",
    "text_primary": "#e0e0e0",
    "text_secondary": "#888899",
    "text_muted": "#555566",
    "border": "#2a2a3e",
    "border_light": "#3a3a5e",
    "user_bubble": "#1a3a5c",
    "ai_bubble": "#1e1e32",
    "scrollbar_bg": "#12121a",
    "scrollbar_handle": "#2a2a3e",
}

FONTS = {
    "family": "Segoe UI, Inter, Roboto, system-ui, sans-serif",
    "mono": "Cascadia Code, Consolas, monospace",
}


def build_stylesheet() -> str:
    c = COLORS
    return f"""
    QMainWindow, QWidget {{
        background-color: {c['bg_primary']};
        color: {c['text_primary']};
        font-family: {FONTS['family']};
        font-size: 13px;
    }}
    QScrollBar:vertical {{
        background: {c['scrollbar_bg']}; width: 8px; border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['scrollbar_handle']}; min-height: 30px; border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c['border_light']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    QScrollBar:horizontal {{
        background: {c['scrollbar_bg']}; height: 8px; border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['scrollbar_handle']}; min-width: 30px; border-radius: 4px;
    }}
    QPushButton {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600; font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {c['bg_card_hover']};
        border-color: {c['accent']};
    }}
    QPushButton:pressed {{ background-color: {c['accent_dim']}; }}
    QPushButton#primary {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 {c['accent']}, stop:1 {c['secondary']});
        color: #0a0a0f; border: none; font-weight: 700;
    }}
    QPushButton#primary:hover {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 {c['accent_dim']}, stop:1 {c['secondary_dim']});
    }}
    QPushButton#danger {{
        border-color: {c['error']}; color: {c['error']};
    }}
    QPushButton#danger:hover {{
        background-color: rgba(255, 82, 82, 0.15);
    }}
    QLineEdit {{
        background-color: {c['bg_input']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 10px 14px; font-size: 14px;
        selection-background-color: {c['accent_dim']};
    }}
    QLineEdit:focus {{ border-color: {c['accent']}; }}
    QListWidget {{
        background-color: {c['bg_surface']};
        border: 1px solid {c['border']};
        border-radius: 8px; padding: 4px; outline: none;
    }}
    QListWidget::item {{
        padding: 8px 12px; border-radius: 6px; margin: 2px 0;
    }}
    QListWidget::item:selected {{
        background-color: {c['accent_glow']}; color: {c['accent']};
    }}
    QListWidget::item:hover {{ background-color: {c['bg_card']}; }}
    QTextBrowser {{
        background-color: {c['bg_surface']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 8px; padding: 12px;
        font-size: 14px;
    }}
    QProgressBar {{
        background-color: {c['bg_surface']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        text-align: center;
        color: {c['text_secondary']}; font-size: 11px; height: 20px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 {c['accent']}, stop:1 {c['secondary']});
        border-radius: 5px;
    }}
    QGroupBox {{
        color: {c['text_secondary']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        margin-top: 12px; padding-top: 20px;
        font-weight: 600; font-size: 11px;
        letter-spacing: 1px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px; left: 16px;
    }}
    QLabel#header_title {{
        font-size: 22px; font-weight: 800; letter-spacing: 3px;
        color: {c['text_primary']};
    }}
    QLabel#badge {{
        background-color: {c['bg_card']};
        color: {c['accent']};
        border: 1px solid {c['accent']};
        border-radius: 12px; padding: 4px 12px;
        font-size: 11px; font-weight: 700;
    }}
    QLabel#section_title {{
        font-size: 13px; font-weight: 700;
        color: {c['text_secondary']};
        letter-spacing: 2px; padding: 8px 0;
    }}
    QLabel#metric {{
        font-family: {FONTS['mono']};
        font-size: 12px; color: {c['accent']};
    }}
    QLabel#footer_label {{
        color: {c['text_muted']}; font-size: 11px;
    }}
    QSplitter::handle {{
        background-color: {c['border']}; width: 1px;
    }}
    QSplitter::handle:hover {{ background-color: {c['accent']}; }}
    QToolTip {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px; padding: 6px 10px; font-size: 12px;
    }}
    """


# ═══════════════════════════════════════════════════════════════════════════
# Background Workers
# ═══════════════════════════════════════════════════════════════════════════

class IngestionWorker(QThread):
    """Background thread for PDF ingestion."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, pipeline, file_paths: List[str]):
        super().__init__()
        self.pipeline = pipeline
        self.file_paths = file_paths

    def run(self):
        try:
            results = self.pipeline.ingest_documents(
                self.file_paths,
                progress_callback=lambda c, t, f: self.progress.emit(c, t, f),
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class QueryWorker(QThread):
    """Background thread for RAG query execution."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pipeline, question: str, max_tokens: int = 256):
        super().__init__()
        self.pipeline = pipeline
        self.question = question
        self.max_tokens = max_tokens

    def run(self):
        try:
            response = self.pipeline.query(
                self.question, max_tokens=self.max_tokens,
            )
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Drop Zone Widget
# ═══════════════════════════════════════════════════════════════════════════

class DropZone(QFrame):
    """Drag-and-drop zone for PDF files."""
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setMaximumHeight(160)
        self._hovering = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("📄")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 32px; background: transparent; border: none;")
        layout.addWidget(icon)

        txt = QLabel("Drop PDF files here")
        txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        txt.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 13px; "
            f"font-weight: 600; background: transparent; border: none;"
        )
        layout.addWidget(txt)

        hint = QLabel("or click Browse below")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; "
            f"background: transparent; border: none;"
        )
        layout.addWidget(hint)
        self._update_style()

    def _update_style(self):
        c = COLORS
        if self._hovering:
            self.setStyleSheet(f"""
                DropZone {{
                    background-color: {c['accent_glow']};
                    border: 2px dashed {c['accent']};
                    border-radius: 12px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                DropZone {{
                    background-color: {c['bg_surface']};
                    border: 2px dashed {c['border']};
                    border-radius: 12px;
                }}
                DropZone:hover {{
                    border-color: {c['border_light']};
                    background-color: {c['bg_card']};
                }}
            """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            if any(u.toLocalFile().lower().endswith('.pdf') for u in event.mimeData().urls()):
                event.acceptProposedAction()
                self._hovering = True
                self._update_style()

    def dragLeaveEvent(self, event):
        self._hovering = False
        self._update_style()

    def dropEvent(self, event: QDropEvent):
        self._hovering = False
        self._update_style()
        files = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile().lower().endswith('.pdf')]
        if files:
            self.files_dropped.emit(files)


# ═══════════════════════════════════════════════════════════════════════════
# Main Application Window
# ═══════════════════════════════════════════════════════════════════════════

class AirGappedCopilotGUI(QMainWindow):
    """Main application window for AirGapped Copilot."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AirGapped Copilot — Qualcomm Snapdragon NPU")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        self.pipeline = None
        self._pending_files: List[str] = []
        self._worker: Optional[QThread] = None

        self._build_ui()
        self._init_pipeline()

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status_footer)
        self._status_timer.start(2000)

    def _init_pipeline(self):
        try:
            from src.rag_pipeline import RAGPipeline
            self.pipeline = RAGPipeline()
            self._update_npu_status()
            self._refresh_file_list()
            self._append_system_message(
                "🚀 AirGapped Copilot initialized successfully.\n"
                "Drop PDF documents on the left panel to begin."
            )
        except Exception as e:
            log.error(f"Pipeline init failed: {e}")
            self._append_system_message(
                f"⚠️ Pipeline initialization error: {e}\n"
                "Some features may be unavailable."
            )

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_header())

        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        body_splitter.setHandleWidth(1)
        body_splitter.addWidget(self._build_left_panel())
        body_splitter.addWidget(self._build_right_panel())
        body_splitter.setSizes([420, 880])
        body_splitter.setStyleSheet(
            f"QSplitter {{ background-color: {COLORS['bg_primary']}; }}"
        )

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(12, 8, 12, 8)
        bl.addWidget(body_splitter)
        main_layout.addWidget(body, stretch=1)

        main_layout.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(64)
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {COLORS['bg_surface']},
                    stop:0.5 {COLORS['bg_card']},
                    stop:1 {COLORS['bg_surface']});
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("⚡ AIRGAPPED COPILOT")
        title.setObjectName("header_title")
        layout.addWidget(title)
        layout.addStretch()

        badge = QLabel("⬡ QUALCOMM AI HUB")
        badge.setObjectName("badge")
        layout.addWidget(badge)
        layout.addSpacing(12)

        self.airgap_badge = QLabel("🛡️ AIR-GAPPED: 0 KB/s (OFFLINE)")
        self.airgap_badge.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(0, 230, 118, 0.08);
                color: {COLORS['success']};
                border: 1px solid {COLORS['success']};
                border-radius: 12px; padding: 4px 14px;
                font-size: 11px; font-weight: 700; letter-spacing: 1px;
            }}
        """)
        layout.addWidget(self.airgap_badge)
        return header

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 4, 8)
        layout.setSpacing(12)

        title = QLabel("📂  DOCUMENT INGESTION WORKSPACE")
        title.setObjectName("section_title")
        layout.addWidget(title)

        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        browse = QPushButton("📁  Browse Files")
        browse.clicked.connect(self._browse_files)
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(browse)

        pgroup = QGroupBox("Pending Files")
        pl = QVBoxLayout(pgroup)
        self.pending_list = QListWidget()
        self.pending_list.setMaximumHeight(120)
        pl.addWidget(self.pending_list)
        layout.addWidget(pgroup)

        self.process_btn = QPushButton("⚡  Process Documents")
        self.process_btn.setObjectName("primary")
        self.process_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_btn.clicked.connect(self._process_documents)
        layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        igroup = QGroupBox("Indexed Documents")
        il = QVBoxLayout(igroup)
        self.indexed_list = QListWidget()
        il.addWidget(self.indexed_list)
        self.chunk_count_label = QLabel("0 chunks indexed")
        self.chunk_count_label.setObjectName("metric")
        il.addWidget(self.chunk_count_label)
        layout.addWidget(igroup, stretch=1)

        clear = QPushButton("🗑️  Clear Database")
        clear.setObjectName("danger")
        clear.setCursor(Qt.CursorShape.PointingHandCursor)
        clear.clicked.connect(self._clear_database)
        layout.addWidget(clear)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel("🔍  INTERACTIVE QUERY & CITATION ENGINE")
        title.setObjectName("section_title")
        layout.addWidget(title)

        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.setFont(QFont("Segoe UI", 13))
        layout.addWidget(self.chat_display, stretch=3)

        # Metrics bar
        mf = QFrame()
        mf.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px; padding: 6px 12px;
            }}
        """)
        ml = QHBoxLayout(mf)
        ml.setContentsMargins(12, 4, 12, 4)

        self.latency_label = QLabel("⏱️ Latency: — ms")
        self.latency_label.setObjectName("metric")
        ml.addWidget(self.latency_label)
        ml.addStretch()
        self.tps_label = QLabel("⚡ Tokens/sec: —")
        self.tps_label.setObjectName("metric")
        ml.addWidget(self.tps_label)
        ml.addStretch()
        self.chunks_used_label = QLabel("📎 Context: —")
        self.chunks_used_label.setObjectName("metric")
        ml.addWidget(self.chunks_used_label)
        layout.addWidget(mf)

        # Citation inspector
        cg = QGroupBox("Citation Inspector")
        cl = QVBoxLayout(cg)
        self.citation_display = QTextBrowser()
        self.citation_display.setMaximumHeight(180)
        self.citation_display.setFont(QFont("Segoe UI", 12))
        self.citation_display.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {COLORS['bg_surface']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px; padding: 10px;
            }}
        """)
        cl.addWidget(self.citation_display)
        layout.addWidget(cg)

        # Query input
        inf = QFrame()
        inf.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
            }}
        """)
        inl = QHBoxLayout(inf)
        inl.setContentsMargins(6, 6, 6, 6)
        inl.setSpacing(8)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText(
            "Ask a question about your documents... (Enter to send)"
        )
        self.query_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px; padding: 12px 16px;
                font-size: 14px; color: {COLORS['text_primary']};
            }}
            QLineEdit:focus {{ border-color: {COLORS['accent']}; }}
        """)
        self.query_input.returnPressed.connect(self._send_query)
        inl.addWidget(self.query_input, stretch=1)

        self.send_btn = QPushButton("Send ➤")
        self.send_btn.setObjectName("primary")
        self.send_btn.setFixedWidth(100)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_query)
        inl.addWidget(self.send_btn)
        layout.addWidget(inf)
        return panel

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setFixedHeight(40)
        footer.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_surface']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(24, 0, 24, 0)

        self.npu_indicator = QLabel("●")
        self.npu_indicator.setStyleSheet(f"color: {COLORS['warning']}; font-size: 10px;")
        layout.addWidget(self.npu_indicator)

        self.npu_status_label = QLabel("Initializing...")
        self.npu_status_label.setObjectName("footer_label")
        layout.addWidget(self.npu_status_label)
        layout.addSpacing(24)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(div)
        layout.addSpacing(24)

        self.inference_label = QLabel("Inference: —")
        self.inference_label.setObjectName("footer_label")
        layout.addWidget(self.inference_label)
        layout.addStretch()

        self.footer_tps = QLabel("Decode: — tok/s")
        self.footer_tps.setObjectName("footer_label")
        layout.addWidget(self.footer_tps)
        layout.addSpacing(24)

        self.model_status_label = QLabel("Models: Loading...")
        self.model_status_label.setObjectName("footer_label")
        layout.addWidget(self.model_status_label)
        return footer

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------
    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF Documents", str(Path.home()),
            "PDF Files (*.pdf);;All Files (*)",
        )
        if files:
            self._on_files_dropped(files)

    def _on_files_dropped(self, files: List[str]):
        for f in files:
            if f not in self._pending_files:
                self._pending_files.append(f)
                item = QListWidgetItem(f"📄 {Path(f).name}")
                item.setToolTip(f)
                self.pending_list.addItem(item)

    def _process_documents(self):
        if not self._pending_files:
            self._append_system_message("⚠️ No files to process. Drop PDFs first.")
            return
        if self.pipeline is None:
            self._append_system_message("❌ Pipeline not initialized.")
            return

        self.process_btn.setEnabled(False)
        self.process_btn.setText("⏳ Processing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._worker = IngestionWorker(self.pipeline, self._pending_files.copy())
        self._worker.progress.connect(self._on_ingestion_progress)
        self._worker.finished.connect(self._on_ingestion_finished)
        self._worker.error.connect(self._on_ingestion_error)
        self._worker.start()

    def _on_ingestion_progress(self, current: int, total: int, filename: str):
        pct = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Processing {filename}... ({current}/{total})")

    def _on_ingestion_finished(self, results):
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡  Process Documents")

        success = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_created for r in results)
        total_ms = sum(r.latency_ms for r in results)

        parts = [f"✅ Ingestion complete: {success}/{len(results)} files processed"]
        parts.append(f"   📊 {total_chunks} chunks in {total_ms:.0f}ms")
        for r in results:
            if r.success:
                parts.append(f"   ✅ {r.filename}: {r.chunks_created} chunks ({r.latency_ms:.0f}ms)")
            else:
                parts.append(f"   ❌ {r.filename}: {r.error_message}")

        self._append_system_message("\n".join(parts))
        self._pending_files.clear()
        self.pending_list.clear()
        self._refresh_file_list()

    def _on_ingestion_error(self, error: str):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("⚡  Process Documents")
        self._append_system_message(f"❌ Ingestion error: {error}")

    def _send_query(self):
        question = self.query_input.text().strip()
        if not question:
            return
        if self.pipeline is None:
            self._append_system_message("❌ Pipeline not initialized.")
            return

        self._append_user_message(question)
        self.query_input.clear()
        self.send_btn.setEnabled(False)
        self.send_btn.setText("⏳...")
        self.query_input.setEnabled(False)

        self._worker = QueryWorker(self.pipeline, question)
        self._worker.finished.connect(self._on_query_finished)
        self._worker.error.connect(self._on_query_error)
        self._worker.start()

    def _on_query_finished(self, response):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send ➤")
        self.query_input.setEnabled(True)
        self.query_input.setFocus()

        self._append_ai_message(response.answer)
        self.latency_label.setText(f"⏱️ Latency: {response.latency_ms:.0f} ms")
        self.tps_label.setText(f"⚡ Tokens/sec: {response.tokens_per_sec:.1f}")
        self.chunks_used_label.setText(f"📎 Context: {response.context_chunks_used} chunks")
        self.footer_tps.setText(f"Decode: {response.tokens_per_sec:.1f} tok/s")
        self._display_citations(response.citations)

    def _on_query_error(self, error: str):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send ➤")
        self.query_input.setEnabled(True)
        self._append_system_message(f"❌ Query error: {error}")

    def _clear_database(self):
        if self.pipeline:
            self.pipeline.clear_database()
            self._refresh_file_list()
            self._append_system_message("🗑️ Vector database cleared.")

    # ------------------------------------------------------------------
    # Chat helpers
    # ------------------------------------------------------------------
    def _append_user_message(self, text: str):
        c = COLORS
        html = f"""
        <div style="text-align: right; margin: 8px 0;">
            <span style="display:inline-block; background-color:{c['user_bubble']};
                color:{c['text_primary']}; padding:10px 16px;
                border-radius:14px 14px 4px 14px; max-width:75%;
                font-size:14px; line-height:1.5;">{text}</span>
            <br/><span style="color:{c['text_muted']}; font-size:10px;">You</span>
        </div>"""
        self.chat_display.append(html)
        self._scroll_chat()

    def _append_ai_message(self, text: str):
        c = COLORS
        escaped = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        escaped = escaped.replace("\n", "<br/>")
        html = f"""
        <div style="text-align:left; margin:8px 0;">
            <span style="color:{c['accent']}; font-size:10px; font-weight:700;">
                ⚡ AirGapped Copilot</span><br/>
            <span style="display:inline-block; background-color:{c['ai_bubble']};
                color:{c['text_primary']}; padding:10px 16px;
                border-radius:14px 14px 14px 4px; max-width:85%;
                font-size:14px; line-height:1.6;">{escaped}</span>
        </div>"""
        self.chat_display.append(html)
        self._scroll_chat()

    def _append_system_message(self, text: str):
        c = COLORS
        escaped = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        escaped = escaped.replace("\n", "<br/>")
        html = f"""
        <div style="text-align:center; margin:6px 20px; padding:8px 14px;
            background-color:{c['bg_card']}; border:1px solid {c['border']};
            border-radius:8px; color:{c['text_secondary']};
            font-size:12px; line-height:1.5;">
            {escaped}
        </div>"""
        self.chat_display.append(html)
        self._scroll_chat()

    def _scroll_chat(self):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

    def _display_citations(self, citations):
        c = COLORS
        if not citations:
            self.citation_display.setHtml(
                f'<p style="color:{c["text_muted"]}; font-style:italic;">'
                f'No citations available.</p>'
            )
            return

        parts = []
        for i, cit in enumerate(citations, 1):
            score_pct = cit.relevance_score * 100
            sc = c['success'] if score_pct > 70 else (c['warning'] if score_pct > 40 else c['error'])
            dt = cit.text[:200] + "..." if len(cit.text) > 200 else cit.text
            dt = dt.replace("<","&lt;").replace(">","&gt;")
            parts.append(f"""
                <div style="margin:6px 0; padding:8px 12px;
                    background-color:{c['bg_card']}; border-left:3px solid {sc};
                    border-radius:4px;">
                    <b style="color:{c['accent']};">[Citation {i}]</b>
                    <span style="color:{c['text_muted']};">
                        📄 {cit.source_file} — Page {cit.page_number}
                    </span>
                    <span style="color:{sc}; float:right;">{score_pct:.0f}% match</span>
                    <br/><span style="color:{c['text_secondary']}; font-size:12px;">{dt}</span>
                </div>""")
        self.citation_display.setHtml("".join(parts))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _refresh_file_list(self):
        self.indexed_list.clear()
        if not self.pipeline:
            return
        files = self.pipeline.get_indexed_files()
        total = 0
        for f in files:
            name, count = f["source_file"], f["chunk_count"]
            total += count
            self.indexed_list.addItem(QListWidgetItem(f"✅  {name}  ({count} chunks)"))
        self.chunk_count_label.setText(f"{total:,} chunks indexed")

    def _update_npu_status(self):
        if not self.pipeline:
            return
        status = self.pipeline.npu_status
        is_npu = "HTP" in status or "Hexagon" in status
        self.npu_status_label.setText(status)
        if is_npu:
            self.npu_indicator.setStyleSheet(f"color:{COLORS['success']}; font-size:10px;")
            self.inference_label.setText("Inference: Hexagon NPU")
        else:
            self.npu_indicator.setStyleSheet(f"color:{COLORS['warning']}; font-size:10px;")
            self.inference_label.setText("Inference: CPU Fallback")
        self.model_status_label.setText(f"Models: Loaded | Emb: {self.pipeline.embedding_status}")

    def _update_status_footer(self):
        self._update_npu_status()


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-16s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AirGapped Copilot")
    app.setOrganizationName("Snapdragon AI Lab")
    app.setStyleSheet(build_stylesheet())

    window = AirGappedCopilotGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
