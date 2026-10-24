"""Modern design system and dark theme stylesheet for the Face Mask Detection UI."""

DARK_THEME_QSS = """
/* Base Window and Fonts */
QWidget {
    background-color: #0f172a;
    color: #f1f5f9;
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 13px;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #0f172a;
    border-radius: 8px;
    padding: 6px;
}

QTabBar::tab {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 10px 22px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background-color: #2563eb;
    color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background-color: #334155;
    color: #e2e8f0;
}

/* Group Boxes / Cards */
QGroupBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 24px;
    font-size: 14px;
    font-weight: 700;
    color: #38bdf8;
    padding: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #38bdf8;
}

/* Push Buttons */
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #1d4ed8;
}

QPushButton:pressed {
    background-color: #1e40af;
}

QPushButton:disabled {
    background-color: #475569;
    color: #94a3b8;
}

QPushButton#btn-danger {
    background-color: #ef4444;
}

QPushButton#btn-danger:hover {
    background-color: #dc2626;
}

QPushButton#btn-success {
    background-color: #10b981;
}

QPushButton#btn-success:hover {
    background-color: #059669;
}

QPushButton#btn-secondary {
    background-color: #334155;
    color: #e2e8f0;
}

QPushButton#btn-secondary:hover {
    background-color: #475569;
}

/* Form inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0f172a;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #38bdf8;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #475569;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #334155;
    border-radius: 6px;
    background-color: #0f172a;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    height: 22px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #38bdf8);
    border-radius: 5px;
}

/* Text Edit / Console */
QTextEdit, QPlainTextEdit {
    background-color: #090d16;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #10b981;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background-color: #0f172a;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #334155;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #475569;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #334155;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #38bdf8;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

/* Status Chips and Cards */
.card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
}

.chip-success {
    background-color: #064e3b;
    color: #34d399;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: bold;
}

.chip-warning {
    background-color: #78350f;
    color: #fbbf24;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: bold;
}

.chip-danger {
    background-color: #7f1d1d;
    color: #f87171;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: bold;
}

.chip-info {
    background-color: #1e3a8a;
    color: #60a5fa;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: bold;
}
"""
