DARK_STYLE = """
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    QPushButton {
        background-color: #404040;
        border: 2px solid #555555;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
    }
    QPushButton:hover {
        background-color: #505050;
        border-color: #666666;
    }
    QPushButton:pressed {
        background-color: #353535;
    }
    QGroupBox {
        font-weight: bold;
        border: 2px solid #555555;
        border-radius: 8px;
        margin: 5px;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }
    QLineEdit {
        background-color: #404040;
        border: 2px solid #555555;
        border-radius: 5px;
        padding: 8px;
        color: #ffffff;
        font-size: 14px;
    }
    QLineEdit:focus {
        border-color: #0078d4;
    }
    QTextEdit {
        background-color: #404040;
        border: 2px solid #555555;
        border-radius: 5px;
        color: #ffffff;
        font-size: 12px;
    }
    QLabel {
        color: #ffffff;
    }
    QTabWidget::pane {
        border: 1px solid #555555;
        border-radius: 8px;
        background-color: #2b2b2b;
    }
    QTabBar::tab {
        background-color: #404040;
        color: #ffffff;
        padding: 8px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background-color: #505050;
        border-bottom: 2px solid #0078d4;
    }
    QComboBox {
        background-color: #404040;
        border: 2px solid #555555;
        border-radius: 5px;
        padding: 8px;
        color: #ffffff;
    }
    QComboBox QAbstractItemView {
        background-color: #404040;
        color: #ffffff;
        selection-background-color: #505050;
    }
    QCheckBox {
        color: #ffffff;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #555555;
        border-radius: 4px;
        background-color: #404040;
    }
    QCheckBox::indicator:checked {
        background-color: #0078d4;
        border: 2px solid #0078d4;
    }
    QSlider::groove:horizontal {
        border: 1px solid #555555;
        height: 8px;
        background: #404040;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #ffffff;
        border: 1px solid #555555;
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }
    QSlider::sub-page:horizontal {
        background: #0078d4;
        border-radius: 4px;
    }
"""