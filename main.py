#!/usr/bin/env python3
"""Axiom Scientific Suite - Main Entry Point"""
import sys
import os
import traceback
import logging

# Fix for PyInstaller frozen apps
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    BASE_DIR = os.path.dirname(sys.executable)
    # Ensure modules directory is on path
    sys.path.insert(0, BASE_DIR)
    os.chdir(BASE_DIR)
    # Set up logging to file for debugging
    log_file = os.path.join(BASE_DIR, 'quantumres.log')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info("Axiom starting from frozen exe")
    logging.info(f"BASE_DIR: {BASE_DIR}")
    logging.info(f"sys.path: {sys.path}")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logging.basicConfig(level=logging.INFO)

os.environ['QT_API'] = 'pyqt5'
# Ensure matplotlib uses non-interactive backend initially
os.environ['MPLBACKEND'] = 'Agg'

# Apply dark matplotlib style globally
try:
    from mpl_style import apply_axiom_style
    apply_axiom_style()
except Exception:
    pass


def show_error_dialog(title, message):
    """Show error as a simple Qt message box or print to console."""
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()
    except Exception:
        print(f"FATAL ERROR: {title}\n{message}", file=sys.stderr)


def main():
    try:
        logging.info("Importing PyQt5...")
        from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel
        from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QPainter, QPen
        from PyQt5.QtCore import Qt, QTimer

        logging.info("Creating QApplication...")
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)
        app.setApplicationName("Axiom Scientific Suite")
        app.setOrganizationName("Axiom")
        app.setStyle("Fusion")

        # Create splash screen
        splash_pix = QPixmap(560, 340)
        splash_pix.fill(QColor(18, 18, 24))
        painter = QPainter(splash_pix)
        painter.setRenderHint(QPainter.Antialiasing)

        # Subtle border glow
        pen = QPen(QColor(60, 120, 200, 80))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRoundedRect(2, 2, 556, 336, 12, 12)

        # App name - top area
        painter.setPen(QColor(90, 160, 230))
        painter.setFont(QFont("Segoe UI", 32, QFont.Bold))
        painter.drawText(splash_pix.rect().adjusted(0, 55, 0, 0), Qt.AlignHCenter | Qt.AlignTop, "Axiom")

        # Subtitle - below name, clearly separated
        painter.setPen(QColor(140, 175, 220))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(splash_pix.rect().adjusted(0, 120, 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                         "Universal Scientific Computing Platform")

        # Version
        painter.setPen(QColor(100, 100, 120))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(splash_pix.rect().adjusted(0, 155, 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                         "v1.0")

        # Decorative line
        painter.setPen(QPen(QColor(60, 120, 200, 120), 1))
        painter.drawLine(140, 190, 420, 190)

        # Loading text - well below everything else
        painter.setPen(QColor(120, 120, 140))
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(splash_pix.rect().adjusted(0, 0, 0, -30), Qt.AlignHCenter | Qt.AlignBottom,
                         "Initializing scientific modules...")

        painter.end()

        splash = QSplashScreen(splash_pix)
        splash.show()
        app.processEvents()

        # Apply theme
        from themes import get_stylesheet, get_palette, THEMES
        default_theme = "Axiom Dark"
        app.setPalette(get_palette(default_theme))
        app.setStyleSheet(get_stylesheet(default_theme))
        app._current_theme = default_theme

        font = QFont("Segoe UI", 10)
        font.setPixelSize(13)
        app.setFont(font)

        logging.info("Creating main window...")
        splash.showMessage("  Preparing workspace...",
                          Qt.AlignBottom | Qt.AlignHCenter, QColor(150, 150, 150))
        app.processEvents()

        from app import QuantumResMainWindow
        window = QuantumResMainWindow(splash=splash)

        tab_count = window.tabs.count()
        logging.info(f"Main window created, {tab_count} tabs")

        # Update window title with module count
        window.setWindowTitle(f"Axiom Scientific Suite \u2014 {tab_count} Modules")

        splash.showMessage(f"  Ready!  {tab_count} modules available.",
                          Qt.AlignBottom | Qt.AlignHCenter, QColor(100, 200, 100))
        app.processEvents()

        window.show()
        splash.finish(window)

        # Log startup summary
        logging.info(f"Axiom started: {tab_count} modules, startup OK")
        logging.info("Window shown, entering event loop")
        sys.exit(app.exec_())

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        logging.critical(f"Fatal error: {error_msg}")
        show_error_dialog("Axiom - Startup Error", error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
