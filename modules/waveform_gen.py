"""
Waveform Generator & Oscilloscope Simulator Module
Real-time waveform generation with multi-channel display, math operations,
Fourier synthesis, Lissajous figures, and oscilloscope-style controls.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QCheckBox, QTabWidget, QSlider, QFileDialog, QTableWidget,
    QTableWidgetItem, QSplitter, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import csv
import os


WAVEFORM_TYPES = ["Sine", "Square", "Triangle", "Sawtooth", "Pulse", "Noise", "DC"]
CHANNEL_COLORS = ["#00FF00", "#FFFF00", "#00FFFF", "#FF6600"]
CHANNEL_NAMES = ["CH1", "CH2", "CH3", "CH4"]
MATH_OPS = ["None", "CH1+CH2", "CH1-CH2", "CH1*CH2", "AM (CH1 carrier, CH2 mod)", "FM (CH1 carrier, CH2 mod)"]


class ChannelControls(QGroupBox):
    """Per-channel waveform parameter controls."""

    def __init__(self, index, color, parent=None):
        super().__init__(f"Channel {index + 1}", parent)
        self.index = index
        self.setStyleSheet(f"QGroupBox {{ color: {color}; font-weight: bold; }}")
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(4)

        self.enabled = QCheckBox("Enable")
        self.enabled.setChecked(self.index == 0)
        layout.addWidget(self.enabled, 0, 0, 1, 2)

        layout.addWidget(QLabel("Type:"), 1, 0)
        self.waveform_type = QComboBox()
        self.waveform_type.addItems(WAVEFORM_TYPES)
        layout.addWidget(self.waveform_type, 1, 1)

        layout.addWidget(QLabel("Freq (Hz):"), 2, 0)
        self.frequency = QDoubleSpinBox()
        self.frequency.setRange(0.1, 10000.0)
        self.frequency.setValue(100.0)
        self.frequency.setDecimals(1)
        layout.addWidget(self.frequency, 2, 1)

        layout.addWidget(QLabel("Amp (V):"), 3, 0)
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(0.0, 10.0)
        self.amplitude.setValue(1.0)
        self.amplitude.setDecimals(2)
        layout.addWidget(self.amplitude, 3, 1)

        layout.addWidget(QLabel("Phase (\u00b0):"), 4, 0)
        self.phase = QDoubleSpinBox()
        self.phase.setRange(0.0, 360.0)
        self.phase.setValue(0.0)
        self.phase.setDecimals(1)
        layout.addWidget(self.phase, 4, 1)

        layout.addWidget(QLabel("DC Offset (V):"), 5, 0)
        self.dc_offset = QDoubleSpinBox()
        self.dc_offset.setRange(-10.0, 10.0)
        self.dc_offset.setValue(0.0)
        self.dc_offset.setDecimals(2)
        layout.addWidget(self.dc_offset, 5, 1)

        layout.addWidget(QLabel("Duty (%):"), 6, 0)
        self.duty_cycle = QDoubleSpinBox()
        self.duty_cycle.setRange(1.0, 99.0)
        self.duty_cycle.setValue(50.0)
        layout.addWidget(self.duty_cycle, 6, 1)

    def get_params(self):
        return {
            "enabled": self.enabled.isChecked(),
            "type": self.waveform_type.currentText(),
            "freq": self.frequency.value(),
            "amp": self.amplitude.value(),
            "phase": np.radians(self.phase.value()),
            "dc_offset": self.dc_offset.value(),
            "duty": self.duty_cycle.value() / 100.0,
        }


class WaveformGenWidget(QWidget):
    """Real-time waveform generator and oscilloscope simulator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None
        self._time_offset = 0.0
        self._running = False
        self._sample_rate = 44100
        self._num_points = 2048
        self._last_data = {}
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_display)

    def set_logger(self, fn):
        self._log = fn

    def _emit_log(self, msg):
        if self._log:
            self._log(msg)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(340)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)

        # Channel controls
        self.channels = []
        for i in range(4):
            ch = ChannelControls(i, CHANNEL_COLORS[i])
            self.channels.append(ch)
            ctrl_layout.addWidget(ch)

        # Oscilloscope controls
        scope_grp = QGroupBox("Oscilloscope")
        scope_lay = QGridLayout(scope_grp)

        scope_lay.addWidget(QLabel("Timebase (ms/div):"), 0, 0)
        self.timebase = QDoubleSpinBox()
        self.timebase.setRange(0.01, 1000.0)
        self.timebase.setValue(1.0)
        self.timebase.setDecimals(2)
        scope_lay.addWidget(self.timebase, 0, 1)

        scope_lay.addWidget(QLabel("V/div:"), 1, 0)
        self.v_per_div = QDoubleSpinBox()
        self.v_per_div.setRange(0.01, 10.0)
        self.v_per_div.setValue(0.5)
        self.v_per_div.setDecimals(2)
        scope_lay.addWidget(self.v_per_div, 1, 1)

        scope_lay.addWidget(QLabel("Trigger Level (V):"), 2, 0)
        self.trigger_level = QDoubleSpinBox()
        self.trigger_level.setRange(-10.0, 10.0)
        self.trigger_level.setValue(0.0)
        self.trigger_level.setDecimals(2)
        scope_lay.addWidget(self.trigger_level, 2, 1)

        scope_lay.addWidget(QLabel("Trigger Source:"), 3, 0)
        self.trigger_src = QComboBox()
        self.trigger_src.addItems(CHANNEL_NAMES)
        scope_lay.addWidget(self.trigger_src, 3, 1)

        ctrl_layout.addWidget(scope_grp)

        # Math operations
        math_grp = QGroupBox("Waveform Math")
        math_lay = QVBoxLayout(math_grp)
        self.math_op = QComboBox()
        self.math_op.addItems(MATH_OPS)
        math_lay.addWidget(self.math_op)
        self.math_enabled = QCheckBox("Show Math Result")
        math_lay.addWidget(self.math_enabled)
        ctrl_layout.addWidget(math_grp)

        # Buttons
        btn_lay = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.run)
        btn_lay.addWidget(self.btn_run)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        btn_lay.addWidget(self.btn_stop)
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export)
        btn_lay.addWidget(self.btn_export)
        ctrl_layout.addLayout(btn_lay)

        ctrl_layout.addStretch()
        ctrl_scroll.setWidget(ctrl_widget)
        main_layout.addWidget(ctrl_scroll)

        # Right: display area with tabs
        right_splitter = QSplitter(Qt.Vertical)

        self.tabs = QTabWidget()

        # Time-domain tab
        self.fig_time = Figure(figsize=(8, 4), facecolor="#1a1a2e")
        self.canvas_time = FigureCanvas(self.fig_time)
        self.ax_time = self.fig_time.add_subplot(111)
        self._style_axis(self.ax_time, "Time (ms)", "Voltage (V)")
        self.tabs.addTab(self.canvas_time, "Time Domain")

        # FFT / Spectrum tab
        self.fig_fft = Figure(figsize=(8, 4), facecolor="#1a1a2e")
        self.canvas_fft = FigureCanvas(self.fig_fft)
        self.ax_fft = self.fig_fft.add_subplot(111)
        self._style_axis(self.ax_fft, "Frequency (Hz)", "Magnitude (dB)")
        self.tabs.addTab(self.canvas_fft, "Spectrum")

        # Lissajous tab
        self.fig_lissa = Figure(figsize=(5, 5), facecolor="#1a1a2e")
        self.canvas_lissa = FigureCanvas(self.fig_lissa)
        self.ax_lissa = self.fig_lissa.add_subplot(111)
        self._style_axis(self.ax_lissa, "CH1 (V)", "CH2 (V)")
        self.tabs.addTab(self.canvas_lissa, "Lissajous (X-Y)")

        # Fourier synthesis tab
        fourier_widget = QWidget()
        fourier_layout = QVBoxLayout(fourier_widget)
        harm_ctrl = QHBoxLayout()
        harm_ctrl.addWidget(QLabel("Harmonics:"))
        self.harmonic_sliders = []
        self.harmonic_labels = []
        harmonics_grid = QGridLayout()
        for i in range(20):
            lbl = QLabel(f"H{i + 1}: 0%")
            lbl.setMinimumWidth(60)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(100 if i == 0 else 0)
            slider.valueChanged.connect(self._on_harmonic_changed)
            self.harmonic_sliders.append(slider)
            self.harmonic_labels.append(lbl)
            row = i // 4
            col = (i % 4) * 2
            harmonics_grid.addWidget(lbl, row, col)
            harmonics_grid.addWidget(slider, row, col + 1)
        fourier_layout.addLayout(harmonics_grid)

        self.fig_fourier = Figure(figsize=(8, 3), facecolor="#1a1a2e")
        self.canvas_fourier = FigureCanvas(self.fig_fourier)
        self.ax_fourier = self.fig_fourier.add_subplot(111)
        self._style_axis(self.ax_fourier, "Time (ms)", "Amplitude")
        fourier_layout.addWidget(self.canvas_fourier)
        self.tabs.addTab(fourier_widget, "Fourier Synthesis")

        right_splitter.addWidget(self.tabs)

        # Measurements table
        self.meas_table = QTableWidget(4, 7)
        self.meas_table.setHorizontalHeaderLabels(
            ["Channel", "Freq (Hz)", "Period (ms)", "Amp (V)", "RMS (V)", "Vpp (V)", "Rise (us)"]
        )
        self.meas_table.setMaximumHeight(160)
        for i in range(4):
            self.meas_table.setItem(i, 0, QTableWidgetItem(CHANNEL_NAMES[i]))
        right_splitter.addWidget(self.meas_table)

        main_layout.addWidget(right_splitter, 1)

    def _style_axis(self, ax, xlabel, ylabel):
        ax.set_facecolor("#0d0d1a")
        ax.set_xlabel(xlabel, color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, color="#333355", linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#444466")

    # ---------------------------------------------------------- Generation
    @staticmethod
    def _generate_waveform(t, params):
        wtype = params["type"]
        freq = params["freq"]
        amp = params["amp"]
        phase = params["phase"]
        dc = params["dc_offset"]
        duty = params["duty"]
        omega = 2.0 * np.pi * freq
        phi = omega * t + phase

        if wtype == "Sine":
            y = amp * np.sin(phi)
        elif wtype == "Square":
            y = amp * np.sign(np.sin(phi))
        elif wtype == "Triangle":
            y = amp * (2.0 / np.pi) * np.arcsin(np.sin(phi))
        elif wtype == "Sawtooth":
            y = amp * (2.0 * (freq * t + phase / (2 * np.pi)) % 1.0 - 1.0)
        elif wtype == "Pulse":
            cycle_pos = (phi / (2 * np.pi)) % 1.0
            y = np.where(cycle_pos < duty, amp, -amp)
        elif wtype == "Noise":
            y = amp * np.random.randn(len(t))
        elif wtype == "DC":
            y = np.full_like(t, amp)
        else:
            y = np.zeros_like(t)
        return y + dc

    def _compute_math(self, ch_data):
        op = self.math_op.currentText()
        if op == "None" or not self.math_enabled.isChecked():
            return None
        d0 = ch_data.get(0)
        d1 = ch_data.get(1)
        if d0 is None or d1 is None:
            return None
        if op == "CH1+CH2":
            return d0 + d1
        elif op == "CH1-CH2":
            return d0 - d1
        elif op == "CH1*CH2":
            return d0 * d1
        elif op.startswith("AM"):
            mod_index = 1.0
            return d0 * (1.0 + mod_index * d1 / (np.max(np.abs(d1)) + 1e-12))
        elif op.startswith("FM"):
            fm_dev = 50.0
            p0 = self.channels[0].get_params()
            phase_mod = 2 * np.pi * fm_dev * np.cumsum(d1) / self._sample_rate
            omega_c = 2 * np.pi * p0["freq"]
            t = np.arange(len(d0)) / self._sample_rate
            return p0["amp"] * np.sin(omega_c * t + phase_mod + p0["phase"])
        return None

    def _apply_trigger(self, t, y_trigger):
        level = self.trigger_level.value()
        for i in range(1, len(y_trigger)):
            if y_trigger[i - 1] < level <= y_trigger[i]:
                return i
        return 0

    # --------------------------------------------------------- Measurements
    def _measure_channel(self, y, row):
        if y is None or len(y) == 0:
            for col in range(1, 7):
                self.meas_table.setItem(row, col, QTableWidgetItem("---"))
            return
        vpp = float(np.max(y) - np.min(y))
        amp_val = vpp / 2.0
        rms = float(np.sqrt(np.mean(y ** 2)))
        # Estimate frequency via zero crossings
        crossings = np.where(np.diff(np.sign(y - np.mean(y))))[0]
        if len(crossings) >= 2:
            avg_half_period = np.mean(np.diff(crossings)) / self._sample_rate
            freq_est = 1.0 / (2.0 * avg_half_period) if avg_half_period > 0 else 0
            period_ms = (1.0 / freq_est * 1000) if freq_est > 0 else 0
        else:
            freq_est = 0
            period_ms = 0
        # Rise time (10% to 90%)
        y_min, y_max = np.min(y), np.max(y)
        thresh_lo = y_min + 0.1 * (y_max - y_min)
        thresh_hi = y_min + 0.9 * (y_max - y_min)
        rise_samples = 0
        in_rise = False
        for i in range(1, len(y)):
            if not in_rise and y[i] >= thresh_lo and y[i - 1] < thresh_lo:
                in_rise = True
                start_idx = i
            elif in_rise and y[i] >= thresh_hi:
                rise_samples = i - start_idx
                break
        rise_us = rise_samples / self._sample_rate * 1e6

        self.meas_table.setItem(row, 1, QTableWidgetItem(f"{freq_est:.1f}"))
        self.meas_table.setItem(row, 2, QTableWidgetItem(f"{period_ms:.3f}"))
        self.meas_table.setItem(row, 3, QTableWidgetItem(f"{amp_val:.4f}"))
        self.meas_table.setItem(row, 4, QTableWidgetItem(f"{rms:.4f}"))
        self.meas_table.setItem(row, 5, QTableWidgetItem(f"{vpp:.4f}"))
        self.meas_table.setItem(row, 6, QTableWidgetItem(f"{rise_us:.1f}"))

    # --------------------------------------------------------- Display
    def _update_display(self):
        tb_ms = self.timebase.value()
        total_time = tb_ms * 10 / 1000.0  # 10 divisions
        t = np.linspace(self._time_offset, self._time_offset + total_time, self._num_points)
        self._time_offset += total_time * 0.5  # scroll forward

        ch_data = {}
        for i, ch in enumerate(self.channels):
            params = ch.get_params()
            if params["enabled"]:
                ch_data[i] = self._generate_waveform(t, params)

        # Triggering
        trig_idx = 0
        trig_ch = self.trigger_src.currentIndex()
        if trig_ch in ch_data:
            trig_idx = self._apply_trigger(t, ch_data[trig_ch])

        self._last_data = {"t": t, "channels": ch_data}

        # --- Time domain plot ---
        self.ax_time.clear()
        self._style_axis(self.ax_time, "Time (ms)", "Voltage (V)")
        t_ms = (t - t[trig_idx]) * 1000
        vdiv = self.v_per_div.value()
        for i, y in ch_data.items():
            self.ax_time.plot(t_ms, y, color=CHANNEL_COLORS[i], linewidth=1.0,
                              label=CHANNEL_NAMES[i], alpha=0.9)
        math_result = self._compute_math(ch_data)
        if math_result is not None:
            self.ax_time.plot(t_ms, math_result, color="#FF00FF", linewidth=1.2,
                              label="Math", linestyle="--")
        # Trigger line
        self.ax_time.axhline(self.trigger_level.value(), color="#FF4444",
                             linewidth=0.7, linestyle=":", alpha=0.6)
        self.ax_time.set_ylim(-vdiv * 5, vdiv * 5)
        self.ax_time.set_xlim(t_ms[0], t_ms[-1])
        self.ax_time.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
                            edgecolor="#444466", labelcolor="white")
        self.canvas_time.draw_idle()

        # --- FFT / Spectrum ---
        self.ax_fft.clear()
        self._style_axis(self.ax_fft, "Frequency (Hz)", "Magnitude (dB)")
        dt = total_time / self._num_points
        freqs = np.fft.rfftfreq(self._num_points, d=dt)
        for i, y in ch_data.items():
            spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            spectrum_db = 20 * np.log10(spectrum + 1e-12)
            self.ax_fft.plot(freqs, spectrum_db, color=CHANNEL_COLORS[i],
                             linewidth=0.8, label=CHANNEL_NAMES[i])
        self.ax_fft.set_xlim(0, min(freqs[-1], 20000))
        self.ax_fft.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
                           edgecolor="#444466", labelcolor="white")
        self.canvas_fft.draw_idle()

        # --- Lissajous ---
        if 0 in ch_data and 1 in ch_data:
            self.ax_lissa.clear()
            self._style_axis(self.ax_lissa, "CH1 (V)", "CH2 (V)")
            self.ax_lissa.plot(ch_data[0], ch_data[1], color="#00FFAA",
                               linewidth=0.8, alpha=0.8)
            self.ax_lissa.set_aspect("equal", adjustable="datalim")
            self.canvas_lissa.draw_idle()

        # Measurements
        for i in range(4):
            self._measure_channel(ch_data.get(i), i)

    def _on_harmonic_changed(self, _=None):
        """Update Fourier synthesis display."""
        for i, (slider, lbl) in enumerate(zip(self.harmonic_sliders, self.harmonic_labels)):
            lbl.setText(f"H{i + 1}: {slider.value()}%")

        t = np.linspace(0, 0.01, 2048)
        y = np.zeros_like(t)
        base_freq = self.channels[0].frequency.value()
        for i, slider in enumerate(self.harmonic_sliders):
            coeff = slider.value() / 100.0
            if coeff > 0:
                y += coeff * np.sin(2 * np.pi * base_freq * (i + 1) * t)

        self.ax_fourier.clear()
        self._style_axis(self.ax_fourier, "Time (ms)", "Amplitude")
        self.ax_fourier.plot(t * 1000, y, color="#00FF88", linewidth=1.0)
        self.ax_fourier.set_xlim(0, t[-1] * 1000)
        self.canvas_fourier.draw_idle()

    # --------------------------------------------------------- Run / Stop
    def run(self):
        if self._running:
            return
        self._running = True
        self._time_offset = 0.0
        self._timer.start(33)  # ~30 fps
        self._emit_log("Waveform generator started.")

    def _stop(self):
        self._running = False
        self._timer.stop()
        self._emit_log("Waveform generator stopped.")

    # --------------------------------------------------------- Export
    def export(self):
        if not self._last_data:
            self._emit_log("No data to export. Run the generator first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Waveform", "", "CSV Files (*.csv);;PNG Image (*.png)"
        )
        if not path:
            return

        t = self._last_data["t"]
        ch = self._last_data["channels"]

        if path.lower().endswith(".csv"):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["Time (s)"] + [CHANNEL_NAMES[i] for i in sorted(ch.keys())]
                writer.writerow(header)
                for idx in range(len(t)):
                    row = [f"{t[idx]:.8f}"]
                    for ci in sorted(ch.keys()):
                        row.append(f"{ch[ci][idx]:.6f}")
                    writer.writerow(row)
            self._emit_log(f"Waveform data exported to {path}")

        elif path.lower().endswith(".png"):
            self.fig_time.savefig(path, dpi=150, facecolor=self.fig_time.get_facecolor(),
                                  bbox_inches="tight")
            self._emit_log(f"Waveform plot exported to {path}")
        else:
            self._emit_log("Unsupported export format.")
