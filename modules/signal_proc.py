"""
Signal Processing Widget for PyQt5 Scientific Suite.

Provides interactive signal generation, FFT analysis, filtering,
windowing, spectrogram display, convolution, and correlation tools
with real-time matplotlib plotting.
"""

try:
    import wave
except ImportError:
    wave = None
import struct as _struct

import numpy as np
from numpy import pi
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq, ifft
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QTabWidget, QLineEdit, QCheckBox, QSplitter, QFileDialog,
    QMessageBox, QFrame, QSizePolicy, QTextEdit, QScrollArea,
    QDialog, QDialogButtonBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


class SignalProcessingWidget(QWidget):
    """Main widget for interactive signal processing and analysis."""

    signal_updated = pyqtSignal()

    # ------------------------------------------------------------------ #
    #  Signal type / filter catalogues                                    #
    # ------------------------------------------------------------------ #
    SIGNAL_TYPES = [
        "Sine", "Square", "Sawtooth", "Chirp",
        "White Noise", "Pink Noise", "Pulse", "Custom Formula",
        "Sweep Sine", "Impulse", "Step", "MLS"
    ]
    FILTER_TYPES = ["Low-pass", "High-pass", "Band-pass", "Band-stop"]
    FILTER_DESIGNS = ["Butterworth", "Chebyshev Type I", "Chebyshev Type II", "Bessel"]
    WINDOW_TYPES = ["Rectangular", "Hanning", "Hamming", "Blackman", "Kaiser"]
    FFT_MODES = ["Magnitude Spectrum", "Phase Spectrum", "Power Spectral Density"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._time = None
        self._signal = None
        self._filtered_signal = None
        self._sample_rate = 1000.0
        self._duration = 1.0
        self._amplitude = 1.0
        self._frequency = 10.0
        self._secondary_signal = None
        self._init_ui()
        self._connect_signals()
        self._generate_signal()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def set_logger(self, fn):
        """Attach an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def load_file(self, path: str):
        """Load a signal from a text/csv/npy file at *path*."""
        self._log(f"Loading signal from {path}")
        try:
            if path.endswith(".npy"):
                data = np.load(path)
            else:
                data = np.loadtxt(path, delimiter=",")
            if data.ndim == 2:
                if data.shape[1] >= 2:
                    self._time = data[:, 0]
                    self._signal = data[:, 1]
                    self._sample_rate = 1.0 / np.mean(np.diff(self._time))
                else:
                    self._signal = data[:, 0]
                    self._sample_rate = float(self.spin_sample_rate.value())
                    self._time = np.arange(len(self._signal)) / self._sample_rate
            else:
                self._signal = data
                self._sample_rate = float(self.spin_sample_rate.value())
                self._time = np.arange(len(self._signal)) / self._sample_rate
            self._duration = self._time[-1] - self._time[0]
            self.spin_duration.setValue(self._duration)
            self._filtered_signal = None
            self._update_all_plots()
            self._log(f"Loaded {len(self._signal)} samples at {self._sample_rate:.1f} Hz")
        except Exception as exc:
            self._log(f"Error loading file: {exc}")
            QMessageBox.warning(self, "Load Error", str(exc))

    def run(self):
        """Generate the signal and refresh all plots (convenience entry-point)."""
        self._generate_signal()

    # ------------------------------------------------------------------ #
    #  UI construction                                                    #
    # ------------------------------------------------------------------ #
    def _init_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # --- Left: control panel inside a scroll area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(420)
        ctrl_container = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_container)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)

        ctrl_layout.addWidget(self._build_signal_group())
        ctrl_layout.addWidget(self._build_filter_group())
        ctrl_layout.addWidget(self._build_fir_design_group())
        ctrl_layout.addWidget(self._build_window_group())
        ctrl_layout.addWidget(self._build_fft_group())
        ctrl_layout.addWidget(self._build_modulation_group())
        ctrl_layout.addWidget(self._build_noise_reduction_group())
        ctrl_layout.addWidget(self._build_operations_group())
        ctrl_layout.addWidget(self._build_actions_group())
        ctrl_layout.addStretch()
        scroll.setWidget(ctrl_container)
        splitter.addWidget(scroll)

        # --- Right: plots ---
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        plot_layout.addWidget(self.tabs)

        self._fig_time, self._ax_time, self._canvas_time = self._make_plot_tab("Time Domain")
        self._fig_freq, self._ax_freq, self._canvas_freq = self._make_plot_tab("Frequency Domain")
        self._fig_spec, self._ax_spec, self._canvas_spec = self._make_plot_tab("Spectrogram")
        self._fig_ops, self._ax_ops, self._canvas_ops = self._make_plot_tab("Operations")
        self._fig_fir, self._ax_fir, self._canvas_fir = self._make_plot_tab("FIR Design")
        self._fig_ceps, self._ax_ceps, self._canvas_ceps = self._make_plot_tab("Cepstrum")

        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Log output at the bottom
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        root.addWidget(self.log_output)

    # -- group builders --------------------------------------------------
    def _build_signal_group(self) -> QGroupBox:
        grp = QGroupBox("Signal Generation")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Type:"), 0, 0)
        self.combo_signal = QComboBox()
        self.combo_signal.addItems(self.SIGNAL_TYPES)
        lay.addWidget(self.combo_signal, 0, 1)

        lay.addWidget(QLabel("Frequency (Hz):"), 1, 0)
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.01, 100000)
        self.spin_freq.setValue(10.0)
        self.spin_freq.setDecimals(2)
        lay.addWidget(self.spin_freq, 1, 1)

        lay.addWidget(QLabel("Amplitude:"), 2, 0)
        self.spin_amplitude = QDoubleSpinBox()
        self.spin_amplitude.setRange(0.001, 1000)
        self.spin_amplitude.setValue(1.0)
        self.spin_amplitude.setDecimals(3)
        lay.addWidget(self.spin_amplitude, 2, 1)

        lay.addWidget(QLabel("Sample Rate (Hz):"), 3, 0)
        self.spin_sample_rate = QDoubleSpinBox()
        self.spin_sample_rate.setRange(1, 1e7)
        self.spin_sample_rate.setValue(1000)
        self.spin_sample_rate.setDecimals(0)
        lay.addWidget(self.spin_sample_rate, 3, 1)

        lay.addWidget(QLabel("Duration (s):"), 4, 0)
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.001, 3600)
        self.spin_duration.setValue(1.0)
        self.spin_duration.setDecimals(3)
        lay.addWidget(self.spin_duration, 4, 1)

        lay.addWidget(QLabel("Phase (deg):"), 5, 0)
        self.spin_phase = QDoubleSpinBox()
        self.spin_phase.setRange(-360, 360)
        self.spin_phase.setValue(0)
        lay.addWidget(self.spin_phase, 5, 1)

        lay.addWidget(QLabel("DC Offset:"), 6, 0)
        self.spin_dc = QDoubleSpinBox()
        self.spin_dc.setRange(-1000, 1000)
        self.spin_dc.setValue(0)
        lay.addWidget(self.spin_dc, 6, 1)

        lay.addWidget(QLabel("Chirp End Freq (Hz):"), 7, 0)
        self.spin_chirp_end = QDoubleSpinBox()
        self.spin_chirp_end.setRange(0.01, 100000)
        self.spin_chirp_end.setValue(100.0)
        lay.addWidget(self.spin_chirp_end, 7, 1)

        lay.addWidget(QLabel("Custom Formula:"), 8, 0)
        self.edit_formula = QLineEdit("np.sin(2*pi*5*t) + 0.5*np.sin(2*pi*20*t)")
        lay.addWidget(self.edit_formula, 8, 1)

        self.chk_add_noise = QCheckBox("Add Gaussian noise")
        lay.addWidget(self.chk_add_noise, 9, 0, 1, 2)

        lay.addWidget(QLabel("Noise Std Dev:"), 10, 0)
        self.spin_noise_std = QDoubleSpinBox()
        self.spin_noise_std.setRange(0, 100)
        self.spin_noise_std.setValue(0.1)
        self.spin_noise_std.setDecimals(4)
        lay.addWidget(self.spin_noise_std, 10, 1)

        return grp

    def _build_filter_group(self) -> QGroupBox:
        grp = QGroupBox("Filtering")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Filter Type:"), 0, 0)
        self.combo_filt_type = QComboBox()
        self.combo_filt_type.addItems(self.FILTER_TYPES)
        lay.addWidget(self.combo_filt_type, 0, 1)

        lay.addWidget(QLabel("Design:"), 1, 0)
        self.combo_filt_design = QComboBox()
        self.combo_filt_design.addItems(self.FILTER_DESIGNS)
        lay.addWidget(self.combo_filt_design, 1, 1)

        lay.addWidget(QLabel("Order:"), 2, 0)
        self.spin_filt_order = QSpinBox()
        self.spin_filt_order.setRange(1, 20)
        self.spin_filt_order.setValue(4)
        lay.addWidget(self.spin_filt_order, 2, 1)

        lay.addWidget(QLabel("Low Cutoff (Hz):"), 3, 0)
        self.spin_cutoff_low = QDoubleSpinBox()
        self.spin_cutoff_low.setRange(0.01, 500000)
        self.spin_cutoff_low.setValue(50.0)
        lay.addWidget(self.spin_cutoff_low, 3, 1)

        lay.addWidget(QLabel("High Cutoff (Hz):"), 4, 0)
        self.spin_cutoff_high = QDoubleSpinBox()
        self.spin_cutoff_high.setRange(0.01, 500000)
        self.spin_cutoff_high.setValue(200.0)
        lay.addWidget(self.spin_cutoff_high, 4, 1)

        lay.addWidget(QLabel("Ripple (dB):"), 5, 0)
        self.spin_ripple = QDoubleSpinBox()
        self.spin_ripple.setRange(0.01, 20)
        self.spin_ripple.setValue(1.0)
        lay.addWidget(self.spin_ripple, 5, 1)

        self.btn_apply_filter = QPushButton("Apply Filter")
        lay.addWidget(self.btn_apply_filter, 6, 0, 1, 2)

        return grp

    def _build_window_group(self) -> QGroupBox:
        grp = QGroupBox("Windowing")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Window:"), 0, 0)
        self.combo_window = QComboBox()
        self.combo_window.addItems(self.WINDOW_TYPES)
        lay.addWidget(self.combo_window, 0, 1)

        lay.addWidget(QLabel("Kaiser Beta:"), 1, 0)
        self.spin_kaiser_beta = QDoubleSpinBox()
        self.spin_kaiser_beta.setRange(0, 40)
        self.spin_kaiser_beta.setValue(5.0)
        lay.addWidget(self.spin_kaiser_beta, 1, 1)

        self.chk_apply_window = QCheckBox("Apply window before FFT")
        self.chk_apply_window.setChecked(True)
        lay.addWidget(self.chk_apply_window, 2, 0, 1, 2)

        return grp

    def _build_fft_group(self) -> QGroupBox:
        grp = QGroupBox("FFT Analysis")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Display:"), 0, 0)
        self.combo_fft_mode = QComboBox()
        self.combo_fft_mode.addItems(self.FFT_MODES)
        lay.addWidget(self.combo_fft_mode, 0, 1)

        self.chk_log_scale = QCheckBox("Log scale (dB)")
        self.chk_log_scale.setChecked(True)
        lay.addWidget(self.chk_log_scale, 1, 0, 1, 2)

        self.chk_one_sided = QCheckBox("One-sided spectrum")
        self.chk_one_sided.setChecked(True)
        lay.addWidget(self.chk_one_sided, 2, 0, 1, 2)

        lay.addWidget(QLabel("Zero-pad factor:"), 3, 0)
        self.spin_zeropad = QSpinBox()
        self.spin_zeropad.setRange(1, 16)
        self.spin_zeropad.setValue(1)
        lay.addWidget(self.spin_zeropad, 3, 1)

        return grp

    def _build_fir_design_group(self) -> QGroupBox:
        grp = QGroupBox("FIR Filter Design")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Method:"), 0, 0)
        self.combo_fir_method = QComboBox()
        self.combo_fir_method.addItems(["Parks-McClellan (Remez)", "Window Method", "Least Squares"])
        lay.addWidget(self.combo_fir_method, 0, 1)

        lay.addWidget(QLabel("Num Taps:"), 1, 0)
        self.spin_fir_taps = QSpinBox()
        self.spin_fir_taps.setRange(3, 1001)
        self.spin_fir_taps.setValue(51)
        self.spin_fir_taps.setSingleStep(2)
        lay.addWidget(self.spin_fir_taps, 1, 1)

        lay.addWidget(QLabel("Band Edges (Hz):"), 2, 0)
        self.edit_fir_bands = QLineEdit("0, 50, 100, 500")
        self.edit_fir_bands.setToolTip("Comma-separated band edge frequencies")
        lay.addWidget(self.edit_fir_bands, 2, 1)

        lay.addWidget(QLabel("Desired Gains:"), 3, 0)
        self.edit_fir_desired = QLineEdit("1, 0")
        self.edit_fir_desired.setToolTip("Desired gain in each band (one per band pair)")
        lay.addWidget(self.edit_fir_desired, 3, 1)

        self.btn_design_fir = QPushButton("Design FIR Filter")
        lay.addWidget(self.btn_design_fir, 4, 0, 1, 2)

        self.btn_apply_fir = QPushButton("Apply Designed FIR to Signal")
        lay.addWidget(self.btn_apply_fir, 5, 0, 1, 2)

        return grp

    def _build_modulation_group(self) -> QGroupBox:
        grp = QGroupBox("Modulation / Demodulation")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Scheme:"), 0, 0)
        self.combo_mod_scheme = QComboBox()
        self.combo_mod_scheme.addItems(["AM", "FM", "BPSK"])
        lay.addWidget(self.combo_mod_scheme, 0, 1)

        lay.addWidget(QLabel("Carrier Freq (Hz):"), 1, 0)
        self.spin_carrier_freq = QDoubleSpinBox()
        self.spin_carrier_freq.setRange(1, 1e6)
        self.spin_carrier_freq.setValue(100.0)
        lay.addWidget(self.spin_carrier_freq, 1, 1)

        lay.addWidget(QLabel("Mod Index / Dev:"), 2, 0)
        self.spin_mod_index = QDoubleSpinBox()
        self.spin_mod_index.setRange(0.01, 1000)
        self.spin_mod_index.setValue(1.0)
        self.spin_mod_index.setDecimals(3)
        lay.addWidget(self.spin_mod_index, 2, 1)

        self.btn_modulate = QPushButton("Modulate")
        lay.addWidget(self.btn_modulate, 3, 0)

        self.btn_demodulate = QPushButton("Demodulate")
        lay.addWidget(self.btn_demodulate, 3, 1)

        return grp

    def _build_noise_reduction_group(self) -> QGroupBox:
        grp = QGroupBox("Noise Reduction / Cepstrum")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Method:"), 0, 0)
        self.combo_nr_method = QComboBox()
        self.combo_nr_method.addItems(["Wiener Filter", "Spectral Subtraction"])
        lay.addWidget(self.combo_nr_method, 0, 1)

        lay.addWidget(QLabel("Noise Est. Frames:"), 1, 0)
        self.spin_noise_frames = QSpinBox()
        self.spin_noise_frames.setRange(1, 100)
        self.spin_noise_frames.setValue(5)
        self.spin_noise_frames.setToolTip("Number of initial frames for noise estimation")
        lay.addWidget(self.spin_noise_frames, 1, 1)

        self.btn_reduce_noise = QPushButton("Reduce Noise")
        lay.addWidget(self.btn_reduce_noise, 2, 0, 1, 2)

        self.btn_cepstrum = QPushButton("Compute Cepstrum")
        lay.addWidget(self.btn_cepstrum, 3, 0, 1, 2)

        return grp

    def _build_operations_group(self) -> QGroupBox:
        grp = QGroupBox("Operations")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Second signal:"), 0, 0)
        self.combo_op_signal = QComboBox()
        self.combo_op_signal.addItems(["Sine", "Square", "Impulse", "Step"])
        lay.addWidget(self.combo_op_signal, 0, 1)

        lay.addWidget(QLabel("Freq (Hz):"), 1, 0)
        self.spin_op_freq = QDoubleSpinBox()
        self.spin_op_freq.setRange(0.01, 100000)
        self.spin_op_freq.setValue(20.0)
        lay.addWidget(self.spin_op_freq, 1, 1)

        self.btn_convolve = QPushButton("Convolve")
        lay.addWidget(self.btn_convolve, 2, 0)

        self.btn_correlate = QPushButton("Cross-Correlate")
        lay.addWidget(self.btn_correlate, 2, 1)

        self.btn_autocorr = QPushButton("Auto-Correlate")
        lay.addWidget(self.btn_autocorr, 3, 0, 1, 2)

        return grp

    def _build_actions_group(self) -> QGroupBox:
        grp = QGroupBox("Actions")
        lay = QVBoxLayout(grp)

        self.btn_generate = QPushButton("Generate Signal")
        self.btn_generate.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.btn_generate)

        self.btn_load = QPushButton("Load from File...")
        lay.addWidget(self.btn_load)

        self.btn_load_wav = QPushButton("Import WAV...")
        lay.addWidget(self.btn_load_wav)

        self.btn_export = QPushButton("Export Signal...")
        lay.addWidget(self.btn_export)

        self.btn_export_wav = QPushButton("Export as WAV...")
        lay.addWidget(self.btn_export_wav)

        self.btn_clear = QPushButton("Clear Plots")
        lay.addWidget(self.btn_clear)

        self.btn_copy_plot = QPushButton("Copy Plot")
        self.btn_copy_plot.setToolTip("Copy current plot to clipboard as image")
        self.btn_copy_plot.clicked.connect(self._copy_plot_to_clipboard)
        lay.addWidget(self.btn_copy_plot)

        return grp

    def _make_plot_tab(self, title):
        fig = Figure(figsize=(6, 4), tight_layout=True)
        style_figure(fig)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(toolbar)
        vbox.addWidget(canvas)
        self.tabs.addTab(container, title)
        return fig, ax, canvas

    # ------------------------------------------------------------------ #
    #  Signal connections                                                  #
    # ------------------------------------------------------------------ #
    def _connect_signals(self):
        self.btn_generate.clicked.connect(self._generate_signal)
        self.btn_apply_filter.clicked.connect(self._apply_filter)
        self.btn_convolve.clicked.connect(self._convolve)
        self.btn_correlate.clicked.connect(self._cross_correlate)
        self.btn_autocorr.clicked.connect(self._auto_correlate)
        self.btn_load.clicked.connect(self._browse_load)
        self.btn_load_wav.clicked.connect(self._import_wav)
        self.btn_export.clicked.connect(self._export_signal)
        self.btn_export_wav.clicked.connect(self._export_wav)
        self.btn_clear.clicked.connect(self._clear_plots)
        self.combo_fft_mode.currentIndexChanged.connect(self._update_freq_plot)
        self.chk_log_scale.stateChanged.connect(self._update_freq_plot)
        self.chk_one_sided.stateChanged.connect(self._update_freq_plot)
        self.combo_window.currentIndexChanged.connect(self._update_freq_plot)
        self.spin_zeropad.valueChanged.connect(self._update_freq_plot)
        # New feature connections
        self.btn_design_fir.clicked.connect(self._design_fir_filter)
        self.btn_apply_fir.clicked.connect(self._apply_fir_filter)
        self.btn_modulate.clicked.connect(self._modulate_signal)
        self.btn_demodulate.clicked.connect(self._demodulate_signal)
        self.btn_reduce_noise.clicked.connect(self._reduce_noise)
        self.btn_cepstrum.clicked.connect(self._compute_cepstrum)

    # ------------------------------------------------------------------ #
    #  Signal generation                                                  #
    # ------------------------------------------------------------------ #
    def _generate_signal(self):
        self._sample_rate = self.spin_sample_rate.value()
        self._duration = self.spin_duration.value()
        self._amplitude = self.spin_amplitude.value()
        self._frequency = self.spin_freq.value()
        phase_rad = np.deg2rad(self.spin_phase.value())
        dc = self.spin_dc.value()
        n_samples = int(self._sample_rate * self._duration)
        self._time = np.linspace(0, self._duration, n_samples, endpoint=False)
        t = self._time
        kind = self.combo_signal.currentText()

        try:
            if kind == "Sine":
                self._signal = self._amplitude * np.sin(2 * pi * self._frequency * t + phase_rad)
            elif kind == "Square":
                self._signal = self._amplitude * sp_signal.square(2 * pi * self._frequency * t + phase_rad)
            elif kind == "Sawtooth":
                self._signal = self._amplitude * sp_signal.sawtooth(2 * pi * self._frequency * t + phase_rad)
            elif kind == "Chirp":
                f1 = self.spin_chirp_end.value()
                self._signal = self._amplitude * sp_signal.chirp(t, self._frequency, self._duration, f1, phi=np.rad2deg(phase_rad))
            elif kind == "White Noise":
                self._signal = self._amplitude * np.random.randn(n_samples)
            elif kind == "Pink Noise":
                self._signal = self._amplitude * self._generate_pink_noise(n_samples)
            elif kind == "Pulse":
                self._signal = self._amplitude * sp_signal.unit_impulse(n_samples, 'mid')
            elif kind == "Custom Formula":
                formula = self.edit_formula.text()
                local_ns = {"np": np, "pi": pi, "t": t, "sin": np.sin,
                            "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}
                self._signal = self._amplitude * np.asarray(eval(formula, {"__builtins__": {}}, local_ns), dtype=float)
            elif kind == "Sweep Sine":
                f0, f1 = self._frequency, self.spin_chirp_end.value()
                self._signal = self._amplitude * sp_signal.chirp(
                    t, f0, self._duration, f1, method='logarithmic',
                    phi=np.rad2deg(phase_rad))
            elif kind == "Impulse":
                self._signal = self._amplitude * sp_signal.unit_impulse(n_samples, 0)
            elif kind == "Step":
                self._signal = self._amplitude * np.ones(n_samples)
                self._signal[:n_samples // 2] = 0.0
            elif kind == "MLS":
                nbits = max(4, min(24, int(np.log2(n_samples))))
                mls = sp_signal.max_len_seq(nbits)[0].astype(float)
                mls = 2.0 * mls - 1.0  # convert to +/-1
                if len(mls) < n_samples:
                    mls = np.tile(mls, n_samples // len(mls) + 1)
                self._signal = self._amplitude * mls[:n_samples]
            else:
                self._signal = np.zeros(n_samples)

            self._signal += dc

            if self.chk_add_noise.isChecked():
                noise_std = self.spin_noise_std.value()
                self._signal += noise_std * np.random.randn(n_samples)

            self._filtered_signal = None
            self._update_all_plots()
            self._log(f"Generated '{kind}': {n_samples} samples, {self._sample_rate:.0f} Hz, {self._duration:.3f} s")
            self.signal_updated.emit()
        except Exception as exc:
            self._log(f"Signal generation error: {exc}")
            QMessageBox.warning(self, "Generation Error", str(exc))

    @staticmethod
    def _generate_pink_noise(n):
        """Generate approximate pink (1/f) noise using the Voss-McCartney algorithm."""
        num_rows = 16
        array = np.empty((n, num_rows))
        array.fill(np.nan)
        array[0, :] = np.random.randn(num_rows)
        array[:, 0] = np.random.randn(n)
        cols = np.random.geometric(0.5, n)
        cols = np.clip(cols, 0, num_rows - 1)
        for i in range(1, n):
            array[i, :] = array[i - 1, :]
            array[i, cols[i]] = np.random.randn()
        total = np.nansum(array, axis=1)
        total -= np.mean(total)
        mx = np.max(np.abs(total))
        if mx > 0:
            total /= mx
        return total

    # ------------------------------------------------------------------ #
    #  Windowing                                                          #
    # ------------------------------------------------------------------ #
    def _get_window(self, n):
        name = self.combo_window.currentText()
        if name == "Hanning":
            return np.hanning(n)
        if name == "Hamming":
            return np.hamming(n)
        if name == "Blackman":
            return np.blackman(n)
        if name == "Kaiser":
            beta = self.spin_kaiser_beta.value()
            return np.kaiser(n, beta)
        return np.ones(n)  # Rectangular

    # ------------------------------------------------------------------ #
    #  FFT helpers                                                        #
    # ------------------------------------------------------------------ #
    def _compute_fft(self, sig):
        n = len(sig)
        zeropad_factor = self.spin_zeropad.value()
        n_fft = n * zeropad_factor
        if self.chk_apply_window.isChecked():
            win = self._get_window(n)
            sig = sig * win
        spectrum = fft(sig, n=n_fft)
        freqs = fftfreq(n_fft, d=1.0 / self._sample_rate)
        return freqs, spectrum

    # ------------------------------------------------------------------ #
    #  Filtering                                                          #
    # ------------------------------------------------------------------ #
    def _apply_filter(self):
        if self._signal is None:
            return
        ftype = self.combo_filt_type.currentText()
        design = self.combo_filt_design.currentText()
        order = self.spin_filt_order.value()
        nyq = self._sample_rate / 2.0

        low = self.spin_cutoff_low.value()
        high = self.spin_cutoff_high.value()
        ripple = self.spin_ripple.value()

        try:
            btype_map = {
                "Low-pass": "low", "High-pass": "high",
                "Band-pass": "band", "Band-stop": "bandstop"
            }
            btype = btype_map[ftype]

            if btype in ("band", "bandstop"):
                Wn = [low / nyq, high / nyq]
            elif btype == "low":
                Wn = low / nyq
            else:
                Wn = low / nyq

            if design == "Butterworth":
                b, a = sp_signal.butter(order, Wn, btype=btype)
            elif design == "Chebyshev Type I":
                b, a = sp_signal.cheby1(order, ripple, Wn, btype=btype)
            elif design == "Chebyshev Type II":
                b, a = sp_signal.cheby2(order, ripple, Wn, btype=btype)
            elif design == "Bessel":
                b, a = sp_signal.bessel(order, Wn, btype=btype, norm='phase')
            else:
                return

            self._filtered_signal = sp_signal.filtfilt(b, a, self._signal)
            self._update_all_plots()
            self._log(f"Applied {design} {ftype} filter (order={order})")
        except Exception as exc:
            self._log(f"Filter error: {exc}")
            QMessageBox.warning(self, "Filter Error", str(exc))

    # ------------------------------------------------------------------ #
    #  Convolution / Correlation                                          #
    # ------------------------------------------------------------------ #
    def _make_secondary_signal(self):
        kind = self.combo_op_signal.currentText()
        freq = self.spin_op_freq.value()
        n = len(self._time)
        t = self._time
        if kind == "Sine":
            return np.sin(2 * pi * freq * t)
        if kind == "Square":
            return sp_signal.square(2 * pi * freq * t)
        if kind == "Impulse":
            return sp_signal.unit_impulse(n, 'mid')
        if kind == "Step":
            s = np.zeros(n)
            s[n // 2:] = 1.0
            return s
        return np.zeros(n)

    def _convolve(self):
        if self._signal is None:
            return
        h = self._make_secondary_signal()
        result = np.convolve(self._signal, h, mode='full')
        t_out = np.arange(len(result)) / self._sample_rate
        self._plot_operation(t_out, result, "Convolution")
        self._log("Computed convolution")

    def _cross_correlate(self):
        if self._signal is None:
            return
        h = self._make_secondary_signal()
        result = np.correlate(self._signal, h, mode='full')
        lags = np.arange(-(len(h) - 1), len(self._signal)) / self._sample_rate
        self._plot_operation(lags, result, "Cross-Correlation")
        self._log("Computed cross-correlation")

    def _auto_correlate(self):
        if self._signal is None:
            return
        result = np.correlate(self._signal, self._signal, mode='full')
        n = len(self._signal)
        lags = np.arange(-(n - 1), n) / self._sample_rate
        self._plot_operation(lags, result, "Auto-Correlation")
        self._log("Computed auto-correlation")

    # ------------------------------------------------------------------ #
    #  Plot updates                                                       #
    # ------------------------------------------------------------------ #
    def _update_all_plots(self):
        self._update_time_plot()
        self._update_freq_plot()
        self._update_spectrogram()

    def _update_time_plot(self):
        ax = self._ax_time
        ax.clear()
        if self._signal is not None:
            ax.plot(self._time, self._signal, linewidth=0.7, label="Signal", color="#1f77b4")
        if self._filtered_signal is not None:
            ax.plot(self._time, self._filtered_signal, linewidth=0.9,
                    label="Filtered", color="#d62728", alpha=0.85)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain")
        ax.grid(True, alpha=0.3)
        self._canvas_time.draw_idle()

    def _update_freq_plot(self):
        ax = self._ax_freq
        ax.clear()
        sig = self._filtered_signal if self._filtered_signal is not None else self._signal
        if sig is None:
            self._canvas_freq.draw_idle()
            return

        freqs, spectrum = self._compute_fft(sig)
        n_fft = len(freqs)
        mode = self.combo_fft_mode.currentText()
        log_scale = self.chk_log_scale.isChecked()
        one_sided = self.chk_one_sided.isChecked()

        if one_sided:
            half = n_fft // 2
            freqs = freqs[:half]
            spectrum = spectrum[:half]

        if mode == "Magnitude Spectrum":
            mag = np.abs(spectrum) / len(sig)
            if one_sided:
                mag[1:] *= 2
            if log_scale:
                mag = 20 * np.log10(mag + 1e-12)
                ylabel = "Magnitude (dB)"
            else:
                ylabel = "Magnitude"
            ax.plot(freqs, mag, linewidth=0.7, color="#2ca02c")
            ax.set_ylabel(ylabel)
        elif mode == "Phase Spectrum":
            phase = np.angle(spectrum, deg=True)
            ax.plot(freqs, phase, linewidth=0.7, color="#9467bd")
            ax.set_ylabel("Phase (degrees)")
        elif mode == "Power Spectral Density":
            psd = (np.abs(spectrum) ** 2) / (len(sig) * self._sample_rate)
            if one_sided:
                psd[1:] *= 2
            if log_scale:
                psd = 10 * np.log10(psd + 1e-12)
                ylabel = "PSD (dB/Hz)"
            else:
                ylabel = "PSD (V^2/Hz)"
            ax.plot(freqs, psd, linewidth=0.7, color="#ff7f0e")
            ax.set_ylabel(ylabel)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(mode)
        ax.grid(True, alpha=0.3)
        self._canvas_freq.draw_idle()

    def _update_spectrogram(self):
        ax = self._ax_spec
        ax.clear()
        sig = self._filtered_signal if self._filtered_signal is not None else self._signal
        if sig is None or len(sig) < 16:
            self._canvas_spec.draw_idle()
            return

        nperseg = min(256, len(sig))
        noverlap = nperseg // 2
        win = self._get_window(nperseg)

        f, t_spec, Sxx = sp_signal.spectrogram(
            sig, fs=self._sample_rate, window=win,
            nperseg=nperseg, noverlap=noverlap, mode='magnitude'
        )
        Sxx_db = 20 * np.log10(Sxx + 1e-12)
        im = ax.pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='inferno')
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Spectrogram (STFT)")

        # Reuse or create colorbar
        if hasattr(self, '_spec_cbar') and self._spec_cbar is not None:
            try:
                self._spec_cbar.remove()
            except Exception:
                pass
        self._spec_cbar = self._fig_spec.colorbar(im, ax=ax, label="dB")
        self._canvas_spec.draw_idle()

    def _plot_operation(self, x, y, title):
        ax = self._ax_ops
        ax.clear()
        ax.plot(x, y, linewidth=0.7, color="#17becf")
        ax.set_xlabel("Time / Lag (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self._canvas_ops.draw_idle()
        self.tabs.setCurrentIndex(3)

    def _clear_plots(self):
        for ax, canvas in [(self._ax_time, self._canvas_time),
                           (self._ax_freq, self._canvas_freq),
                           (self._ax_spec, self._canvas_spec),
                           (self._ax_ops, self._canvas_ops)]:
            ax.clear()
            canvas.draw_idle()
        self._signal = None
        self._filtered_signal = None
        self._time = None
        self._log("Plots cleared")

    # ------------------------------------------------------------------ #
    #  FIR Filter Design                                                  #
    # ------------------------------------------------------------------ #
    _fir_coeffs = None

    def _design_fir_filter(self):
        """Design an FIR filter using the selected method and display response."""
        try:
            num_taps = self.spin_fir_taps.value()
            if num_taps % 2 == 0:
                num_taps += 1  # ensure odd for type I filter
            bands_text = self.edit_fir_bands.text().strip()
            desired_text = self.edit_fir_desired.text().strip()
            bands = [float(x.strip()) for x in bands_text.split(",")]
            desired = [float(x.strip()) for x in desired_text.split(",")]
            nyq = self._sample_rate / 2.0
            method = self.combo_fir_method.currentText()

            if "Parks-McClellan" in method or "Remez" in method:
                bands_norm = [b / nyq for b in bands]
                coeffs = sp_signal.remez(num_taps, bands_norm, desired)
            elif "Window" in method:
                if len(bands) >= 2:
                    cutoff = bands[1] / nyq
                    coeffs = sp_signal.firwin(num_taps, cutoff)
                else:
                    coeffs = sp_signal.firwin(num_taps, 0.5)
            else:  # Least Squares
                bands_norm = [b / nyq for b in bands]
                coeffs = sp_signal.firls(num_taps, bands_norm, desired)

            self._fir_coeffs = coeffs

            # Plot impulse and frequency response
            ax = self._ax_fir
            ax.clear()
            w, h = sp_signal.freqz(coeffs, worN=2048, fs=self._sample_rate)
            ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12), color="#2ca02c", lw=1.5)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title(f"FIR Filter ({method}, {num_taps} taps)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=-80)

            # Inset: impulse response
            inset = ax.inset_axes([0.55, 0.55, 0.4, 0.35])
            inset.stem(coeffs, linefmt="C0-", markerfmt="C0.", basefmt=" ")
            inset.set_title("Impulse Response", fontsize=8)
            inset.tick_params(labelsize=7)

            self._canvas_fir.draw_idle()
            self.tabs.setCurrentIndex(4)
            self._log(f"Designed FIR filter: {method}, {num_taps} taps")
        except Exception as exc:
            self._log(f"FIR design error: {exc}")
            QMessageBox.warning(self, "FIR Design Error", str(exc))

    def _apply_fir_filter(self):
        """Apply the most recently designed FIR filter to the current signal."""
        if self._signal is None:
            return
        if self._fir_coeffs is None:
            QMessageBox.information(self, "No Filter", "Design an FIR filter first.")
            return
        try:
            self._filtered_signal = sp_signal.lfilter(self._fir_coeffs, 1.0, self._signal)
            self._update_all_plots()
            self._log("Applied designed FIR filter to signal")
        except Exception as exc:
            self._log(f"FIR apply error: {exc}")

    # ------------------------------------------------------------------ #
    #  Modulation / Demodulation                                          #
    # ------------------------------------------------------------------ #
    def _modulate_signal(self):
        """Modulate the current signal using the selected scheme."""
        if self._signal is None:
            return
        try:
            scheme = self.combo_mod_scheme.currentText()
            fc = self.spin_carrier_freq.value()
            mod_idx = self.spin_mod_index.value()
            t = self._time
            msg = self._signal / (np.max(np.abs(self._signal)) + 1e-12)

            if scheme == "AM":
                carrier = np.cos(2 * pi * fc * t)
                modulated = (1.0 + mod_idx * msg) * carrier
            elif scheme == "FM":
                deviation = mod_idx * fc
                phase = 2 * pi * fc * t + 2 * pi * deviation * np.cumsum(msg) / self._sample_rate
                modulated = np.cos(phase)
            elif scheme == "BPSK":
                bits = (msg >= 0).astype(float) * 2 - 1
                modulated = bits * np.cos(2 * pi * fc * t)
            else:
                return

            self._filtered_signal = modulated
            self._update_all_plots()
            self._log(f"Modulated signal: {scheme}, carrier={fc:.0f} Hz")
        except Exception as exc:
            self._log(f"Modulation error: {exc}")
            QMessageBox.warning(self, "Modulation Error", str(exc))

    def _demodulate_signal(self):
        """Demodulate the filtered signal (or signal) using the selected scheme."""
        sig = self._filtered_signal if self._filtered_signal is not None else self._signal
        if sig is None:
            return
        try:
            scheme = self.combo_mod_scheme.currentText()
            fc = self.spin_carrier_freq.value()
            t = self._time

            if scheme == "AM":
                analytic = sp_signal.hilbert(sig)
                envelope = np.abs(analytic)
                demod = envelope - np.mean(envelope)
            elif scheme == "FM":
                analytic = sp_signal.hilbert(sig)
                inst_phase = np.unwrap(np.angle(analytic))
                demod = np.diff(inst_phase) * self._sample_rate / (2 * pi)
                demod = np.append(demod, demod[-1])
                demod -= np.mean(demod)
            elif scheme == "BPSK":
                ref = np.cos(2 * pi * fc * t)
                product = sig * ref
                # Low-pass to recover
                nyq = self._sample_rate / 2.0
                cutoff = min(fc * 0.3, nyq * 0.9)
                b, a = sp_signal.butter(4, cutoff / nyq, btype='low')
                demod = sp_signal.filtfilt(b, a, product)
            else:
                return

            self._plot_operation(self._time[:len(demod)], demod, f"{scheme} Demodulated")
            self._log(f"Demodulated signal: {scheme}")
        except Exception as exc:
            self._log(f"Demodulation error: {exc}")
            QMessageBox.warning(self, "Demodulation Error", str(exc))

    # ------------------------------------------------------------------ #
    #  Noise Reduction                                                    #
    # ------------------------------------------------------------------ #
    def _reduce_noise(self):
        """Apply noise reduction to the current signal."""
        if self._signal is None:
            return
        try:
            method = self.combo_nr_method.currentText()
            n_frames = self.spin_noise_frames.value()
            nperseg = min(256, len(self._signal) // 4)
            noverlap = nperseg // 2

            if method == "Wiener Filter":
                # Simple Wiener filter using scipy
                try:
                    self._filtered_signal = sp_signal.wiener(self._signal, mysize=nperseg)
                except Exception:
                    # Fallback: frequency-domain Wiener
                    self._filtered_signal = self._spectral_subtraction(
                        self._signal, nperseg, noverlap, n_frames)
            else:  # Spectral Subtraction
                self._filtered_signal = self._spectral_subtraction(
                    self._signal, nperseg, noverlap, n_frames)

            self._update_all_plots()
            self._log(f"Applied noise reduction: {method}")
        except Exception as exc:
            self._log(f"Noise reduction error: {exc}")
            QMessageBox.warning(self, "Noise Reduction Error", str(exc))

    @staticmethod
    def _spectral_subtraction(sig, nperseg, noverlap, n_noise_frames):
        """Frequency-domain spectral subtraction noise reduction."""
        hop = nperseg - noverlap
        window = np.hanning(nperseg)
        n_frames_total = (len(sig) - nperseg) // hop + 1
        if n_frames_total < 2:
            return sig.copy()

        # Estimate noise spectrum from first n_noise_frames frames
        noise_spec = np.zeros(nperseg)
        n_est = min(n_noise_frames, n_frames_total)
        for i in range(n_est):
            frame = sig[i * hop: i * hop + nperseg] * window
            noise_spec += np.abs(fft(frame)) ** 2
        noise_spec /= n_est

        # Subtract noise and reconstruct
        output = np.zeros(len(sig))
        win_sum = np.zeros(len(sig))
        for i in range(n_frames_total):
            start = i * hop
            frame = sig[start: start + nperseg] * window
            spec = fft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)
            clean_mag = np.maximum(mag ** 2 - noise_spec, 0.0) ** 0.5
            clean = np.real(ifft(clean_mag * np.exp(1j * phase)))
            output[start: start + nperseg] += clean * window
            win_sum[start: start + nperseg] += window ** 2
        win_sum = np.maximum(win_sum, 1e-12)
        return output / win_sum

    # ------------------------------------------------------------------ #
    #  Cepstral Analysis                                                  #
    # ------------------------------------------------------------------ #
    def _compute_cepstrum(self):
        """Compute the real cepstrum and display it for pitch detection."""
        if self._signal is None:
            return
        try:
            sig = self._filtered_signal if self._filtered_signal is not None else self._signal
            spectrum = fft(sig)
            log_spectrum = np.log(np.abs(spectrum) + 1e-12)
            cepstrum = np.real(ifft(log_spectrum))
            n = len(cepstrum)
            quefrency = np.arange(n) / self._sample_rate

            # Find pitch: look for peak in cepstrum between plausible pitch periods
            min_f0, max_f0 = 50.0, 800.0  # Hz
            min_q = int(self._sample_rate / max_f0)
            max_q = min(int(self._sample_rate / min_f0), n // 2)
            if min_q < max_q:
                search_region = cepstrum[min_q:max_q]
                peak_idx = np.argmax(search_region) + min_q
                pitch_freq = self._sample_rate / peak_idx if peak_idx > 0 else 0
            else:
                pitch_freq = 0

            ax = self._ax_ceps
            ax.clear()
            half = n // 2
            ax.plot(quefrency[:half] * 1000, cepstrum[:half], color="#9467bd", lw=0.8)
            ax.set_xlabel("Quefrency (ms)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Real Cepstrum (Estimated Pitch: {pitch_freq:.1f} Hz)")
            ax.grid(True, alpha=0.3)
            if pitch_freq > 0:
                q_ms = (1.0 / pitch_freq) * 1000
                ax.axvline(q_ms, color="red", ls="--", alpha=0.7, label=f"Pitch: {pitch_freq:.1f} Hz")
                ax.legend(fontsize=8)
            self._canvas_ceps.draw_idle()
            self.tabs.setCurrentIndex(5)
            self._log(f"Computed cepstrum. Estimated pitch: {pitch_freq:.1f} Hz")
        except Exception as exc:
            self._log(f"Cepstrum error: {exc}")
            QMessageBox.warning(self, "Cepstrum Error", str(exc))

    # ------------------------------------------------------------------ #
    #  WAV File I/O                                                       #
    # ------------------------------------------------------------------ #
    def _import_wav(self):
        """Import a WAV audio file as the current signal."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import WAV", "", "WAV Files (*.wav)")
        if not path:
            return
        try:
            with wave.open(path, "rb") as wf:
                n_channels = wf.getnchannels()
                samp_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if samp_width == 1:
                fmt = f"<{n_frames * n_channels}B"
                data = np.array(_struct.unpack(fmt, raw), dtype=float) - 128.0
                data /= 128.0
            elif samp_width == 2:
                fmt = f"<{n_frames * n_channels}h"
                data = np.array(_struct.unpack(fmt, raw), dtype=float) / 32768.0
            elif samp_width == 4:
                fmt = f"<{n_frames * n_channels}i"
                data = np.array(_struct.unpack(fmt, raw), dtype=float) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {samp_width}")

            if n_channels > 1:
                data = data.reshape(-1, n_channels)
                data = data[:, 0]  # take first channel

            self._signal = data
            self._sample_rate = float(framerate)
            self._duration = len(data) / self._sample_rate
            self._time = np.arange(len(data)) / self._sample_rate
            self._filtered_signal = None
            self.spin_sample_rate.setValue(self._sample_rate)
            self.spin_duration.setValue(self._duration)
            self._update_all_plots()
            self._log(f"Imported WAV: {path} ({framerate} Hz, {n_channels} ch, {n_frames} frames)")
        except Exception as exc:
            self._log(f"WAV import error: {exc}")
            QMessageBox.warning(self, "WAV Import Error", str(exc))

    def _export_wav(self):
        """Export the current signal as a 16-bit WAV file."""
        sig = self._filtered_signal if self._filtered_signal is not None else self._signal
        if sig is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", "", "WAV Files (*.wav)")
        if not path:
            return
        try:
            # Normalize to 16-bit range
            max_val = np.max(np.abs(sig)) + 1e-12
            normalized = (sig / max_val * 32767).astype(np.int16)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(self._sample_rate))
                wf.writeframes(normalized.tobytes())
            self._log(f"Exported WAV: {path}")
        except Exception as exc:
            self._log(f"WAV export error: {exc}")
            QMessageBox.warning(self, "WAV Export Error", str(exc))

    # ------------------------------------------------------------------ #
    #  File I/O                                                           #
    # ------------------------------------------------------------------ #
    def _browse_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Signal", "",
            "All Supported (*.csv *.txt *.npy);;CSV (*.csv);;Text (*.txt);;NumPy (*.npy)"
        )
        if path:
            self.load_file(path)

    def _export_signal(self):
        if self._signal is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Signal", "",
            "CSV (*.csv);;NumPy (*.npy)"
        )
        if not path:
            return
        try:
            sig = self._filtered_signal if self._filtered_signal is not None else self._signal
            if path.endswith(".npy"):
                np.save(path, np.column_stack([self._time, sig]))
            else:
                np.savetxt(path, np.column_stack([self._time, sig]),
                           delimiter=",", header="time,signal", comments="")
            self._log(f"Exported signal to {path}")
        except Exception as exc:
            self._log(f"Export error: {exc}")
            QMessageBox.warning(self, "Export Error", str(exc))

    # ------------------------------------------------------------------ #
    #  Clipboard                                                          #
    # ------------------------------------------------------------------ #
    def _copy_plot_to_clipboard(self):
        """Copy current plot to clipboard as image."""
        import io
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QImage
        # Determine which figure is currently visible based on the active tab
        fig_map = {
            0: self._fig_time,
            1: self._fig_freq,
            2: self._fig_spec,
            3: self._fig_ops,
            4: self._fig_fir,
            5: self._fig_ceps,
        }
        _figure = fig_map.get(self.tabs.currentIndex(), self._fig_time)
        buf = io.BytesIO()
        _figure.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                        facecolor=_figure.get_facecolor())
        buf.seek(0)
        img = QImage()
        img.loadFromData(buf.read())
        QApplication.clipboard().setImage(img)
        self._log("Plot copied to clipboard.")

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #
    def _log(self, msg: str):
        self.log_output.append(msg)
        if self._logger is not None:
            self._logger(msg)
