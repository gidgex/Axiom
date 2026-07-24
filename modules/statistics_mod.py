"""
Statistics Module for QuantumRes Scientific Suite.

Provides descriptive statistics, hypothesis testing, correlation analysis,
probability distribution tools, and regression modeling in a PyQt5 widget.
"""

import numpy as np
import datetime
import io
import base64
from scipy import stats as sp_stats
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QLabel,
    QPushButton, QComboBox, QLineEdit, QGroupBox, QFormLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGridLayout, QHeaderView, QMessageBox, QPlainTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass


class StatisticsWidget(QWidget):
    """Main statistics widget offering R/SPSS-like analytical capabilities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._init_ui()

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    def set_logger(self, fn):
        """Attach an external logging callback ``fn(message)``."""
        self._logger = fn

    def _log(self, msg):
        if self._logger:
            self._logger(msg)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Vertical)

        # --- Data input area ---
        data_group = QGroupBox("Data Input")
        data_lay = QVBoxLayout(data_group)
        hint = QLabel(
            "Enter numeric data separated by commas, spaces, or newlines. "
            "For two-sample tests use the separator '|' between groups."
        )
        hint.setWordWrap(True)
        data_lay.addWidget(hint)
        self.data_edit = QPlainTextEdit()
        self.data_edit.setPlaceholderText(
            "e.g.  1.2, 3.4, 5.6, 7.8  or  group1 | group2"
        )
        self.data_edit.setMaximumHeight(120)
        data_lay.addWidget(self.data_edit)
        splitter.addWidget(data_group)

        # --- Analysis tabs ---
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_descriptive_tab(), "Descriptive")
        self.tabs.addTab(self._build_hypothesis_tab(), "Hypothesis Tests")
        self.tabs.addTab(self._build_correlation_tab(), "Correlation")
        self.tabs.addTab(self._build_distribution_tab(), "Distributions")
        self.tabs.addTab(self._build_regression_tab(), "Regression")
        self.tabs.addTab(self._build_advanced_tab(), "Advanced")
        splitter.addWidget(self.tabs)

        # --- Results display ---
        results_group = QGroupBox("Results")
        res_lay = QVBoxLayout(results_group)
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setFont(QFont("Consolas", 10))
        res_lay.addWidget(self.results_display)
        splitter.addWidget(results_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Data parsing helpers
    # ------------------------------------------------------------------
    def _parse_single(self):
        """Return a single 1-D numpy array from the data input."""
        text = self.data_edit.toPlainText().strip()
        if not text:
            raise ValueError("Data input is empty.")
        text = text.replace("\n", ",").replace("\t", ",").replace(" ", ",")
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        return np.array([float(t) for t in tokens])

    def _parse_groups(self):
        """Return two 1-D numpy arrays split by '|'."""
        text = self.data_edit.toPlainText().strip()
        if "|" not in text:
            raise ValueError("Use '|' to separate two groups.")
        parts = text.split("|")
        if len(parts) != 2:
            raise ValueError("Exactly two groups separated by '|' are required.")
        groups = []
        for p in parts:
            p = p.replace("\n", ",").replace("\t", ",").replace(" ", ",")
            tokens = [t.strip() for t in p.split(",") if t.strip()]
            groups.append(np.array([float(t) for t in tokens]))
        return groups[0], groups[1]

    def _parse_matrix(self):
        """Return a 2-D numpy array (rows = observations, cols = variables)."""
        text = self.data_edit.toPlainText().strip()
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = [t.strip() for t in line.replace("\t", ",").replace(" ", ",").split(",") if t.strip()]
            rows.append([float(t) for t in tokens])
        if not rows:
            raise ValueError("Data input is empty.")
        return np.array(rows)

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    def _show(self, text):
        self.results_display.setPlainText(text)
        self._log(text)

    @staticmethod
    def _clean_num(x, tol=1e-10):
        """Clean floating-point noise for display."""
        if isinstance(x, (float,)):
            rounded = round(x)
            if abs(x - rounded) < tol:
                return int(rounded)
            return round(x, 10)
        return x

    def _fmt(self, val, decimals=6):
        if val is None:
            return "N/A"
        val = self._clean_num(val)
        if isinstance(val, int):
            return str(val)
        return f"{val:.{decimals}g}"

    # ==================================================================
    # 1. DESCRIPTIVE STATISTICS
    # ==================================================================
    def _build_descriptive_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        btn = QPushButton("Compute Descriptive Statistics")
        btn.clicked.connect(self._run_descriptive)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _run_descriptive(self):
        try:
            data = self._parse_single()
        except Exception as e:
            self._show(f"Error: {e}")
            return

        n = len(data)
        mean = np.mean(data)
        median = np.median(data)
        mode_res = sp_stats.mode(data, keepdims=True)
        mode_val = mode_res.mode[0] if mode_res.mode.size > 0 else None
        mode_count = mode_res.count[0] if mode_res.count.size > 0 else 0
        std_pop = np.std(data, ddof=0)
        std_sample = np.std(data, ddof=1) if n > 1 else float("nan")
        var_pop = np.var(data, ddof=0)
        var_sample = np.var(data, ddof=1) if n > 1 else float("nan")
        skew = sp_stats.skew(data, bias=False) if n > 2 else float("nan")
        kurt = sp_stats.kurtosis(data, bias=False) if n > 3 else float("nan")
        sem = sp_stats.sem(data) if n > 1 else float("nan")
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min
        ci95_low, ci95_high = sp_stats.t.interval(
            0.95, df=n - 1, loc=mean, scale=sem
        ) if n > 1 else (float("nan"), float("nan"))

        lines = [
            "=" * 50,
            "       DESCRIPTIVE STATISTICS",
            "=" * 50,
            f"  N                : {n}",
            f"  Mean             : {self._fmt(mean)}",
            f"  Median           : {self._fmt(median)}",
            f"  Mode             : {self._fmt(mode_val)}  (count={mode_count})",
            f"  Std Dev (sample) : {self._fmt(std_sample)}",
            f"  Std Dev (pop)    : {self._fmt(std_pop)}",
            f"  Variance (sample): {self._fmt(var_sample)}",
            f"  Variance (pop)   : {self._fmt(var_pop)}",
            f"  Std Error Mean   : {self._fmt(sem)}",
            f"  Skewness         : {self._fmt(skew)}",
            f"  Kurtosis (excess): {self._fmt(kurt)}",
            "-" * 50,
            f"  Min              : {self._fmt(data_min)}",
            f"  Q1 (25th pctl)   : {self._fmt(q1)}",
            f"  Q2 (50th pctl)   : {self._fmt(q2)}",
            f"  Q3 (75th pctl)   : {self._fmt(q3)}",
            f"  Max              : {self._fmt(data_max)}",
            f"  IQR              : {self._fmt(iqr)}",
            f"  Range            : {self._fmt(data_range)}",
            "-" * 50,
            f"  95% CI for Mean  : [{self._fmt(ci95_low)}, {self._fmt(ci95_high)}]",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    # ==================================================================
    # 2. HYPOTHESIS TESTING
    # ==================================================================
    def _build_hypothesis_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        form = QFormLayout()
        self.hyp_combo = QComboBox()
        self.hyp_combo.addItems([
            "One-Sample t-test",
            "Two-Sample t-test (independent)",
            "Paired t-test",
            "Chi-Square Goodness of Fit",
            "One-Way ANOVA",
            "Mann-Whitney U",
            "Wilcoxon Signed-Rank",
        ])
        form.addRow("Test:", self.hyp_combo)

        self.hyp_mu = QDoubleSpinBox()
        self.hyp_mu.setRange(-1e12, 1e12)
        self.hyp_mu.setDecimals(6)
        self.hyp_mu.setValue(0.0)
        form.addRow("Hypothesized mean (mu0):", self.hyp_mu)

        self.hyp_alpha = QDoubleSpinBox()
        self.hyp_alpha.setRange(0.001, 0.5)
        self.hyp_alpha.setDecimals(3)
        self.hyp_alpha.setValue(0.05)
        self.hyp_alpha.setSingleStep(0.01)
        form.addRow("Significance level (alpha):", self.hyp_alpha)

        self.hyp_equal_var = QCheckBox("Assume equal variances")
        self.hyp_equal_var.setChecked(True)
        form.addRow(self.hyp_equal_var)

        lay.addLayout(form)

        btn = QPushButton("Run Test")
        btn.clicked.connect(self._run_hypothesis)
        lay.addWidget(btn)
        lay.addStretch()
        return w

    def _run_hypothesis(self):
        test = self.hyp_combo.currentText()
        alpha = self.hyp_alpha.value()
        try:
            if test == "One-Sample t-test":
                self._hyp_one_sample_t(alpha)
            elif test == "Two-Sample t-test (independent)":
                self._hyp_two_sample_t(alpha)
            elif test == "Paired t-test":
                self._hyp_paired_t(alpha)
            elif test == "Chi-Square Goodness of Fit":
                self._hyp_chi_square(alpha)
            elif test == "One-Way ANOVA":
                self._hyp_anova(alpha)
            elif test == "Mann-Whitney U":
                self._hyp_mann_whitney(alpha)
            elif test == "Wilcoxon Signed-Rank":
                self._hyp_wilcoxon(alpha)
        except Exception as e:
            self._show(f"Error: {e}")

    def _decision(self, p, alpha):
        return "Reject H0" if p < alpha else "Fail to reject H0"

    def _hyp_one_sample_t(self, alpha):
        data = self._parse_single()
        mu0 = self.hyp_mu.value()
        t_stat, p_val = sp_stats.ttest_1samp(data, mu0)
        lines = [
            "=" * 50,
            "       ONE-SAMPLE t-TEST",
            "=" * 50,
            f"  N          : {len(data)}",
            f"  Mean       : {self._fmt(np.mean(data))}",
            f"  Std Dev    : {self._fmt(np.std(data, ddof=1))}",
            f"  H0: mu = {mu0}",
            f"  t-statistic: {self._fmt(t_stat)}",
            f"  p-value    : {self._fmt(p_val)}",
            f"  alpha      : {alpha}",
            f"  Decision   : {self._decision(p_val, alpha)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    def _hyp_two_sample_t(self, alpha):
        g1, g2 = self._parse_groups()
        equal_var = self.hyp_equal_var.isChecked()
        t_stat, p_val = sp_stats.ttest_ind(g1, g2, equal_var=equal_var)
        label = "Student's" if equal_var else "Welch's"
        lines = [
            "=" * 50,
            f"       TWO-SAMPLE t-TEST ({label})",
            "=" * 50,
            f"  Group 1: N={len(g1)}, Mean={self._fmt(np.mean(g1))}, SD={self._fmt(np.std(g1, ddof=1))}",
            f"  Group 2: N={len(g2)}, Mean={self._fmt(np.mean(g2))}, SD={self._fmt(np.std(g2, ddof=1))}",
            f"  Equal variances assumed: {equal_var}",
            f"  t-statistic: {self._fmt(t_stat)}",
            f"  p-value    : {self._fmt(p_val)}",
            f"  alpha      : {alpha}",
            f"  Decision   : {self._decision(p_val, alpha)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    def _hyp_paired_t(self, alpha):
        g1, g2 = self._parse_groups()
        if len(g1) != len(g2):
            raise ValueError("Paired t-test requires equal-length groups.")
        t_stat, p_val = sp_stats.ttest_rel(g1, g2)
        diff = g1 - g2
        lines = [
            "=" * 50,
            "       PAIRED t-TEST",
            "=" * 50,
            f"  N pairs       : {len(g1)}",
            f"  Mean diff     : {self._fmt(np.mean(diff))}",
            f"  SD diff       : {self._fmt(np.std(diff, ddof=1))}",
            f"  t-statistic   : {self._fmt(t_stat)}",
            f"  p-value       : {self._fmt(p_val)}",
            f"  alpha         : {alpha}",
            f"  Decision      : {self._decision(p_val, alpha)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    def _hyp_chi_square(self, alpha):
        data = self._parse_single()
        observed = data.astype(int)
        chi2, p_val = sp_stats.chisquare(observed)
        lines = [
            "=" * 50,
            "       CHI-SQUARE GOODNESS OF FIT",
            "=" * 50,
            f"  Observed frequencies: {[self._clean_num(v) for v in observed.tolist()]}",
            f"  Expected (uniform)  : {[self._clean_num(v) for v in np.full_like(observed, np.mean(observed), dtype=float).tolist()]}",
            f"  Chi-Square stat     : {self._fmt(chi2)}",
            f"  Degrees of freedom  : {len(observed) - 1}",
            f"  p-value             : {self._fmt(p_val)}",
            f"  alpha               : {alpha}",
            f"  Decision            : {self._decision(p_val, alpha)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    def _hyp_anova(self, alpha):
        text = self.data_edit.toPlainText().strip()
        parts = text.split("|")
        if len(parts) < 2:
            raise ValueError("ANOVA requires at least 2 groups separated by '|'.")
        groups = []
        for p in parts:
            p = p.replace("\n", ",").replace("\t", ",").replace(" ", ",")
            tokens = [t.strip() for t in p.split(",") if t.strip()]
            groups.append(np.array([float(t) for t in tokens]))
        f_stat, p_val = sp_stats.f_oneway(*groups)
        lines = [
            "=" * 50,
            "       ONE-WAY ANOVA",
            "=" * 50,
        ]
        for i, g in enumerate(groups, 1):
            lines.append(f"  Group {i}: N={len(g)}, Mean={self._fmt(np.mean(g))}, SD={self._fmt(np.std(g, ddof=1))}")
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        lines += [
            f"  Number of groups   : {k}",
            f"  df between         : {k - 1}",
            f"  df within          : {n_total - k}",
            f"  F-statistic        : {self._fmt(f_stat)}",
            f"  p-value            : {self._fmt(p_val)}",
            f"  alpha              : {alpha}",
            f"  Decision           : {self._decision(p_val, alpha)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    def _hyp_mann_whitney(self, alpha):
        g1, g2 = self._parse_groups()
        u_stat, p_val = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        lines = [
            "=" * 50,
            "       MANN-WHITNEY U TEST",
            "=" * 50,
            f"  Group 1: N={len(g1)}, Median={self._fmt(np.median(g1))}",
            f"  Group 2: N={len(g2)}, Median={self._fmt(np.median(g2))}",
            f"  U-statistic: {self._fmt(u_stat)}",
            f"  p-value    : {self._fmt(p_val)}",
            f"  alpha      : {alpha}",
            f"  Decision   : {self._decision(p_val, alpha)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    def _hyp_wilcoxon(self, alpha):
        g1, g2 = self._parse_groups()
        if len(g1) != len(g2):
            raise ValueError("Wilcoxon test requires equal-length paired samples.")
        w_stat, p_val = sp_stats.wilcoxon(g1, g2)
        lines = [
            "=" * 50,
            "       WILCOXON SIGNED-RANK TEST",
            "=" * 50,
            f"  N pairs      : {len(g1)}",
            f"  W-statistic  : {self._fmt(w_stat)}",
            f"  p-value      : {self._fmt(p_val)}",
            f"  alpha        : {alpha}",
            f"  Decision     : {self._decision(p_val, alpha)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    # ==================================================================
    # 3. CORRELATION
    # ==================================================================
    def _build_correlation_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        info = QLabel(
            "Enter a matrix: each row is an observation, columns are variables "
            "(comma or space separated, one row per line)."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QFormLayout()
        self.corr_method = QComboBox()
        self.corr_method.addItems(["Pearson", "Spearman", "Kendall"])
        form.addRow("Method:", self.corr_method)
        lay.addLayout(form)

        btn = QPushButton("Compute Correlation Matrix")
        btn.clicked.connect(self._run_correlation)
        lay.addWidget(btn)

        self._corr_fig = Figure(figsize=(5, 4))
        style_figure(self._corr_fig)
        self.corr_canvas = FigureCanvas(self._corr_fig)
        lay.addWidget(self.corr_canvas)
        lay.addStretch()
        return w

    def _run_correlation(self):
        try:
            mat = self._parse_matrix()
        except Exception as e:
            self._show(f"Error: {e}")
            return

        if mat.ndim != 2 or mat.shape[1] < 2:
            self._show("Error: Need at least 2 variables (columns).")
            return

        method = self.corr_method.currentText().lower()
        n_vars = mat.shape[1]
        corr = np.zeros((n_vars, n_vars))
        pvals = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if method == "pearson":
                    r, p = sp_stats.pearsonr(mat[:, i], mat[:, j])
                elif method == "spearman":
                    r, p = sp_stats.spearmanr(mat[:, i], mat[:, j])
                else:
                    r, p = sp_stats.kendalltau(mat[:, i], mat[:, j])
                corr[i, j] = r
                pvals[i, j] = p

        var_names = [f"V{k+1}" for k in range(n_vars)]
        header = "         " + "  ".join(f"{v:>9s}" for v in var_names)
        lines = [
            "=" * max(60, len(header)),
            f"       CORRELATION MATRIX ({method.upper()})",
            "=" * max(60, len(header)),
            header,
        ]
        for i, vn in enumerate(var_names):
            row = f"  {vn:>5s}  " + "  ".join(f"{corr[i,j]:9.4f}" for j in range(n_vars))
            lines.append(row)
        lines.append("")
        lines.append("  p-values:")
        lines.append(header)
        for i, vn in enumerate(var_names):
            row = f"  {vn:>5s}  " + "  ".join(f"{pvals[i,j]:9.6f}" for j in range(n_vars))
            lines.append(row)
        lines.append("=" * max(60, len(header)))
        self._show("\n".join(lines))

        # Heat map
        fig = self.corr_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        cax = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        ax.set_xticklabels(var_names)
        ax.set_yticklabels(var_names)
        for i in range(n_vars):
            for j in range(n_vars):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{method.capitalize()} Correlation")
        fig.tight_layout()
        self.corr_canvas.draw()

    # ==================================================================
    # 4. PROBABILITY DISTRIBUTIONS
    # ==================================================================
    def _build_distribution_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        form = QFormLayout()
        self.dist_combo = QComboBox()
        self.dist_combo.addItems([
            "Normal", "t", "Chi-Square", "F",
            "Poisson", "Binomial", "Exponential",
        ])
        self.dist_combo.currentTextChanged.connect(self._on_dist_changed)
        form.addRow("Distribution:", self.dist_combo)

        self.dist_param1_label = QLabel("mu:")
        self.dist_param1 = QDoubleSpinBox()
        self.dist_param1.setRange(-1e6, 1e6)
        self.dist_param1.setDecimals(4)
        self.dist_param1.setValue(0.0)
        form.addRow(self.dist_param1_label, self.dist_param1)

        self.dist_param2_label = QLabel("sigma:")
        self.dist_param2 = QDoubleSpinBox()
        self.dist_param2.setRange(0.0001, 1e6)
        self.dist_param2.setDecimals(4)
        self.dist_param2.setValue(1.0)
        form.addRow(self.dist_param2_label, self.dist_param2)

        self.dist_plot_type = QComboBox()
        self.dist_plot_type.addItems(["PDF/PMF", "CDF"])
        form.addRow("Plot type:", self.dist_plot_type)

        self.dist_quantile_input = QDoubleSpinBox()
        self.dist_quantile_input.setRange(0.0001, 0.9999)
        self.dist_quantile_input.setDecimals(4)
        self.dist_quantile_input.setValue(0.975)
        self.dist_quantile_input.setSingleStep(0.01)
        form.addRow("Quantile probability:", self.dist_quantile_input)

        lay.addLayout(form)

        btn_row = QHBoxLayout()
        btn_plot = QPushButton("Plot Distribution")
        btn_plot.clicked.connect(self._run_dist_plot)
        btn_row.addWidget(btn_plot)
        btn_quant = QPushButton("Compute Quantile")
        btn_quant.clicked.connect(self._run_quantile)
        btn_row.addWidget(btn_quant)
        lay.addLayout(btn_row)

        self._dist_fig = Figure(figsize=(5, 4))
        style_figure(self._dist_fig)
        self.dist_canvas = FigureCanvas(self._dist_fig)
        lay.addWidget(self.dist_canvas)
        return w

    def _on_dist_changed(self, name):
        param_map = {
            "Normal": ("mu", "sigma", 0, 1),
            "t": ("df", "", 5, 1),
            "Chi-Square": ("df", "", 5, 1),
            "F": ("df1", "df2", 5, 10),
            "Poisson": ("lambda", "", 5, 1),
            "Binomial": ("n", "p", 10, 0.5),
            "Exponential": ("lambda", "", 1, 1),
        }
        p1_label, p2_label, p1_val, p2_val = param_map.get(name, ("p1", "p2", 0, 1))
        self.dist_param1_label.setText(f"{p1_label}:")
        self.dist_param2_label.setText(f"{p2_label}:" if p2_label else "(unused)")
        self.dist_param1.setValue(p1_val)
        self.dist_param2.setValue(p2_val)
        self.dist_param2.setEnabled(bool(p2_label))

    def _get_dist(self):
        name = self.dist_combo.currentText()
        p1 = self.dist_param1.value()
        p2 = self.dist_param2.value()
        if name == "Normal":
            return sp_stats.norm(loc=p1, scale=p2), False
        elif name == "t":
            return sp_stats.t(df=p1), False
        elif name == "Chi-Square":
            return sp_stats.chi2(df=p1), False
        elif name == "F":
            return sp_stats.f(dfn=p1, dfd=p2), False
        elif name == "Poisson":
            return sp_stats.poisson(mu=p1), True
        elif name == "Binomial":
            return sp_stats.binom(n=int(p1), p=p2), True
        elif name == "Exponential":
            return sp_stats.expon(scale=1.0 / p1 if p1 != 0 else 1), False
        raise ValueError(f"Unknown distribution: {name}")

    def _run_dist_plot(self):
        try:
            dist, discrete = self._get_dist()
        except Exception as e:
            self._show(f"Error: {e}")
            return

        fig = self.dist_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        plot_type = self.dist_plot_type.currentText()
        name = self.dist_combo.currentText()

        if discrete:
            lo, hi = dist.ppf(0.001), dist.ppf(0.999)
            lo = max(int(lo), 0)
            hi = int(hi) + 1
            x = np.arange(lo, hi + 1)
            if plot_type == "PDF/PMF":
                y = dist.pmf(x)
                ax.bar(x, y, color="steelblue", alpha=0.8)
                ax.set_ylabel("PMF")
            else:
                y = dist.cdf(x)
                ax.step(x, y, where="mid", color="steelblue")
                ax.set_ylabel("CDF")
        else:
            lo, hi = dist.ppf(0.001), dist.ppf(0.999)
            x = np.linspace(lo, hi, 500)
            if plot_type == "PDF/PMF":
                y = dist.pdf(x)
                ax.plot(x, y, color="steelblue", linewidth=2)
                ax.fill_between(x, y, alpha=0.2, color="steelblue")
                ax.set_ylabel("PDF")
            else:
                y = dist.cdf(x)
                ax.plot(x, y, color="steelblue", linewidth=2)
                ax.set_ylabel("CDF")

        ax.set_title(f"{name} Distribution - {plot_type}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.dist_canvas.draw()
        self._log(f"Plotted {name} {plot_type}")

    def _run_quantile(self):
        try:
            dist, _ = self._get_dist()
        except Exception as e:
            self._show(f"Error: {e}")
            return

        prob = self.dist_quantile_input.value()
        q = dist.ppf(prob)
        name = self.dist_combo.currentText()
        lines = [
            "=" * 50,
            "       QUANTILE CALCULATOR",
            "=" * 50,
            f"  Distribution : {name}",
            f"  Probability  : {prob}",
            f"  Quantile     : {self._fmt(q)}",
            f"  Mean         : {self._fmt(dist.mean())}",
            f"  Variance     : {self._fmt(dist.var())}",
            f"  Std Dev      : {self._fmt(dist.std())}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    # ==================================================================
    # 5. REGRESSION
    # ==================================================================
    def _build_regression_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel(
            "For simple/polynomial: enter two columns (X, Y) per row.\n"
            "For multiple: enter N columns per row; last column is the response Y."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QFormLayout()
        self.reg_type = QComboBox()
        self.reg_type.addItems(["Linear", "Polynomial", "Multiple"])
        form.addRow("Type:", self.reg_type)

        self.reg_degree = QSpinBox()
        self.reg_degree.setRange(1, 10)
        self.reg_degree.setValue(2)
        form.addRow("Poly degree:", self.reg_degree)
        lay.addLayout(form)

        btn = QPushButton("Run Regression")
        btn.clicked.connect(self._run_regression)
        lay.addWidget(btn)

        self._reg_fig = Figure(figsize=(5, 4))
        style_figure(self._reg_fig)
        self.reg_canvas = FigureCanvas(self._reg_fig)
        lay.addWidget(self.reg_canvas)
        return w

    def _run_regression(self):
        try:
            mat = self._parse_matrix()
        except Exception as e:
            self._show(f"Error: {e}")
            return

        reg_type = self.reg_type.currentText()
        try:
            if reg_type == "Linear":
                self._reg_linear(mat)
            elif reg_type == "Polynomial":
                self._reg_polynomial(mat)
            elif reg_type == "Multiple":
                self._reg_multiple(mat)
        except Exception as e:
            self._show(f"Error: {e}")

    def _reg_linear(self, mat):
        if mat.shape[1] < 2:
            raise ValueError("Need at least 2 columns (X, Y).")
        x = mat[:, 0]
        y = mat[:, 1]
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)
        r_sq = r_val ** 2
        y_pred = slope * x + intercept
        residuals = y - y_pred
        n = len(x)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - 2) if n > 2 else float("nan")
        mse = ss_res / (n - 2) if n > 2 else float("nan")
        f_stat = (ss_tot - ss_res) / mse if mse > 0 else float("nan")
        f_pval = 1 - sp_stats.f.cdf(f_stat, 1, n - 2) if n > 2 else float("nan")

        lines = [
            "=" * 50,
            "       LINEAR REGRESSION",
            "=" * 50,
            f"  Y = {self._fmt(slope)} * X + {self._fmt(intercept)}",
            "",
            f"  Slope        : {self._fmt(slope)}",
            f"  Intercept    : {self._fmt(intercept)}",
            f"  Std Err      : {self._fmt(std_err)}",
            f"  R            : {self._fmt(r_val)}",
            f"  R-squared    : {self._fmt(r_sq)}",
            f"  Adj R-squared: {self._fmt(adj_r_sq)}",
            f"  F-statistic  : {self._fmt(f_stat)}",
            f"  p-value (F)  : {self._fmt(f_pval)}",
            f"  p-value (slope): {self._fmt(p_val)}",
            f"  MSE          : {self._fmt(mse)}",
            f"  N            : {n}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

        # Plot
        fig = self.reg_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color="steelblue", label="Data", zorder=5)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label="Fit")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Linear Regression (R\u00b2={r_sq:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.reg_canvas.draw()

    def _reg_polynomial(self, mat):
        if mat.shape[1] < 2:
            raise ValueError("Need at least 2 columns (X, Y).")
        x = mat[:, 0]
        y = mat[:, 1]
        degree = self.reg_degree.value()
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        n = len(x)
        p = degree + 1
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p) if n > p else float("nan")

        lines = [
            "=" * 50,
            f"       POLYNOMIAL REGRESSION (degree={degree})",
            "=" * 50,
        ]
        eq_parts = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power == 0:
                eq_parts.append(f"{c:+.6f}")
            elif power == 1:
                eq_parts.append(f"{c:+.6f}*X")
            else:
                eq_parts.append(f"{c:+.6f}*X^{power}")
        lines.append(f"  Y = {'  '.join(eq_parts)}")
        lines.append("")
        lines.append("  Coefficients:")
        for i, c in enumerate(coeffs):
            lines.append(f"    X^{degree-i} : {self._fmt(c)}")
        lines += [
            f"  R-squared    : {self._fmt(r_sq)}",
            f"  Adj R-squared: {self._fmt(adj_r_sq)}",
            f"  SS residual  : {self._fmt(ss_res)}",
            f"  N            : {n}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

        fig = self.reg_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color="steelblue", label="Data", zorder=5)
        x_line = np.linspace(x.min(), x.max(), 300)
        ax.plot(x_line, poly(x_line), "r-", linewidth=2, label=f"Poly (deg {degree})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Polynomial Regression (R\u00b2={r_sq:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.reg_canvas.draw()

    def _reg_multiple(self, mat):
        if mat.shape[1] < 3:
            raise ValueError("Multiple regression requires at least 3 columns (X1..Xk, Y).")
        X = mat[:, :-1]
        y = mat[:, -1]
        n, k = X.shape
        # Add intercept column
        X_aug = np.column_stack([np.ones(n), X])
        try:
            # OLS: beta = (X'X)^-1 X'y
            XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
            beta = XtX_inv @ X_aug.T @ y
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix - cannot compute regression (collinear predictors?).")

        y_pred = X_aug @ beta
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
        mse = ss_res / (n - k - 1) if n > k + 1 else float("nan")
        f_stat = ((ss_tot - ss_res) / k) / mse if mse > 0 else float("nan")
        f_pval = 1 - sp_stats.f.cdf(f_stat, k, n - k - 1) if n > k + 1 else float("nan")

        # Standard errors and p-values for each coefficient
        se_beta = np.sqrt(np.diag(XtX_inv) * mse) if mse > 0 else np.full(k + 1, float("nan"))
        t_vals = beta / se_beta
        p_vals = 2 * (1 - sp_stats.t.cdf(np.abs(t_vals), df=n - k - 1)) if n > k + 1 else np.full(k + 1, float("nan"))

        lines = [
            "=" * 60,
            "       MULTIPLE REGRESSION",
            "=" * 60,
            f"  N = {n}, Predictors = {k}",
            "",
            "  Coefficients:",
            f"  {'Variable':>12s}  {'Coeff':>12s}  {'Std Err':>12s}  {'t-value':>10s}  {'p-value':>10s}",
            f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}",
            f"  {'(Intercept)':>12s}  {beta[0]:12.6f}  {se_beta[0]:12.6f}  {t_vals[0]:10.4f}  {p_vals[0]:10.6f}",
        ]
        for i in range(k):
            vname = f"X{i+1}"
            lines.append(
                f"  {vname:>12s}  {beta[i+1]:12.6f}  {se_beta[i+1]:12.6f}  {t_vals[i+1]:10.4f}  {p_vals[i+1]:10.6f}"
            )
        lines += [
            "",
            f"  R-squared    : {self._fmt(r_sq)}",
            f"  Adj R-squared: {self._fmt(adj_r_sq)}",
            f"  F-statistic  : {self._fmt(f_stat)}  (df1={k}, df2={n-k-1})",
            f"  p-value (F)  : {self._fmt(f_pval)}",
            f"  MSE          : {self._fmt(mse)}",
            "=" * 60,
        ]
        self._show("\n".join(lines))

        # Plot predicted vs actual
        fig = self.reg_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(y_pred, y, color="steelblue", alpha=0.7)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Multiple Regression: Actual vs Predicted (R\u00b2={r_sq:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.reg_canvas.draw()

    # ==================================================================
    # 6. ADVANCED STATISTICS
    # ==================================================================
    def _build_advanced_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        form = QFormLayout()
        self.adv_combo = QComboBox()
        self.adv_combo.addItems([
            "Power Analysis",
            "Bootstrap CI",
            "Bayesian Inference (Binomial)",
            "Bayesian Inference (Normal)",
            "Cohen's d",
            "Eta-Squared (from ANOVA)",
            "Odds Ratio",
            "QQ Plot & Normality Tests",
            "Generate Statistical Report",
        ])
        form.addRow("Analysis:", self.adv_combo)

        self.adv_param1_label = QLabel("Effect size (d):")
        self.adv_param1 = QDoubleSpinBox()
        self.adv_param1.setRange(0.001, 1000)
        self.adv_param1.setDecimals(4)
        self.adv_param1.setValue(0.5)
        form.addRow(self.adv_param1_label, self.adv_param1)

        self.adv_param2_label = QLabel("Alpha:")
        self.adv_param2 = QDoubleSpinBox()
        self.adv_param2.setRange(0.001, 0.999)
        self.adv_param2.setDecimals(4)
        self.adv_param2.setValue(0.05)
        form.addRow(self.adv_param2_label, self.adv_param2)

        self.adv_param3_label = QLabel("Power:")
        self.adv_param3 = QDoubleSpinBox()
        self.adv_param3.setRange(0.01, 0.999)
        self.adv_param3.setDecimals(4)
        self.adv_param3.setValue(0.80)
        form.addRow(self.adv_param3_label, self.adv_param3)

        self.adv_n_bootstrap = QSpinBox()
        self.adv_n_bootstrap.setRange(100, 100000)
        self.adv_n_bootstrap.setValue(5000)
        form.addRow("Bootstrap iterations:", self.adv_n_bootstrap)

        lay.addLayout(form)

        btn = QPushButton("Run Analysis")
        btn.clicked.connect(self._run_advanced)
        lay.addWidget(btn)

        self._adv_fig = Figure(figsize=(5, 4))
        style_figure(self._adv_fig)
        self.adv_canvas = FigureCanvas(self._adv_fig)
        lay.addWidget(self.adv_canvas)
        return w

    def _run_advanced(self):
        analysis = self.adv_combo.currentText()
        try:
            if analysis == "Power Analysis":
                self._adv_power_analysis()
            elif analysis == "Bootstrap CI":
                self._adv_bootstrap_ci()
            elif analysis == "Bayesian Inference (Binomial)":
                self._adv_bayesian_binomial()
            elif analysis == "Bayesian Inference (Normal)":
                self._adv_bayesian_normal()
            elif analysis == "Cohen's d":
                self._adv_cohens_d()
            elif analysis == "Eta-Squared (from ANOVA)":
                self._adv_eta_squared()
            elif analysis == "Odds Ratio":
                self._adv_odds_ratio()
            elif analysis == "QQ Plot & Normality Tests":
                self._adv_qq_normality()
            elif analysis == "Generate Statistical Report":
                self._adv_generate_report()
        except Exception as e:
            self._show(f"Error: {e}")

    # -- Power Analysis --
    def _adv_power_analysis(self):
        d = self.adv_param1.value()   # effect size
        alpha = self.adv_param2.value()
        power = self.adv_param3.value()

        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)
        n = ((z_alpha + z_beta) / d) ** 2
        n = int(np.ceil(n))

        # Also compute power curve
        ns = np.arange(5, max(n * 3, 200))
        powers = []
        for ni in ns:
            se = 1.0 / np.sqrt(ni)
            z_crit = z_alpha
            achieved_power = 1 - sp_stats.norm.cdf(z_crit - d / se)
            powers.append(achieved_power)

        lines = [
            "=" * 50,
            "       POWER ANALYSIS",
            "=" * 50,
            f"  Effect size (d) : {d}",
            f"  Alpha           : {alpha}",
            f"  Desired power   : {power}",
            f"  Required N      : {n} per group",
            f"  Total N (2-grp) : {n * 2}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

        fig = self.adv_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(ns, powers, 'b-', linewidth=2)
        ax.axhline(power, color='r', linestyle='--', label=f'Target power={power}')
        ax.axvline(n, color='g', linestyle='--', label=f'Required N={n}')
        ax.set_xlabel("Sample size per group")
        ax.set_ylabel("Statistical Power")
        ax.set_title(f"Power Analysis (d={d}, alpha={alpha})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.adv_canvas.draw()

    # -- Bootstrap Confidence Intervals --
    def _adv_bootstrap_ci(self):
        data = self._parse_single()
        n_boot = self.adv_n_bootstrap.value()
        alpha = self.adv_param2.value()
        n = len(data)

        boot_means = np.array([
            np.mean(np.random.choice(data, size=n, replace=True))
            for _ in range(n_boot)
        ])
        ci_low = np.percentile(boot_means, 100 * alpha / 2)
        ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
        boot_se = np.std(boot_means, ddof=1)

        lines = [
            "=" * 50,
            "       BOOTSTRAP CONFIDENCE INTERVAL",
            "=" * 50,
            f"  N observations   : {n}",
            f"  Bootstrap iters  : {n_boot}",
            f"  Confidence level : {(1 - alpha) * 100:.1f}%",
            f"  Sample mean      : {self._fmt(np.mean(data))}",
            f"  Bootstrap SE     : {self._fmt(boot_se)}",
            f"  CI lower         : {self._fmt(ci_low)}",
            f"  CI upper         : {self._fmt(ci_high)}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

        fig = self.adv_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.hist(boot_means, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(ci_low, color='red', linestyle='--', linewidth=2, label=f'CI low={ci_low:.4f}')
        ax.axvline(ci_high, color='red', linestyle='--', linewidth=2, label=f'CI high={ci_high:.4f}')
        ax.axvline(np.mean(data), color='green', linewidth=2, label=f'Sample mean={np.mean(data):.4f}')
        ax.set_xlabel("Bootstrap Mean")
        ax.set_ylabel("Frequency")
        ax.set_title("Bootstrap Distribution of the Mean")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self.adv_canvas.draw()

    # -- Bayesian Inference (Binomial) --
    def _adv_bayesian_binomial(self):
        data = self._parse_single()
        successes = int(np.sum(data > 0))
        n = len(data)
        # Beta prior: param1 = alpha_prior, param2 = beta_prior
        a_prior = max(self.adv_param1.value(), 0.01)
        b_prior = max(self.adv_param3.value(), 0.01)

        a_post = a_prior + successes
        b_post = b_prior + (n - successes)
        posterior = sp_stats.beta(a_post, b_post)
        x = np.linspace(0, 1, 500)

        lines = [
            "=" * 50,
            "       BAYESIAN INFERENCE (BINOMIAL)",
            "=" * 50,
            f"  Observations     : {n}",
            f"  Successes (>0)   : {successes}",
            f"  Prior: Beta({a_prior:.2f}, {b_prior:.2f})",
            f"  Posterior: Beta({a_post:.2f}, {b_post:.2f})",
            f"  Posterior mean   : {self._fmt(posterior.mean())}",
            f"  Posterior median : {self._fmt(posterior.median())}",
            f"  95% Credible Int : [{self._fmt(posterior.ppf(0.025))}, {self._fmt(posterior.ppf(0.975))}]",
            "=" * 50,
        ]
        self._show("\n".join(lines))

        fig = self.adv_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        prior = sp_stats.beta(a_prior, b_prior)
        ax.plot(x, prior.pdf(x), 'b--', linewidth=1.5, label=f'Prior Beta({a_prior:.1f},{b_prior:.1f})')
        ax.plot(x, posterior.pdf(x), 'r-', linewidth=2, label=f'Posterior Beta({a_post:.1f},{b_post:.1f})')
        ax.fill_between(x, posterior.pdf(x), alpha=0.2, color='red')
        ax.set_xlabel("p (probability of success)")
        ax.set_ylabel("Density")
        ax.set_title("Bayesian Binomial Inference")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.adv_canvas.draw()

    # -- Bayesian Inference (Normal) --
    def _adv_bayesian_normal(self):
        data = self._parse_single()
        n = len(data)
        x_bar = np.mean(data)
        s2 = np.var(data, ddof=1)

        # Prior: Normal(mu0, sigma0^2); param1=mu0, param3=sigma0
        mu0 = self.adv_param1.value()
        sigma0 = max(self.adv_param3.value(), 0.01)
        sigma0_sq = sigma0 ** 2

        # Known-variance approximation: posterior for mu
        known_var = s2  # approximate
        post_var = 1.0 / (1.0 / sigma0_sq + n / known_var)
        post_mean = post_var * (mu0 / sigma0_sq + n * x_bar / known_var)
        post_std = np.sqrt(post_var)

        lines = [
            "=" * 50,
            "       BAYESIAN INFERENCE (NORMAL MEAN)",
            "=" * 50,
            f"  N                : {n}",
            f"  Sample mean      : {self._fmt(x_bar)}",
            f"  Sample variance  : {self._fmt(s2)}",
            f"  Prior: N({mu0}, {sigma0}^2)",
            f"  Posterior mean   : {self._fmt(post_mean)}",
            f"  Posterior SD     : {self._fmt(post_std)}",
            f"  95% Credible Int : [{self._fmt(post_mean - 1.96*post_std)}, {self._fmt(post_mean + 1.96*post_std)}]",
            "=" * 50,
        ]
        self._show("\n".join(lines))

        fig = self.adv_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        x_range = np.linspace(post_mean - 4*post_std, post_mean + 4*post_std, 500)
        prior_dist = sp_stats.norm(mu0, sigma0)
        post_dist = sp_stats.norm(post_mean, post_std)
        ax.plot(x_range, prior_dist.pdf(x_range), 'b--', linewidth=1.5, label='Prior')
        ax.plot(x_range, post_dist.pdf(x_range), 'r-', linewidth=2, label='Posterior')
        ax.fill_between(x_range, post_dist.pdf(x_range), alpha=0.2, color='red')
        ax.axvline(x_bar, color='green', linestyle=':', label=f'Sample mean={x_bar:.3f}')
        ax.set_xlabel("mu")
        ax.set_ylabel("Density")
        ax.set_title("Bayesian Normal Mean Inference")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.adv_canvas.draw()

    # -- Cohen's d --
    def _adv_cohens_d(self):
        g1, g2 = self._parse_groups()
        n1, n2 = len(g1), len(g2)
        m1, m2 = np.mean(g1), np.mean(g2)
        s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
        sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (m1 - m2) / sp if sp > 0 else 0.0

        interpretation = "negligible"
        if abs(d) >= 0.2: interpretation = "small"
        if abs(d) >= 0.5: interpretation = "medium"
        if abs(d) >= 0.8: interpretation = "large"

        lines = [
            "=" * 50,
            "       COHEN'S d EFFECT SIZE",
            "=" * 50,
            f"  Group 1: N={n1}, Mean={self._fmt(m1)}, SD={self._fmt(s1)}",
            f"  Group 2: N={n2}, Mean={self._fmt(m2)}, SD={self._fmt(s2)}",
            f"  Pooled SD       : {self._fmt(sp)}",
            f"  Cohen's d       : {self._fmt(d)}",
            f"  |d|             : {self._fmt(abs(d))}",
            f"  Interpretation  : {interpretation}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    # -- Eta-Squared --
    def _adv_eta_squared(self):
        text = self.data_edit.toPlainText().strip()
        parts = text.split("|")
        if len(parts) < 2:
            raise ValueError("Eta-squared requires at least 2 groups separated by '|'.")
        groups = []
        for p in parts:
            p = p.replace("\n", ",").replace("\t", ",").replace(" ", ",")
            tokens = [t.strip() for t in p.split(",") if t.strip()]
            groups.append(np.array([float(t) for t in tokens]))
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        interpretation = "negligible"
        if eta_sq >= 0.01: interpretation = "small"
        if eta_sq >= 0.06: interpretation = "medium"
        if eta_sq >= 0.14: interpretation = "large"

        lines = [
            "=" * 50,
            "       ETA-SQUARED EFFECT SIZE",
            "=" * 50,
            f"  Number of groups : {len(groups)}",
            f"  SS between       : {self._fmt(ss_between)}",
            f"  SS total         : {self._fmt(ss_total)}",
            f"  Eta-squared      : {self._fmt(eta_sq)}",
            f"  Interpretation   : {interpretation}",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    # -- Odds Ratio --
    def _adv_odds_ratio(self):
        # Expects 4 numbers: a, b, c, d for 2x2 contingency table
        data = self._parse_single()
        if len(data) != 4:
            raise ValueError("Odds ratio requires exactly 4 values: a, b, c, d for 2x2 table.")
        a, b, c, d = data
        odds_ratio = (a * d) / (b * c) if (b * c) != 0 else float("inf")
        log_or = np.log(odds_ratio) if odds_ratio > 0 and np.isfinite(odds_ratio) else float("nan")
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a,b,c,d) > 0 else float("nan")
        ci_low = np.exp(log_or - 1.96 * se_log_or) if np.isfinite(se_log_or) else float("nan")
        ci_high = np.exp(log_or + 1.96 * se_log_or) if np.isfinite(se_log_or) else float("nan")

        lines = [
            "=" * 50,
            "       ODDS RATIO",
            "=" * 50,
            f"  2x2 Table:",
            f"           |  Outcome+ |  Outcome- ",
            f"  Exposed+ |  {a:>8.0f} |  {b:>8.0f}",
            f"  Exposed- |  {c:>8.0f} |  {d:>8.0f}",
            f"",
            f"  Odds Ratio       : {self._fmt(odds_ratio)}",
            f"  Log(OR)          : {self._fmt(log_or)}",
            f"  SE(Log OR)       : {self._fmt(se_log_or)}",
            f"  95% CI           : [{self._fmt(ci_low)}, {self._fmt(ci_high)}]",
            "=" * 50,
        ]
        self._show("\n".join(lines))

    # -- QQ Plot & Normality Tests --
    def _adv_qq_normality(self):
        data = self._parse_single()
        n = len(data)

        # Normality tests
        shapiro_stat, shapiro_p = sp_stats.shapiro(data) if n >= 3 else (float("nan"), float("nan"))
        ad_result = sp_stats.anderson(data, dist='norm') if n >= 8 else None
        ks_stat, ks_p = sp_stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))

        lines = [
            "=" * 50,
            "       NORMALITY TESTS",
            "=" * 50,
            f"  N                : {n}",
            f"  Mean             : {self._fmt(np.mean(data))}",
            f"  Std Dev          : {self._fmt(np.std(data, ddof=1))}",
            f"  Skewness         : {self._fmt(sp_stats.skew(data, bias=False))}",
            f"  Kurtosis         : {self._fmt(sp_stats.kurtosis(data, bias=False))}",
            "",
            f"  Shapiro-Wilk     : W={self._fmt(shapiro_stat)}, p={self._fmt(shapiro_p)}",
        ]
        if ad_result is not None:
            lines.append(f"  Anderson-Darling : A2={self._fmt(ad_result.statistic)}")
            for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
                flag = " *" if ad_result.statistic > cv else ""
                lines.append(f"    Sig. level {sl}%: critical={self._fmt(cv)}{flag}")
        lines += [
            f"  Kolmogorov-Smirnov: D={self._fmt(ks_stat)}, p={self._fmt(ks_p)}",
            "",
            "  (* = reject normality at that significance level)",
            "=" * 50,
        ]
        self._show("\n".join(lines))

        # QQ plot
        fig = self.adv_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        sorted_data = np.sort(data)
        theoretical = sp_stats.norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n), loc=np.mean(data), scale=np.std(data, ddof=1))
        ax.scatter(theoretical, sorted_data, c='steelblue', s=20, edgecolors='k', linewidths=0.3)
        lims = [min(theoretical.min(), sorted_data.min()), max(theoretical.max(), sorted_data.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Reference line')
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title("QQ Plot (Normal)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.adv_canvas.draw()

    # -- Generate Statistical Report (HTML) --
    def _adv_generate_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Statistical Report", "stats_report.html",
            "HTML files (*.html);;All files (*)"
        )
        if not path:
            return

        try:
            data = self._parse_single()
        except Exception:
            self._show("Error: Cannot generate report without valid data.")
            return

        n = len(data)
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data, ddof=1)
        skew = sp_stats.skew(data, bias=False) if n > 2 else float("nan")
        kurt = sp_stats.kurtosis(data, bias=False) if n > 3 else float("nan")
        shapiro_stat, shapiro_p = sp_stats.shapiro(data) if n >= 3 else (float("nan"), float("nan"))
        ks_stat, ks_p = sp_stats.kstest(data, 'norm', args=(mean, std))

        # QQ plot
        fig_qq = Figure(figsize=(5, 4), dpi=100)
        style_figure(fig_qq)
        ax_qq = fig_qq.add_subplot(111)
        sorted_data = np.sort(data)
        theoretical = sp_stats.norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n), loc=mean, scale=std)
        ax_qq.scatter(theoretical, sorted_data, c='steelblue', s=15, edgecolors='k', linewidths=0.3)
        lims = [min(theoretical.min(), sorted_data.min()), max(theoretical.max(), sorted_data.max())]
        ax_qq.plot(lims, lims, 'r--', linewidth=1.5)
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles")
        ax_qq.set_title("QQ Plot")
        ax_qq.grid(True, alpha=0.3)
        fig_qq.tight_layout()

        buf_qq = io.BytesIO()
        fig_qq.savefig(buf_qq, format="png", dpi=100, bbox_inches="tight")
        buf_qq.seek(0)
        qq_b64 = base64.b64encode(buf_qq.read()).decode("utf-8")

        # Histogram
        fig_hist = Figure(figsize=(5, 4), dpi=100)
        style_figure(fig_hist)
        ax_hist = fig_hist.add_subplot(111)
        ax_hist.hist(data, bins='auto', color='steelblue', alpha=0.7, edgecolor='white')
        ax_hist.axvline(mean, color='red', linewidth=2, label=f'Mean={mean:.3f}')
        ax_hist.axvline(median, color='green', linewidth=2, linestyle='--', label=f'Median={median:.3f}')
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Histogram")
        ax_hist.legend()
        fig_hist.tight_layout()

        buf_hist = io.BytesIO()
        fig_hist.savefig(buf_hist, format="png", dpi=100, bbox_inches="tight")
        buf_hist.seek(0)
        hist_b64 = base64.b64encode(buf_hist.read()).decode("utf-8")

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        normality_status = "Data appears normal" if shapiro_p > 0.05 else "Data may not be normal"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Statistical Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #27ae60; padding-bottom: 10px; }}
  h2 {{ color: #27ae60; }}
  table {{ border-collapse: collapse; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 14px; text-align: left; }}
  th {{ background-color: #27ae60; color: white; }}
  tr:nth-child(even) {{ background-color: #f9f9f9; }}
  .meta {{ color: #777; font-size: 0.9em; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
  .flag {{ color: #e74c3c; font-weight: bold; }}
  .ok {{ color: #27ae60; font-weight: bold; }}
</style></head>
<body>
<h1>Statistical Analysis Report</h1>
<p class="meta">Generated: {ts} | QuantumRes Scientific Suite</p>

<h2>Descriptive Statistics</h2>
<table>
<tr><th>Statistic</th><th>Value</th></tr>
<tr><td>N</td><td>{n}</td></tr>
<tr><td>Mean</td><td>{mean:.6f}</td></tr>
<tr><td>Median</td><td>{median:.6f}</td></tr>
<tr><td>Std Deviation</td><td>{std:.6f}</td></tr>
<tr><td>Skewness</td><td>{skew:.6f}</td></tr>
<tr><td>Kurtosis</td><td>{kurt:.6f}</td></tr>
<tr><td>Min</td><td>{np.min(data):.6f}</td></tr>
<tr><td>Max</td><td>{np.max(data):.6f}</td></tr>
</table>

<h2>Normality Assessment</h2>
<p class="{'ok' if shapiro_p > 0.05 else 'flag'}">{normality_status}</p>
<table>
<tr><th>Test</th><th>Statistic</th><th>p-value</th></tr>
<tr><td>Shapiro-Wilk</td><td>{shapiro_stat:.6f}</td><td>{shapiro_p:.6f}</td></tr>
<tr><td>Kolmogorov-Smirnov</td><td>{ks_stat:.6f}</td><td>{ks_p:.6f}</td></tr>
</table>

<h2>Visualizations</h2>
<h3>Histogram</h3>
<img src="data:image/png;base64,{hist_b64}" alt="Histogram">
<h3>QQ Plot</h3>
<img src="data:image/png;base64,{qq_b64}" alt="QQ Plot">

</body></html>"""

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            self._show(f"Statistical report saved to: {path}")
            self._log(f"Report saved: {path}")
        except Exception as exc:
            self._show(f"Error saving report: {exc}")
