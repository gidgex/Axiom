"""
Machine Learning Tools Widget for QuantumRes Scientific Suite.

Provides a comprehensive AI/ML environment with dataset loading, preprocessing,
classification, regression, clustering, neural networks, model evaluation,
hyperparameter tuning, and interactive visualisation.
"""

import os
import io
import traceback
from functools import partial

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QLabel, QComboBox, QLineEdit, QGroupBox,
    QSplitter, QTextEdit, QToolBar, QAction, QMenu, QMessageBox,
    QInputDialog, QHeaderView, QDialog, QFormLayout, QDialogButtonBox,
    QCheckBox, QSpinBox, QTabWidget, QApplication, QProgressBar,
    QDoubleSpinBox, QSizePolicy, QGridLayout, QScrollArea, QFrame,
    QStackedWidget, QRadioButton, QButtonGroup, QSlider
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import (
    make_moons, make_circles, make_blobs, load_iris, load_digits,
    make_regression, make_classification,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import joblib
import datetime
import tempfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS = [
    "KNN", "Decision Tree", "Random Forest", "SVM",
    "Logistic Regression", "Naive Bayes", "MLP Classifier",
]

REGRESSION_MODELS = [
    "Linear Regression", "Ridge", "Lasso", "SVR",
    "Decision Tree Regressor", "Random Forest Regressor", "MLP Regressor",
]

CLUSTERING_MODELS = [
    "K-Means", "DBSCAN", "Agglomerative", "Gaussian Mixture",
]

SYNTHETIC_DATASETS = [
    "Moons", "Circles", "Blobs", "Iris", "Digits",
    "Synthetic Classification", "Synthetic Regression",
]

SCALERS = ["None", "StandardScaler", "MinMaxScaler"]


# ---------------------------------------------------------------------------
# Hyperparameter dialog
# ---------------------------------------------------------------------------

class HyperparamDialog(QDialog):
    """Dynamic dialog that builds controls for model hyperparameters."""

    def __init__(self, model_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Hyperparameters - {model_name}")
        self.setMinimumWidth(360)
        self._params = {}
        self._widgets = {}
        layout = QFormLayout(self)

        specs = self._specs_for(model_name)
        for name, (default, ptype, *extra) in specs.items():
            if ptype == "int":
                lo, hi = extra if extra else (1, 1000)
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setValue(default)
            elif ptype == "float":
                lo, hi = extra if extra else (0.0, 100.0)
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setDecimals(4)
                w.setSingleStep(0.01)
                w.setValue(default)
            elif ptype == "combo":
                w = QComboBox()
                w.addItems(extra[0])
                w.setCurrentText(str(default))
            elif ptype == "text":
                w = QLineEdit(str(default))
            else:
                continue
            self._widgets[name] = (w, ptype)
            layout.addRow(f"{name}:", w)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    # ------------------------------------------------------------------
    @staticmethod
    def _specs_for(model_name):
        """Return {param_name: (default, type, *extra)} for each model."""
        specs = {
            "KNN": {
                "n_neighbors": (5, "int", 1, 100),
                "weights": ("uniform", "combo", ["uniform", "distance"]),
                "metric": ("minkowski", "combo", ["minkowski", "euclidean", "manhattan"]),
            },
            "Decision Tree": {
                "max_depth": (5, "int", 1, 100),
                "min_samples_split": (2, "int", 2, 50),
                "criterion": ("gini", "combo", ["gini", "entropy"]),
            },
            "Random Forest": {
                "n_estimators": (100, "int", 10, 1000),
                "max_depth": (10, "int", 1, 100),
                "min_samples_split": (2, "int", 2, 50),
            },
            "SVM": {
                "C": (1.0, "float", 0.001, 1000.0),
                "kernel": ("rbf", "combo", ["rbf", "linear", "poly", "sigmoid"]),
                "gamma": ("scale", "combo", ["scale", "auto"]),
            },
            "Logistic Regression": {
                "C": (1.0, "float", 0.001, 1000.0),
                "max_iter": (200, "int", 50, 5000),
                "solver": ("lbfgs", "combo", ["lbfgs", "liblinear", "saga"]),
            },
            "Naive Bayes": {
                "var_smoothing": (1e-9, "float", 0.0, 1.0),
            },
            "MLP Classifier": {
                "hidden_layer_sizes": ("100", "text"),
                "activation": ("relu", "combo", ["relu", "tanh", "logistic"]),
                "max_iter": (300, "int", 50, 5000),
                "learning_rate_init": (0.001, "float", 0.0001, 1.0),
                "solver": ("adam", "combo", ["adam", "sgd", "lbfgs"]),
            },
            "Linear Regression": {},
            "Ridge": {
                "alpha": (1.0, "float", 0.0001, 1000.0),
            },
            "Lasso": {
                "alpha": (1.0, "float", 0.0001, 1000.0),
                "max_iter": (1000, "int", 100, 10000),
            },
            "SVR": {
                "C": (1.0, "float", 0.001, 1000.0),
                "kernel": ("rbf", "combo", ["rbf", "linear", "poly", "sigmoid"]),
                "epsilon": (0.1, "float", 0.0, 10.0),
            },
            "Decision Tree Regressor": {
                "max_depth": (5, "int", 1, 100),
                "min_samples_split": (2, "int", 2, 50),
            },
            "Random Forest Regressor": {
                "n_estimators": (100, "int", 10, 1000),
                "max_depth": (10, "int", 1, 100),
            },
            "MLP Regressor": {
                "hidden_layer_sizes": ("100", "text"),
                "activation": ("relu", "combo", ["relu", "tanh", "logistic"]),
                "max_iter": (300, "int", 50, 5000),
                "learning_rate_init": (0.001, "float", 0.0001, 1.0),
            },
            "K-Means": {
                "n_clusters": (3, "int", 2, 50),
                "max_iter": (300, "int", 50, 2000),
                "n_init": (10, "int", 1, 50),
            },
            "DBSCAN": {
                "eps": (0.5, "float", 0.01, 50.0),
                "min_samples": (5, "int", 1, 100),
            },
            "Agglomerative": {
                "n_clusters": (3, "int", 2, 50),
                "linkage": ("ward", "combo", ["ward", "complete", "average", "single"]),
            },
            "Gaussian Mixture": {
                "n_components": (3, "int", 1, 50),
                "covariance_type": ("full", "combo", ["full", "tied", "diag", "spherical"]),
                "max_iter": (100, "int", 10, 1000),
            },
        }
        return specs.get(model_name, {})

    # ------------------------------------------------------------------
    def get_params(self):
        """Return dict of parameter values."""
        params = {}
        for name, (widget, ptype) in self._widgets.items():
            if ptype == "int":
                params[name] = widget.value()
            elif ptype == "float":
                params[name] = widget.value()
            elif ptype == "combo":
                params[name] = widget.currentText()
            elif ptype == "text":
                params[name] = widget.text()
        return params


# ---------------------------------------------------------------------------
# Worker thread for long-running training
# ---------------------------------------------------------------------------

class TrainWorker(QThread):
    """Runs model training in a background thread."""
    finished = pyqtSignal(object)   # result dict
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, task_fn, parent=None):
        super().__init__(parent)
        self._task_fn = task_fn

    def run(self):
        try:
            result = self._task_fn()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class MLToolsWidget(QWidget):
    """Comprehensive Machine Learning tools widget for the QuantumRes suite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._df = None           # loaded / generated DataFrame
        self._X = None
        self._y = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._model = None
        self._model_params = {}
        self._task_type = "classification"  # classification | regression | clustering
        self._scaler = None
        self._pca = None
        self._label_encoder = None
        self._worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_logger(self, fn):
        """Set an external logging callback ``fn(message: str)``."""
        self._logger = fn

    def load_file(self, path):
        """Load a CSV file programmatically."""
        try:
            self._df = pd.read_csv(path)
            self._log(f"Loaded file: {path}  ({self._df.shape[0]} rows, {self._df.shape[1]} cols)")
            self._refresh_column_combos()
            self._preview_data()
        except Exception as exc:
            self._log(f"Error loading file: {exc}")

    def run(self):
        """Trigger the current pipeline (equivalent to pressing Run)."""
        self._on_run()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg):
        self._results_text.append(msg)
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- Left: controls ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(310)
        left_scroll.setMaximumWidth(420)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # -- Data section --
        grp_data = QGroupBox("Data")
        dl = QVBoxLayout(grp_data)

        row = QHBoxLayout()
        self._btn_load_csv = QPushButton("Load CSV")
        self._btn_load_csv.clicked.connect(self._on_load_csv)
        row.addWidget(self._btn_load_csv)

        self._combo_synthetic = QComboBox()
        self._combo_synthetic.addItems(["-- Synthetic --"] + SYNTHETIC_DATASETS)
        row.addWidget(self._combo_synthetic)

        self._btn_generate = QPushButton("Generate")
        self._btn_generate.clicked.connect(self._on_generate_dataset)
        row.addWidget(self._btn_generate)
        dl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Samples:"))
        self._spin_samples = QSpinBox()
        self._spin_samples.setRange(50, 100000)
        self._spin_samples.setValue(500)
        row2.addWidget(self._spin_samples)
        row2.addWidget(QLabel("Noise:"))
        self._spin_noise = QDoubleSpinBox()
        self._spin_noise.setRange(0.0, 5.0)
        self._spin_noise.setValue(0.2)
        self._spin_noise.setSingleStep(0.05)
        row2.addWidget(self._spin_noise)
        dl.addLayout(row2)

        self._lbl_data_info = QLabel("No data loaded.")
        dl.addWidget(self._lbl_data_info)
        left_layout.addWidget(grp_data)

        # -- Column selection --
        grp_cols = QGroupBox("Columns")
        cl = QGridLayout(grp_cols)
        cl.addWidget(QLabel("Features:"), 0, 0)
        self._txt_features = QLineEdit()
        self._txt_features.setPlaceholderText("col1,col2,... or leave blank for all numeric")
        cl.addWidget(self._txt_features, 0, 1)
        cl.addWidget(QLabel("Target:"), 1, 0)
        self._combo_target = QComboBox()
        cl.addWidget(self._combo_target, 1, 1)
        left_layout.addWidget(grp_cols)

        # -- Preprocessing --
        grp_pre = QGroupBox("Preprocessing")
        pl = QVBoxLayout(grp_pre)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Scaler:"))
        self._combo_scaler = QComboBox()
        self._combo_scaler.addItems(SCALERS)
        r1.addWidget(self._combo_scaler)
        pl.addLayout(r1)

        r2 = QHBoxLayout()
        self._chk_pca = QCheckBox("PCA")
        r2.addWidget(self._chk_pca)
        r2.addWidget(QLabel("Components:"))
        self._spin_pca = QSpinBox()
        self._spin_pca.setRange(1, 500)
        self._spin_pca.setValue(2)
        r2.addWidget(self._spin_pca)
        pl.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Test size:"))
        self._spin_test = QDoubleSpinBox()
        self._spin_test.setRange(0.05, 0.95)
        self._spin_test.setValue(0.25)
        self._spin_test.setSingleStep(0.05)
        r3.addWidget(self._spin_test)
        r3.addWidget(QLabel("Random seed:"))
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 99999)
        self._spin_seed.setValue(42)
        r3.addWidget(self._spin_seed)
        pl.addLayout(r3)

        left_layout.addWidget(grp_pre)

        # -- Task / Model --
        grp_model = QGroupBox("Model")
        ml = QVBoxLayout(grp_model)

        r_task = QHBoxLayout()
        self._bg_task = QButtonGroup(self)
        for label in ["Classification", "Regression", "Clustering"]:
            rb = QRadioButton(label)
            self._bg_task.addButton(rb)
            r_task.addWidget(rb)
            if label == "Classification":
                rb.setChecked(True)
        self._bg_task.buttonClicked.connect(self._on_task_changed)
        ml.addLayout(r_task)

        self._combo_model = QComboBox()
        self._combo_model.addItems(CLASSIFICATION_MODELS)
        ml.addWidget(self._combo_model)

        r_hyper = QHBoxLayout()
        self._btn_hyperparams = QPushButton("Hyperparameters...")
        self._btn_hyperparams.clicked.connect(self._on_edit_hyperparams)
        r_hyper.addWidget(self._btn_hyperparams)
        self._btn_run = QPushButton("Run")
        self._btn_run.setStyleSheet("font-weight:bold;")
        self._btn_run.clicked.connect(self._on_run)
        r_hyper.addWidget(self._btn_run)
        ml.addLayout(r_hyper)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        ml.addWidget(self._progress)
        left_layout.addWidget(grp_model)

        # -- Visualization --
        grp_viz = QGroupBox("Visualization")
        vl = QVBoxLayout(grp_viz)
        self._combo_viz = QComboBox()
        self._combo_viz.addItems([
            "Scatter / Predictions",
            "Decision Boundary",
            "Confusion Matrix",
            "ROC Curve",
            "Learning Curve",
            "Feature Importance",
            "Residuals",
            "Cluster Scatter",
        ])
        vl.addWidget(self._combo_viz)
        self._btn_plot = QPushButton("Plot")
        self._btn_plot.clicked.connect(self._on_plot)
        vl.addWidget(self._btn_plot)
        left_layout.addWidget(grp_viz)

        # -- Cross-validation --
        grp_cv = QGroupBox("Cross-Validation")
        cvl = QHBoxLayout(grp_cv)
        cvl.addWidget(QLabel("Folds:"))
        self._spin_cv = QSpinBox()
        self._spin_cv.setRange(2, 20)
        self._spin_cv.setValue(5)
        cvl.addWidget(self._spin_cv)
        self._btn_cv = QPushButton("Run CV")
        self._btn_cv.clicked.connect(self._on_cross_validate)
        cvl.addWidget(self._btn_cv)
        left_layout.addWidget(grp_cv)

        # -- Advanced Tools --
        grp_adv = QGroupBox("Advanced Tools")
        adv_layout = QVBoxLayout(grp_adv)

        self._btn_feature_imp = QPushButton("Feature Importance")
        self._btn_feature_imp.setToolTip("Rank features by permutation importance with visualization")
        self._btn_feature_imp.clicked.connect(self._feature_importance_ranking)
        adv_layout.addWidget(self._btn_feature_imp)

        self._btn_gridsearch = QPushButton("Hyperparameter Tuning (GridSearch)")
        self._btn_gridsearch.setToolTip("Automated GridSearchCV for the current model")
        self._btn_gridsearch.clicked.connect(self._grid_search_tuning)
        adv_layout.addWidget(self._btn_gridsearch)

        self._btn_explain = QPushButton("Explain Prediction")
        self._btn_explain.setToolTip("Approximate SHAP-like feature contributions for a sample")
        self._btn_explain.clicked.connect(self._explain_prediction)
        adv_layout.addWidget(self._btn_explain)

        self._btn_forecast = QPushButton("Time Series Forecast")
        self._btn_forecast.setToolTip("Simple ARIMA-like time series forecasting")
        self._btn_forecast.clicked.connect(self._time_series_forecast)
        adv_layout.addWidget(self._btn_forecast)

        self._btn_anomaly = QPushButton("Anomaly Detection")
        self._btn_anomaly.setToolTip("Detect anomalies using Isolation Forest or LOF")
        self._btn_anomaly.clicked.connect(self._anomaly_detection)
        adv_layout.addWidget(self._btn_anomaly)

        left_layout.addWidget(grp_adv)

        left_layout.addStretch()

        # ---- Right: results & plots ----
        right_tabs = QTabWidget()
        splitter.addWidget(right_tabs)

        # -- Data preview tab --
        self._table_preview = QTableWidget()
        self._table_preview.setEditTriggers(QTableWidget.NoEditTriggers)
        right_tabs.addTab(self._table_preview, "Data Preview")

        # -- Plot tab --
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(7, 5), dpi=100)
        style_figure(self._figure)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, plot_widget)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas)
        right_tabs.addTab(plot_widget, "Plot")

        # -- Results text tab --
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setFont(QFont("Consolas", 9))
        right_tabs.addTab(self._results_text, "Results")

        # -- Metrics tab --
        self._metrics_table = QTableWidget()
        self._metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._metrics_table.horizontalHeader().setStretchLastSection(True)
        right_tabs.addTab(self._metrics_table, "Metrics")

        self._right_tabs = right_tabs
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self.load_file(path)

    def _on_generate_dataset(self):
        name = self._combo_synthetic.currentText()
        if name.startswith("--"):
            return
        n = self._spin_samples.value()
        noise = self._spin_noise.value()
        seed = self._spin_seed.value()
        try:
            if name == "Moons":
                X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
                self._df = pd.DataFrame(X, columns=["x1", "x2"])
                self._df["target"] = y
            elif name == "Circles":
                X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=seed)
                self._df = pd.DataFrame(X, columns=["x1", "x2"])
                self._df["target"] = y
            elif name == "Blobs":
                X, y = make_blobs(n_samples=n, centers=4, random_state=seed)
                self._df = pd.DataFrame(X, columns=["x1", "x2"])
                self._df["target"] = y
            elif name == "Iris":
                ds = load_iris()
                self._df = pd.DataFrame(ds.data, columns=ds.feature_names)
                self._df["target"] = ds.target
            elif name == "Digits":
                ds = load_digits()
                self._df = pd.DataFrame(ds.data, columns=[f"px{i}" for i in range(ds.data.shape[1])])
                self._df["target"] = ds.target
            elif name == "Synthetic Classification":
                X, y = make_classification(
                    n_samples=n, n_features=10, n_informative=5,
                    n_classes=3, random_state=seed,
                )
                self._df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
                self._df["target"] = y
            elif name == "Synthetic Regression":
                X, y = make_regression(
                    n_samples=n, n_features=5, noise=noise * 10, random_state=seed,
                )
                self._df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
                self._df["target"] = y
            else:
                return

            self._log(f"Generated dataset '{name}': {self._df.shape[0]} rows, {self._df.shape[1]} cols")
            self._refresh_column_combos()
            self._preview_data()
        except Exception as exc:
            self._log(f"Error generating dataset: {exc}")

    def _refresh_column_combos(self):
        if self._df is None:
            return
        cols = list(self._df.columns)
        self._combo_target.clear()
        self._combo_target.addItems(cols)
        if "target" in cols:
            self._combo_target.setCurrentText("target")
        elif len(cols):
            self._combo_target.setCurrentIndex(len(cols) - 1)
        self._lbl_data_info.setText(
            f"{self._df.shape[0]} rows x {self._df.shape[1]} cols | "
            f"dtypes: {', '.join(self._df.dtypes.astype(str).unique())}"
        )

    def _preview_data(self):
        if self._df is None:
            return
        preview = self._df.head(200)
        self._table_preview.setRowCount(len(preview))
        self._table_preview.setColumnCount(len(preview.columns))
        self._table_preview.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for r in range(len(preview)):
            for c in range(len(preview.columns)):
                val = preview.iloc[r, c]
                self._table_preview.setItem(r, c, QTableWidgetItem(str(val)))
        self._table_preview.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Task switching
    # ------------------------------------------------------------------

    def _on_task_changed(self, btn):
        text = btn.text().lower()
        self._task_type = text
        self._combo_model.clear()
        if text == "classification":
            self._combo_model.addItems(CLASSIFICATION_MODELS)
        elif text == "regression":
            self._combo_model.addItems(REGRESSION_MODELS)
        elif text == "clustering":
            self._combo_model.addItems(CLUSTERING_MODELS)

    # ------------------------------------------------------------------
    # Hyperparameter editing
    # ------------------------------------------------------------------

    def _on_edit_hyperparams(self):
        model_name = self._combo_model.currentText()
        if not model_name:
            return
        dlg = HyperparamDialog(model_name, self)
        if dlg.exec_() == QDialog.Accepted:
            self._model_params = dlg.get_params()
            self._log(f"Hyperparameters updated: {self._model_params}")

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------

    def _prepare_data(self):
        """Build X, y arrays from current DataFrame and settings."""
        if self._df is None:
            raise ValueError("No data loaded.")

        target_col = self._combo_target.currentText()
        feature_text = self._txt_features.text().strip()

        if feature_text:
            feature_cols = [c.strip() for c in feature_text.split(",") if c.strip()]
        else:
            feature_cols = [c for c in self._df.columns if c != target_col]
            # keep only numeric
            feature_cols = [c for c in feature_cols if np.issubdtype(self._df[c].dtype, np.number)]

        if not feature_cols:
            raise ValueError("No numeric feature columns found.")

        X = self._df[feature_cols].values.astype(np.float64)

        # Handle target
        self._label_encoder = None
        if self._task_type != "clustering":
            if target_col not in self._df.columns:
                raise ValueError(f"Target column '{target_col}' not found.")
            y_raw = self._df[target_col]
            if not np.issubdtype(y_raw.dtype, np.number):
                self._label_encoder = LabelEncoder()
                y = self._label_encoder.fit_transform(y_raw)
            else:
                y = y_raw.values.astype(np.float64)
        else:
            y = None

        # Handle NaNs
        mask = ~np.isnan(X).any(axis=1)
        if y is not None:
            mask &= ~np.isnan(y)
        X = X[mask]
        if y is not None:
            y = y[mask]

        # Scaling
        scaler_name = self._combo_scaler.currentText()
        self._scaler = None
        if scaler_name == "StandardScaler":
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        elif scaler_name == "MinMaxScaler":
            self._scaler = MinMaxScaler()
            X = self._scaler.fit_transform(X)

        # PCA
        self._pca = None
        if self._chk_pca.isChecked():
            n_comp = min(self._spin_pca.value(), X.shape[1], X.shape[0])
            self._pca = PCA(n_components=n_comp)
            X = self._pca.fit_transform(X)
            self._log(f"PCA: explained variance = {self._pca.explained_variance_ratio_.sum():.4f}")

        self._X = X
        self._y = y

        # Train/test split
        if self._task_type != "clustering":
            test_size = self._spin_test.value()
            seed = self._spin_seed.value()
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed,
            )
            self._log(
                f"Train/test split: train={len(self._X_train)}, test={len(self._X_test)}"
            )
        else:
            self._X_train = X
            self._X_test = None
            self._y_train = y
            self._y_test = None

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _build_model(self):
        name = self._combo_model.currentText()
        p = dict(self._model_params)

        # Parse hidden_layer_sizes from text
        if "hidden_layer_sizes" in p:
            raw = p["hidden_layer_sizes"]
            try:
                p["hidden_layer_sizes"] = tuple(int(x) for x in raw.split(","))
            except Exception:
                p["hidden_layer_sizes"] = (100,)

        seed = self._spin_seed.value()

        builders = {
            # Classification
            "KNN": lambda: KNeighborsClassifier(**p),
            "Decision Tree": lambda: DecisionTreeClassifier(random_state=seed, **p),
            "Random Forest": lambda: RandomForestClassifier(random_state=seed, **p),
            "SVM": lambda: SVC(probability=True, random_state=seed, **p),
            "Logistic Regression": lambda: LogisticRegression(random_state=seed, **p),
            "Naive Bayes": lambda: GaussianNB(**p),
            "MLP Classifier": lambda: MLPClassifier(random_state=seed, **p),
            # Regression
            "Linear Regression": lambda: LinearRegression(**p),
            "Ridge": lambda: Ridge(**p),
            "Lasso": lambda: Lasso(**p),
            "SVR": lambda: SVR(**p),
            "Decision Tree Regressor": lambda: DecisionTreeRegressor(random_state=seed, **p),
            "Random Forest Regressor": lambda: RandomForestRegressor(random_state=seed, **p),
            "MLP Regressor": lambda: MLPRegressor(random_state=seed, **p),
            # Clustering
            "K-Means": lambda: KMeans(random_state=seed, **p),
            "DBSCAN": lambda: DBSCAN(**p),
            "Agglomerative": lambda: AgglomerativeClustering(**p),
            "Gaussian Mixture": lambda: GaussianMixture(random_state=seed, **p),
        }
        if name not in builders:
            raise ValueError(f"Unknown model: {name}")
        return builders[name]()

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    def _on_run(self):
        try:
            self._prepare_data()
        except Exception as exc:
            self._log(f"Data preparation error: {exc}")
            return

        self._progress.setVisible(True)
        self._btn_run.setEnabled(False)

        def task():
            return self._train_and_evaluate()

        self._worker = TrainWorker(task)
        self._worker.finished.connect(self._on_train_done)
        self._worker.error.connect(self._on_train_error)
        self._worker.start()

    @pyqtSlot(object)
    def _on_train_done(self, result):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._display_metrics(result)
        self._right_tabs.setCurrentIndex(2)  # Results tab

    @pyqtSlot(str)
    def _on_train_error(self, tb):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._log(f"Training error:\n{tb}")

    def _train_and_evaluate(self):
        """Train model and collect metrics. Runs in worker thread."""
        model = self._build_model()
        self._model = model
        name = self._combo_model.currentText()
        result = {"model": name}

        if self._task_type == "classification":
            model.fit(self._X_train, self._y_train)
            y_pred = model.predict(self._X_test)

            result["accuracy"] = accuracy_score(self._y_test, y_pred)
            avg = "binary" if len(np.unique(self._y_test)) == 2 else "weighted"
            result["precision"] = precision_score(self._y_test, y_pred, average=avg, zero_division=0)
            result["recall"] = recall_score(self._y_test, y_pred, average=avg, zero_division=0)
            result["f1"] = f1_score(self._y_test, y_pred, average=avg, zero_division=0)
            result["confusion_matrix"] = confusion_matrix(self._y_test, y_pred)
            result["classification_report"] = classification_report(
                self._y_test, y_pred, zero_division=0,
            )

            # ROC (binary or OvR)
            if hasattr(model, "predict_proba"):
                classes = np.unique(self._y_test)
                if len(classes) == 2:
                    proba = model.predict_proba(self._X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self._y_test, proba)
                    result["roc_auc"] = auc(fpr, tpr)
                    result["roc_fpr"] = fpr
                    result["roc_tpr"] = tpr
                else:
                    # store per-class ROC
                    result["roc_multiclass"] = {}
                    proba = model.predict_proba(self._X_test)
                    for i, c in enumerate(classes):
                        binary_y = (self._y_test == c).astype(int)
                        fpr, tpr, _ = roc_curve(binary_y, proba[:, i])
                        result["roc_multiclass"][c] = (fpr, tpr, auc(fpr, tpr))

        elif self._task_type == "regression":
            model.fit(self._X_train, self._y_train)
            y_pred = model.predict(self._X_test)

            result["r2"] = r2_score(self._y_test, y_pred)
            result["mse"] = mean_squared_error(self._y_test, y_pred)
            result["rmse"] = np.sqrt(result["mse"])
            result["mae"] = mean_absolute_error(self._y_test, y_pred)
            result["y_pred"] = y_pred

        elif self._task_type == "clustering":
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(self._X_train)
            else:
                model.fit(self._X_train)
                labels = model.predict(self._X_train)

            result["labels"] = labels
            n_labels = len(set(labels) - {-1})
            result["n_clusters"] = n_labels
            if n_labels >= 2 and n_labels < len(self._X_train):
                result["silhouette"] = silhouette_score(self._X_train, labels)
            if self._y_train is not None:
                result["has_true_labels"] = True

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    def _display_metrics(self, result):
        self._log("=" * 60)
        self._log(f"Model: {result['model']}  |  Task: {self._task_type}")
        self._log("-" * 60)

        rows = []
        if self._task_type == "classification":
            for m in ["accuracy", "precision", "recall", "f1"]:
                if m in result:
                    val = f"{result[m]:.4f}"
                    rows.append((m.capitalize(), val))
                    self._log(f"  {m.capitalize():>12s}: {val}")
            if "roc_auc" in result:
                rows.append(("ROC AUC", f"{result['roc_auc']:.4f}"))
                self._log(f"  {'ROC AUC':>12s}: {result['roc_auc']:.4f}")
            if "classification_report" in result:
                self._log("\n" + result["classification_report"])

        elif self._task_type == "regression":
            for m in ["r2", "mse", "rmse", "mae"]:
                if m in result:
                    val = f"{result[m]:.4f}"
                    rows.append((m.upper(), val))
                    self._log(f"  {m.upper():>12s}: {val}")

        elif self._task_type == "clustering":
            rows.append(("Clusters", str(result.get("n_clusters", "?"))))
            self._log(f"  {'Clusters':>12s}: {result.get('n_clusters', '?')}")
            if "silhouette" in result:
                val = f"{result['silhouette']:.4f}"
                rows.append(("Silhouette", val))
                self._log(f"  {'Silhouette':>12s}: {val}")

        self._log("=" * 60)

        # Populate metrics table
        self._metrics_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(k))
            self._metrics_table.setItem(i, 1, QTableWidgetItem(v))

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _on_cross_validate(self):
        if self._df is None:
            self._log("No data loaded for cross-validation.")
            return
        if self._task_type == "clustering":
            self._log("Cross-validation is not applicable for clustering.")
            return
        try:
            self._prepare_data()
            model = self._build_model()
            folds = self._spin_cv.value()
            scoring = "accuracy" if self._task_type == "classification" else "r2"
            scores = cross_val_score(model, self._X, self._y, cv=folds, scoring=scoring)
            self._log(
                f"Cross-Validation ({folds}-fold, {scoring}):\n"
                f"  scores = {np.round(scores, 4)}\n"
                f"  mean   = {scores.mean():.4f}\n"
                f"  std    = {scores.std():.4f}"
            )
        except Exception as exc:
            self._log(f"Cross-validation error: {exc}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _on_plot(self):
        choice = self._combo_viz.currentText()
        try:
            self._figure.clear()
            if choice == "Scatter / Predictions":
                self._plot_scatter()
            elif choice == "Decision Boundary":
                self._plot_decision_boundary()
            elif choice == "Confusion Matrix":
                self._plot_confusion_matrix()
            elif choice == "ROC Curve":
                self._plot_roc()
            elif choice == "Learning Curve":
                self._plot_learning_curve()
            elif choice == "Feature Importance":
                self._plot_feature_importance()
            elif choice == "Residuals":
                self._plot_residuals()
            elif choice == "Cluster Scatter":
                self._plot_cluster_scatter()
            self._canvas.draw()
            self._right_tabs.setCurrentIndex(1)  # Plot tab
        except Exception as exc:
            self._log(f"Plot error: {exc}\n{traceback.format_exc()}")

    # -- individual plot methods --

    def _plot_scatter(self):
        ax = self._figure.add_subplot(111)
        if self._X is None:
            return
        X = self._X
        if X.shape[1] < 2:
            ax.set_title("Need at least 2 features for scatter")
            return
        if self._task_type == "clustering" and hasattr(self, "_last_result"):
            c = self._last_result.get("labels", None)
        elif self._y is not None:
            c = self._y
        else:
            c = None
        scatter = ax.scatter(X[:, 0], X[:, 1], c=c, cmap="viridis", s=15, alpha=0.7, edgecolors="k", linewidths=0.3)
        if c is not None:
            self._figure.colorbar(scatter, ax=ax)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Data Scatter (first 2 features)")

    def _plot_decision_boundary(self):
        ax = self._figure.add_subplot(111)
        if self._model is None or self._X_train is None:
            ax.set_title("Train a model first")
            return
        if self._X_train.shape[1] != 2:
            ax.set_title("Decision boundary requires exactly 2 features (use PCA=2)")
            return
        if self._task_type not in ("classification", "clustering"):
            ax.set_title("Decision boundary available for classification / clustering")
            return

        X = self._X_train
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        if hasattr(self._model, "predict"):
            Z = self._model.predict(grid)
        else:
            ax.set_title("Model has no predict method")
            return

        Z = Z.reshape(xx.shape)
        cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFFFAA", "#FFAAFF"])
        cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        if self._y_train is not None:
            ax.scatter(X[:, 0], X[:, 1], c=self._y_train, cmap=cmap_bold, s=15, edgecolors="k", linewidths=0.3)
        else:
            ax.scatter(X[:, 0], X[:, 1], c=Z[::1], s=15, edgecolors="k", linewidths=0.3)
        ax.set_title("Decision Boundary")

    def _plot_confusion_matrix(self):
        ax = self._figure.add_subplot(111)
        if not hasattr(self, "_last_result") or "confusion_matrix" not in self._last_result:
            ax.set_title("Run a classification model first")
            return
        cm = self._last_result["confusion_matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        self._figure.colorbar(im, ax=ax)
        n = cm.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        # annotate cells
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    def _plot_roc(self):
        ax = self._figure.add_subplot(111)
        if not hasattr(self, "_last_result"):
            ax.set_title("Run a classification model first")
            return
        r = self._last_result
        if "roc_fpr" in r:
            ax.plot(r["roc_fpr"], r["roc_tpr"], label=f"AUC = {r['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        elif "roc_multiclass" in r:
            for cls, (fpr, tpr, a) in r["roc_multiclass"].items():
                ax.plot(fpr, tpr, label=f"Class {cls} (AUC={a:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        else:
            ax.set_title("ROC data not available (model may lack predict_proba)")
            return
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

    def _plot_learning_curve(self):
        ax = self._figure.add_subplot(111)
        if self._model is None or self._X is None or self._y is None:
            ax.set_title("Train a supervised model first")
            return
        model = self._build_model()
        scoring = "accuracy" if self._task_type == "classification" else "r2"
        train_sizes, train_scores, val_scores = learning_curve(
            model, self._X, self._y, cv=5, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="orange")
        ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training")
        ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation")
        ax.set_xlabel("Training Size")
        ax.set_ylabel(scoring.capitalize())
        ax.set_title("Learning Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self):
        ax = self._figure.add_subplot(111)
        if self._model is None:
            ax.set_title("Train a model first")
            return
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            coef = self._model.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        else:
            ax.set_title("Model does not expose feature importances or coefficients")
            return

        n_feat = len(importances)
        if self._df is not None:
            target_col = self._combo_target.currentText()
            feature_text = self._txt_features.text().strip()
            if feature_text:
                names = [c.strip() for c in feature_text.split(",")]
            else:
                names = [c for c in self._df.columns if c != target_col and np.issubdtype(self._df[c].dtype, np.number)]
            # PCA renames
            if self._pca is not None:
                names = [f"PC{i+1}" for i in range(n_feat)]
            elif len(names) != n_feat:
                names = [f"f{i}" for i in range(n_feat)]
        else:
            names = [f"f{i}" for i in range(n_feat)]

        indices = np.argsort(importances)[::-1]
        ax.barh(range(n_feat), importances[indices], color="steelblue")
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels([names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")

    def _plot_residuals(self):
        ax = self._figure.add_subplot(111)
        if not hasattr(self, "_last_result") or "y_pred" not in self._last_result:
            ax.set_title("Run a regression model first")
            return
        y_pred = self._last_result["y_pred"]
        residuals = self._y_test - y_pred
        ax.scatter(y_pred, residuals, s=15, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid(True, alpha=0.3)

    def _plot_cluster_scatter(self):
        ax = self._figure.add_subplot(111)
        if not hasattr(self, "_last_result") or "labels" not in self._last_result:
            ax.set_title("Run a clustering model first")
            return
        X = self._X_train
        labels = self._last_result["labels"]
        if X.shape[1] < 2:
            ax.set_title("Need >= 2 features")
            return
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15, alpha=0.7, edgecolors="k", linewidths=0.3)
        self._figure.colorbar(scatter, ax=ax)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"Clusters ({self._last_result.get('n_clusters', '?')} found)")

    # ------------------------------------------------------------------
    # Model Export / Import
    # ------------------------------------------------------------------

    def export_model(self, path=None):
        """Save the trained model to a .pkl file using joblib."""
        if self._model is None:
            self._log("No trained model to export.")
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Model", "model.pkl",
                "Pickle files (*.pkl);;All files (*)"
            )
        if not path:
            return
        bundle = {
            "model": self._model,
            "scaler": self._scaler,
            "pca": self._pca,
            "label_encoder": self._label_encoder,
            "task_type": self._task_type,
            "model_name": self._combo_model.currentText(),
            "model_params": self._model_params,
        }
        joblib.dump(bundle, path)
        self._log(f"Model exported to: {path}")

    def load_model(self, path=None):
        """Load a previously saved model from a .pkl file."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Model", "", "Pickle files (*.pkl);;All files (*)"
            )
        if not path:
            return
        try:
            bundle = joblib.load(path)
            self._model = bundle.get("model")
            self._scaler = bundle.get("scaler")
            self._pca = bundle.get("pca")
            self._label_encoder = bundle.get("label_encoder")
            self._task_type = bundle.get("task_type", "classification")
            name = bundle.get("model_name", "Unknown")
            self._model_params = bundle.get("model_params", {})
            self._log(f"Model loaded from: {path} ({name}, task={self._task_type})")
        except Exception as exc:
            self._log(f"Error loading model: {exc}")

    # ------------------------------------------------------------------
    # Prediction Interface
    # ------------------------------------------------------------------

    def predict(self, data=None):
        """Make predictions on new data using the loaded/trained model.
        *data*: numpy array or None (prompts for CSV file).
        Returns prediction array."""
        if self._model is None:
            self._log("No trained model available for prediction.")
            return None
        if data is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open data for prediction", "",
                "CSV files (*.csv);;All files (*)"
            )
            if not path:
                return None
            try:
                df = pd.read_csv(path)
                data = df.select_dtypes(include=[np.number]).values
            except Exception as exc:
                self._log(f"Error reading prediction data: {exc}")
                return None
        try:
            X = np.asarray(data, dtype=np.float64)
            if self._scaler is not None:
                X = self._scaler.transform(X)
            if self._pca is not None:
                X = self._pca.transform(X)
            preds = self._model.predict(X)
            if self._label_encoder is not None and self._task_type == "classification":
                preds_decoded = self._label_encoder.inverse_transform(preds.astype(int))
                self._log(f"Predictions (decoded): {preds_decoded[:20]}...")
            else:
                self._log(f"Predictions: {preds[:20]}...")
            return preds
        except Exception as exc:
            self._log(f"Prediction error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def generate_polynomial_features(self, degree=2, interaction_only=False):
        """Auto-generate polynomial and interaction features from current data.
        Updates the internal DataFrame with the new features."""
        if self._df is None:
            self._log("No data loaded for feature engineering.")
            return
        target_col = self._combo_target.currentText()
        feature_text = self._txt_features.text().strip()
        if feature_text:
            feature_cols = [c.strip() for c in feature_text.split(",") if c.strip()]
        else:
            feature_cols = [c for c in self._df.columns if c != target_col
                           and np.issubdtype(self._df[c].dtype, np.number)]
        if not feature_cols:
            self._log("No numeric feature columns found.")
            return

        X = self._df[feature_cols].values.astype(np.float64)
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        X_poly = poly.fit_transform(X)
        new_names = poly.get_feature_names_out(feature_cols)

        df_new = pd.DataFrame(X_poly, columns=new_names)
        if target_col in self._df.columns:
            df_new[target_col] = self._df[target_col].values
        self._df = df_new
        self._refresh_column_combos()
        self._preview_data()
        n_new = len(new_names) - len(feature_cols)
        self._log(
            f"Feature engineering: degree={degree}, interaction_only={interaction_only}. "
            f"Added {n_new} new features ({len(new_names)} total)."
        )

    # ------------------------------------------------------------------
    # Pipeline Builder
    # ------------------------------------------------------------------

    def build_pipeline(self):
        """Chain current preprocessing + model as a reusable sklearn Pipeline.
        Returns the Pipeline object."""
        steps = []
        scaler_name = self._combo_scaler.currentText()
        if scaler_name == "StandardScaler":
            steps.append(("scaler", StandardScaler()))
        elif scaler_name == "MinMaxScaler":
            steps.append(("scaler", MinMaxScaler()))
        if self._chk_pca.isChecked():
            n_comp = self._spin_pca.value()
            steps.append(("pca", PCA(n_components=n_comp)))
        model = self._build_model()
        steps.append(("model", model))
        pipeline = Pipeline(steps)
        self._log(f"Pipeline built with {len(steps)} step(s): {[s[0] for s in steps]}")
        return pipeline

    def export_pipeline(self, path=None):
        """Build and export the full pipeline to a .pkl file."""
        pipeline = self.build_pipeline()
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Pipeline", "pipeline.pkl",
                "Pickle files (*.pkl);;All files (*)"
            )
        if path:
            joblib.dump(pipeline, path)
            self._log(f"Pipeline exported to: {path}")

    # ------------------------------------------------------------------
    # Model Comparison Table
    # ------------------------------------------------------------------

    def compare_models(self, model_list=None, cv_folds=5):
        """Train multiple models and compare metrics side-by-side.
        *model_list*: list of model name strings, or None for all models in current task.
        Returns a pandas DataFrame of comparison results."""
        if self._df is None:
            self._log("No data loaded for model comparison.")
            return None
        try:
            self._prepare_data()
        except Exception as exc:
            self._log(f"Data preparation error: {exc}")
            return None

        if self._task_type == "clustering":
            self._log("Model comparison is not supported for clustering.")
            return None

        if model_list is None:
            if self._task_type == "classification":
                model_list = CLASSIFICATION_MODELS
            else:
                model_list = REGRESSION_MODELS

        results = []
        for name in model_list:
            try:
                old_params = dict(self._model_params)
                self._model_params = {}
                self._combo_model.setCurrentText(name)
                model = self._build_model()
                self._model_params = old_params

                scoring = "accuracy" if self._task_type == "classification" else "r2"
                scores = cross_val_score(model, self._X, self._y, cv=cv_folds, scoring=scoring)

                model.fit(self._X_train, self._y_train)
                y_pred = model.predict(self._X_test)

                row = {"Model": name, f"CV {scoring} (mean)": round(scores.mean(), 4),
                       f"CV {scoring} (std)": round(scores.std(), 4)}
                if self._task_type == "classification":
                    row["Accuracy"] = round(accuracy_score(self._y_test, y_pred), 4)
                    avg = "binary" if len(np.unique(self._y_test)) == 2 else "weighted"
                    row["F1"] = round(f1_score(self._y_test, y_pred, average=avg, zero_division=0), 4)
                else:
                    row["R2"] = round(r2_score(self._y_test, y_pred), 4)
                    row["RMSE"] = round(np.sqrt(mean_squared_error(self._y_test, y_pred)), 4)
                    row["MAE"] = round(mean_absolute_error(self._y_test, y_pred), 4)
                results.append(row)
                self._log(f"  Compared: {name}")
            except Exception as exc:
                self._log(f"  Skipped {name}: {exc}")

        if not results:
            self._log("No models compared successfully.")
            return None

        df_comp = pd.DataFrame(results)
        self._log("\n" + "=" * 70)
        self._log("MODEL COMPARISON TABLE")
        self._log("-" * 70)
        self._log(df_comp.to_string(index=False))
        self._log("=" * 70)

        # Show in metrics table
        cols = list(df_comp.columns)
        self._metrics_table.setRowCount(len(df_comp))
        self._metrics_table.setColumnCount(len(cols))
        self._metrics_table.setHorizontalHeaderLabels(cols)
        for r in range(len(df_comp)):
            for c in range(len(cols)):
                self._metrics_table.setItem(r, c, QTableWidgetItem(str(df_comp.iloc[r, c])))
        self._right_tabs.setCurrentIndex(3)
        return df_comp

    # ------------------------------------------------------------------
    # Feature Importance Ranking with Visualization
    # ------------------------------------------------------------------

    def _feature_importance_ranking(self):
        """Compute and visualize feature importance using permutation importance."""
        if self._model is None or self._X_test is None:
            self._log("Train a model first before computing feature importance.")
            return
        try:
            from sklearn.inspection import permutation_importance
            self._log("Computing permutation importance (this may take a moment)...")
            scoring = 'accuracy' if self._task_type == 'classification' else 'r2'
            r = permutation_importance(
                self._model, self._X_test, self._y_test,
                n_repeats=10, random_state=42, scoring=scoring
            )
            importance_mean = r.importances_mean
            importance_std = r.importances_std

            # Get feature names
            if self._df is not None:
                feature_txt = self._txt_features.text().strip()
                if feature_txt:
                    feature_names = [c.strip() for c in feature_txt.split(',')]
                else:
                    target_col = self._combo_target.currentText()
                    feature_names = [c for c in self._df.select_dtypes(include=[np.number]).columns
                                     if c != target_col]
            else:
                feature_names = [f"Feature {i}" for i in range(len(importance_mean))]

            # If PCA was applied, use component names
            if self._pca is not None:
                feature_names = [f"PC{i+1}" for i in range(len(importance_mean))]

            # Sort by importance
            sorted_idx = np.argsort(importance_mean)
            sorted_names = [feature_names[i] if i < len(feature_names) else f"F{i}"
                           for i in sorted_idx]
            sorted_imp = importance_mean[sorted_idx]
            sorted_std = importance_std[sorted_idx]

            # Plot
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            y_pos = np.arange(len(sorted_names))
            ax.barh(y_pos, sorted_imp, xerr=sorted_std, align='center',
                    color='#3498db', alpha=0.8, edgecolor='white')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names, fontsize=8)
            ax.set_xlabel('Permutation Importance (decrease in score)')
            ax.set_title('Feature Importance Ranking')
            ax.grid(True, axis='x', alpha=0.3)
            self._figure.tight_layout()
            self._canvas.draw()
            self._right_tabs.setCurrentIndex(1)

            # Log results
            self._log("\n" + "=" * 50)
            self._log("FEATURE IMPORTANCE RANKING")
            self._log("-" * 50)
            for name, imp, std in zip(reversed(sorted_names),
                                       reversed(sorted_imp),
                                       reversed(sorted_std)):
                self._log(f"  {name:20s}: {imp:+.4f} +/- {std:.4f}")
            self._log("=" * 50)
        except Exception as exc:
            self._log(f"Feature importance error: {exc}")

    # ------------------------------------------------------------------
    # Automated Hyperparameter Tuning (GridSearchCV)
    # ------------------------------------------------------------------

    def _grid_search_tuning(self):
        """Automated hyperparameter tuning using GridSearchCV."""
        if self._X_train is None:
            self._log("Prepare data first (run the pipeline).")
            return
        model_name = self._combo_model.currentText()
        self._log(f"Starting GridSearchCV for {model_name}...")

        try:
            from sklearn.model_selection import GridSearchCV

            # Define parameter grids for each model
            param_grids = {
                "KNN": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
                "Decision Tree": {"max_depth": [3, 5, 10, 15, None], "min_samples_split": [2, 5, 10]},
                "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]},
                "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
                "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "max_iter": [200, 500]},
                "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.01, 0.1, 1.0, 10.0]},
                "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "epsilon": [0.01, 0.1, 0.5]},
                "Decision Tree Regressor": {"max_depth": [3, 5, 10, 15, None], "min_samples_split": [2, 5, 10]},
                "Random Forest Regressor": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]},
            }

            grid = param_grids.get(model_name)
            if grid is None:
                self._log(f"No predefined grid for {model_name}. Using default parameters.")
                return

            # Build base estimator
            estimator = self._build_model(model_name, {})
            scoring = 'accuracy' if self._task_type == 'classification' else 'r2'

            gs = GridSearchCV(estimator, grid, cv=min(5, len(self._X_train)),
                              scoring=scoring, n_jobs=-1, refit=True)
            gs.fit(self._X_train, self._y_train)

            self._log("\n" + "=" * 50)
            self._log(f"GRID SEARCH RESULTS - {model_name}")
            self._log("-" * 50)
            self._log(f"Best score ({scoring}): {gs.best_score_:.4f}")
            self._log(f"Best params: {gs.best_params_}")
            self._log("-" * 50)

            # Show top 5 parameter combinations
            results_df = pd.DataFrame(gs.cv_results_)
            results_df = results_df.sort_values('rank_test_score')
            for idx, row in results_df.head(5).iterrows():
                self._log(f"  Rank {int(row['rank_test_score'])}: "
                         f"score={row['mean_test_score']:.4f} +/- {row['std_test_score']:.4f} "
                         f"params={row['params']}")
            self._log("=" * 50)

            # Apply best model
            self._model = gs.best_estimator_
            self._model_params = gs.best_params_
            self._log(f"Best model applied. Re-evaluate on test set for final metrics.")

            # Evaluate on test set
            y_pred = self._model.predict(self._X_test)
            if self._task_type == 'classification':
                acc = accuracy_score(self._y_test, y_pred)
                self._log(f"Test accuracy with best params: {acc:.4f}")
            else:
                r2 = r2_score(self._y_test, y_pred)
                self._log(f"Test R2 with best params: {r2:.4f}")

        except Exception as exc:
            self._log(f"GridSearch error: {exc}")

    # ------------------------------------------------------------------
    # Explainability: Permutation-based feature contribution
    # ------------------------------------------------------------------

    def _explain_prediction(self):
        """Approximate SHAP-like feature contribution for individual predictions."""
        if self._model is None or self._X_test is None:
            self._log("Train a model first.")
            return
        try:
            n_test = self._X_test.shape[0]
            sample_idx, ok = QInputDialog.getInt(
                self, "Explain Prediction",
                f"Test sample index (0 to {n_test - 1}):", 0, 0, n_test - 1
            )
            if not ok:
                return

            sample = self._X_test[sample_idx:sample_idx + 1]
            baseline_pred = self._model.predict(sample)[0]

            # Get feature names
            if self._df is not None:
                feature_txt = self._txt_features.text().strip()
                if feature_txt:
                    feature_names = [c.strip() for c in feature_txt.split(',')]
                else:
                    target_col = self._combo_target.currentText()
                    feature_names = [c for c in self._df.select_dtypes(include=[np.number]).columns
                                     if c != target_col]
            else:
                feature_names = [f"Feature {i}" for i in range(sample.shape[1])]

            if self._pca is not None:
                feature_names = [f"PC{i+1}" for i in range(sample.shape[1])]

            # Compute contribution by permuting each feature
            n_features = sample.shape[1]
            contributions = np.zeros(n_features)
            n_repeats = 50

            for feat_idx in range(n_features):
                deltas = []
                for _ in range(n_repeats):
                    perturbed = sample.copy()
                    # Replace with random value from training set
                    rand_idx = np.random.randint(0, self._X_train.shape[0])
                    perturbed[0, feat_idx] = self._X_train[rand_idx, feat_idx]
                    new_pred = self._model.predict(perturbed)[0]
                    if self._task_type == 'classification':
                        deltas.append(1.0 if new_pred != baseline_pred else 0.0)
                    else:
                        deltas.append(baseline_pred - new_pred)
                contributions[feat_idx] = np.mean(deltas)

            # Plot
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            sorted_idx = np.argsort(np.abs(contributions))
            names = [feature_names[i] if i < len(feature_names) else f"F{i}"
                     for i in sorted_idx]
            vals = contributions[sorted_idx]
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
            y_pos = np.arange(len(names))
            ax.barh(y_pos, vals, color=colors, alpha=0.8, edgecolor='white')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Feature Contribution')
            ax.set_title(f'Prediction Explanation (sample #{sample_idx}, pred={baseline_pred})')
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(True, axis='x', alpha=0.3)
            self._figure.tight_layout()
            self._canvas.draw()
            self._right_tabs.setCurrentIndex(1)

            self._log(f"\nExplanation for test sample #{sample_idx}:")
            self._log(f"  Prediction: {baseline_pred}")
            if self._y_test is not None:
                self._log(f"  True value: {self._y_test[sample_idx]}")
            for name, val in zip(reversed(names), reversed(vals)):
                self._log(f"  {name:20s}: {val:+.4f}")

        except Exception as exc:
            self._log(f"Explanation error: {exc}")

    # ------------------------------------------------------------------
    # Time Series Forecasting (Simple ARIMA-like)
    # ------------------------------------------------------------------

    def _time_series_forecast(self):
        """Simple autoregressive time series forecasting."""
        if self._df is None or self._df.empty:
            self._log("Load data first.")
            return
        num_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            self._log("No numeric columns available for forecasting.")
            return

        col, ok = QInputDialog.getItem(
            self, "Time Series Forecast", "Column to forecast:", num_cols, editable=False
        )
        if not ok:
            return

        horizon, ok = QInputDialog.getInt(
            self, "Forecast Horizon", "Steps to forecast:", 20, 1, 500
        )
        if not ok:
            return

        ar_order, ok = QInputDialog.getInt(
            self, "AR Order", "Autoregressive order (lags):", 5, 1, 50
        )
        if not ok:
            return

        try:
            series = self._df[col].dropna().values.astype(float)
            n = len(series)
            if n < ar_order + 10:
                self._log(f"Need at least {ar_order + 10} data points.")
                return

            # Difference the series for stationarity (d=1)
            diff_series = np.diff(series)

            # Build AR model using least squares
            X_ar = np.zeros((len(diff_series) - ar_order, ar_order))
            y_ar = diff_series[ar_order:]
            for i in range(ar_order):
                X_ar[:, i] = diff_series[ar_order - i - 1:len(diff_series) - i - 1]

            # Fit using least squares
            coeffs, residuals, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)

            # Forecast
            forecast_diff = []
            last_values = list(diff_series[-ar_order:])
            for _ in range(horizon):
                pred = np.dot(coeffs, last_values[-ar_order:][::-1])
                forecast_diff.append(pred)
                last_values.append(pred)

            # Integrate (undo differencing)
            forecast = np.zeros(horizon)
            forecast[0] = series[-1] + forecast_diff[0]
            for i in range(1, horizon):
                forecast[i] = forecast[i - 1] + forecast_diff[i]

            # Compute simple confidence intervals from residuals
            if len(residuals) > 0:
                residual_std = np.sqrt(residuals[0] / len(y_ar))
            else:
                residual_std = np.std(y_ar - X_ar @ coeffs)
            ci_mult = np.sqrt(np.arange(1, horizon + 1))
            upper = forecast + 1.96 * residual_std * ci_mult
            lower = forecast - 1.96 * residual_std * ci_mult

            # Plot
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            t_hist = np.arange(n)
            t_fore = np.arange(n, n + horizon)

            ax.plot(t_hist, series, 'b-', linewidth=1, label='Historical')
            ax.plot(t_fore, forecast, 'r-', linewidth=1.5, label='Forecast')
            ax.fill_between(t_fore, lower, upper, alpha=0.2, color='red', label='95% CI')
            ax.axvline(n - 0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time Index')
            ax.set_ylabel(col)
            ax.set_title(f'Time Series Forecast: {col} (AR({ar_order}), d=1)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            self._figure.tight_layout()
            self._canvas.draw()
            self._right_tabs.setCurrentIndex(1)

            self._log(f"\nTime Series Forecast for '{col}':")
            self._log(f"  Model: AR({ar_order}) with d=1 differencing")
            self._log(f"  AR coefficients: {np.round(coeffs, 4).tolist()}")
            self._log(f"  Residual std: {residual_std:.4f}")
            self._log(f"  Forecast ({horizon} steps): "
                      f"[{forecast[0]:.4f}, ..., {forecast[-1]:.4f}]")

        except Exception as exc:
            self._log(f"Forecast error: {exc}")

    # ------------------------------------------------------------------
    # Anomaly Detection (Isolation Forest, LOF)
    # ------------------------------------------------------------------

    def _anomaly_detection(self):
        """Detect anomalies using Isolation Forest or Local Outlier Factor."""
        if self._df is None or self._df.empty:
            self._log("Load data first.")
            return

        methods = ["Isolation Forest", "Local Outlier Factor (LOF)"]
        method, ok = QInputDialog.getItem(
            self, "Anomaly Detection", "Method:", methods, editable=False
        )
        if not ok:
            return

        contamination, ok = QInputDialog.getDouble(
            self, "Contamination", "Expected anomaly fraction (0.01-0.5):",
            0.05, 0.01, 0.5, 3
        )
        if not ok:
            return

        try:
            # Prepare numeric data
            num_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                self._log("No numeric columns for anomaly detection.")
                return

            X = self._df[num_cols].dropna().values
            if len(X) < 10:
                self._log("Need at least 10 data points.")
                return

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if "Isolation" in method:
                from sklearn.ensemble import IsolationForest
                detector = IsolationForest(
                    contamination=contamination, random_state=42, n_estimators=100
                )
                labels = detector.fit_predict(X_scaled)
                scores = detector.decision_function(X_scaled)
            else:
                from sklearn.neighbors import LocalOutlierFactor
                detector = LocalOutlierFactor(
                    n_neighbors=20, contamination=contamination
                )
                labels = detector.fit_predict(X_scaled)
                scores = detector.negative_outlier_factor_

            anomalies = labels == -1
            n_anomalies = anomalies.sum()

            # Add results to dataframe
            valid_idx = self._df[num_cols].dropna().index
            self._df.loc[valid_idx, 'anomaly_label'] = labels
            self._df.loc[valid_idx, 'anomaly_score'] = scores

            # Plot
            self._figure.clear()
            if X.shape[1] >= 2:
                # Use first 2 features or PCA
                if X.shape[1] > 2:
                    from sklearn.decomposition import PCA as PCA_
                    pca = PCA_(n_components=2)
                    X_2d = pca.fit_transform(X_scaled)
                    xlabel, ylabel = 'PC1', 'PC2'
                else:
                    X_2d = X_scaled
                    xlabel, ylabel = num_cols[0], num_cols[1]

                ax = self._figure.add_subplot(111)
                normal = ~anomalies
                ax.scatter(X_2d[normal, 0], X_2d[normal, 1], c='#3498db',
                          s=15, alpha=0.5, label=f'Normal ({normal.sum()})')
                ax.scatter(X_2d[anomalies, 0], X_2d[anomalies, 1], c='#e74c3c',
                          s=30, marker='x', linewidth=2, label=f'Anomaly ({n_anomalies})')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'Anomaly Detection: {method}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax = self._figure.add_subplot(111)
                ax.plot(scores, 'b-', linewidth=0.5, alpha=0.7)
                ax.scatter(np.where(anomalies)[0], scores[anomalies],
                          c='red', s=20, zorder=5, label=f'Anomalies ({n_anomalies})')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Anomaly Score')
                ax.set_title(f'Anomaly Detection: {method}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            self._figure.tight_layout()
            self._canvas.draw()
            self._right_tabs.setCurrentIndex(1)

            self._log(f"\n{'=' * 50}")
            self._log(f"ANOMALY DETECTION - {method}")
            self._log(f"{'-' * 50}")
            self._log(f"  Features used: {len(num_cols)}")
            self._log(f"  Samples: {len(X)}")
            self._log(f"  Contamination: {contamination}")
            self._log(f"  Anomalies found: {n_anomalies} ({n_anomalies/len(X)*100:.1f}%)")
            self._log(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            self._log(f"  Added 'anomaly_label' and 'anomaly_score' columns to data.")
            self._log(f"{'=' * 50}")

        except Exception as exc:
            self._log(f"Anomaly detection error: {exc}")

    # ------------------------------------------------------------------
    # Generate Prediction Report (HTML)
    # ------------------------------------------------------------------

    def generate_report(self, path=None):
        """Generate an HTML report with model info, metrics, and plots.
        Returns the file path of the generated report."""
        if not hasattr(self, "_last_result") or self._last_result is None:
            self._log("No results available. Train a model first.")
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Report", "ml_report.html",
                "HTML files (*.html);;All files (*)"
            )
        if not path:
            return None

        r = self._last_result
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate plot as base64
        import base64
        fig_tmp = Figure(figsize=(6, 4), dpi=100)
        style_figure(fig_tmp)
        ax = fig_tmp.add_subplot(111)
        if self._task_type == "classification" and "confusion_matrix" in r:
            cm = r["confusion_matrix"]
            ax.imshow(cm, cmap="Blues", aspect="auto")
            n = cm.shape[0]
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
        elif self._task_type == "regression" and "y_pred" in r:
            residuals = self._y_test - r["y_pred"]
            ax.scatter(r["y_pred"], residuals, s=10, alpha=0.6)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
        else:
            ax.text(0.5, 0.5, "No plot available", ha="center", va="center")

        buf = io.BytesIO()
        fig_tmp.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        # Build metrics HTML
        metrics_rows = ""
        if self._task_type == "classification":
            for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if m in r:
                    metrics_rows += f"<tr><td><strong>{m.capitalize()}</strong></td><td>{r[m]:.4f}</td></tr>\n"
        elif self._task_type == "regression":
            for m in ["r2", "mse", "rmse", "mae"]:
                if m in r:
                    metrics_rows += f"<tr><td><strong>{m.upper()}</strong></td><td>{r[m]:.4f}</td></tr>\n"

        data_info = ""
        if self._df is not None:
            data_info = f"{self._df.shape[0]} rows x {self._df.shape[1]} cols"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ML Prediction Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2980b9; }}
  table {{ border-collapse: collapse; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 14px; text-align: left; }}
  th {{ background-color: #3498db; color: white; }}
  tr:nth-child(even) {{ background-color: #f9f9f9; }}
  .meta {{ color: #777; font-size: 0.9em; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
</style></head>
<body>
<h1>Machine Learning Prediction Report</h1>
<p class="meta">Generated: {ts} | QuantumRes Scientific Suite</p>

<h2>Model Information</h2>
<table>
<tr><td><strong>Model</strong></td><td>{r.get('model', 'N/A')}</td></tr>
<tr><td><strong>Task Type</strong></td><td>{self._task_type}</td></tr>
<tr><td><strong>Dataset</strong></td><td>{data_info}</td></tr>
<tr><td><strong>Parameters</strong></td><td>{self._model_params or 'defaults'}</td></tr>
</table>

<h2>Performance Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{metrics_rows}
</table>

<h2>Visualization</h2>
<img src="data:image/png;base64,{img_b64}" alt="Model Plot">

{"<h2>Classification Report</h2><pre>" + r.get("classification_report", "") + "</pre>" if "classification_report" in r else ""}

</body></html>"""

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            self._log(f"Report saved to: {path}")
            return path
        except Exception as exc:
            self._log(f"Error saving report: {exc}")
            return None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = MLToolsWidget()
    win.setWindowTitle("ML Tools - QuantumRes")
    win.resize(1100, 700)
    win.show()
    sys.exit(app.exec_())
