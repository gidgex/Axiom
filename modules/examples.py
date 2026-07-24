"""Built-in example datasets and tutorials for Axiom Scientific Suite."""
import numpy as np


def _iris_data():
    """Generate a synthetic Iris-like dataset (150 samples, 4 features + species)."""
    rng = np.random.RandomState(42)
    # Setosa
    setosa = rng.normal(loc=[5.0, 3.4, 1.5, 0.2], scale=[0.35, 0.38, 0.17, 0.10], size=(50, 4))
    # Versicolor
    versicolor = rng.normal(loc=[5.9, 2.8, 4.3, 1.3], scale=[0.52, 0.31, 0.47, 0.20], size=(50, 4))
    # Virginica
    virginica = rng.normal(loc=[6.6, 3.0, 5.6, 2.0], scale=[0.64, 0.32, 0.55, 0.27], size=(50, 4))
    data = np.vstack([setosa, versicolor, virginica])
    species = np.array(["setosa"] * 50 + ["versicolor"] * 50 + ["virginica"] * 50)
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    return data, species, columns


def _temperature_series():
    """Generate a realistic synthetic daily temperature time series (365 days)."""
    t = np.arange(365)
    seasonal = 15 + 12 * np.sin(2 * np.pi * (t - 80) / 365)
    rng = np.random.RandomState(7)
    noise = rng.normal(0, 2.5, 365)
    trend = 0.005 * t
    return t, seasonal + noise + trend


def _materials_data():
    """Synthetic materials property dataset: density, thermal conductivity, Young's modulus."""
    materials = [
        "Aluminum", "Copper", "Steel", "Titanium", "Nickel",
        "Brass", "Bronze", "Zinc", "Lead", "Tungsten",
        "Silver", "Gold", "Platinum", "Iron", "Magnesium",
    ]
    rng = np.random.RandomState(99)
    density = np.array([2.70, 8.96, 7.85, 4.51, 8.90,
                        8.50, 8.80, 7.13, 11.34, 19.25,
                        10.49, 19.30, 21.45, 7.87, 1.74])
    conductivity = np.array([237, 401, 50, 22, 91,
                             109, 50, 116, 35, 174,
                             429, 318, 72, 80, 156], dtype=float)
    youngs_modulus = np.array([69, 130, 200, 116, 200,
                               100, 110, 108, 16, 411,
                               83, 79, 168, 211, 45], dtype=float)
    # Add small measurement noise
    conductivity += rng.normal(0, 2, len(materials))
    youngs_modulus += rng.normal(0, 1.5, len(materials))
    return materials, density, conductivity, youngs_modulus


EXAMPLES = {
    "2D Plotter": {
        "Damped Oscillation": {
            "description": "Exponentially decaying sine wave",
            "code": "x = np.linspace(0, 10, 500)\ny = np.exp(-x/3) * np.sin(5*x)",
            "x": lambda: np.linspace(0, 10, 500),
            "y": lambda: np.exp(-np.linspace(0, 10, 500) / 3) * np.sin(5 * np.linspace(0, 10, 500)),
        },
        "Gaussian Peak": {
            "description": "Normalized Gaussian centered at x=5 with sigma=0.8",
            "code": "x = np.linspace(0, 10, 500)\ny = (1/(0.8*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-5)/0.8)**2)",
            "x": lambda: np.linspace(0, 10, 500),
            "y": lambda: (1 / (0.8 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((np.linspace(0, 10, 500) - 5) / 0.8) ** 2
            ),
        },
        "Noisy Sine": {
            "description": "Sine wave with additive Gaussian noise (SNR ~10 dB)",
            "code": (
                "x = np.linspace(0, 4*np.pi, 600)\n"
                "rng = np.random.RandomState(0)\n"
                "y = np.sin(x) + 0.3*rng.randn(len(x))"
            ),
            "x": lambda: np.linspace(0, 4 * np.pi, 600),
            "y": lambda: np.sin(np.linspace(0, 4 * np.pi, 600))
            + 0.3 * np.random.RandomState(0).randn(600),
        },
        "Power Spectrum": {
            "description": "1/f noise power spectral density on log-log axes",
            "code": (
                "freq = np.logspace(-2, 2, 1000)\n"
                "rng = np.random.RandomState(3)\n"
                "psd = (1/freq) * (1 + 0.2*rng.randn(len(freq)))\n"
                "psd = np.clip(psd, 1e-4, None)"
            ),
            "x": lambda: np.logspace(-2, 2, 1000),
            "y": lambda: np.clip(
                (1 / np.logspace(-2, 2, 1000))
                * (1 + 0.2 * np.random.RandomState(3).randn(1000)),
                1e-4,
                None,
            ),
        },
        "Bessel Functions": {
            "description": "First three Bessel functions of the first kind J0, J1, J2",
            "code": (
                "from scipy.special import jv\n"
                "x = np.linspace(0, 20, 500)\n"
                "y0 = jv(0, x)\ny1 = jv(1, x)\ny2 = jv(2, x)"
            ),
            "x": lambda: np.linspace(0, 20, 500),
            "y": lambda: np.column_stack([
                np.sinc(np.linspace(0, 20, 500) / np.pi),  # approximate J0
                np.sin(np.linspace(0, 20, 500)) / np.linspace(1e-9, 20, 500),
            ]),
        },
    },

    "Data Analysis": {
        "Iris Dataset": {
            "description": "Classic 150-sample dataset with 4 features across 3 species",
            "data": lambda: _iris_data()[0],
            "labels": lambda: _iris_data()[1],
            "columns": lambda: _iris_data()[2],
        },
        "Temperature Series": {
            "description": "Synthetic daily temperature over one year with seasonal trend and noise",
            "x": lambda: _temperature_series()[0],
            "y": lambda: _temperature_series()[1],
            "columns": lambda: ["day", "temperature_C"],
        },
        "Materials Properties": {
            "description": "Physical properties of 15 metals: density, thermal conductivity, Young's modulus",
            "data": lambda: np.column_stack([
                _materials_data()[1],
                _materials_data()[2],
                _materials_data()[3],
            ]),
            "labels": lambda: _materials_data()[0],
            "columns": lambda: ["density_g_cm3", "conductivity_W_mK", "youngs_modulus_GPa"],
        },
        "Random Walk Portfolio": {
            "description": "Simulated 5-asset portfolio with correlated random walks (252 trading days)",
            "data": lambda: (
                lambda rng: np.cumsum(
                    rng.multivariate_normal(
                        mean=[0.0003, 0.0001, 0.0005, -0.0001, 0.0002],
                        cov=np.array([
                            [1.0, 0.6, 0.3, -0.2, 0.1],
                            [0.6, 1.0, 0.4, -0.1, 0.2],
                            [0.3, 0.4, 1.0, 0.0, 0.3],
                            [-0.2, -0.1, 0.0, 1.0, -0.3],
                            [0.1, 0.2, 0.3, -0.3, 1.0],
                        ]) * 0.0004,
                        size=252,
                    ),
                    axis=0,
                )
            )(np.random.RandomState(11)),
            "columns": lambda: ["Asset_A", "Asset_B", "Asset_C", "Asset_D", "Asset_E"],
        },
    },

    "Math Engine": {
        "3x3 Eigenvalue Problem": {
            "description": "Symmetric 3x3 matrix with known eigenvalues (1, 2, 3)",
            "code": (
                "A = np.array([[2, -1, 0],\n"
                "              [-1, 2, -1],\n"
                "              [0, -1, 2]])\n"
                "eigenvalues, eigenvectors = np.linalg.eigh(A)"
            ),
            "matrix": lambda: np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float),
        },
        "Linear System 4x4": {
            "description": "Solve Ax = b for a well-conditioned 4x4 system",
            "code": (
                "A = np.array([[4, 1, -1, 0],\n"
                "              [1, 4, 0, -1],\n"
                "              [-1, 0, 4, 1],\n"
                "              [0, -1, 1, 4]], dtype=float)\n"
                "b = np.array([15, 10, 10, 15], dtype=float)\n"
                "x = np.linalg.solve(A, b)"
            ),
            "A": lambda: np.array(
                [[4, 1, -1, 0], [1, 4, 0, -1], [-1, 0, 4, 1], [0, -1, 1, 4]], dtype=float
            ),
            "b": lambda: np.array([15, 10, 10, 15], dtype=float),
        },
        "Vandermonde Interpolation": {
            "description": "Polynomial interpolation through 6 data points via Vandermonde matrix",
            "code": (
                "x_pts = np.array([0, 1, 2, 3, 4, 5], dtype=float)\n"
                "y_pts = np.array([1.0, 2.7, 7.4, 20.1, 54.6, 148.4])\n"
                "V = np.vander(x_pts, increasing=True)\n"
                "coeffs = np.linalg.solve(V, y_pts)"
            ),
            "x": lambda: np.array([0, 1, 2, 3, 4, 5], dtype=float),
            "y": lambda: np.array([1.0, 2.7, 7.4, 20.1, 54.6, 148.4]),
        },
    },

    "Curve Fitting": {
        "Noisy Gaussian": {
            "description": "Gaussian peak (amp=5, mu=3, sigma=0.6) with noise for fitting",
            "code": (
                "x = np.linspace(0, 6, 200)\n"
                "rng = np.random.RandomState(42)\n"
                "y = 5 * np.exp(-0.5*((x - 3)/0.6)**2) + 0.3*rng.randn(200)"
            ),
            "x": lambda: np.linspace(0, 6, 200),
            "y": lambda: 5 * np.exp(-0.5 * ((np.linspace(0, 6, 200) - 3) / 0.6) ** 2)
            + 0.3 * np.random.RandomState(42).randn(200),
            "true_params": {"amplitude": 5.0, "mu": 3.0, "sigma": 0.6},
        },
        "Exponential Decay": {
            "description": "Exponential decay (A=10, tau=2.5) with Poisson-like noise",
            "code": (
                "x = np.linspace(0, 15, 150)\n"
                "rng = np.random.RandomState(7)\n"
                "y_true = 10 * np.exp(-x / 2.5)\n"
                "y = y_true + rng.normal(0, 0.4, len(x))"
            ),
            "x": lambda: np.linspace(0, 15, 150),
            "y": lambda: 10 * np.exp(-np.linspace(0, 15, 150) / 2.5)
            + np.random.RandomState(7).normal(0, 0.4, 150),
            "true_params": {"A": 10.0, "tau": 2.5},
        },
        "Double Peak": {
            "description": "Sum of two Gaussians (partially overlapping) with noise",
            "code": (
                "x = np.linspace(-2, 8, 300)\n"
                "rng = np.random.RandomState(5)\n"
                "y = (3*np.exp(-0.5*((x-1.5)/0.5)**2) +\n"
                "     4.5*np.exp(-0.5*((x-4.0)/0.8)**2) +\n"
                "     0.2*rng.randn(300))"
            ),
            "x": lambda: np.linspace(-2, 8, 300),
            "y": lambda: (
                3 * np.exp(-0.5 * ((np.linspace(-2, 8, 300) - 1.5) / 0.5) ** 2)
                + 4.5 * np.exp(-0.5 * ((np.linspace(-2, 8, 300) - 4.0) / 0.8) ** 2)
                + 0.2 * np.random.RandomState(5).randn(300)
            ),
            "true_params": {
                "peak1": {"amplitude": 3.0, "mu": 1.5, "sigma": 0.5},
                "peak2": {"amplitude": 4.5, "mu": 4.0, "sigma": 0.8},
            },
        },
        "Logistic Growth": {
            "description": "Logistic growth curve (L=100, k=0.5, x0=10) with noise for fitting",
            "code": (
                "x = np.linspace(0, 25, 180)\n"
                "rng = np.random.RandomState(13)\n"
                "y = 100 / (1 + np.exp(-0.5*(x - 10))) + 2*rng.randn(180)"
            ),
            "x": lambda: np.linspace(0, 25, 180),
            "y": lambda: 100 / (1 + np.exp(-0.5 * (np.linspace(0, 25, 180) - 10)))
            + 2 * np.random.RandomState(13).randn(180),
            "true_params": {"L": 100.0, "k": 0.5, "x0": 10.0},
        },
    },

    "FEM Solver": {
        "Heat Sink": {
            "description": "Rectangular domain with central heat source, fixed-temperature boundaries",
            "domain": {"width": 0.1, "height": 0.05, "unit": "m"},
            "boundary_conditions": {
                "left": {"type": "dirichlet", "value": 300.0},
                "right": {"type": "dirichlet", "value": 300.0},
                "top": {"type": "neumann", "flux": 0.0},
                "bottom": {"type": "neumann", "flux": 0.0},
            },
            "source": {"x_center": 0.05, "y_center": 0.025, "radius": 0.01, "power": 1e5},
            "material": {"conductivity": 237.0, "name": "Aluminum"},
            "mesh_density": 20,
        },
        "Cantilever Beam": {
            "description": "2D cantilever beam with point load at free end",
            "domain": {"length": 1.0, "height": 0.1, "unit": "m"},
            "boundary_conditions": {
                "left": {"type": "fixed"},
                "right_tip": {"type": "point_load", "Fy": -1000.0},
            },
            "material": {"E": 210e9, "nu": 0.3, "name": "Steel"},
            "mesh_density": 16,
        },
        "Thermal Gradient": {
            "description": "L-shaped domain with temperature gradient between two faces",
            "domain": {"type": "L-shape", "outer": 0.1, "cutout": 0.05, "unit": "m"},
            "boundary_conditions": {
                "hot_face": {"type": "dirichlet", "value": 400.0},
                "cold_face": {"type": "dirichlet", "value": 300.0},
                "insulated": {"type": "neumann", "flux": 0.0},
            },
            "material": {"conductivity": 50.0, "name": "Steel"},
            "mesh_density": 24,
        },
    },

    "Signal Processing": {
        "Mixed Frequencies": {
            "description": "Superposition of 50 Hz, 120 Hz, and 300 Hz sinusoids sampled at 2 kHz",
            "code": (
                "fs = 2000\n"
                "t = np.arange(0, 1, 1/fs)\n"
                "signal = (1.0*np.sin(2*np.pi*50*t) +\n"
                "          0.5*np.sin(2*np.pi*120*t) +\n"
                "          0.3*np.sin(2*np.pi*300*t))"
            ),
            "x": lambda: np.arange(0, 1, 1 / 2000),
            "y": lambda: (
                1.0 * np.sin(2 * np.pi * 50 * np.arange(0, 1, 1 / 2000))
                + 0.5 * np.sin(2 * np.pi * 120 * np.arange(0, 1, 1 / 2000))
                + 0.3 * np.sin(2 * np.pi * 300 * np.arange(0, 1, 1 / 2000))
            ),
            "sample_rate": 2000,
            "frequencies_hz": [50, 120, 300],
        },
        "Chirp Signal": {
            "description": "Linear chirp sweeping from 10 Hz to 500 Hz over 2 seconds",
            "code": (
                "fs = 4000\n"
                "t = np.arange(0, 2, 1/fs)\n"
                "f0, f1 = 10, 500\n"
                "phase = 2*np.pi * (f0*t + (f1-f0)/(2*2)*t**2)\n"
                "signal = np.sin(phase)"
            ),
            "x": lambda: np.arange(0, 2, 1 / 4000),
            "y": lambda: np.sin(
                2 * np.pi * (10 * np.arange(0, 2, 1 / 4000)
                             + (500 - 10) / 4 * np.arange(0, 2, 1 / 4000) ** 2)
            ),
            "sample_rate": 4000,
        },
        "AM Modulated Signal": {
            "description": "Amplitude-modulated carrier (1 kHz) with 50 Hz modulating signal",
            "code": (
                "fs = 8000\n"
                "t = np.arange(0, 0.5, 1/fs)\n"
                "carrier = np.sin(2*np.pi*1000*t)\n"
                "modulator = 1 + 0.7*np.sin(2*np.pi*50*t)\n"
                "signal = modulator * carrier"
            ),
            "x": lambda: np.arange(0, 0.5, 1 / 8000),
            "y": lambda: (
                (1 + 0.7 * np.sin(2 * np.pi * 50 * np.arange(0, 0.5, 1 / 8000)))
                * np.sin(2 * np.pi * 1000 * np.arange(0, 0.5, 1 / 8000))
            ),
            "sample_rate": 8000,
        },
    },

    "Statistics": {
        "Two-Sample Test Data": {
            "description": "Two groups with slightly different means for t-test demonstration",
            "code": (
                "rng = np.random.RandomState(22)\n"
                "group_a = rng.normal(loc=50.0, scale=5.0, size=40)\n"
                "group_b = rng.normal(loc=53.0, scale=5.5, size=45)"
            ),
            "group_a": lambda: np.random.RandomState(22).normal(50.0, 5.0, 40),
            "group_b": lambda: np.random.RandomState(23).normal(53.0, 5.5, 45),
            "true_params": {"mean_a": 50.0, "mean_b": 53.0, "std_a": 5.0, "std_b": 5.5},
        },
        "ANOVA Groups": {
            "description": "Four treatment groups for one-way ANOVA (unequal means)",
            "code": (
                "rng = np.random.RandomState(10)\n"
                "control = rng.normal(20.0, 3.0, 30)\n"
                "treatment_a = rng.normal(22.5, 3.5, 30)\n"
                "treatment_b = rng.normal(25.0, 3.0, 30)\n"
                "treatment_c = rng.normal(21.0, 2.8, 30)"
            ),
            "groups": lambda: {
                "control": np.random.RandomState(10).normal(20.0, 3.0, 30),
                "treatment_a": np.random.RandomState(11).normal(22.5, 3.5, 30),
                "treatment_b": np.random.RandomState(12).normal(25.0, 3.0, 30),
                "treatment_c": np.random.RandomState(13).normal(21.0, 2.8, 30),
            },
            "true_means": {"control": 20.0, "treatment_a": 22.5, "treatment_b": 25.0, "treatment_c": 21.0},
        },
        "Correlation Matrix Demo": {
            "description": "5 correlated variables for correlation/regression analysis",
            "code": (
                "rng = np.random.RandomState(55)\n"
                "cov = np.array([[1, .8, .3, -.2, .1],\n"
                "                [.8, 1, .5, -.1, .2],\n"
                "                [.3, .5, 1, .4, .6],\n"
                "                [-.2,-.1, .4, 1, .3],\n"
                "                [.1, .2, .6, .3, 1]])\n"
                "data = rng.multivariate_normal(np.zeros(5), cov, 100)"
            ),
            "data": lambda: np.random.RandomState(55).multivariate_normal(
                np.zeros(5),
                np.array([
                    [1, .8, .3, -.2, .1],
                    [.8, 1, .5, -.1, .2],
                    [.3, .5, 1, .4, .6],
                    [-.2, -.1, .4, 1, .3],
                    [.1, .2, .6, .3, 1],
                ]),
                100,
            ),
            "columns": lambda: ["X1", "X2", "X3", "X4", "X5"],
        },
    },

    "ML / AI": {
        "Classification Demo": {
            "description": "Two interleaving half-moon clusters (300 samples) for binary classification",
            "code": (
                "rng = np.random.RandomState(1)\n"
                "n = 150\n"
                "theta1 = np.linspace(0, np.pi, n)\n"
                "theta2 = np.linspace(0, np.pi, n)\n"
                "X = np.vstack([\n"
                "    np.column_stack([np.cos(theta1), np.sin(theta1)]) + 0.1*rng.randn(n,2),\n"
                "    np.column_stack([1-np.cos(theta2), 1-np.sin(theta2)-0.5]) + 0.1*rng.randn(n,2)\n"
                "])\n"
                "y = np.array([0]*n + [1]*n)"
            ),
            "data": lambda: (
                lambda rng, n: np.vstack([
                    np.column_stack([np.cos(np.linspace(0, np.pi, n)),
                                     np.sin(np.linspace(0, np.pi, n))]) + 0.1 * rng.randn(n, 2),
                    np.column_stack([1 - np.cos(np.linspace(0, np.pi, n)),
                                     1 - np.sin(np.linspace(0, np.pi, n)) - 0.5]) + 0.1 * rng.randn(n, 2),
                ])
            )(np.random.RandomState(1), 150),
            "labels": lambda: np.array([0] * 150 + [1] * 150),
        },
        "Regression Demo": {
            "description": "Nonlinear regression: sinusoidal with noise (200 samples)",
            "code": (
                "rng = np.random.RandomState(9)\n"
                "X = np.sort(rng.uniform(0, 6, 200))\n"
                "y = np.sin(X) * X + 0.5*rng.randn(200)"
            ),
            "x": lambda: np.sort(np.random.RandomState(9).uniform(0, 6, 200)),
            "y": lambda: (
                lambda X: np.sin(X) * X + 0.5 * np.random.RandomState(9).randn(200)
            )(np.sort(np.random.RandomState(9).uniform(0, 6, 200))),
        },
        "Clustering Demo": {
            "description": "Four well-separated Gaussian blobs (400 samples) for unsupervised learning",
            "code": (
                "rng = np.random.RandomState(33)\n"
                "centers = np.array([[-3, -3], [3, -3], [-3, 3], [3, 3]])\n"
                "X = np.vstack([rng.normal(c, 0.8, (100, 2)) for c in centers])\n"
                "y = np.repeat([0, 1, 2, 3], 100)"
            ),
            "data": lambda: np.vstack([
                np.random.RandomState(33 + i).normal(c, 0.8, (100, 2))
                for i, c in enumerate([[-3, -3], [3, -3], [-3, 3], [3, 3]])
            ]),
            "labels": lambda: np.repeat([0, 1, 2, 3], 100),
        },
    },
}


def get_example_names(module_name: str) -> list:
    """Return a list of example names available for a given module.

    Parameters
    ----------
    module_name : str
        The module key, e.g. "2D Plotter", "Curve Fitting".

    Returns
    -------
    list of str
        Example names, or empty list if the module has no examples.
    """
    return list(EXAMPLES.get(module_name, {}).keys())


def get_example(module_name: str, example_name: str) -> dict:
    """Retrieve a specific example by module and example name.

    Parameters
    ----------
    module_name : str
        The module key.
    example_name : str
        The example key within that module.

    Returns
    -------
    dict
        The example dictionary with description, data, code, etc.

    Raises
    ------
    KeyError
        If module or example is not found.
    """
    try:
        return EXAMPLES[module_name][example_name]
    except KeyError:
        available = get_example_names(module_name)
        raise KeyError(
            f"Example '{example_name}' not found in module '{module_name}'. "
            f"Available: {available}"
        )


def load_example_data(module_name: str, example_name: str) -> dict:
    """Load an example and evaluate any lambda data generators.

    Returns a new dict where all callable values have been invoked,
    producing concrete numpy arrays.

    Parameters
    ----------
    module_name : str
        The module key.
    example_name : str
        The example key.

    Returns
    -------
    dict
        A copy of the example with all lambdas resolved to arrays/values.
    """
    ex = get_example(module_name, example_name)
    result = {}
    for key, val in ex.items():
        if callable(val):
            result[key] = val()
        else:
            result[key] = val
    return result


def list_all_examples() -> dict:
    """Return a nested dict of {module: {example_name: description}}.

    Useful for building menus or overview displays.
    """
    overview = {}
    for module, examples in EXAMPLES.items():
        overview[module] = {
            name: info.get("description", "") for name, info in examples.items()
        }
    return overview
