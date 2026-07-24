"""
Axiom Scientific Suite - Global Matplotlib Dark Style
Apply this to all matplotlib figures for consistent dark-themed plots.
"""
import matplotlib
import matplotlib.pyplot as plt

# Dark style parameters matching Axiom's dark theme
AXIOM_DARK_STYLE = {
    'figure.facecolor': '#1e1e1e',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#555555',
    'axes.labelcolor': '#cccccc',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.grid': True,
    'axes.prop_cycle': matplotlib.cycler('color', [
        '#4a90d9', '#e8308a', '#40d8e0', '#e07830', '#30a848',
        '#a87eff', '#ff6666', '#ffcc44', '#66ffcc', '#ff66cc',
    ]),
    'grid.color': '#333333',
    'grid.alpha': 0.4,
    'grid.linewidth': 0.5,
    'text.color': '#cccccc',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.facecolor': '#2a2a2a',
    'legend.edgecolor': '#555555',
    'legend.fontsize': 9,
    'legend.labelcolor': '#cccccc',
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'savefig.facecolor': '#1e1e1e',
    'savefig.edgecolor': '#1e1e1e',
    'savefig.dpi': 150,
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
}


def apply_axiom_style():
    """Apply Axiom dark style globally to all matplotlib plots."""
    for key, val in AXIOM_DARK_STYLE.items():
        try:
            matplotlib.rcParams[key] = val
        except (KeyError, ValueError):
            pass


def style_figure(fig):
    """Apply dark styling to an existing figure."""
    fig.set_facecolor('#1e1e1e')
    for ax in fig.get_axes():
        style_axes(ax)


def style_axes(ax):
    """Apply dark styling to an existing axes."""
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#aaaaaa', which='both')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#cccccc')
    for spine in ax.spines.values():
        spine.set_color('#555555')
    ax.grid(True, alpha=0.4, color='#333333', linewidth=0.5)


# ---------------------------------------------------------------------------
# Journal / Presentation Plot Style Presets
# ---------------------------------------------------------------------------
# Each preset is a dict of matplotlib rcParams that can be applied to produce
# publication-ready figures matching a specific journal's style guide.

# Conversion helpers
_MM_TO_IN = 1 / 25.4
_CM_TO_IN = 1 / 2.54

JOURNAL_STYLES = {
    "Nature": {
        # Nature: 89 mm single column, 183 mm double column
        # Helvetica font family, 7 pt label text
        'figure.figsize': (89 * _MM_TO_IN, 60 * _MM_TO_IN),  # single column default
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.prop_cycle': matplotlib.cycler('color', [
            '#0C5DA5', '#FF2C00', '#00B945', '#FF9500',
            '#845B97', '#474747', '#9e9e9e',
        ]),
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'legend.fontsize': 6,
        'legend.frameon': False,
        'legend.labelcolor': 'black',
        '_axiom_meta': {
            'single_column_mm': 89,
            'double_column_mm': 183,
            'description': 'Nature journal style (Helvetica, 7pt)',
        },
    },

    "Science": {
        # Science/AAAS: similar to Nature but uses Arial
        'figure.figsize': (89 * _MM_TO_IN, 60 * _MM_TO_IN),
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.prop_cycle': matplotlib.cycler('color', [
            '#0C5DA5', '#FF2C00', '#00B945', '#FF9500',
            '#845B97', '#474747', '#9e9e9e',
        ]),
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'legend.fontsize': 6,
        'legend.frameon': False,
        'legend.labelcolor': 'black',
        '_axiom_meta': {
            'single_column_mm': 89,
            'double_column_mm': 183,
            'description': 'Science/AAAS journal style (Arial, 7pt)',
        },
    },

    "APS (Physical Review)": {
        # APS / Physical Review: 8.6 cm single column, Times New Roman, 10 pt
        'figure.figsize': (8.6 * _CM_TO_IN, 6.5 * _CM_TO_IN),
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.prop_cycle': matplotlib.cycler('color', [
            '#000000', '#E41A1C', '#377EB8', '#4DAF4A',
            '#984EA3', '#FF7F00', '#A65628',
        ]),
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.labelcolor': 'black',
        'mathtext.fontset': 'cm',
        '_axiom_meta': {
            'single_column_cm': 8.6,
            'double_column_cm': 17.1,
            'description': 'APS Physical Review style (Times New Roman, 10pt)',
        },
    },

    "IEEE": {
        # IEEE: 3.5 in single column, 7.16 in double column, Times New Roman, 8 pt
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.prop_cycle': matplotlib.cycler('color', [
            '#000000', '#0072BD', '#D95319', '#EDB120',
            '#7E2F8E', '#77AC30', '#4DBEEE',
        ]),
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'legend.fontsize': 7,
        'legend.frameon': True,
        'legend.edgecolor': '#cccccc',
        'legend.labelcolor': 'black',
        'mathtext.fontset': 'cm',
        '_axiom_meta': {
            'single_column_in': 3.5,
            'double_column_in': 7.16,
            'description': 'IEEE style (Times New Roman, 8pt)',
        },
    },

    "Presentation": {
        # Large fonts, thick lines, high contrast for projector / slide use
        'figure.figsize': (10, 7),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.dpi': 150,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#222222',
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'axes.prop_cycle': matplotlib.cycler('color', [
            '#2176FF', '#E63946', '#06D6A0', '#FFB703',
            '#8338EC', '#FB5607', '#3A86FF',
        ]),
        'grid.color': '#dddddd',
        'grid.alpha': 0.6,
        'grid.linewidth': 0.8,
        'text.color': '#222222',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 2.5,
        'lines.markersize': 9,
        'legend.fontsize': 13,
        'legend.frameon': True,
        'legend.edgecolor': '#cccccc',
        'legend.facecolor': 'white',
        'legend.labelcolor': '#222222',
        '_axiom_meta': {
            'description': 'Presentation style: large fonts, thick lines, high contrast',
        },
    },
}


def get_journal_names() -> list:
    """Return a sorted list of all available journal/presentation style names.

    Returns
    -------
    list of str
        Names that can be passed to :func:`apply_journal_style`.
    """
    return sorted(JOURNAL_STYLES.keys())


def apply_journal_style(name: str):
    """Apply a journal or presentation style preset to matplotlib rcParams.

    This overrides current rcParams with the values in the selected preset.
    Keys prefixed with ``_axiom_`` are metadata and are skipped.

    Parameters
    ----------
    name : str
        One of the keys in :data:`JOURNAL_STYLES`.  Use
        :func:`get_journal_names` to list valid options.

    Raises
    ------
    ValueError
        If *name* is not a recognized style.
    """
    if name not in JOURNAL_STYLES:
        available = ", ".join(get_journal_names())
        raise ValueError(
            f"Unknown journal style '{name}'. Available styles: {available}"
        )
    style = JOURNAL_STYLES[name]
    for key, val in style.items():
        if key.startswith('_'):
            continue  # skip metadata keys
        try:
            matplotlib.rcParams[key] = val
        except (KeyError, ValueError):
            pass


def get_journal_figure_size(name: str, double_column: bool = False) -> tuple:
    """Return the (width, height) in inches for a journal style.

    Parameters
    ----------
    name : str
        Journal style name.
    double_column : bool
        If True, return the double-column width instead of single-column.

    Returns
    -------
    tuple of float
        (width_inches, height_inches).  Height is the preset default.
    """
    if name not in JOURNAL_STYLES:
        raise ValueError(f"Unknown journal style '{name}'.")

    meta = JOURNAL_STYLES[name].get('_axiom_meta', {})
    default_w, default_h = JOURNAL_STYLES[name].get('figure.figsize', (6, 4))

    if double_column:
        if 'double_column_mm' in meta:
            return (meta['double_column_mm'] * _MM_TO_IN, default_h)
        elif 'double_column_cm' in meta:
            return (meta['double_column_cm'] * _CM_TO_IN, default_h)
        elif 'double_column_in' in meta:
            return (meta['double_column_in'], default_h)

    return (default_w, default_h)
