"""
Interactive Periodic Table Widget for QuantumRes.
Displays all 118 elements with detailed properties, search, comparison, and trend visualization.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QDialog, QScrollArea, QFrame,
    QGroupBox, QSplitter, QTextEdit, QSizePolicy, QMessageBox,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass
import matplotlib.patches as mpatches


# ── Element Data ────────────────────────────────────────────────────────────
# Each element: (Z, symbol, name, mass, category, electron_config,
#                electronegativity, ionization_energy_eV, density_g_cm3,
#                melting_K, boiling_K, discovery_year, oxidation_states)
# Categories: AM=Alkali Metal, AEM=Alkaline Earth, TM=Transition Metal,
#             PTM=Post-Transition, Met=Metalloid, NM=Nonmetal,
#             Hal=Halogen, NG=Noble Gas, Lan=Lanthanide, Act=Actinide

ELEMENTS = [
    (1, "H", "Hydrogen", 1.008, "NM", "1s1", 2.20, 13.598, 0.00009, 14.01, 20.28, 1766, "-1,+1"),
    (2, "He", "Helium", 4.003, "NG", "1s2", None, 24.587, 0.000179, 0.95, 4.22, 1868, "0"),
    (3, "Li", "Lithium", 6.941, "AM", "[He]2s1", 0.98, 5.392, 0.534, 453.7, 1615, 1817, "+1"),
    (4, "Be", "Beryllium", 9.012, "AEM", "[He]2s2", 1.57, 9.323, 1.85, 1560, 2744, 1798, "+2"),
    (5, "B", "Boron", 10.81, "Met", "[He]2s2 2p1", 2.04, 8.298, 2.34, 2349, 4200, 1808, "+3"),
    (6, "C", "Carbon", 12.011, "NM", "[He]2s2 2p2", 2.55, 11.260, 2.267, 3823, 4098, -3750, "-4,+4"),
    (7, "N", "Nitrogen", 14.007, "NM", "[He]2s2 2p3", 3.04, 14.534, 0.0012506, 63.15, 77.36, 1772, "-3,+3,+5"),
    (8, "O", "Oxygen", 15.999, "NM", "[He]2s2 2p4", 3.44, 13.618, 0.001429, 54.36, 90.20, 1774, "-2"),
    (9, "F", "Fluorine", 18.998, "Hal", "[He]2s2 2p5", 3.98, 17.423, 0.001696, 53.53, 85.03, 1886, "-1"),
    (10, "Ne", "Neon", 20.180, "NG", "[He]2s2 2p6", None, 21.565, 0.0009, 24.56, 27.07, 1898, "0"),
    (11, "Na", "Sodium", 22.990, "AM", "[Ne]3s1", 0.93, 5.139, 0.971, 370.9, 1156, 1807, "+1"),
    (12, "Mg", "Magnesium", 24.305, "AEM", "[Ne]3s2", 1.31, 7.646, 1.738, 923, 1363, 1755, "+2"),
    (13, "Al", "Aluminium", 26.982, "PTM", "[Ne]3s2 3p1", 1.61, 5.986, 2.698, 933.5, 2792, 1825, "+3"),
    (14, "Si", "Silicon", 28.086, "Met", "[Ne]3s2 3p2", 1.90, 8.152, 2.329, 1687, 3538, 1824, "-4,+4"),
    (15, "P", "Phosphorus", 30.974, "NM", "[Ne]3s2 3p3", 2.19, 10.487, 1.82, 317.3, 553.7, 1669, "-3,+3,+5"),
    (16, "S", "Sulfur", 32.06, "NM", "[Ne]3s2 3p4", 2.58, 10.360, 2.067, 388.4, 717.8, -500, "-2,+4,+6"),
    (17, "Cl", "Chlorine", 35.45, "Hal", "[Ne]3s2 3p5", 3.16, 12.968, 0.003214, 171.6, 239.1, 1774, "-1,+1,+5,+7"),
    (18, "Ar", "Argon", 39.948, "NG", "[Ne]3s2 3p6", None, 15.760, 0.001784, 83.80, 87.30, 1894, "0"),
    (19, "K", "Potassium", 39.098, "AM", "[Ar]4s1", 0.82, 4.341, 0.862, 336.5, 1032, 1807, "+1"),
    (20, "Ca", "Calcium", 40.078, "AEM", "[Ar]4s2", 1.00, 6.113, 1.55, 1115, 1757, 1808, "+2"),
    (21, "Sc", "Scandium", 44.956, "TM", "[Ar]3d1 4s2", 1.36, 6.561, 2.989, 1814, 3109, 1879, "+3"),
    (22, "Ti", "Titanium", 47.867, "TM", "[Ar]3d2 4s2", 1.54, 6.828, 4.507, 1941, 3560, 1791, "+2,+3,+4"),
    (23, "V", "Vanadium", 50.942, "TM", "[Ar]3d3 4s2", 1.63, 6.746, 6.11, 2183, 3680, 1801, "+2,+3,+4,+5"),
    (24, "Cr", "Chromium", 51.996, "TM", "[Ar]3d5 4s1", 1.66, 6.767, 7.15, 2180, 2944, 1797, "+2,+3,+6"),
    (25, "Mn", "Manganese", 54.938, "TM", "[Ar]3d5 4s2", 1.55, 7.434, 7.44, 1519, 2334, 1774, "+2,+4,+7"),
    (26, "Fe", "Iron", 55.845, "TM", "[Ar]3d6 4s2", 1.83, 7.902, 7.874, 1811, 3134, -5000, "+2,+3"),
    (27, "Co", "Cobalt", 58.933, "TM", "[Ar]3d7 4s2", 1.88, 7.881, 8.90, 1768, 3200, 1735, "+2,+3"),
    (28, "Ni", "Nickel", 58.693, "TM", "[Ar]3d8 4s2", 1.91, 7.640, 8.908, 1728, 3186, 1751, "+2,+3"),
    (29, "Cu", "Copper", 63.546, "TM", "[Ar]3d10 4s1", 1.90, 7.726, 8.96, 1357.8, 2835, -9000, "+1,+2"),
    (30, "Zn", "Zinc", 65.38, "TM", "[Ar]3d10 4s2", 1.65, 9.394, 7.134, 692.7, 1180, -1000, "+2"),
    (31, "Ga", "Gallium", 69.723, "PTM", "[Ar]3d10 4s2 4p1", 1.81, 5.999, 5.907, 302.9, 2477, 1875, "+3"),
    (32, "Ge", "Germanium", 72.63, "Met", "[Ar]3d10 4s2 4p2", 2.01, 7.900, 5.323, 1211.4, 3106, 1886, "+2,+4"),
    (33, "As", "Arsenic", 74.922, "Met", "[Ar]3d10 4s2 4p3", 2.18, 9.815, 5.776, 1090, 887, -2500, "-3,+3,+5"),
    (34, "Se", "Selenium", 78.971, "NM", "[Ar]3d10 4s2 4p4", 2.55, 9.752, 4.809, 494, 958, 1817, "-2,+4,+6"),
    (35, "Br", "Bromine", 79.904, "Hal", "[Ar]3d10 4s2 4p5", 2.96, 11.814, 3.122, 265.8, 332, 1826, "-1,+1,+5"),
    (36, "Kr", "Krypton", 83.798, "NG", "[Ar]3d10 4s2 4p6", 3.00, 14.000, 0.003749, 115.8, 119.9, 1898, "0,+2"),
    (37, "Rb", "Rubidium", 85.468, "AM", "[Kr]5s1", 0.82, 4.177, 1.532, 312.5, 961, 1861, "+1"),
    (38, "Sr", "Strontium", 87.62, "AEM", "[Kr]5s2", 0.95, 5.695, 2.64, 1050, 1655, 1790, "+2"),
    (39, "Y", "Yttrium", 88.906, "TM", "[Kr]4d1 5s2", 1.22, 6.217, 4.469, 1799, 3609, 1794, "+3"),
    (40, "Zr", "Zirconium", 91.224, "TM", "[Kr]4d2 5s2", 1.33, 6.634, 6.506, 2128, 4682, 1789, "+4"),
    (41, "Nb", "Niobium", 92.906, "TM", "[Kr]4d4 5s1", 1.60, 6.759, 8.57, 2750, 5017, 1801, "+3,+5"),
    (42, "Mo", "Molybdenum", 95.95, "TM", "[Kr]4d5 5s1", 2.16, 7.092, 10.22, 2896, 4912, 1781, "+4,+6"),
    (43, "Tc", "Technetium", 98.0, "TM", "[Kr]4d5 5s2", 1.90, 7.28, 11.5, 2430, 4538, 1937, "+4,+7"),
    (44, "Ru", "Ruthenium", 101.07, "TM", "[Kr]4d7 5s1", 2.20, 7.361, 12.37, 2607, 4423, 1844, "+3,+4"),
    (45, "Rh", "Rhodium", 102.906, "TM", "[Kr]4d8 5s1", 2.28, 7.459, 12.41, 2237, 3968, 1803, "+3"),
    (46, "Pd", "Palladium", 106.42, "TM", "[Kr]4d10", 2.20, 8.337, 12.02, 1828.1, 3236, 1803, "+2,+4"),
    (47, "Ag", "Silver", 107.868, "TM", "[Kr]4d10 5s1", 1.93, 7.576, 10.501, 1234.9, 2435, -5000, "+1"),
    (48, "Cd", "Cadmium", 112.414, "TM", "[Kr]4d10 5s2", 1.69, 8.994, 8.65, 594.2, 1040, 1817, "+2"),
    (49, "In", "Indium", 114.818, "PTM", "[Kr]4d10 5s2 5p1", 1.78, 5.786, 7.31, 429.8, 2345, 1863, "+3"),
    (50, "Sn", "Tin", 118.710, "PTM", "[Kr]4d10 5s2 5p2", 1.96, 7.344, 7.287, 505.1, 2875, -3500, "+2,+4"),
    (51, "Sb", "Antimony", 121.760, "Met", "[Kr]4d10 5s2 5p3", 2.05, 8.640, 6.685, 903.8, 1860, -3000, "-3,+3,+5"),
    (52, "Te", "Tellurium", 127.60, "Met", "[Kr]4d10 5s2 5p4", 2.10, 9.010, 6.232, 722.7, 1261, 1783, "-2,+4,+6"),
    (53, "I", "Iodine", 126.904, "Hal", "[Kr]4d10 5s2 5p5", 2.66, 10.451, 4.93, 386.9, 457.6, 1811, "-1,+1,+5,+7"),
    (54, "Xe", "Xenon", 131.293, "NG", "[Kr]4d10 5s2 5p6", 2.60, 12.130, 0.005887, 161.4, 165.1, 1898, "0,+2,+4,+6"),
    (55, "Cs", "Caesium", 132.905, "AM", "[Xe]6s1", 0.79, 3.894, 1.873, 301.7, 944, 1860, "+1"),
    (56, "Ba", "Barium", 137.327, "AEM", "[Xe]6s2", 0.89, 5.212, 3.594, 1000, 2170, 1808, "+2"),
    (57, "La", "Lanthanum", 138.905, "Lan", "[Xe]5d1 6s2", 1.10, 5.577, 6.145, 1193, 3737, 1839, "+3"),
    (58, "Ce", "Cerium", 140.116, "Lan", "[Xe]4f1 5d1 6s2", 1.12, 5.539, 6.770, 1068, 3716, 1803, "+3,+4"),
    (59, "Pr", "Praseodymium", 140.908, "Lan", "[Xe]4f3 6s2", 1.13, 5.473, 6.773, 1208, 3793, 1885, "+3"),
    (60, "Nd", "Neodymium", 144.242, "Lan", "[Xe]4f4 6s2", 1.14, 5.525, 7.007, 1297, 3347, 1885, "+3"),
    (61, "Pm", "Promethium", 145.0, "Lan", "[Xe]4f5 6s2", 1.13, 5.582, 7.26, 1315, 3273, 1945, "+3"),
    (62, "Sm", "Samarium", 150.36, "Lan", "[Xe]4f6 6s2", 1.17, 5.644, 7.52, 1345, 2067, 1879, "+2,+3"),
    (63, "Eu", "Europium", 151.964, "Lan", "[Xe]4f7 6s2", 1.20, 5.670, 5.243, 1099, 1802, 1901, "+2,+3"),
    (64, "Gd", "Gadolinium", 157.25, "Lan", "[Xe]4f7 5d1 6s2", 1.20, 6.150, 7.895, 1585, 3546, 1880, "+3"),
    (65, "Tb", "Terbium", 158.925, "Lan", "[Xe]4f9 6s2", 1.20, 5.864, 8.229, 1629, 3503, 1843, "+3"),
    (66, "Dy", "Dysprosium", 162.500, "Lan", "[Xe]4f10 6s2", 1.22, 5.939, 8.55, 1680, 2840, 1886, "+3"),
    (67, "Ho", "Holmium", 164.930, "Lan", "[Xe]4f11 6s2", 1.23, 6.022, 8.795, 1734, 2993, 1878, "+3"),
    (68, "Er", "Erbium", 167.259, "Lan", "[Xe]4f12 6s2", 1.24, 6.108, 9.066, 1802, 3141, 1842, "+3"),
    (69, "Tm", "Thulium", 168.934, "Lan", "[Xe]4f13 6s2", 1.25, 6.184, 9.321, 1818, 2223, 1879, "+3"),
    (70, "Yb", "Ytterbium", 173.045, "Lan", "[Xe]4f14 6s2", 1.10, 6.254, 6.965, 1097, 1469, 1878, "+2,+3"),
    (71, "Lu", "Lutetium", 174.967, "Lan", "[Xe]4f14 5d1 6s2", 1.27, 5.426, 9.84, 1925, 3675, 1907, "+3"),
    (72, "Hf", "Hafnium", 178.49, "TM", "[Xe]4f14 5d2 6s2", 1.30, 6.825, 13.31, 2506, 4876, 1923, "+4"),
    (73, "Ta", "Tantalum", 180.948, "TM", "[Xe]4f14 5d3 6s2", 1.50, 7.550, 16.654, 3290, 5731, 1802, "+5"),
    (74, "W", "Tungsten", 183.84, "TM", "[Xe]4f14 5d4 6s2", 2.36, 7.864, 19.25, 3695, 5828, 1783, "+4,+6"),
    (75, "Re", "Rhenium", 186.207, "TM", "[Xe]4f14 5d5 6s2", 1.90, 7.833, 21.02, 3459, 5869, 1925, "+4,+7"),
    (76, "Os", "Osmium", 190.23, "TM", "[Xe]4f14 5d6 6s2", 2.20, 8.438, 22.587, 3306, 5285, 1803, "+3,+4"),
    (77, "Ir", "Iridium", 192.217, "TM", "[Xe]4f14 5d7 6s2", 2.20, 8.967, 22.56, 2719, 4701, 1803, "+3,+4"),
    (78, "Pt", "Platinum", 195.084, "TM", "[Xe]4f14 5d9 6s1", 2.28, 8.959, 21.46, 2041.4, 4098, 1735, "+2,+4"),
    (79, "Au", "Gold", 196.967, "TM", "[Xe]4f14 5d10 6s1", 2.54, 9.226, 19.282, 1337.3, 3129, -6000, "+1,+3"),
    (80, "Hg", "Mercury", 200.592, "TM", "[Xe]4f14 5d10 6s2", 2.00, 10.438, 13.5336, 234.3, 629.9, -2000, "+1,+2"),
    (81, "Tl", "Thallium", 204.38, "PTM", "[Xe]4f14 5d10 6s2 6p1", 1.62, 6.108, 11.85, 577, 1746, 1861, "+1,+3"),
    (82, "Pb", "Lead", 207.2, "PTM", "[Xe]4f14 5d10 6s2 6p2", 1.87, 7.417, 11.342, 600.6, 2022, -7000, "+2,+4"),
    (83, "Bi", "Bismuth", 208.980, "PTM", "[Xe]4f14 5d10 6s2 6p3", 2.02, 7.286, 9.807, 544.7, 1837, 1753, "+3,+5"),
    (84, "Po", "Polonium", 209.0, "Met", "[Xe]4f14 5d10 6s2 6p4", 2.00, 8.414, 9.32, 527, 1235, 1898, "+2,+4"),
    (85, "At", "Astatine", 210.0, "Hal", "[Xe]4f14 5d10 6s2 6p5", 2.20, 9.318, 7.0, 575, 610, 1940, "-1,+1"),
    (86, "Rn", "Radon", 222.0, "NG", "[Xe]4f14 5d10 6s2 6p6", None, 10.749, 0.00973, 202, 211.5, 1900, "0,+2"),
    (87, "Fr", "Francium", 223.0, "AM", "[Rn]7s1", 0.70, 4.073, 1.87, 300, 950, 1939, "+1"),
    (88, "Ra", "Radium", 226.0, "AEM", "[Rn]7s2", 0.90, 5.278, 5.5, 973, 2010, 1898, "+2"),
    (89, "Ac", "Actinium", 227.0, "Act", "[Rn]6d1 7s2", 1.10, 5.17, 10.07, 1323, 3471, 1899, "+3"),
    (90, "Th", "Thorium", 232.038, "Act", "[Rn]6d2 7s2", 1.30, 6.308, 11.72, 2115, 5061, 1829, "+4"),
    (91, "Pa", "Protactinium", 231.036, "Act", "[Rn]5f2 6d1 7s2", 1.50, 5.89, 15.37, 1841, 4300, 1913, "+4,+5"),
    (92, "U", "Uranium", 238.029, "Act", "[Rn]5f3 6d1 7s2", 1.38, 6.194, 18.95, 1405.3, 4404, 1789, "+3,+4,+5,+6"),
    (93, "Np", "Neptunium", 237.0, "Act", "[Rn]5f4 6d1 7s2", 1.36, 6.266, 20.45, 917, 4175, 1940, "+3,+4,+5,+6"),
    (94, "Pu", "Plutonium", 244.0, "Act", "[Rn]5f6 7s2", 1.28, 6.026, 19.84, 912.5, 3501, 1940, "+3,+4,+5,+6"),
    (95, "Am", "Americium", 243.0, "Act", "[Rn]5f7 7s2", 1.30, 5.974, 13.69, 1449, 2880, 1944, "+3,+4,+5,+6"),
    (96, "Cm", "Curium", 247.0, "Act", "[Rn]5f7 6d1 7s2", 1.30, 5.992, 13.51, 1613, 3383, 1944, "+3"),
    (97, "Bk", "Berkelium", 247.0, "Act", "[Rn]5f9 7s2", 1.30, 6.198, 14.79, 1259, 2900, 1949, "+3,+4"),
    (98, "Cf", "Californium", 251.0, "Act", "[Rn]5f10 7s2", 1.30, 6.282, 15.1, 1173, 1743, 1950, "+2,+3"),
    (99, "Es", "Einsteinium", 252.0, "Act", "[Rn]5f11 7s2", 1.30, 6.42, 8.84, 1133, 1269, 1952, "+2,+3"),
    (100, "Fm", "Fermium", 257.0, "Act", "[Rn]5f12 7s2", 1.30, 6.50, None, 1800, None, 1952, "+2,+3"),
    (101, "Md", "Mendelevium", 258.0, "Act", "[Rn]5f13 7s2", 1.30, 6.58, None, 1100, None, 1955, "+2,+3"),
    (102, "No", "Nobelium", 259.0, "Act", "[Rn]5f14 7s2", 1.30, 6.65, None, 1100, None, 1958, "+2,+3"),
    (103, "Lr", "Lawrencium", 266.0, "Act", "[Rn]5f14 7s2 7p1", 1.30, 4.90, None, 1900, None, 1961, "+3"),
    (104, "Rf", "Rutherfordium", 267.0, "TM", "[Rn]5f14 6d2 7s2", None, 6.0, None, None, None, 1964, "+4"),
    (105, "Db", "Dubnium", 268.0, "TM", "[Rn]5f14 6d3 7s2", None, None, None, None, None, 1967, "+5"),
    (106, "Sg", "Seaborgium", 269.0, "TM", "[Rn]5f14 6d4 7s2", None, None, None, None, None, 1974, "+6"),
    (107, "Bh", "Bohrium", 270.0, "TM", "[Rn]5f14 6d5 7s2", None, None, None, None, None, 1981, "+7"),
    (108, "Hs", "Hassium", 277.0, "TM", "[Rn]5f14 6d6 7s2", None, None, None, None, None, 1984, "+8"),
    (109, "Mt", "Meitnerium", 278.0, "TM", "[Rn]5f14 6d7 7s2", None, None, None, None, None, 1982, None),
    (110, "Ds", "Darmstadtium", 281.0, "TM", "[Rn]5f14 6d8 7s2", None, None, None, None, None, 1994, None),
    (111, "Rg", "Roentgenium", 282.0, "TM", "[Rn]5f14 6d9 7s2", None, None, None, None, None, 1994, None),
    (112, "Cn", "Copernicium", 285.0, "TM", "[Rn]5f14 6d10 7s2", None, None, None, None, None, 1996, "+2"),
    (113, "Nh", "Nihonium", 286.0, "PTM", "[Rn]5f14 6d10 7s2 7p1", None, None, None, None, None, 2003, None),
    (114, "Fl", "Flerovium", 289.0, "PTM", "[Rn]5f14 6d10 7s2 7p2", None, None, None, None, None, 1998, None),
    (115, "Mc", "Moscovium", 290.0, "PTM", "[Rn]5f14 6d10 7s2 7p3", None, None, None, None, None, 2003, None),
    (116, "Lv", "Livermorium", 293.0, "PTM", "[Rn]5f14 6d10 7s2 7p4", None, None, None, None, None, 2000, None),
    (117, "Ts", "Tennessine", 294.0, "Hal", "[Rn]5f14 6d10 7s2 7p5", None, None, None, None, None, 2010, None),
    (118, "Og", "Oganesson", 294.0, "NG", "[Rn]5f14 6d10 7s2 7p6", None, None, None, None, None, 2002, None),
]

# ── Refined Category Colors (muted pastels for professional look) ───────────
CATEGORY_COLORS = {
    "AM":  "#E8837C",   # Alkali metals
    "AEM": "#F5C87A",   # Alkaline earth metals
    "TM":  "#7AB8D4",   # Transition metals
    "PTM": "#8FBF8F",   # Post-transition metals
    "Met": "#C9B06B",   # Metalloids
    "NM":  "#7ED47E",   # Nonmetals
    "Hal": "#C9CC5E",   # Halogens
    "NG":  "#9BCFCF",   # Noble gases
    "Lan": "#C49ED0",   # Lanthanides
    "Act": "#DB8FA0",   # Actinides
}

CATEGORY_NAMES = {
    "AM": "Alkali Metal", "AEM": "Alkaline Earth Metal", "TM": "Transition Metal",
    "PTM": "Post-Transition Metal", "Met": "Metalloid", "NM": "Nonmetal",
    "Hal": "Halogen", "NG": "Noble Gas", "Lan": "Lanthanide", "Act": "Actinide",
}

# Standard periodic table layout: (row, col) for each Z
PT_LAYOUT = {
    1:(0,0), 2:(0,17),
    3:(1,0), 4:(1,1), 5:(1,12), 6:(1,13), 7:(1,14), 8:(1,15), 9:(1,16), 10:(1,17),
    11:(2,0), 12:(2,1), 13:(2,12), 14:(2,13), 15:(2,14), 16:(2,15), 17:(2,16), 18:(2,17),
    19:(3,0), 20:(3,1),
    21:(3,2), 22:(3,3), 23:(3,4), 24:(3,5), 25:(3,6), 26:(3,7), 27:(3,8),
    28:(3,9), 29:(3,10), 30:(3,11), 31:(3,12), 32:(3,13), 33:(3,14), 34:(3,15), 35:(3,16), 36:(3,17),
    37:(4,0), 38:(4,1),
    39:(4,2), 40:(4,3), 41:(4,4), 42:(4,5), 43:(4,6), 44:(4,7), 45:(4,8),
    46:(4,9), 47:(4,10), 48:(4,11), 49:(4,12), 50:(4,13), 51:(4,14), 52:(4,15), 53:(4,16), 54:(4,17),
    55:(5,0), 56:(5,1),
    71:(5,2), 72:(5,3), 73:(5,4), 74:(5,5), 75:(5,6), 76:(5,7), 77:(5,8),
    78:(5,9), 79:(5,10), 80:(5,11), 81:(5,12), 82:(5,13), 83:(5,14), 84:(5,15), 85:(5,16), 86:(5,17),
    87:(6,0), 88:(6,1),
    103:(6,2), 104:(6,3), 105:(6,4), 106:(6,5), 107:(6,6), 108:(6,7), 109:(6,8),
    110:(6,9), 111:(6,10), 112:(6,11), 113:(6,12), 114:(6,13), 115:(6,14), 116:(6,15), 117:(6,16), 118:(6,17),
}
for i, z in enumerate(range(57, 71)):
    PT_LAYOUT[z] = (8, 2 + i)
for i, z in enumerate(range(89, 103)):
    PT_LAYOUT[z] = (9, 2 + i)


# ── Extended Element Data Dictionaries ──────────────────────────────────────

# Block assignment by atomic number
ELEMENT_BLOCK = {
    1: "s", 2: "s",
    3: "s", 4: "s", 5: "p", 6: "p", 7: "p", 8: "p", 9: "p", 10: "p",
    11: "s", 12: "s", 13: "p", 14: "p", 15: "p", 16: "p", 17: "p", 18: "p",
    19: "s", 20: "s", 21: "d", 22: "d", 23: "d", 24: "d", 25: "d", 26: "d",
    27: "d", 28: "d", 29: "d", 30: "d", 31: "p", 32: "p", 33: "p", 34: "p",
    35: "p", 36: "p", 37: "s", 38: "s", 39: "d", 40: "d", 41: "d", 42: "d",
    43: "d", 44: "d", 45: "d", 46: "d", 47: "d", 48: "d", 49: "p", 50: "p",
    51: "p", 52: "p", 53: "p", 54: "p", 55: "s", 56: "s",
    57: "f", 58: "f", 59: "f", 60: "f", 61: "f", 62: "f", 63: "f", 64: "f",
    65: "f", 66: "f", 67: "f", 68: "f", 69: "f", 70: "f", 71: "d",
    72: "d", 73: "d", 74: "d", 75: "d", 76: "d", 77: "d", 78: "d", 79: "d",
    80: "d", 81: "p", 82: "p", 83: "p", 84: "p", 85: "p", 86: "p",
    87: "s", 88: "s",
    89: "f", 90: "f", 91: "f", 92: "f", 93: "f", 94: "f", 95: "f", 96: "f",
    97: "f", 98: "f", 99: "f", 100: "f", 101: "f", 102: "f", 103: "d",
    104: "d", 105: "d", 106: "d", 107: "d", 108: "d", 109: "d", 110: "d",
    111: "d", 112: "d", 113: "p", 114: "p", 115: "p", 116: "p", 117: "p", 118: "p",
}

# Period number for each element
ELEMENT_PERIOD = {}
_period_ranges = [(1, 2, 1), (3, 10, 2), (11, 18, 3), (19, 36, 4),
                  (37, 54, 5), (55, 86, 6), (87, 118, 7)]
for _lo, _hi, _p in _period_ranges:
    for _z in range(_lo, _hi + 1):
        ELEMENT_PERIOD[_z] = _p

# Group number for each element (IUPAC 1-18, None for Lan/Act)
ELEMENT_GROUP = {
    1: 1, 2: 18,
    3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18,
    11: 1, 12: 2, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9,
    28: 10, 29: 11, 30: 12, 31: 13, 32: 14, 33: 15, 34: 16, 35: 17, 36: 18,
    37: 1, 38: 2, 39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9,
    46: 10, 47: 11, 48: 12, 49: 13, 50: 14, 51: 15, 52: 16, 53: 17, 54: 18,
    55: 1, 56: 2,
    71: 3, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10, 79: 11,
    80: 12, 81: 13, 82: 14, 83: 15, 84: 16, 85: 17, 86: 18,
    87: 1, 88: 2,
    103: 3, 104: 4, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9, 110: 10,
    111: 11, 112: 12, 113: 13, 114: 14, 115: 15, 116: 16, 117: 17, 118: 18,
}

# Atomic radius in pm (empirical)
ATOMIC_RADIUS = {
    1: 25, 2: 31, 3: 145, 4: 105, 5: 85, 6: 70, 7: 65, 8: 60, 9: 50, 10: 38,
    11: 180, 12: 150, 13: 125, 14: 110, 15: 100, 16: 100, 17: 100, 18: 71,
    19: 220, 20: 180, 21: 160, 22: 140, 23: 135, 24: 140, 25: 140, 26: 140,
    27: 135, 28: 135, 29: 135, 30: 135, 31: 130, 32: 125, 33: 115, 34: 115,
    35: 115, 36: 88, 37: 235, 38: 200, 39: 180, 40: 155, 41: 145, 42: 145,
    43: 135, 44: 130, 45: 135, 46: 140, 47: 160, 48: 155, 49: 155, 50: 145,
    51: 145, 52: 140, 53: 140, 54: 108, 55: 260, 56: 215, 57: 195, 58: 185,
    59: 185, 60: 185, 61: 185, 62: 185, 63: 185, 64: 180, 65: 175, 66: 175,
    67: 175, 68: 175, 69: 175, 70: 175, 71: 175, 72: 155, 73: 145, 74: 135,
    75: 135, 76: 130, 77: 135, 78: 135, 79: 135, 80: 150, 81: 190, 82: 180,
    83: 160, 84: 190, 85: None, 86: 120, 87: 260, 88: 215, 89: 195, 90: 180,
    91: 180, 92: 175, 93: 175, 94: 175, 95: 175,
}

# Covalent radius in pm
COVALENT_RADIUS = {
    1: 31, 2: 28, 3: 128, 4: 96, 5: 84, 6: 76, 7: 71, 8: 66, 9: 57, 10: 58,
    11: 166, 12: 141, 13: 121, 14: 111, 15: 107, 16: 105, 17: 102, 18: 106,
    19: 203, 20: 176, 21: 170, 22: 160, 23: 153, 24: 139, 25: 150, 26: 142,
    27: 138, 28: 124, 29: 132, 30: 122, 31: 122, 32: 120, 33: 119, 34: 120,
    35: 120, 36: 116, 37: 220, 38: 195, 39: 190, 40: 175, 41: 164, 42: 154,
    43: 147, 44: 146, 45: 142, 46: 139, 47: 145, 48: 144, 49: 142, 50: 139,
    51: 139, 52: 138, 53: 139, 54: 140, 55: 244, 56: 215, 57: 207, 58: 204,
    59: 203, 60: 201, 61: 199, 62: 198, 63: 198, 64: 196, 65: 194, 66: 192,
    67: 192, 68: 189, 69: 190, 70: 187, 71: 187, 72: 175, 73: 170, 74: 162,
    75: 151, 76: 144, 77: 141, 78: 136, 79: 136, 80: 132, 81: 145, 82: 146,
    83: 148, 84: 140, 85: 150, 86: 150,
}

# Van der Waals radius in pm
VDW_RADIUS = {
    1: 120, 2: 140, 3: 182, 4: 153, 5: 192, 6: 170, 7: 155, 8: 152, 9: 147, 10: 154,
    11: 227, 12: 173, 13: 184, 14: 210, 15: 180, 16: 180, 17: 175, 18: 188,
    19: 275, 20: 231, 29: 140, 30: 139, 31: 187, 32: 211, 33: 185, 34: 190,
    35: 185, 36: 202, 37: 303, 38: 249, 47: 172, 48: 158, 49: 193, 50: 217,
    51: 206, 52: 206, 53: 198, 54: 216, 55: 343, 56: 268, 79: 166, 80: 155,
    81: 196, 82: 202, 83: 207, 84: 197, 85: 202, 86: 220,
}

# Crystal structure at standard conditions
CRYSTAL_STRUCTURE = {
    1: "HCP", 2: "HCP", 3: "BCC", 4: "HCP", 5: "Rhombohedral", 6: "Hexagonal",
    7: "HCP", 8: "Cubic", 9: "Cubic", 10: "FCC", 11: "BCC", 12: "HCP",
    13: "FCC", 14: "Diamond cubic", 15: "Orthorhombic", 16: "Orthorhombic",
    17: "Orthorhombic", 18: "FCC", 19: "BCC", 20: "FCC", 21: "HCP", 22: "HCP",
    23: "BCC", 24: "BCC", 25: "BCC", 26: "BCC", 27: "HCP", 28: "FCC",
    29: "FCC", 30: "HCP", 31: "Orthorhombic", 32: "Diamond cubic",
    33: "Rhombohedral", 34: "Hexagonal", 35: "Orthorhombic", 36: "FCC",
    37: "BCC", 38: "FCC", 39: "HCP", 40: "HCP", 41: "BCC", 42: "BCC",
    43: "HCP", 44: "HCP", 45: "FCC", 46: "FCC", 47: "FCC", 48: "HCP",
    49: "Tetragonal", 50: "Tetragonal", 51: "Rhombohedral", 52: "Hexagonal",
    53: "Orthorhombic", 54: "FCC", 55: "BCC", 56: "BCC",
    57: "DHCP", 58: "DHCP", 59: "DHCP", 60: "DHCP", 62: "Rhombohedral",
    63: "BCC", 64: "HCP", 65: "HCP", 66: "HCP", 67: "HCP", 68: "HCP",
    69: "HCP", 70: "FCC", 71: "HCP",
    72: "HCP", 73: "BCC", 74: "BCC", 75: "HCP", 76: "HCP", 77: "FCC",
    78: "FCC", 79: "FCC", 80: "Rhombohedral", 81: "HCP", 82: "FCC",
    83: "Rhombohedral", 84: "Cubic", 87: "BCC", 88: "BCC",
    89: "FCC", 90: "FCC", 91: "Tetragonal", 92: "Orthorhombic",
    93: "Orthorhombic", 94: "Monoclinic", 95: "DHCP", 96: "DHCP",
}

# State at room temperature (25 C): "Solid", "Liquid", "Gas"
STATE_AT_RT = {
    1: "Gas", 2: "Gas", 3: "Solid", 4: "Solid", 5: "Solid", 6: "Solid",
    7: "Gas", 8: "Gas", 9: "Gas", 10: "Gas", 11: "Solid", 12: "Solid",
    13: "Solid", 14: "Solid", 15: "Solid", 16: "Solid", 17: "Gas", 18: "Gas",
    19: "Solid", 20: "Solid", 21: "Solid", 22: "Solid", 23: "Solid", 24: "Solid",
    25: "Solid", 26: "Solid", 27: "Solid", 28: "Solid", 29: "Solid", 30: "Solid",
    31: "Solid", 32: "Solid", 33: "Solid", 34: "Solid", 35: "Liquid", 36: "Gas",
    37: "Solid", 38: "Solid", 39: "Solid", 40: "Solid", 41: "Solid", 42: "Solid",
    43: "Solid", 44: "Solid", 45: "Solid", 46: "Solid", 47: "Solid", 48: "Solid",
    49: "Solid", 50: "Solid", 51: "Solid", 52: "Solid", 53: "Solid", 54: "Gas",
    55: "Solid", 56: "Solid", 57: "Solid", 58: "Solid", 59: "Solid", 60: "Solid",
    61: "Solid", 62: "Solid", 63: "Solid", 64: "Solid", 65: "Solid", 66: "Solid",
    67: "Solid", 68: "Solid", 69: "Solid", 70: "Solid", 71: "Solid",
    72: "Solid", 73: "Solid", 74: "Solid", 75: "Solid", 76: "Solid", 77: "Solid",
    78: "Solid", 79: "Solid", 80: "Liquid", 81: "Solid", 82: "Solid", 83: "Solid",
    84: "Solid", 85: "Solid", 86: "Gas", 87: "Solid", 88: "Solid",
    89: "Solid", 90: "Solid", 91: "Solid", 92: "Solid", 93: "Solid", 94: "Solid",
    95: "Solid", 96: "Solid", 97: "Solid", 98: "Solid", 99: "Solid", 100: "Solid",
    101: "Solid", 102: "Solid", 103: "Solid",
}

# Electron affinity in eV (first)
ELECTRON_AFFINITY = {
    1: 0.754, 2: -0.50, 3: 0.618, 4: -0.50, 5: 0.280, 6: 1.263, 7: -0.07,
    8: 1.461, 9: 3.401, 10: -1.20, 11: 0.548, 12: -0.40, 13: 0.441, 14: 1.389,
    15: 0.746, 16: 2.077, 17: 3.617, 18: -1.00, 19: 0.501, 20: 0.024,
    21: 0.188, 22: 0.079, 23: 0.525, 24: 0.666, 25: -0.50, 26: 0.151,
    27: 0.662, 28: 1.156, 29: 1.235, 30: -0.60, 31: 0.300, 32: 1.233,
    33: 0.804, 34: 2.021, 35: 3.364, 36: -1.00, 37: 0.486, 38: 0.048,
    39: 0.307, 40: 0.426, 41: 0.893, 42: 0.746, 43: 0.550, 44: 1.050,
    45: 1.137, 46: 0.562, 47: 1.302, 48: -0.70, 49: 0.300, 50: 1.112,
    51: 1.047, 52: 1.971, 53: 3.059, 54: -0.80, 55: 0.472, 56: 0.145,
    79: 2.309, 80: -0.50, 81: 0.200, 82: 0.364, 83: 0.946,
}

# Discoverer (key elements)
DISCOVERER = {
    1: "Henry Cavendish", 2: "Pierre Janssen / Joseph Lockyer",
    3: "Johan August Arfwedson", 4: "Louis Nicolas Vauquelin",
    5: "Joseph Louis Gay-Lussac / Louis Jacques Thenard",
    6: "Known since antiquity", 7: "Daniel Rutherford",
    8: "Carl Wilhelm Scheele / Joseph Priestley", 9: "Henri Moissan",
    10: "William Ramsay / Morris Travers", 11: "Humphry Davy",
    12: "Joseph Black (recognized)", 13: "Hans Christian Oersted",
    14: "Jons Jacob Berzelius", 15: "Hennig Brand", 16: "Known since antiquity",
    17: "Carl Wilhelm Scheele", 18: "Lord Rayleigh / William Ramsay",
    19: "Humphry Davy", 20: "Humphry Davy",
    21: "Lars Fredrik Nilson", 22: "William Gregor",
    23: "Andres Manuel del Rio", 24: "Louis Nicolas Vauquelin",
    25: "Johan Gottlieb Gahn", 26: "Known since antiquity",
    27: "Georg Brandt", 28: "Axel Fredrik Cronstedt",
    29: "Known since antiquity", 30: "Known since antiquity",
    31: "Paul Emile Lecoq de Boisbaudran", 32: "Clemens Winkler",
    33: "Albertus Magnus", 34: "Jons Jacob Berzelius",
    35: "Antoine Jerome Balard / Carl Jacob Lowig", 36: "William Ramsay / Morris Travers",
    37: "Robert Bunsen / Gustav Kirchhoff", 38: "Adair Crawford / William Cruickshank",
    47: "Known since antiquity", 48: "Friedrich Stromeyer / Karl Samuel Leberecht Hermann",
    50: "Known since antiquity", 53: "Bernard Courtois",
    54: "William Ramsay / Morris Travers",
    55: "Robert Bunsen / Gustav Kirchhoff", 56: "Carl Wilhelm Scheele (identified)",
    74: "Juan Jose Elhuyar / Fausto Elhuyar", 78: "Antonio de Ulloa",
    79: "Known since antiquity", 80: "Known since antiquity",
    82: "Known since antiquity", 83: "Claude Francois Geoffroy",
    86: "Friedrich Ernst Dorn", 88: "Marie Curie / Pierre Curie",
    92: "Martin Heinrich Klaproth", 93: "Edwin McMillan / Philip Abelson",
    94: "Glenn T. Seaborg et al.",
}

# Etymology / Named after
ETYMOLOGY = {
    1: "Greek 'hydro' + 'genes' (water-forming)", 2: "Greek 'helios' (sun)",
    3: "Greek 'lithos' (stone)", 4: "Greek 'beryllos' (beryl mineral)",
    5: "Arabic 'buraq' (borax)", 6: "Latin 'carbo' (charcoal)",
    7: "Greek 'nitron' + 'genes' (niter-forming)", 8: "Greek 'oxy' + 'genes' (acid-forming)",
    9: "Latin 'fluere' (to flow)", 10: "Greek 'neos' (new)",
    11: "Latin 'natrium' (soda)", 12: "Greek 'Magnesia' (district in Thessaly)",
    13: "Latin 'alumen' (alum)", 14: "Latin 'silex' (flint)",
    15: "Greek 'phosphoros' (light-bearing)", 16: "Latin 'sulpur'",
    17: "Greek 'chloros' (greenish-yellow)", 18: "Greek 'argon' (idle/lazy)",
    19: "English 'potash'; Latin 'kalium'", 20: "Latin 'calx' (lime)",
    21: "Latin 'Scandia' (Scandinavia)", 22: "Greek 'Titans' (mythology)",
    23: "Old Norse 'Vanadis' (goddess Freyja)", 24: "Greek 'chroma' (color)",
    25: "Latin 'magnes' (magnet)", 26: "Anglo-Saxon 'iren'; Latin 'ferrum'",
    27: "German 'Kobold' (goblin)", 28: "German 'Kupfernickel' (false copper)",
    29: "Latin 'cuprum' (from Cyprus)", 30: "German 'Zink'",
    31: "Latin 'Gallia' (France)", 32: "Latin 'Germania' (Germany)",
    33: "Greek 'arsenikon' (yellow pigment)", 34: "Greek 'selene' (moon)",
    35: "Greek 'bromos' (stench)", 36: "Greek 'kryptos' (hidden)",
    37: "Latin 'rubidus' (deep red)", 38: "Strontian, village in Scotland",
    39: "Ytterby, village in Sweden", 40: "Persian 'zargun' (gold-colored)",
    41: "Greek 'Niobe' (mythology)", 42: "Greek 'molybdos' (lead)",
    43: "Greek 'technetos' (artificial)", 44: "Latin 'Ruthenia' (Russia)",
    45: "Greek 'rhodon' (rose)", 46: "Asteroid Pallas",
    47: "Anglo-Saxon 'seolfor'; Latin 'argentum'", 48: "Latin 'cadmia' (zinc ore)",
    49: "Indigo spectral line", 50: "Anglo-Saxon 'tin'; Latin 'stannum'",
    51: "Greek 'anti + monos' (not alone); Latin 'stibium'",
    52: "Latin 'tellus' (earth)", 53: "Greek 'iodes' (violet)",
    54: "Greek 'xenos' (stranger)", 55: "Latin 'caesius' (sky blue)",
    56: "Greek 'barys' (heavy)", 57: "Greek 'lanthanein' (to lie hidden)",
    58: "Asteroid Ceres", 59: "Greek 'prasios + didymos' (green twin)",
    60: "Greek 'neos + didymos' (new twin)",
    61: "Greek 'Prometheus' (mythology)",
    62: "Samarskite mineral (Col. Samarsky-Bykhovets)",
    63: "Europe", 64: "Johan Gadolin (chemist)",
    65: "Ytterby, Sweden", 66: "Greek 'dysprositos' (hard to get at)",
    67: "Latin 'Holmia' (Stockholm)", 68: "Ytterby, Sweden",
    69: "Thule (ancient name for Scandinavia)", 70: "Ytterby, Sweden",
    71: "Latin 'Lutetia' (Paris)",
    72: "Latin 'Hafnia' (Copenhagen)", 73: "Greek 'Tantalos' (mythology)",
    74: "Swedish 'tung sten' (heavy stone)", 75: "Latin 'Rhenus' (Rhine river)",
    76: "Greek 'osme' (smell)", 77: "Greek 'iris' (rainbow)",
    78: "Spanish 'platina' (little silver)", 79: "Anglo-Saxon 'gold'; Latin 'aurum'",
    80: "Planet Mercury; Latin 'hydrargyrum' (liquid silver)",
    81: "Greek 'thallos' (green shoot)", 82: "Anglo-Saxon 'lead'; Latin 'plumbum'",
    83: "German 'Bisemutum'", 84: "Latin 'Polonia' (Poland)",
    85: "Greek 'astatos' (unstable)", 86: "Radium emanation",
    87: "France", 88: "Latin 'radius' (ray)",
    89: "Greek 'aktinos' (ray)", 90: "Thor (Norse god)",
    91: "Greek 'protos + aktinos' (first ray)", 92: "Planet Uranus",
    93: "Planet Neptune", 94: "Dwarf planet Pluto",
    95: "Americas", 96: "Marie and Pierre Curie",
    97: "Berkeley, California", 98: "State of California",
    99: "Albert Einstein", 100: "Enrico Fermi",
    101: "Dmitri Mendeleev", 102: "Alfred Nobel",
    103: "Ernest O. Lawrence",
    104: "Ernest Rutherford", 105: "Dubna, Russia",
    106: "Glenn T. Seaborg", 107: "Niels Bohr",
    108: "State of Hesse, Germany", 109: "Lise Meitner",
    110: "Darmstadt, Germany", 111: "Wilhelm Rontgen",
    112: "Nicolaus Copernicus", 113: "Nihon (Japan)",
    114: "Flerov Laboratory", 115: "Moscow Oblast, Russia",
    116: "Lawrence Livermore National Laboratory",
    117: "State of Tennessee", 118: "Yuri Oganessian",
}

# Uses / applications (brief)
USES = {
    1: "Rocket fuel, ammonia synthesis, fuel cells, hydrogenation of fats.",
    2: "Cryogenics, MRI coolant, party balloons, leak detection, welding shield gas.",
    3: "Lithium-ion batteries, psychiatric medication, lightweight alloys, glass/ceramics.",
    4: "Aerospace alloys (with Cu), X-ray windows, nuclear reactors, gyroscopes.",
    5: "Borosilicate glass, fiberglass, detergents, semiconductors (doping).",
    6: "Steel production, carbon fiber, activated charcoal, diamonds, graphite lubricant.",
    7: "Fertilizer (ammonia), explosives, cryopreservation, food packaging atmosphere.",
    8: "Respiration, steel-making, medical oxygen, welding, water treatment.",
    9: "Toothpaste (fluoride), Teflon (PTFE), refrigerants, uranium enrichment (UF6).",
    10: "Neon signs and lighting, high-voltage indicators, lasers, cryogenic refrigerant.",
    11: "Table salt (NaCl), street lamps, chemical reagent, heat transfer fluid.",
    12: "Lightweight alloys (aircraft), fireworks (white light), Epsom salts, chlorophyll.",
    13: "Aircraft construction, beverage cans, foil, electrical transmission lines.",
    14: "Semiconductors (computer chips), solar cells, glass, silicone polymers.",
    15: "Fertilizers, matches, detergents, pesticides, steel production.",
    16: "Sulfuric acid production, vulcanizing rubber, gunpowder, fungicides.",
    17: "Water purification, PVC production, bleach, disinfectants.",
    18: "Welding shield gas, fluorescent lighting, insulating windows.",
    19: "Fertilizers (potash), soap making, gunpowder, potassium hydroxide.",
    20: "Cement/concrete, plaster, calcium supplements, steel deoxidizer.",
    21: "Aerospace alloys, sports equipment, mercury vapor lamps.",
    22: "Aerospace/aircraft, medical implants, pigment (TiO2), corrosion-resistant piping.",
    23: "Steel alloys (tools), vanadium redox batteries, catalysts.",
    24: "Stainless steel, chrome plating, pigments, tanning leather.",
    25: "Steel production, batteries (MnO2), water purification, pigments.",
    26: "Steel/construction, vehicles, machinery, hemoglobin in blood.",
    27: "Superalloys (jet engines), lithium-ion battery cathodes, magnets, pigments.",
    28: "Stainless steel, coins, nickel-cadmium batteries, electroplating.",
    29: "Electrical wiring, plumbing, electronics, coinage, antimicrobial surfaces.",
    30: "Galvanizing steel, brass alloys, batteries, dietary supplement.",
    31: "Semiconductors (GaAs), LEDs, solar cells, thermometers.",
    32: "Fiber optics, infrared optics, transistors, PET catalysts.",
    33: "Semiconductors (GaAs), wood preservatives, pesticides (historical).",
    34: "Photocopiers, glass coloring, solar cells, rubber vulcanization.",
    35: "Flame retardants, water purification, photography, sedatives (historical).",
    36: "Fluorescent lamps, flash photography, insulating windows.",
    37: "Fireworks (purple), photocells, atomic clocks research.",
    38: "Fireworks (red), ferrite magnets, CRT glass, flares.",
    39: "Phosphors in LEDs/CRTs, ceramic superconductors, laser crystals.",
    40: "Nuclear reactor cladding, ceramics, surgical instruments.",
    41: "Superconducting magnets (MRI, LHC), rocket nozzles, steel alloys.",
    42: "High-strength steel alloys, catalysts, lubricant additive (MoS2).",
    43: "Nuclear medicine (99mTc radioisotope imaging), calibration source.",
    44: "Catalysts, electrical contacts, solar cells, wear-resistant alloys.",
    45: "Catalytic converters, jewelry, electronics contacts, mirrors.",
    46: "Catalytic converters, electronics, dental alloys, hydrogen purification.",
    47: "Jewelry, electronics (solder, contacts), photography, antimicrobial, mirrors.",
    48: "Ni-Cd batteries, pigments, coatings, nuclear reactor control rods.",
    49: "Touch screens (ITO), solders, semiconductors.",
    50: "Tin cans (coating), solder, bronze alloys, tin foil.",
    51: "Flame retardants, lead-acid batteries, semiconductors.",
    52: "Thermoelectric devices, semiconductor alloys, rubber vulcanization.",
    53: "Disinfectants, medical contrast agents, photography, table salt iodization.",
    54: "Anesthetics, ion propulsion, lighting, bubble chambers.",
    55: "Atomic clocks, photoelectric cells, ion propulsion, drilling fluids.",
    56: "Medical imaging (barium swallow), drilling fluids, fireworks (green).",
    57: "Camera/telescope lenses, catalysts, hydrogen storage, lighter flints.",
    58: "Catalytic converters, self-cleaning ovens, glass polishing, lighter flints.",
    59: "Aircraft engines alloy, magnets, glass coloring (green), arc lighting.",
    60: "NdFeB magnets (speakers, hard drives, EVs), lasers, glass coloring.",
    62: "Magnets (SmCo), nuclear reactor control, cancer treatment (153Sm).",
    63: "Red/blue phosphors (TVs/LEDs), nuclear control rods, Euro banknote security.",
    64: "MRI contrast agent (Gd-DTPA), neutron capture, magnets.",
    72: "Nuclear reactor control rods, plasma cutting tips, super alloys.",
    73: "Capacitors (electronics), surgical instruments, jet engine parts.",
    74: "Incandescent bulb filaments, cutting tools, armor-piercing ammo, X-ray targets.",
    75: "Jet engine superalloys, catalysts, thermocouples.",
    76: "Fountain pen nibs, instrument pivots, fingerprint detection.",
    77: "Spark plugs, crucibles, standard kilogram (historical).",
    78: "Catalytic converters, jewelry, chemotherapy (cisplatin), electronics.",
    79: "Jewelry, electronics, central bank reserves, dental work, nanotechnology.",
    80: "Thermometers (historical), fluorescent lighting, dental amalgams.",
    81: "Electronics, superconductors, medical imaging (201Tl).",
    82: "Lead-acid batteries, radiation shielding, solder (historical), ammunition.",
    83: "Pharmaceuticals (Pepto-Bismol), cosmetics, fire sprinklers, fusible alloys.",
    84: "Antistatic devices, neutron source (with Be), thermoelectric power (space probes).",
    86: "Radon testing in homes, cancer radiotherapy (historical).",
    88: "Luminescent paint (historical), neutron source, radiotherapy.",
    90: "Nuclear fuel (molten salt reactors), gas lamp mantles, welding rods.",
    92: "Nuclear fuel (power/weapons), armor plating (DU), counterweights.",
    94: "Nuclear weapons, RTGs (space probes), research.",
    95: "Smoke detectors (241Am), radiography.",
}

# Natural abundance in Earth's crust (ppm) and rank
ABUNDANCE = {
    1: (1400, 10), 2: (0.008, 71), 3: (20, 33), 4: (2.8, 47), 5: (10, 38),
    6: (200, 15), 7: (19, 34), 8: (461000, 1), 9: (585, 13), 10: (0.005, 73),
    11: (23600, 6), 12: (23300, 7), 13: (82300, 3), 14: (282000, 2), 15: (1050, 11),
    16: (350, 14), 17: (145, 19), 18: (3.5, 56), 19: (20900, 8), 20: (41500, 5),
    21: (22, 31), 22: (5650, 9), 23: (120, 20), 24: (102, 21), 25: (950, 12),
    26: (56300, 4), 27: (25, 30), 28: (84, 22), 29: (60, 25), 30: (70, 24),
    31: (19, 35), 32: (1.5, 52), 33: (1.8, 51), 34: (0.05, 66), 35: (2.4, 49),
    36: (0.0001, 79), 37: (90, 23), 38: (370, 15), 47: (0.075, 64), 48: (0.15, 61),
    49: (0.25, 57), 50: (2.3, 50), 53: (0.45, 55), 55: (3, 48), 56: (425, 14),
    57: (39, 28), 58: (66.5, 25), 79: (0.004, 74), 80: (0.085, 62),
    82: (14, 36), 83: (0.009, 70), 90: (9.6, 39), 92: (2.7, 48),
}

# Isotopes: (number_of_stable_isotopes, most_common_isotope_mass_number)
ISOTOPE_INFO = {
    1: (2, 1), 2: (2, 4), 3: (2, 7), 4: (1, 9), 5: (2, 11), 6: (2, 12),
    7: (2, 14), 8: (3, 16), 9: (1, 19), 10: (3, 20), 11: (1, 23), 12: (3, 24),
    13: (1, 27), 14: (3, 28), 15: (1, 31), 16: (4, 32), 17: (2, 35), 18: (3, 40),
    19: (2, 39), 20: (5, 40), 21: (1, 45), 22: (5, 48), 23: (1, 51), 24: (4, 52),
    25: (1, 55), 26: (4, 56), 27: (1, 59), 28: (5, 58), 29: (2, 63), 30: (5, 64),
    31: (2, 69), 32: (4, 74), 33: (1, 75), 34: (5, 80), 35: (2, 79), 36: (5, 84),
    37: (1, 85), 38: (4, 88), 39: (1, 89), 40: (4, 90), 41: (1, 93), 42: (6, 98),
    43: (0, 98), 44: (7, 102), 45: (1, 103), 46: (6, 106), 47: (2, 107), 48: (6, 114),
    49: (1, 115), 50: (7, 120), 51: (2, 121), 52: (6, 130), 53: (1, 127), 54: (7, 132),
    55: (1, 133), 56: (6, 138), 57: (1, 139), 58: (3, 140), 59: (1, 141), 60: (5, 142),
    62: (5, 152), 63: (1, 153), 64: (6, 158), 65: (1, 159), 66: (7, 164),
    67: (1, 165), 68: (6, 166), 69: (1, 169), 70: (6, 174), 71: (1, 175),
    72: (5, 180), 73: (1, 181), 74: (4, 184), 75: (1, 187), 76: (5, 192),
    77: (2, 193), 78: (5, 195), 79: (1, 197), 80: (6, 202), 81: (2, 205),
    82: (3, 208), 83: (1, 209), 90: (0, 232), 92: (0, 238),
}


# ── Helper Functions ────────────────────────────────────────────────────────

def _get_element(z):
    """Return element tuple by atomic number (1-based)."""
    return ELEMENTS[z - 1]


def _elem_dict(t):
    """Convert element tuple to dict."""
    keys = ["Z", "symbol", "name", "mass", "category", "electron_config",
            "electronegativity", "ionization_energy", "density",
            "melting_point", "boiling_point", "discovery_year", "oxidation_states"]
    return dict(zip(keys, t))


def _fmt(val, suffix=""):
    """Format a value with optional suffix, returning 'N/A' for None."""
    if val is None:
        return "N/A"
    return f"{val}{suffix}"


def _k_to_c(k):
    """Convert Kelvin to Celsius string."""
    if k is None:
        return "N/A"
    return f"{k - 273.15:.1f}"


# ── Dialogs ─────────────────────────────────────────────────────────────────

class ElementDetailDialog(QDialog):
    """Dialog showing full details for a single element."""

    def __init__(self, elem, parent=None):
        super().__init__(parent)
        d = _elem_dict(elem)
        self.setWindowTitle(f"{d['name']} ({d['symbol']})")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        header = QLabel(f"<h2 style='text-align:center'>{d['Z']}  {d['symbol']}</h2>"
                        f"<h3 style='text-align:center'>{d['name']}</h3>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        cat_name = CATEGORY_NAMES.get(d["category"], d["category"])
        props = [
            ("Category", cat_name),
            ("Atomic Mass", f"{d['mass']:.3f} u"),
            ("Electron Config", d["electron_config"]),
            ("Electronegativity", _fmt(d["electronegativity"])),
            ("Ionization Energy", _fmt(d["ionization_energy"], " eV")),
            ("Density", _fmt(d["density"], " g/cm\u00b3")),
            ("Melting Point", _fmt(d["melting_point"], " K")),
            ("Boiling Point", _fmt(d["boiling_point"], " K")),
            ("Discovery Year", _fmt(d["discovery_year"])),
            ("Oxidation States", d["oxidation_states"] or "N/A"),
        ]
        for name, val in props:
            row = QHBoxLayout()
            lbl = QLabel(f"<b>{name}:</b>")
            lbl.setFixedWidth(160)
            row.addWidget(lbl)
            row.addWidget(QLabel(str(val)))
            row.addStretch()
            layout.addLayout(row)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class CompareDialog(QDialog):
    """Side-by-side comparison of two elements."""

    def __init__(self, elem1, elem2, parent=None):
        super().__init__(parent)
        d1, d2 = _elem_dict(elem1), _elem_dict(elem2)
        self.setWindowTitle(f"Compare: {d1['symbol']} vs {d2['symbol']}")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        headers = ["Property", d1["symbol"], d2["symbol"]]
        for c, h in enumerate(headers):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, c)

        rows = [
            ("Atomic Number", d1["Z"], d2["Z"]),
            ("Name", d1["name"], d2["name"]),
            ("Mass (u)", f"{d1['mass']:.3f}", f"{d2['mass']:.3f}"),
            ("Category", CATEGORY_NAMES.get(d1["category"], ""), CATEGORY_NAMES.get(d2["category"], "")),
            ("Electronegativity", d1["electronegativity"], d2["electronegativity"]),
            ("Ionization (eV)", d1["ionization_energy"], d2["ionization_energy"]),
            ("Density (g/cm3)", d1["density"], d2["density"]),
            ("Melting Pt (K)", d1["melting_point"], d2["melting_point"]),
            ("Boiling Pt (K)", d1["boiling_point"], d2["boiling_point"]),
            ("Oxidation States", d1["oxidation_states"], d2["oxidation_states"]),
        ]
        for r, (prop, v1, v2) in enumerate(rows, 1):
            grid.addWidget(QLabel(prop), r, 0)
            grid.addWidget(QLabel(str(v1) if v1 is not None else "N/A"), r, 1)
            grid.addWidget(QLabel(str(v2) if v2 is not None else "N/A"), r, 2)

        layout.addLayout(grid)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ── Main Widget ─────────────────────────────────────────────────────────────

class PeriodicTableWidget(QWidget):
    """Interactive periodic table widget with comprehensive element data."""

    element_clicked = pyqtSignal(int)  # emits atomic number

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None
        self._buttons = {}       # Z -> QPushButton
        self._selected = []      # for comparison (max 2)
        self._current_z = None   # currently displayed element
        self._init_ui()

    def set_logger(self, fn):
        """Set external logging function."""
        self._log = fn

    def _emit_log(self, msg):
        if self._log:
            self._log(msg)

    # ── UI Construction ─────────────────────────────────────────────────
    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(4)

        # Search bar
        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        search_lbl = QLabel("Search:")
        search_lbl.setStyleSheet("font-weight: bold; color: #c0c0c0;")
        search_row.addWidget(search_lbl)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Element name, symbol, or atomic number...")
        self._search.setStyleSheet(
            "QLineEdit { background: #1e2a38; border: 1px solid #3a5068; border-radius: 4px;"
            "padding: 4px 8px; color: #e0e0e0; font-size: 12px; }"
            "QLineEdit:focus { border-color: #00d4ff; }"
        )
        self._search.textChanged.connect(self._on_search)
        search_row.addWidget(self._search)

        self._compare_btn = QPushButton("Compare Selected (0/2)")
        self._compare_btn.setEnabled(False)
        self._compare_btn.setStyleSheet(
            "QPushButton { background: #2a3f55; color: #b0c4de; border: 1px solid #3a5068;"
            "border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #3a5068; }"
            "QPushButton:disabled { color: #5a6a7a; }"
        )
        self._compare_btn.clicked.connect(self._on_compare)
        search_row.addWidget(self._compare_btn)

        self._clear_sel_btn = QPushButton("Clear Selection")
        self._clear_sel_btn.setStyleSheet(
            "QPushButton { background: #2a3f55; color: #b0c4de; border: 1px solid #3a5068;"
            "border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #3a5068; }"
        )
        self._clear_sel_btn.clicked.connect(self._clear_selection)
        search_row.addWidget(self._clear_sel_btn)
        main.addLayout(search_row)

        # Trend visualization row
        trend_row = QHBoxLayout()
        trend_row.setSpacing(6)
        trend_lbl = QLabel("Trend:")
        trend_lbl.setStyleSheet("font-weight: bold; color: #c0c0c0;")
        trend_row.addWidget(trend_lbl)
        self._trend_prop = QComboBox()
        self._trend_prop.addItems(["Electronegativity", "Ionization Energy", "Density",
                                   "Melting Point", "Boiling Point", "Atomic Mass"])
        self._trend_prop.setStyleSheet(
            "QComboBox { background: #1e2a38; color: #e0e0e0; border: 1px solid #3a5068;"
            "border-radius: 4px; padding: 3px 6px; }"
        )
        trend_row.addWidget(self._trend_prop)

        across_lbl = QLabel("Across:")
        across_lbl.setStyleSheet("font-weight: bold; color: #c0c0c0;")
        trend_row.addWidget(across_lbl)
        self._trend_mode = QComboBox()
        self._trend_mode.addItems(["Period 1", "Period 2", "Period 3", "Period 4",
                                   "Period 5", "Period 6", "Period 7",
                                   "Group 1", "Group 2", "Group 13", "Group 14",
                                   "Group 15", "Group 16", "Group 17", "Group 18"])
        self._trend_mode.setStyleSheet(
            "QComboBox { background: #1e2a38; color: #e0e0e0; border: 1px solid #3a5068;"
            "border-radius: 4px; padding: 3px 6px; }"
        )
        trend_row.addWidget(self._trend_mode)

        _btn_style = (
            "QPushButton { background: #1a3a5c; color: #8ec8e8; border: 1px solid #2a5a7a;"
            "border-radius: 4px; padding: 4px 12px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: #2a5a7a; color: #ffffff; }"
        )

        plot_btn = QPushButton("Plot Trend")
        plot_btn.setStyleSheet(_btn_style)
        plot_btn.clicked.connect(self._plot_trend)
        trend_row.addWidget(plot_btn)
        main.addLayout(trend_row)

        # Feature buttons row
        feature_row = QHBoxLayout()
        feature_row.setSpacing(6)
        shell_btn = QPushButton("Electron Shell Diagram")
        shell_btn.setStyleSheet(_btn_style)
        shell_btn.clicked.connect(self._electron_shell_dialog)
        feature_row.addWidget(shell_btn)

        isotope_btn = QPushButton("Isotope Browser")
        isotope_btn.setStyleSheet(_btn_style)
        isotope_btn.clicked.connect(self._isotope_browser)
        feature_row.addWidget(isotope_btn)

        full_trend_btn = QPushButton("Full-Table Trend Plot")
        full_trend_btn.setStyleSheet(_btn_style)
        full_trend_btn.clicked.connect(self._full_table_trend_plot)
        feature_row.addWidget(full_trend_btn)
        main.addLayout(feature_row)

        # Periodic table grid inside scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        grid_widget = QWidget()
        grid_widget.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(grid_widget)
        self._grid.setSpacing(2)
        self._grid.setContentsMargins(2, 2, 2, 2)

        for elem in ELEMENTS:
            z = elem[0]
            if z not in PT_LAYOUT:
                continue
            row, col = PT_LAYOUT[z]
            btn = self._make_element_button(elem)
            btn.clicked.connect(lambda checked, zz=z: self._on_element_click(zz))
            self._grid.addWidget(btn, row, col)
            self._buttons[z] = btn

        # Lanthanide / Actinide labels
        lan_label = QLabel("57-70")
        lan_label.setAlignment(Qt.AlignCenter)
        lan_label.setStyleSheet("color: #9a7aa8; font-size: 10px; font-weight: bold;")
        self._grid.addWidget(lan_label, 8, 0, 1, 2)
        act_label = QLabel("89-102")
        act_label.setAlignment(Qt.AlignCenter)
        act_label.setStyleSheet("color: #b07080; font-size: 10px; font-weight: bold;")
        self._grid.addWidget(act_label, 9, 0, 1, 2)

        scroll.setWidget(grid_widget)
        main.addWidget(scroll, stretch=3)

        # Legend (horizontal at the bottom)
        legend_frame = QFrame()
        legend_frame.setStyleSheet(
            "QFrame { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 4px; }"
        )
        legend_layout = QHBoxLayout(legend_frame)
        legend_layout.setContentsMargins(8, 4, 8, 4)
        legend_layout.setSpacing(4)
        for cat, name in CATEGORY_NAMES.items():
            color = CATEGORY_COLORS[cat]
            lbl = QLabel(f"  {name}  ")
            lbl.setStyleSheet(
                f"background-color: {color}; color: #1a1a1a; border: 1px solid #5a5a5a;"
                f"border-radius: 3px; font-size: 9px; padding: 2px 4px; font-weight: bold;"
            )
            legend_layout.addWidget(lbl)
        legend_layout.addStretch()
        main.addWidget(legend_frame)

        # Info panel (scrollable HTML)
        self._info = QTextEdit()
        self._info.setReadOnly(True)
        self._info.setMinimumHeight(220)
        self._info.setStyleSheet(
            "QTextEdit { background: #0a1018; border: 1px solid #1a2a3a; border-radius: 4px;"
            "color: #d0d8e0; font-size: 12px; padding: 6px; }"
        )
        self._info.setHtml(self._welcome_html())
        main.addWidget(self._info, stretch=2)

    def _make_element_button(self, elem):
        """Create a styled element button with atomic number, symbol, and mass."""
        z, symbol, name, mass, cat = elem[0], elem[1], elem[2], elem[3], elem[4]
        color = CATEGORY_COLORS.get(cat, "#FFFFFF")

        btn = QPushButton()
        btn.setFixedSize(60, 60)
        btn.setToolTip(f"{name} ({symbol}) - Z={z}")

        # Build rich text label: atomic number top-left, symbol center, mass bottom
        mass_str = f"{mass:.1f}" if mass < 100 else f"{mass:.0f}"
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {color};"
            f"  border: 1px solid #6a6a6a;"
            f"  border-radius: 4px;"
            f"  color: #1a1a1a;"
            f"  padding: 0px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  border: 2px solid #00d4ff;"
            f"  background-color: {color};"
            f"}}"
        )

        # Use a layout inside the button for multi-line content
        btn_layout = QVBoxLayout(btn)
        btn_layout.setContentsMargins(2, 1, 2, 1)
        btn_layout.setSpacing(0)

        z_label = QLabel(str(z))
        z_label.setStyleSheet("color: #1a1a1a; font-size: 8px; background: transparent; border: none;")
        z_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        z_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        btn_layout.addWidget(z_label)

        sym_label = QLabel(symbol)
        sym_label.setStyleSheet(
            "color: #1a1a1a; font-size: 16px; font-weight: bold; background: transparent; border: none;"
        )
        sym_label.setAlignment(Qt.AlignCenter)
        sym_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        btn_layout.addWidget(sym_label)

        mass_label = QLabel(mass_str)
        mass_label.setStyleSheet("color: #2a2a2a; font-size: 7px; background: transparent; border: none;")
        mass_label.setAlignment(Qt.AlignCenter | Qt.AlignBottom)
        mass_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        btn_layout.addWidget(mass_label)

        return btn

    def _welcome_html(self):
        """Return the welcome / placeholder HTML for the info panel."""
        return (
            "<div style='text-align:center; padding: 30px; color: #5a7a9a;'>"
            "<h2 style='color: #4a90b8;'>Periodic Table of Elements</h2>"
            "<p style='font-size: 13px;'>Click any element to view comprehensive details.<br>"
            "Select two elements to enable comparison mode.</p>"
            "<p style='font-size: 11px; color: #3a5a7a;'>"
            "Features: Search | Compare | Trend Plots | Electron Shells | Isotope Browser</p>"
            "</div>"
        )

    # ── Comprehensive Info Panel HTML ──────────────────────────────────
    def _build_info_html(self, z):
        """Build comprehensive, beautifully formatted HTML for the element detail panel."""
        d = _elem_dict(_get_element(z))
        cat = d["category"]
        cat_name = CATEGORY_NAMES.get(cat, cat)
        cat_color = CATEGORY_COLORS.get(cat, "#888")
        block = ELEMENT_BLOCK.get(z, "?")
        period = ELEMENT_PERIOD.get(z, "?")
        group = ELEMENT_GROUP.get(z)
        group_str = str(group) if group else "N/A (Lan/Act)"

        state = STATE_AT_RT.get(z, "Unknown")
        crystal = CRYSTAL_STRUCTURE.get(z)
        a_radius = ATOMIC_RADIUS.get(z)
        c_radius = COVALENT_RADIUS.get(z)
        vdw = VDW_RADIUS.get(z)
        ea = ELECTRON_AFFINITY.get(z)
        discoverer = DISCOVERER.get(z)
        etym = ETYMOLOGY.get(z)
        uses = USES.get(z)
        abundance = ABUNDANCE.get(z)
        iso_info = ISOTOPE_INFO.get(z)

        mp_k = d["melting_point"]
        bp_k = d["boiling_point"]
        mp_c = _k_to_c(mp_k)
        bp_c = _k_to_c(bp_k)

        # CSS styles for sections
        css = (
            "<style>"
            "body { font-family: 'Segoe UI', Arial, sans-serif; color: #d0d8e0; margin: 0; padding: 0; }"
            ".header { text-align: center; padding: 8px 0 4px 0; }"
            ".header .z { font-size: 14px; color: #6a8aaa; }"
            f".header .symbol {{ font-size: 36px; font-weight: bold; color: {cat_color}; }}"
            ".header .name { font-size: 18px; color: #b0c8e0; font-weight: 600; }"
            f".header .cat {{ display: inline-block; background: {cat_color}; color: #1a1a1a;"
            "  padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: bold; margin-top: 2px; }}"
            ".section { margin: 6px 0; padding: 6px 10px; border-radius: 6px; }"
            ".section-title { font-size: 12px; font-weight: bold; color: #6aafcf; margin-bottom: 4px;"
            "  border-bottom: 1px solid #1a3050; padding-bottom: 2px; text-transform: uppercase; letter-spacing: 1px; }"
            ".s-atomic { background: #101820; }"
            ".s-thermo { background: #0f1a14; }"
            ".s-physical { background: #14101a; }"
            ".s-size { background: #1a1510; }"
            ".s-history { background: #10141a; }"
            ".s-uses { background: #0f1018; }"
            "table { width: 100%; border-collapse: collapse; }"
            "td { padding: 2px 6px; font-size: 11px; vertical-align: top; }"
            "td.prop { color: #7a9ab8; font-weight: 600; width: 45%; white-space: nowrap; }"
            "td.val { color: #d0dce8; }"
            "</style>"
        )

        # Header
        header = (
            f"<div class='header'>"
            f"<span class='z'>Z = {d['Z']}</span><br>"
            f"<span class='symbol'>{d['symbol']}</span><br>"
            f"<span class='name'>{d['name']}</span><br>"
            f"<span class='cat'>{cat_name}</span>"
            f"</div>"
        )

        # Atomic Properties section
        atomic_rows = (
            f"<tr><td class='prop'>Atomic Mass</td><td class='val'>{d['mass']:.4f} u</td></tr>"
            f"<tr><td class='prop'>Electron Configuration</td><td class='val'><code>{d['electron_config']}</code></td></tr>"
            f"<tr><td class='prop'>Block / Period / Group</td><td class='val'>{block}-block &middot; Period {period} &middot; Group {group_str}</td></tr>"
            f"<tr><td class='prop'>Electronegativity (Pauling)</td><td class='val'>{_fmt(d['electronegativity'])}</td></tr>"
            f"<tr><td class='prop'>1st Ionization Energy</td><td class='val'>{_fmt(d['ionization_energy'], ' eV')}</td></tr>"
            f"<tr><td class='prop'>Electron Affinity</td><td class='val'>{_fmt(ea, ' eV') if ea is not None else 'N/A'}</td></tr>"
            f"<tr><td class='prop'>Oxidation States</td><td class='val'>{d['oxidation_states'] or 'N/A'}</td></tr>"
        )
        atomic_section = (
            f"<div class='section s-atomic'>"
            f"<div class='section-title'>Atomic Properties</div>"
            f"<table>{atomic_rows}</table></div>"
        )

        # Thermodynamic Properties
        thermo_rows = (
            f"<tr><td class='prop'>Melting Point</td><td class='val'>"
            f"{_fmt(mp_k, ' K')}{(' (' + mp_c + ' C)') if mp_k is not None else ''}</td></tr>"
            f"<tr><td class='prop'>Boiling Point</td><td class='val'>"
            f"{_fmt(bp_k, ' K')}{(' (' + bp_c + ' C)') if bp_k is not None else ''}</td></tr>"
            f"<tr><td class='prop'>State at Room Temp.</td><td class='val'>{state}</td></tr>"
        )
        thermo_section = (
            f"<div class='section s-thermo'>"
            f"<div class='section-title'>Thermodynamic Properties</div>"
            f"<table>{thermo_rows}</table></div>"
        )

        # Physical Properties
        physical_rows = (
            f"<tr><td class='prop'>Density</td><td class='val'>{_fmt(d['density'], ' g/cm3')}</td></tr>"
            f"<tr><td class='prop'>Crystal Structure</td><td class='val'>{crystal or 'N/A'}</td></tr>"
        )
        physical_section = (
            f"<div class='section s-physical'>"
            f"<div class='section-title'>Physical Properties</div>"
            f"<table>{physical_rows}</table></div>"
        )

        # Atomic Size
        size_rows = (
            f"<tr><td class='prop'>Atomic Radius</td><td class='val'>{_fmt(a_radius, ' pm') if a_radius else 'N/A'}</td></tr>"
            f"<tr><td class='prop'>Covalent Radius</td><td class='val'>{_fmt(c_radius, ' pm') if c_radius else 'N/A'}</td></tr>"
            f"<tr><td class='prop'>Van der Waals Radius</td><td class='val'>{_fmt(vdw, ' pm') if vdw else 'N/A'}</td></tr>"
        )
        size_section = (
            f"<div class='section s-size'>"
            f"<div class='section-title'>Atomic Size</div>"
            f"<table>{size_rows}</table></div>"
        )

        # Isotopes
        iso_html = ""
        if iso_info:
            n_stable, common = iso_info
            iso_html = (
                f"<tr><td class='prop'>Stable Isotopes</td><td class='val'>{n_stable}</td></tr>"
                f"<tr><td class='prop'>Most Common Isotope</td><td class='val'>"
                f"<sup>{common}</sup>{d['symbol']}</td></tr>"
            )

        # Abundance
        abund_html = ""
        if abundance:
            ppm, rank = abundance
            abund_html = (
                f"<tr><td class='prop'>Crustal Abundance</td><td class='val'>"
                f"{ppm} ppm (rank #{rank})</td></tr>"
            )

        # History / Discovery
        disc_year = d["discovery_year"]
        if disc_year is not None and disc_year < 0:
            disc_str = f"~{abs(disc_year)} BCE"
        else:
            disc_str = _fmt(disc_year)

        disc_row = f"<tr><td class='prop'>Discovered</td><td class='val'>{disc_str}</td></tr>"
        discoverer_row = f"<tr><td class='prop'>Discoverer</td><td class='val'>{discoverer}</td></tr>" if discoverer else ""
        etym_row = f"<tr><td class='prop'>Etymology</td><td class='val'>{etym}</td></tr>" if etym else ""
        history_rows = disc_row + discoverer_row + etym_row + iso_html + abund_html
        history_section = (
            f"<div class='section s-history'>"
            f"<div class='section-title'>Discovery &amp; Classification</div>"
            f"<table>{history_rows}</table></div>"
        )

        # Uses
        uses_section = ""
        if uses:
            uses_section = (
                f"<div class='section s-uses'>"
                f"<div class='section-title'>Applications &amp; Uses</div>"
                f"<p style='font-size: 11px; color: #b0c0d0; margin: 2px 6px;'>{uses}</p>"
                f"</div>"
            )

        html = (
            f"<html><head>{css}</head><body>"
            f"{header}"
            f"<table style='width:100%'><tr><td style='vertical-align:top; width:50%;'>"
            f"{atomic_section}{thermo_section}{uses_section}"
            f"</td><td style='vertical-align:top; width:50%;'>"
            f"{physical_section}{size_section}{history_section}"
            f"</td></tr></table>"
            f"</body></html>"
        )
        return html

    # ── Handlers ────────────────────────────────────────────────────────
    def _on_element_click(self, z):
        elem = _get_element(z)
        d = _elem_dict(elem)
        self.element_clicked.emit(z)
        self._emit_log(f"Selected element: {d['name']} (Z={z})")
        self._current_z = z

        # Update info panel with comprehensive HTML
        self._info.setHtml(self._build_info_html(z))

        # Update button highlight (selection for comparison)
        cat = d["category"]
        color = CATEGORY_COLORS.get(cat, "#FFFFFF")

        if z in self._selected:
            self._selected.remove(z)
            self._reset_button_style(z)
        else:
            if len(self._selected) >= 2:
                old = self._selected.pop(0)
                self._reset_button_style(old)
            self._selected.append(z)
            # Apply glow border
            self._buttons[z].setStyleSheet(
                f"QPushButton {{"
                f"  background-color: {color};"
                f"  border: 3px solid #00d4ff;"
                f"  border-radius: 4px;"
                f"  color: #1a1a1a;"
                f"  padding: 0px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  border: 3px solid #00d4ff;"
                f"  background-color: {color};"
                f"}}"
            )
        self._compare_btn.setText(f"Compare Selected ({len(self._selected)}/2)")
        self._compare_btn.setEnabled(len(self._selected) == 2)

    def _reset_button_style(self, z):
        """Reset an element button to its default non-selected style."""
        elem = _get_element(z)
        cat = elem[4]
        color = CATEGORY_COLORS.get(cat, "#FFFFFF")
        self._buttons[z].setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {color};"
            f"  border: 1px solid #6a6a6a;"
            f"  border-radius: 4px;"
            f"  color: #1a1a1a;"
            f"  padding: 0px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  border: 2px solid #00d4ff;"
            f"  background-color: {color};"
            f"}}"
        )

    def _clear_selection(self):
        for z in self._selected:
            self._reset_button_style(z)
        self._selected.clear()
        self._compare_btn.setText("Compare Selected (0/2)")
        self._compare_btn.setEnabled(False)

    def _on_compare(self):
        if len(self._selected) != 2:
            return
        e1 = _get_element(self._selected[0])
        e2 = _get_element(self._selected[1])
        self._emit_log(f"Comparing {e1[1]} and {e2[1]}")
        dlg = CompareDialog(e1, e2, self)
        dlg.exec_()

    def _on_search(self, text):
        text = text.strip().lower()
        for z, btn in self._buttons.items():
            elem = _get_element(z)
            match = (
                (not text)
                or text in elem[1].lower()
                or text in elem[2].lower()
                or text == str(elem[0])
            )
            btn.setVisible(match)

    # ── Trend Plotting ──────────────────────────────────────────────────
    def _plot_trend(self):
        prop_map = {
            "Electronegativity": 6, "Ionization Energy": 7, "Density": 8,
            "Melting Point": 9, "Boiling Point": 10, "Atomic Mass": 3,
        }
        prop_name = self._trend_prop.currentText()
        idx = prop_map[prop_name]
        mode = self._trend_mode.currentText()
        self._emit_log(f"Plotting trend: {prop_name} across {mode}")

        elements = self._get_trend_elements(mode)
        if not elements:
            return

        zs, vals, labels = [], [], []
        for elem in elements:
            v = elem[idx]
            if v is not None:
                zs.append(elem[0])
                vals.append(float(v))
                labels.append(elem[1])

        if not vals:
            QMessageBox.information(self, "No Data", "No data available for this property/selection.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Trend: {prop_name} - {mode}")
        dlg.setMinimumSize(640, 420)
        lay = QVBoxLayout(dlg)

        fig = Figure(figsize=(7, 4))
        style_figure(fig)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(range(len(vals)), vals, 'o-', color='#2196F3', markersize=6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel(prop_name)
        ax.set_title(f"{prop_name} across {mode}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        lay.addWidget(canvas)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec_()

    def _get_trend_elements(self, mode):
        """Return list of element tuples for a given period or group."""
        period_ranges = {
            "Period 1": (1, 2), "Period 2": (3, 10), "Period 3": (11, 18),
            "Period 4": (19, 36), "Period 5": (37, 54), "Period 6": (55, 86),
            "Period 7": (87, 118),
        }
        group_members = {
            "Group 1":  [1, 3, 11, 19, 37, 55, 87],
            "Group 2":  [4, 12, 20, 38, 56, 88],
            "Group 13": [5, 13, 31, 49, 81, 113],
            "Group 14": [6, 14, 32, 50, 82, 114],
            "Group 15": [7, 15, 33, 51, 83, 115],
            "Group 16": [8, 16, 34, 52, 84, 116],
            "Group 17": [9, 17, 35, 53, 85, 117],
            "Group 18": [2, 10, 18, 36, 54, 86, 118],
        }
        if mode in period_ranges:
            lo, hi = period_ranges[mode]
            return [_get_element(z) for z in range(lo, hi + 1)]
        elif mode in group_members:
            return [_get_element(z) for z in group_members[mode]]
        return []

    # ── Electron Shell Diagram ─────────────────────────────────────────
    _ORBITAL_ORDER = [
        "1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p",
        "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p"
    ]
    _ORBITAL_MAX = {"s": 2, "p": 6, "d": 10, "f": 14}

    def _electron_shell_dialog(self):
        """Show a visual orbital filling diagram for a selected element."""
        if not self._selected:
            QMessageBox.information(self, "No Selection", "Click an element first.")
            return

        z = self._selected[-1]
        elem = _get_element(z)
        d = _elem_dict(elem)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Electron Shell Diagram - {d['name']} (Z={z})")
        dlg.setMinimumSize(700, 500)
        lay = QVBoxLayout(dlg)

        lay.addWidget(QLabel(f"<h2>{d['name']} ({d['symbol']}), Z = {z}</h2>"
                              f"<p>Config: {d['electron_config']}</p>"))

        fig = Figure(figsize=(9, 5), tight_layout=True)
        style_figure(fig)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Fill orbitals
        remaining = z
        orbital_data = []
        for orb in self._ORBITAL_ORDER:
            if remaining <= 0:
                break
            n = int(orb[0])
            l_type = orb[1]
            max_e = self._ORBITAL_MAX[l_type]
            electrons = min(remaining, max_e)
            orbital_data.append((orb, electrons, max_e))
            remaining -= electrons

        # Draw orbital boxes
        y = 0
        colors = {"s": "#4a90d9", "p": "#e74c3c", "d": "#2ca02c", "f": "#ff7f0e"}
        for orb, electrons, max_e in orbital_data:
            l_type = orb[1]
            color = colors.get(l_type, "#888")
            for i in range(max_e):
                x = i * 1.2
                filled = i < electrons
                rect = mpatches.FancyBboxPatch(
                    (x, y), 0.9, 0.7,
                    boxstyle="round,pad=0.05",
                    facecolor=color if filled else "#f0f0f0",
                    edgecolor="#333",
                    alpha=0.8 if filled else 0.3)
                ax.add_patch(rect)
                if filled:
                    ax.annotate("\u2191", (x + 0.3, y + 0.2), fontsize=10,
                                ha="center", color="white" if filled else "#ccc")
            ax.text(-0.8, y + 0.35, orb, fontsize=10, fontweight="bold",
                    ha="right", va="center")
            ax.text(max_e * 1.2 + 0.2, y + 0.35,
                    f"{electrons}/{max_e}", fontsize=9, va="center", color="#666")
            y += 1.1

        ax.set_xlim(-2, 18)
        ax.set_ylim(-0.5, y + 0.5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Orbital Filling: {d['symbol']} (Z={z})", fontsize=12)

        for l_type, color in colors.items():
            ax.plot([], [], 's', color=color, ms=10, label=f"{l_type} orbital")
        ax.legend(loc="lower right", fontsize=8)

        lay.addWidget(canvas)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec_()

    # ── Isotope Browser ────────────────────────────────────────────────
    _ISOTOPE_DATA = {
        1: [(1, 99.9855, "stable"), (2, 0.0145, "stable"), (3, 0, "12.32 yr")],
        2: [(3, 0.000137, "stable"), (4, 99.999863, "stable")],
        6: [(12, 98.93, "stable"), (13, 1.07, "stable"), (14, 0, "5730 yr")],
        7: [(14, 99.636, "stable"), (15, 0.364, "stable")],
        8: [(16, 99.757, "stable"), (17, 0.038, "stable"), (18, 0.205, "stable")],
        11: [(23, 100, "stable"), (22, 0, "2.605 yr"), (24, 0, "14.96 hr")],
        12: [(24, 78.99, "stable"), (25, 10.0, "stable"), (26, 11.01, "stable")],
        14: [(28, 92.223, "stable"), (29, 4.685, "stable"), (30, 3.092, "stable")],
        15: [(31, 100, "stable"), (32, 0, "14.28 d"), (33, 0, "25.3 d")],
        16: [(32, 94.99, "stable"), (33, 0.75, "stable"), (34, 4.25, "stable"), (36, 0.01, "stable")],
        17: [(35, 75.76, "stable"), (37, 24.24, "stable"), (36, 0, "3.01e5 yr")],
        19: [(39, 93.258, "stable"), (40, 0.012, "1.25e9 yr"), (41, 6.73, "stable")],
        20: [(40, 96.941, "stable"), (42, 0.647, "stable"), (44, 2.086, "stable"), (48, 0.187, "stable")],
        26: [(54, 5.845, "stable"), (56, 91.754, "stable"), (57, 2.119, "stable"), (58, 0.282, "stable")],
        29: [(63, 69.17, "stable"), (65, 30.83, "stable")],
        47: [(107, 51.839, "stable"), (109, 48.161, "stable")],
        53: [(127, 100, "stable"), (129, 0, "1.57e7 yr"), (131, 0, "8.02 d")],
        79: [(197, 100, "stable"), (198, 0, "2.695 d")],
        82: [(204, 1.4, "stable"), (206, 24.1, "stable"), (207, 22.1, "stable"), (208, 52.4, "stable")],
        92: [(234, 0.0054, "2.46e5 yr"), (235, 0.7204, "7.04e8 yr"), (238, 99.2742, "4.47e9 yr")],
    }

    def _isotope_browser(self):
        """Show known isotopes with abundance and half-life for selected element."""
        if not self._selected:
            QMessageBox.information(self, "No Selection", "Click an element first.")
            return

        z = self._selected[-1]
        elem = _get_element(z)
        d = _elem_dict(elem)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Isotope Browser - {d['name']}")
        dlg.setMinimumSize(500, 350)
        lay = QVBoxLayout(dlg)

        lay.addWidget(QLabel(f"<h2>Isotopes of {d['name']} (Z={z})</h2>"))

        isotopes = self._ISOTOPE_DATA.get(z, None)

        if isotopes is None:
            mass_num = int(round(d["mass"]))
            isotopes = [(mass_num, 100.0, "stable")]
            lay.addWidget(QLabel("<i>Detailed isotope data not available for this element. "
                                 "Showing primary isotope only.</i>"))

        table = QTableWidget(len(isotopes), 4)
        table.setHorizontalHeaderLabels(["Isotope", "Mass Number", "Abundance (%)", "Half-Life"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, (mass_num, abundance, half_life) in enumerate(isotopes):
            table.setItem(i, 0, QTableWidgetItem(f"{d['symbol']}-{mass_num}"))
            table.setItem(i, 1, QTableWidgetItem(str(mass_num)))
            table.setItem(i, 2, QTableWidgetItem(f"{abundance:.4f}" if abundance > 0 else "trace"))
            table.setItem(i, 3, QTableWidgetItem(str(half_life)))

        lay.addWidget(table)

        # Bar chart of abundances
        stable = [(iso[0], iso[1]) for iso in isotopes if iso[1] > 0]
        if stable:
            fig = Figure(figsize=(6, 3), tight_layout=True)
            style_figure(fig)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            labels = [f"{d['symbol']}-{m}" for m, _ in stable]
            values = [a for _, a in stable]
            ax.bar(labels, values, color="#2196F3", edgecolor="white")
            ax.set_ylabel("Abundance (%)")
            ax.set_title(f"Natural Abundance of {d['name']} Isotopes")
            for i, v in enumerate(values):
                ax.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=8)
            lay.addWidget(canvas)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec_()

    # ── Full-Table Trend Plot ──────────────────────────────────────────
    def _full_table_trend_plot(self):
        """Plot a property trend across the entire periodic table."""
        prop_map = {
            "Electronegativity": 6, "Ionization Energy": 7, "Density": 8,
            "Melting Point": 9, "Boiling Point": 10, "Atomic Mass": 3,
        }
        prop_name = self._trend_prop.currentText()
        idx = prop_map.get(prop_name, 6)
        self._emit_log(f"Full-table trend plot: {prop_name}")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Full Periodic Table Trend: {prop_name}")
        dlg.setMinimumSize(900, 600)
        lay = QVBoxLayout(dlg)

        fig = Figure(figsize=(12, 7), tight_layout=True)
        style_figure(fig)
        canvas = FigureCanvas(fig)

        # Top: scatter plot Z vs property
        ax1 = fig.add_subplot(211)
        zs, vals, colors_list, symbols = [], [], [], []
        for elem in ELEMENTS:
            v = elem[idx]
            if v is not None:
                zs.append(elem[0])
                vals.append(float(v))
                cat = elem[4]
                colors_list.append(CATEGORY_COLORS.get(cat, "#888888"))
                symbols.append(elem[1])

        ax1.scatter(zs, vals, c=colors_list, s=20, edgecolors="black", linewidths=0.3, zorder=5)
        ax1.plot(zs, vals, '-', color="#ccc", lw=0.5, zorder=1)

        if len(zs) > 10:
            step = max(1, len(zs) // 15)
            for i in range(0, len(zs), step):
                ax1.annotate(symbols[i], (zs[i], vals[i]), fontsize=6,
                             textcoords="offset points", xytext=(0, 5), ha="center")

        ax1.set_xlabel("Atomic Number (Z)")
        ax1.set_ylabel(prop_name)
        ax1.set_title(f"{prop_name} vs Atomic Number")
        ax1.grid(True, alpha=0.3)

        # Bottom: heatmap-style periodic table
        ax2 = fig.add_subplot(212)
        for elem in ELEMENTS:
            z = elem[0]
            if z not in PT_LAYOUT:
                continue
            row, col = PT_LAYOUT[z]
            v = elem[idx]
            if v is not None:
                val = float(v)
            else:
                val = None

            if val is not None and vals:
                vmin, vmax = min(vals), max(vals)
                if vmax > vmin:
                    norm_val = (val - vmin) / (vmax - vmin)
                else:
                    norm_val = 0.5
                import matplotlib.cm as cm
                color = cm.viridis(norm_val)
            else:
                color = "#dddddd"

            rect = mpatches.FancyBboxPatch((col, -row), 0.9, 0.9,
                                            boxstyle="round,pad=0.02",
                                            facecolor=color, edgecolor="#333",
                                            linewidth=0.3)
            ax2.add_patch(rect)
            ax2.text(col + 0.45, -row + 0.45, elem[1], ha="center", va="center",
                     fontsize=5, fontweight="bold")

        ax2.set_xlim(-0.5, 18.5)
        ax2.set_ylim(-10.5, 1.5)
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_title(f"{prop_name} Heatmap (viridis colormap)")

        if vals:
            import matplotlib.cm as cm
            try:
                from matplotlib.colors import Normalize
                sm = cm.ScalarMappable(cmap="viridis",
                                       norm=Normalize(vmin=min(vals), vmax=max(vals)))
                sm.set_array([])
                fig.colorbar(sm, ax=ax2, fraction=0.02, pad=0.04, label=prop_name)
            except Exception:
                pass

        lay.addWidget(canvas)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec_()
