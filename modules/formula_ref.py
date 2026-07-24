"""
formula_ref.py - Comprehensive Searchable Formula Reference
Part of the Axiom Scientific Suite

Provides a browsable, searchable encyclopedia of scientific formulas
with LaTeX rendering via matplotlib mathtext, organized by category
and subcategory with full-text search across names, descriptions,
tags, and variable definitions.
"""

import sys
import io
from collections import OrderedDict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
    QTreeWidgetItem, QLineEdit, QLabel, QScrollArea, QFrame,
    QPushButton, QApplication, QHeaderView, QTableWidget,
    QTableWidgetItem, QSizePolicy, QGroupBox, QTextEdit
)
from PyQt5.QtCore import Qt, QSize, QByteArray, QBuffer
from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# ---------------------------------------------------------------------------
# Formula database
# ---------------------------------------------------------------------------

FORMULAS = [
    # ======================================================================
    # PHYSICS - MECHANICS
    # ======================================================================
    {
        "name": "Newton's Second Law",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$F = ma$",
        "description": "The net force on an object equals its mass times its acceleration.",
        "variables": {"F": "Net force (N)", "m": "Mass (kg)", "a": "Acceleration (m/s^2)"},
        "tags": ["newton", "force", "mass", "acceleration", "dynamics"],
    },
    {
        "name": "Kinematic Equation - Velocity",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$v = v_0 + at$",
        "description": "Velocity as a function of time under constant acceleration.",
        "variables": {"v": "Final velocity", "v_0": "Initial velocity", "a": "Acceleration", "t": "Time"},
        "tags": ["kinematics", "velocity", "acceleration", "motion"],
    },
    {
        "name": "Kinematic Equation - Displacement",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$x = x_0 + v_0 t + \frac{1}{2}at^2$",
        "description": "Position as a function of time under constant acceleration.",
        "variables": {"x": "Final position", "x_0": "Initial position", "v_0": "Initial velocity", "a": "Acceleration", "t": "Time"},
        "tags": ["kinematics", "displacement", "position", "motion"],
    },
    {
        "name": "Kinematic Equation - Velocity-Displacement",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$v^2 = v_0^2 + 2a(x - x_0)$",
        "description": "Relates velocity to displacement without explicit time dependence.",
        "variables": {"v": "Final velocity", "v_0": "Initial velocity", "a": "Acceleration", "x": "Final position", "x_0": "Initial position"},
        "tags": ["kinematics", "velocity", "displacement"],
    },
    {
        "name": "Work-Energy Theorem",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$W_{net} = \Delta KE = \frac{1}{2}mv_f^2 - \frac{1}{2}mv_i^2$",
        "description": "The net work done on an object equals the change in its kinetic energy.",
        "variables": {"W_{net}": "Net work", "m": "Mass", "v_f": "Final velocity", "v_i": "Initial velocity"},
        "tags": ["work", "energy", "kinetic", "theorem"],
    },
    {
        "name": "Kinetic Energy",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$KE = \frac{1}{2}mv^2$",
        "description": "The energy an object possesses due to its motion.",
        "variables": {"KE": "Kinetic energy (J)", "m": "Mass (kg)", "v": "Velocity (m/s)"},
        "tags": ["kinetic", "energy", "motion"],
    },
    {
        "name": "Gravitational Potential Energy",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$PE = mgh$",
        "description": "Energy stored in an object due to its height in a gravitational field.",
        "variables": {"PE": "Potential energy (J)", "m": "Mass (kg)", "g": "Gravitational acceleration", "h": "Height (m)"},
        "tags": ["potential", "energy", "gravity", "height"],
    },
    {
        "name": "Momentum",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$\vec{p} = m\vec{v}$",
        "description": "Linear momentum is the product of mass and velocity.",
        "variables": {"p": "Momentum (kg m/s)", "m": "Mass (kg)", "v": "Velocity (m/s)"},
        "tags": ["momentum", "mass", "velocity", "linear"],
    },
    {
        "name": "Impulse-Momentum Theorem",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$\vec{J} = \vec{F}\Delta t = \Delta \vec{p}$",
        "description": "Impulse equals the change in momentum of an object.",
        "variables": {"J": "Impulse (N s)", "F": "Force (N)", "t": "Time interval (s)", "p": "Momentum"},
        "tags": ["impulse", "momentum", "force", "collision"],
    },
    {
        "name": "Centripetal Force",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$F_c = \frac{mv^2}{r}$",
        "description": "The inward force required to keep an object moving in a circular path.",
        "variables": {"F_c": "Centripetal force", "m": "Mass", "v": "Tangential velocity", "r": "Radius"},
        "tags": ["centripetal", "circular", "force", "radius"],
    },
    {
        "name": "Torque",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$\vec{\tau} = \vec{r} \times \vec{F}$",
        "description": "Torque is the rotational analogue of force, equal to the cross product of the position vector and force.",
        "variables": {"tau": "Torque (N m)", "r": "Position vector (m)", "F": "Force (N)"},
        "tags": ["torque", "rotation", "moment", "cross product"],
    },
    {
        "name": "Angular Momentum",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$\vec{L} = I\vec{\omega}$",
        "description": "Angular momentum of a rigid body equals moment of inertia times angular velocity.",
        "variables": {"L": "Angular momentum", "I": "Moment of inertia", "omega": "Angular velocity (rad/s)"},
        "tags": ["angular", "momentum", "rotation", "inertia"],
    },
    {
        "name": "Moment of Inertia (Point Mass)",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$I = \sum m_i r_i^2$",
        "description": "The moment of inertia is the rotational analogue of mass, summed over all point masses.",
        "variables": {"I": "Moment of inertia (kg m^2)", "m_i": "Mass of i-th particle", "r_i": "Distance from axis"},
        "tags": ["inertia", "rotation", "mass", "axis"],
    },
    {
        "name": "Hooke's Law",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$F = -kx$",
        "description": "The restoring force of a spring is proportional to its displacement from equilibrium.",
        "variables": {"F": "Spring force (N)", "k": "Spring constant (N/m)", "x": "Displacement (m)"},
        "tags": ["spring", "hooke", "elastic", "restoring"],
    },
    {
        "name": "Simple Pendulum Period",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$T = 2\pi\sqrt{\frac{L}{g}}$",
        "description": "The period of a simple pendulum for small oscillations.",
        "variables": {"T": "Period (s)", "L": "Length of pendulum (m)", "g": "Gravitational acceleration (m/s^2)"},
        "tags": ["pendulum", "period", "oscillation", "harmonic"],
    },
    {
        "name": "Damped Oscillation",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$x(t) = A e^{-\gamma t} \cos(\omega' t + \phi)$",
        "description": "Displacement of a damped harmonic oscillator where amplitude decays exponentially.",
        "variables": {"A": "Initial amplitude", "gamma": "Damping coefficient", "omega'": "Damped angular frequency", "phi": "Phase constant"},
        "tags": ["damped", "oscillation", "harmonic", "decay"],
    },
    {
        "name": "Universal Gravitation",
        "category": "Physics",
        "subcategory": "Mechanics",
        "latex": r"$F = G\frac{m_1 m_2}{r^2}$",
        "description": "Every mass attracts every other mass with a force proportional to the product of their masses and inversely proportional to the square of the distance.",
        "variables": {"F": "Gravitational force", "G": "Gravitational constant", "m_1": "Mass 1", "m_2": "Mass 2", "r": "Distance between centres"},
        "tags": ["gravity", "gravitation", "newton", "universal", "inverse square"],
    },
    # ======================================================================
    # PHYSICS - ELECTROMAGNETISM
    # ======================================================================
    {
        "name": "Coulomb's Law",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$F = k_e \frac{q_1 q_2}{r^2}$",
        "description": "The electrostatic force between two point charges is proportional to the product of the charges and inversely proportional to the square of their separation.",
        "variables": {"F": "Electrostatic force", "k_e": "Coulomb constant", "q_1": "Charge 1", "q_2": "Charge 2", "r": "Separation"},
        "tags": ["coulomb", "electric", "charge", "force", "electrostatic"],
    },
    {
        "name": "Electric Field (Point Charge)",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\vec{E} = k_e \frac{q}{r^2}\hat{r}$",
        "description": "The electric field created by a point charge at distance r.",
        "variables": {"E": "Electric field (N/C)", "k_e": "Coulomb constant", "q": "Source charge", "r": "Distance"},
        "tags": ["electric", "field", "charge", "point"],
    },
    {
        "name": "Gauss's Law",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\oint \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$",
        "description": "The total electric flux through a closed surface equals the enclosed charge divided by the permittivity of free space.",
        "variables": {"E": "Electric field", "A": "Surface area element", "Q_enc": "Enclosed charge", "epsilon_0": "Permittivity of free space"},
        "tags": ["gauss", "flux", "electric", "enclosed", "surface"],
    },
    {
        "name": "Parallel Plate Capacitance",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$C = \epsilon_0 \frac{A}{d}$",
        "description": "Capacitance of a parallel plate capacitor with plate area A and separation d.",
        "variables": {"C": "Capacitance (F)", "epsilon_0": "Permittivity of free space", "A": "Plate area", "d": "Plate separation"},
        "tags": ["capacitor", "parallel plate", "capacitance"],
    },
    {
        "name": "Spherical Capacitance",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$C = 4\pi\epsilon_0 \frac{ab}{b-a}$",
        "description": "Capacitance of a spherical capacitor with inner radius a and outer radius b.",
        "variables": {"C": "Capacitance", "epsilon_0": "Permittivity", "a": "Inner radius", "b": "Outer radius"},
        "tags": ["capacitor", "spherical", "capacitance"],
    },
    {
        "name": "Cylindrical Capacitance",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$C = \frac{2\pi\epsilon_0 L}{\ln(b/a)}$",
        "description": "Capacitance per unit length of a cylindrical capacitor.",
        "variables": {"C": "Capacitance", "epsilon_0": "Permittivity", "L": "Length", "a": "Inner radius", "b": "Outer radius"},
        "tags": ["capacitor", "cylindrical", "capacitance"],
    },
    {
        "name": "Ohm's Law",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$V = IR$",
        "description": "The voltage across a resistor is proportional to the current through it.",
        "variables": {"V": "Voltage (V)", "I": "Current (A)", "R": "Resistance (Ohm)"},
        "tags": ["ohm", "voltage", "current", "resistance"],
    },
    {
        "name": "Electrical Power",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$P = IV = I^2 R = \frac{V^2}{R}$",
        "description": "Power dissipated by a resistive circuit element.",
        "variables": {"P": "Power (W)", "I": "Current (A)", "V": "Voltage (V)", "R": "Resistance (Ohm)"},
        "tags": ["power", "current", "voltage", "dissipation"],
    },
    {
        "name": "Kirchhoff's Voltage Law",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\sum_{loop} V_i = 0$",
        "description": "The sum of all voltage drops around any closed loop in a circuit equals zero.",
        "variables": {"V_i": "Voltage drop across element i"},
        "tags": ["kirchhoff", "voltage", "loop", "circuit"],
    },
    {
        "name": "Kirchhoff's Current Law",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\sum_{node} I_i = 0$",
        "description": "The sum of all currents entering and leaving any node in a circuit equals zero.",
        "variables": {"I_i": "Current in branch i"},
        "tags": ["kirchhoff", "current", "node", "junction", "circuit"],
    },
    {
        "name": "Biot-Savart Law",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$d\vec{B} = \frac{\mu_0}{4\pi}\frac{I\, d\vec{l} \times \hat{r}}{r^2}$",
        "description": "Gives the magnetic field contribution from an infinitesimal current element.",
        "variables": {"B": "Magnetic field", "mu_0": "Permeability of free space", "I": "Current", "dl": "Current element", "r": "Distance"},
        "tags": ["biot", "savart", "magnetic", "field", "current"],
    },
    {
        "name": "Ampere's Law",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\oint \vec{B} \cdot d\vec{l} = \mu_0 I_{enc}$",
        "description": "The line integral of the magnetic field around a closed loop equals mu_0 times the enclosed current.",
        "variables": {"B": "Magnetic field", "l": "Path element", "mu_0": "Permeability", "I_enc": "Enclosed current"},
        "tags": ["ampere", "magnetic", "current", "loop"],
    },
    {
        "name": "Faraday's Law of Induction",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\mathcal{E} = -\frac{d\Phi_B}{dt}$",
        "description": "The induced EMF in a loop equals the negative rate of change of magnetic flux through the loop (Lenz's law included).",
        "variables": {"E": "Electromotive force (V)", "Phi_B": "Magnetic flux (Wb)", "t": "Time (s)"},
        "tags": ["faraday", "induction", "emf", "flux", "lenz"],
    },
    {
        "name": "Inductance (Solenoid)",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$L = \mu_0 n^2 A l$",
        "description": "Self-inductance of a solenoid with n turns per unit length, cross-sectional area A, and length l.",
        "variables": {"L": "Inductance (H)", "mu_0": "Permeability", "n": "Turns per unit length", "A": "Cross-sectional area", "l": "Length"},
        "tags": ["inductance", "solenoid", "coil", "magnetic"],
    },
    {
        "name": "LC Circuit Resonant Frequency",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\omega_0 = \frac{1}{\sqrt{LC}}$",
        "description": "The natural resonant angular frequency of an LC circuit.",
        "variables": {"omega_0": "Resonant frequency", "L": "Inductance", "C": "Capacitance"},
        "tags": ["LC", "resonance", "frequency", "oscillation"],
    },
    {
        "name": "RLC Impedance",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$Z = \sqrt{R^2 + (\omega L - \frac{1}{\omega C})^2}$",
        "description": "The impedance magnitude of a series RLC circuit at angular frequency omega.",
        "variables": {"Z": "Impedance (Ohm)", "R": "Resistance", "L": "Inductance", "C": "Capacitance", "omega": "Angular frequency"},
        "tags": ["RLC", "impedance", "circuit", "AC"],
    },
    {
        "name": "Maxwell's Equations (Differential Form)",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": (
            r"$\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0},\quad "
            r"\nabla \cdot \vec{B} = 0$"
            "\n"
            r"$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t},\quad "
            r"\nabla \times \vec{B} = \mu_0 \vec{J} + \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}$"
        ),
        "description": "The four Maxwell equations in differential form unify electricity and magnetism.",
        "variables": {"E": "Electric field", "B": "Magnetic field", "rho": "Charge density", "J": "Current density", "epsilon_0": "Permittivity", "mu_0": "Permeability"},
        "tags": ["maxwell", "differential", "divergence", "curl", "electromagnetic"],
    },
    {
        "name": "Lorentz Force",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\vec{F} = q(\vec{E} + \vec{v} \times \vec{B})$",
        "description": "The total electromagnetic force on a charged particle moving in electric and magnetic fields.",
        "variables": {"F": "Force", "q": "Charge", "E": "Electric field", "v": "Velocity", "B": "Magnetic field"},
        "tags": ["lorentz", "force", "electric", "magnetic", "charge"],
    },
    {
        "name": "Poynting Vector",
        "category": "Physics",
        "subcategory": "Electromagnetism",
        "latex": r"$\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$",
        "description": "The Poynting vector describes the directional energy flux of an electromagnetic field.",
        "variables": {"S": "Energy flux (W/m^2)", "E": "Electric field", "B": "Magnetic field", "mu_0": "Permeability"},
        "tags": ["poynting", "energy", "flux", "electromagnetic", "radiation"],
    },
    # ======================================================================
    # PHYSICS - THERMODYNAMICS
    # ======================================================================
    {
        "name": "Ideal Gas Law",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$PV = nRT$",
        "description": "Relates pressure, volume, temperature, and amount of an ideal gas.",
        "variables": {"P": "Pressure (Pa)", "V": "Volume (m^3)", "n": "Amount of substance (mol)", "R": "Gas constant", "T": "Temperature (K)"},
        "tags": ["ideal", "gas", "pressure", "volume", "temperature"],
    },
    {
        "name": "First Law of Thermodynamics",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$\Delta U = Q - W$",
        "description": "The change in internal energy of a system equals heat added minus work done by the system.",
        "variables": {"U": "Internal energy", "Q": "Heat added", "W": "Work done by system"},
        "tags": ["first law", "internal energy", "heat", "work"],
    },
    {
        "name": "Entropy Change",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$\Delta S = \int \frac{dQ_{rev}}{T}$",
        "description": "Entropy change for a reversible process is the integral of heat divided by temperature.",
        "variables": {"S": "Entropy (J/K)", "Q_rev": "Reversible heat transfer", "T": "Temperature (K)"},
        "tags": ["entropy", "reversible", "second law", "heat"],
    },
    {
        "name": "Second Law (Clausius Inequality)",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$\oint \frac{dQ}{T} \leq 0$",
        "description": "For any cyclic process, the integral of dQ/T is less than or equal to zero, with equality for reversible cycles.",
        "variables": {"Q": "Heat transfer", "T": "Temperature"},
        "tags": ["second law", "clausius", "inequality", "irreversible"],
    },
    {
        "name": "Carnot Efficiency",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$\eta = 1 - \frac{T_C}{T_H}$",
        "description": "The maximum possible efficiency of a heat engine operating between two temperatures.",
        "variables": {"eta": "Efficiency", "T_C": "Cold reservoir temperature (K)", "T_H": "Hot reservoir temperature (K)"},
        "tags": ["carnot", "efficiency", "heat engine", "reversible"],
    },
    {
        "name": "Heat Capacity",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$Q = mc\Delta T$",
        "description": "Heat transferred to a substance relates to its mass, specific heat, and temperature change.",
        "variables": {"Q": "Heat (J)", "m": "Mass (kg)", "c": "Specific heat capacity", "T": "Temperature change (K)"},
        "tags": ["heat", "capacity", "specific heat", "temperature"],
    },
    {
        "name": "Stefan-Boltzmann Law",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$P = \sigma A T^4$",
        "description": "The total power radiated by a black body is proportional to the fourth power of its temperature.",
        "variables": {"P": "Radiated power (W)", "sigma": "Stefan-Boltzmann constant", "A": "Surface area", "T": "Temperature (K)"},
        "tags": ["stefan", "boltzmann", "radiation", "blackbody", "thermal"],
    },
    {
        "name": "Wien's Displacement Law",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$\lambda_{max} = \frac{b}{T}$",
        "description": "The peak wavelength of blackbody radiation is inversely proportional to temperature.",
        "variables": {"lambda_max": "Peak wavelength (m)", "b": "Wien's displacement constant", "T": "Temperature (K)"},
        "tags": ["wien", "displacement", "blackbody", "wavelength", "peak"],
    },
    {
        "name": "Boltzmann Distribution",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$P(E) \propto e^{-E / k_B T}$",
        "description": "The probability of a system being in a state of energy E at temperature T.",
        "variables": {"P(E)": "Probability of energy E", "E": "Energy", "k_B": "Boltzmann constant", "T": "Temperature"},
        "tags": ["boltzmann", "distribution", "probability", "thermal", "statistical"],
    },
    {
        "name": "Partition Function",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$Z = \sum_i e^{-E_i / k_B T}$",
        "description": "The partition function sums over all microstates and encodes the statistical properties of a system.",
        "variables": {"Z": "Partition function", "E_i": "Energy of state i", "k_B": "Boltzmann constant", "T": "Temperature"},
        "tags": ["partition", "function", "statistical", "mechanics", "microstate"],
    },
    {
        "name": "Clausius-Clapeyron Equation",
        "category": "Physics",
        "subcategory": "Thermodynamics",
        "latex": r"$\frac{dP}{dT} = \frac{L}{T\Delta V}$",
        "description": "Describes the slope of the phase boundary in a P-T diagram relating latent heat and volume change.",
        "variables": {"P": "Pressure", "T": "Temperature", "L": "Latent heat", "V": "Volume change"},
        "tags": ["clausius", "clapeyron", "phase", "transition", "latent heat"],
    },
    # ======================================================================
    # PHYSICS - QUANTUM MECHANICS
    # ======================================================================
    {
        "name": "Time-Dependent Schrodinger Equation",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi$",
        "description": "The fundamental equation governing the time evolution of a quantum state.",
        "variables": {"hbar": "Reduced Planck constant", "Psi": "Wave function", "H": "Hamiltonian operator", "t": "Time"},
        "tags": ["schrodinger", "quantum", "wave function", "time dependent", "hamiltonian"],
    },
    {
        "name": "Time-Independent Schrodinger Equation",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$\hat{H}\psi = E\psi$",
        "description": "The eigenvalue equation for stationary states of a quantum system.",
        "variables": {"H": "Hamiltonian operator", "psi": "Eigenstate", "E": "Energy eigenvalue"},
        "tags": ["schrodinger", "quantum", "eigenvalue", "stationary", "time independent"],
    },
    {
        "name": "de Broglie Wavelength",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$\lambda = \frac{h}{p}$",
        "description": "Every particle has an associated wavelength inversely proportional to its momentum.",
        "variables": {"lambda": "de Broglie wavelength", "h": "Planck constant", "p": "Momentum"},
        "tags": ["de broglie", "wavelength", "wave-particle", "duality", "momentum"],
    },
    {
        "name": "Heisenberg Uncertainty Principle",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$\Delta x \, \Delta p \geq \frac{\hbar}{2}$",
        "description": "The position and momentum of a particle cannot both be precisely known simultaneously.",
        "variables": {"Delta_x": "Position uncertainty", "Delta_p": "Momentum uncertainty", "hbar": "Reduced Planck constant"},
        "tags": ["heisenberg", "uncertainty", "position", "momentum", "limit"],
    },
    {
        "name": "Planck-Einstein Relation",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$E = h\nu = \hbar\omega$",
        "description": "The energy of a photon is proportional to its frequency.",
        "variables": {"E": "Photon energy", "h": "Planck constant", "nu": "Frequency", "hbar": "Reduced Planck constant", "omega": "Angular frequency"},
        "tags": ["planck", "einstein", "photon", "energy", "frequency"],
    },
    {
        "name": "Photoelectric Effect",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$KE_{max} = h\nu - \phi$",
        "description": "The maximum kinetic energy of ejected electrons equals the photon energy minus the work function.",
        "variables": {"KE_max": "Max kinetic energy", "h": "Planck constant", "nu": "Frequency", "phi": "Work function"},
        "tags": ["photoelectric", "effect", "electron", "photon", "work function"],
    },
    {
        "name": "Bohr Model Energy Levels",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$E_n = -\frac{13.6 \text{ eV}}{n^2}$",
        "description": "The quantized energy levels of hydrogen in the Bohr model.",
        "variables": {"E_n": "Energy of level n", "n": "Principal quantum number"},
        "tags": ["bohr", "hydrogen", "energy", "level", "quantized"],
    },
    {
        "name": "Hydrogen Energy Levels (General)",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$E_n = -\frac{m_e e^4}{2\hbar^2}\frac{1}{n^2}$",
        "description": "Energy levels of hydrogen from the Schrodinger equation solution.",
        "variables": {"E_n": "Energy of level n", "m_e": "Electron mass", "e": "Electron charge", "hbar": "Reduced Planck constant", "n": "Principal quantum number"},
        "tags": ["hydrogen", "energy", "level", "quantum", "electron"],
    },
    {
        "name": "Tunneling Probability (Rectangular Barrier)",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$T \approx e^{-2\kappa L}$",
        "description": "Approximate transmission probability through a rectangular potential barrier of width L.",
        "variables": {"T": "Transmission coefficient", "kappa": "Decay constant (sqrt(2m(V-E))/hbar)", "L": "Barrier width"},
        "tags": ["tunneling", "barrier", "transmission", "quantum", "probability"],
    },
    {
        "name": "Canonical Commutation Relation",
        "category": "Physics",
        "subcategory": "Quantum Mechanics",
        "latex": r"$[\hat{x}, \hat{p}] = i\hbar$",
        "description": "The fundamental commutation relation between the position and momentum operators.",
        "variables": {"x": "Position operator", "p": "Momentum operator", "hbar": "Reduced Planck constant"},
        "tags": ["commutation", "operator", "position", "momentum", "canonical"],
    },
    # ======================================================================
    # PHYSICS - RELATIVITY
    # ======================================================================
    {
        "name": "Lorentz Factor",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}$",
        "description": "The Lorentz factor appears in all relativistic transformations.",
        "variables": {"gamma": "Lorentz factor", "v": "Relative velocity", "c": "Speed of light"},
        "tags": ["lorentz", "factor", "relativistic", "gamma"],
    },
    {
        "name": "Time Dilation",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$\Delta t = \gamma \Delta t_0$",
        "description": "A moving clock ticks slower relative to a stationary observer.",
        "variables": {"Delta_t": "Dilated time", "Delta_t_0": "Proper time", "gamma": "Lorentz factor"},
        "tags": ["time", "dilation", "relativistic", "clock", "proper time"],
    },
    {
        "name": "Length Contraction",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$L = \frac{L_0}{\gamma}$",
        "description": "A moving object is measured to be shorter along the direction of motion.",
        "variables": {"L": "Contracted length", "L_0": "Proper length", "gamma": "Lorentz factor"},
        "tags": ["length", "contraction", "relativistic", "proper length"],
    },
    {
        "name": "Mass-Energy Equivalence",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$E = mc^2$",
        "description": "The rest energy of an object is equal to its mass times the speed of light squared.",
        "variables": {"E": "Rest energy", "m": "Rest mass", "c": "Speed of light"},
        "tags": ["einstein", "mass", "energy", "equivalence", "E=mc2"],
    },
    {
        "name": "Relativistic Momentum",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$p = \gamma m v$",
        "description": "Momentum of an object at relativistic speeds.",
        "variables": {"p": "Relativistic momentum", "gamma": "Lorentz factor", "m": "Rest mass", "v": "Velocity"},
        "tags": ["momentum", "relativistic", "mass", "velocity"],
    },
    {
        "name": "Relativistic Energy-Momentum Relation",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$E^2 = (pc)^2 + (mc^2)^2$",
        "description": "Relates total energy, momentum, and rest mass of a particle.",
        "variables": {"E": "Total energy", "p": "Momentum", "m": "Rest mass", "c": "Speed of light"},
        "tags": ["energy", "momentum", "relativistic", "invariant"],
    },
    {
        "name": "Spacetime Interval",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2$",
        "description": "The invariant spacetime interval between two events in special relativity.",
        "variables": {"ds": "Spacetime interval", "c": "Speed of light", "t": "Time", "x,y,z": "Spatial coordinates"},
        "tags": ["spacetime", "interval", "invariant", "metric", "minkowski"],
    },
    {
        "name": "Gravitational Redshift",
        "category": "Physics",
        "subcategory": "Relativity",
        "latex": r"$\frac{\Delta\nu}{\nu} = \frac{GM}{rc^2}$",
        "description": "Light escaping a gravitational field is redshifted proportionally to the gravitational potential.",
        "variables": {"nu": "Frequency", "G": "Gravitational constant", "M": "Mass", "r": "Radial distance", "c": "Speed of light"},
        "tags": ["gravitational", "redshift", "general relativity", "frequency"],
    },
    # ======================================================================
    # CHEMISTRY
    # ======================================================================
    {
        "name": "Ideal Gas Law (Chemistry)",
        "category": "Chemistry",
        "subcategory": "General",
        "latex": r"$PV = nRT$",
        "description": "Equation of state for an ideal gas relating pressure, volume, moles, and temperature.",
        "variables": {"P": "Pressure", "V": "Volume", "n": "Moles", "R": "Gas constant (8.314 J/mol K)", "T": "Temperature"},
        "tags": ["ideal", "gas", "pressure", "volume", "temperature", "moles"],
    },
    {
        "name": "Van der Waals Equation",
        "category": "Chemistry",
        "subcategory": "General",
        "latex": r"$\left(P + \frac{an^2}{V^2}\right)(V - nb) = nRT$",
        "description": "A more realistic equation of state that accounts for intermolecular forces and finite molecular size.",
        "variables": {"P": "Pressure", "V": "Volume", "n": "Moles", "a": "Attraction parameter", "b": "Volume parameter", "R": "Gas constant", "T": "Temperature"},
        "tags": ["van der waals", "real gas", "intermolecular", "equation of state"],
    },
    {
        "name": "Arrhenius Equation",
        "category": "Chemistry",
        "subcategory": "Kinetics",
        "latex": r"$k = A e^{-E_a / RT}$",
        "description": "The rate constant of a reaction depends exponentially on the activation energy and temperature.",
        "variables": {"k": "Rate constant", "A": "Pre-exponential factor", "E_a": "Activation energy", "R": "Gas constant", "T": "Temperature"},
        "tags": ["arrhenius", "rate", "activation", "kinetics", "temperature"],
    },
    {
        "name": "Nernst Equation",
        "category": "Chemistry",
        "subcategory": "Electrochemistry",
        "latex": r"$E = E^0 - \frac{RT}{nF}\ln Q$",
        "description": "Relates the cell potential to the standard potential and the reaction quotient.",
        "variables": {"E": "Cell potential", "E_0": "Standard potential", "R": "Gas constant", "T": "Temperature", "n": "Electrons transferred", "F": "Faraday constant", "Q": "Reaction quotient"},
        "tags": ["nernst", "electrochemistry", "cell", "potential", "redox"],
    },
    {
        "name": "Henderson-Hasselbalch Equation",
        "category": "Chemistry",
        "subcategory": "Acid-Base",
        "latex": r"$pH = pK_a + \log\frac{[A^-]}{[HA]}$",
        "description": "Relates pH of a buffer solution to the pKa and the ratio of conjugate base to acid concentrations.",
        "variables": {"pH": "Acidity measure", "pK_a": "Acid dissociation constant", "[A^-]": "Conjugate base concentration", "[HA]": "Acid concentration"},
        "tags": ["henderson", "hasselbalch", "buffer", "pH", "acid", "base"],
    },
    {
        "name": "First-Order Rate Law",
        "category": "Chemistry",
        "subcategory": "Kinetics",
        "latex": r"$[A] = [A]_0 e^{-kt}$",
        "description": "Concentration of a reactant decays exponentially in a first-order reaction.",
        "variables": {"[A]": "Concentration at time t", "[A]_0": "Initial concentration", "k": "Rate constant", "t": "Time"},
        "tags": ["rate", "law", "first order", "kinetics", "decay"],
    },
    {
        "name": "Second-Order Rate Law",
        "category": "Chemistry",
        "subcategory": "Kinetics",
        "latex": r"$\frac{1}{[A]} = \frac{1}{[A]_0} + kt$",
        "description": "Integrated rate law for a second-order reaction in a single reactant.",
        "variables": {"[A]": "Concentration", "[A]_0": "Initial concentration", "k": "Rate constant", "t": "Time"},
        "tags": ["rate", "law", "second order", "kinetics"],
    },
    {
        "name": "Beer-Lambert Law",
        "category": "Chemistry",
        "subcategory": "Spectroscopy",
        "latex": r"$A = \epsilon l c$",
        "description": "Absorbance of light by a solution is proportional to concentration and path length.",
        "variables": {"A": "Absorbance", "epsilon": "Molar absorptivity", "l": "Path length (cm)", "c": "Concentration (mol/L)"},
        "tags": ["beer", "lambert", "absorbance", "spectroscopy", "concentration"],
    },
    {
        "name": "Gibbs Free Energy",
        "category": "Chemistry",
        "subcategory": "Thermodynamics",
        "latex": r"$\Delta G = \Delta H - T\Delta S$",
        "description": "The Gibbs free energy change determines the spontaneity of a process at constant T and P.",
        "variables": {"G": "Gibbs free energy", "H": "Enthalpy", "T": "Temperature", "S": "Entropy"},
        "tags": ["gibbs", "free energy", "enthalpy", "entropy", "spontaneous"],
    },
    {
        "name": "Equilibrium Constant",
        "category": "Chemistry",
        "subcategory": "Equilibrium",
        "latex": r"$\Delta G^0 = -RT \ln K$",
        "description": "Relates the standard Gibbs free energy change to the equilibrium constant.",
        "variables": {"G_0": "Standard Gibbs energy", "R": "Gas constant", "T": "Temperature", "K": "Equilibrium constant"},
        "tags": ["equilibrium", "constant", "gibbs", "thermodynamics"],
    },
    {
        "name": "Raoult's Law",
        "category": "Chemistry",
        "subcategory": "Solutions",
        "latex": r"$P_A = x_A P_A^*$",
        "description": "The partial vapor pressure of a component equals its mole fraction times its pure vapor pressure.",
        "variables": {"P_A": "Partial pressure", "x_A": "Mole fraction", "P_A^*": "Pure vapor pressure"},
        "tags": ["raoult", "vapor", "pressure", "solution", "mole fraction"],
    },
    # ======================================================================
    # MATHEMATICS
    # ======================================================================
    {
        "name": "Quadratic Formula",
        "category": "Mathematics",
        "subcategory": "Algebra",
        "latex": r"$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$",
        "description": "Gives the solutions to any quadratic equation ax^2 + bx + c = 0.",
        "variables": {"x": "Solution", "a": "Quadratic coefficient", "b": "Linear coefficient", "c": "Constant term"},
        "tags": ["quadratic", "roots", "polynomial", "algebra"],
    },
    {
        "name": "Binomial Theorem",
        "category": "Mathematics",
        "subcategory": "Algebra",
        "latex": r"$(a+b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$",
        "description": "Expands powers of a binomial sum as a finite series of terms involving binomial coefficients.",
        "variables": {"a,b": "Terms", "n": "Exponent", "k": "Summation index"},
        "tags": ["binomial", "theorem", "expansion", "combinatorics"],
    },
    {
        "name": "Taylor Series",
        "category": "Mathematics",
        "subcategory": "Analysis",
        "latex": r"$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$",
        "description": "Represents a smooth function as an infinite sum of terms computed from its derivatives at a single point.",
        "variables": {"f": "Function", "a": "Expansion point", "n": "Order", "f^(n)": "n-th derivative"},
        "tags": ["taylor", "series", "expansion", "approximation", "derivative"],
    },
    {
        "name": "Euler's Formula",
        "category": "Mathematics",
        "subcategory": "Complex Analysis",
        "latex": r"$e^{i\theta} = \cos\theta + i\sin\theta$",
        "description": "Establishes the fundamental relationship between the trigonometric functions and the complex exponential.",
        "variables": {"theta": "Angle (radians)", "i": "Imaginary unit", "e": "Euler's number"},
        "tags": ["euler", "complex", "exponential", "trigonometric", "identity"],
    },
    {
        "name": "Fourier Transform",
        "category": "Mathematics",
        "subcategory": "Analysis",
        "latex": r"$\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t)\, e^{-i\omega t}\, dt$",
        "description": "Decomposes a function of time into its constituent frequencies.",
        "variables": {"f(t)": "Time-domain function", "f_hat": "Frequency-domain function", "omega": "Angular frequency"},
        "tags": ["fourier", "transform", "frequency", "spectral", "integral"],
    },
    {
        "name": "Inverse Fourier Transform",
        "category": "Mathematics",
        "subcategory": "Analysis",
        "latex": r"$f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \hat{f}(\omega)\, e^{i\omega t}\, d\omega$",
        "description": "Recovers the time-domain function from its frequency-domain representation.",
        "variables": {"f(t)": "Time-domain function", "f_hat": "Frequency-domain", "omega": "Angular frequency"},
        "tags": ["fourier", "inverse", "transform", "frequency"],
    },
    {
        "name": "Laplace Transform",
        "category": "Mathematics",
        "subcategory": "Analysis",
        "latex": r"$F(s) = \int_0^{\infty} f(t)\, e^{-st}\, dt$",
        "description": "Transforms a function of time into a function of complex frequency, widely used in control theory.",
        "variables": {"F(s)": "Laplace transform", "f(t)": "Original function", "s": "Complex frequency"},
        "tags": ["laplace", "transform", "integral", "control", "differential"],
    },
    {
        "name": "Integration by Parts",
        "category": "Mathematics",
        "subcategory": "Calculus",
        "latex": r"$\int u\, dv = uv - \int v\, du$",
        "description": "A technique for evaluating integrals derived from the product rule for derivatives.",
        "variables": {"u": "First function", "v": "Second function"},
        "tags": ["integration", "parts", "calculus", "technique"],
    },
    {
        "name": "Chain Rule",
        "category": "Mathematics",
        "subcategory": "Calculus",
        "latex": r"$\frac{d}{dx}f(g(x)) = f'(g(x))\cdot g'(x)$",
        "description": "The derivative of a composition of functions equals the product of their individual derivatives.",
        "variables": {"f": "Outer function", "g": "Inner function", "x": "Variable"},
        "tags": ["chain", "rule", "derivative", "composition", "calculus"],
    },
    {
        "name": "Divergence Theorem",
        "category": "Mathematics",
        "subcategory": "Vector Calculus",
        "latex": r"$\oint_S \vec{F}\cdot d\vec{A} = \int_V (\nabla \cdot \vec{F})\, dV$",
        "description": "Relates the flux of a vector field through a closed surface to the divergence within the enclosed volume.",
        "variables": {"F": "Vector field", "S": "Closed surface", "V": "Enclosed volume"},
        "tags": ["divergence", "theorem", "gauss", "flux", "volume", "surface"],
    },
    {
        "name": "Stokes' Theorem",
        "category": "Mathematics",
        "subcategory": "Vector Calculus",
        "latex": r"$\oint_C \vec{F}\cdot d\vec{r} = \int_S (\nabla \times \vec{F})\cdot d\vec{A}$",
        "description": "Relates a line integral around a closed curve to a surface integral of the curl over any surface bounded by that curve.",
        "variables": {"F": "Vector field", "C": "Closed curve", "S": "Bounded surface"},
        "tags": ["stokes", "theorem", "curl", "line integral", "surface integral"],
    },
    {
        "name": "Green's Theorem",
        "category": "Mathematics",
        "subcategory": "Vector Calculus",
        "latex": r"$\oint_C (P\,dx + Q\,dy) = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$",
        "description": "Relates a line integral around a simple closed curve to a double integral over the plane region it encloses.",
        "variables": {"P,Q": "Component functions", "C": "Closed curve", "D": "Enclosed region"},
        "tags": ["green", "theorem", "line integral", "double integral", "plane"],
    },
    {
        "name": "Cauchy-Schwarz Inequality",
        "category": "Mathematics",
        "subcategory": "Linear Algebra",
        "latex": r"$|\langle \vec{u},\vec{v}\rangle|^2 \leq \langle \vec{u},\vec{u}\rangle \cdot \langle \vec{v},\vec{v}\rangle$",
        "description": "The squared inner product of two vectors is at most the product of their squared norms.",
        "variables": {"u,v": "Vectors in inner product space"},
        "tags": ["cauchy", "schwarz", "inequality", "inner product", "norm"],
    },
    {
        "name": "Matrix Determinant (2x2)",
        "category": "Mathematics",
        "subcategory": "Linear Algebra",
        "latex": r"$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$",
        "description": "The determinant of a 2x2 matrix, giving the signed area scale factor of the linear transformation.",
        "variables": {"a,b,c,d": "Matrix entries"},
        "tags": ["determinant", "matrix", "linear algebra", "2x2"],
    },
    {
        "name": "Eigenvalue Equation",
        "category": "Mathematics",
        "subcategory": "Linear Algebra",
        "latex": r"$A\vec{v} = \lambda\vec{v}$",
        "description": "An eigenvector v of matrix A is scaled by eigenvalue lambda under the linear transformation.",
        "variables": {"A": "Square matrix", "v": "Eigenvector", "lambda": "Eigenvalue"},
        "tags": ["eigenvalue", "eigenvector", "matrix", "linear algebra", "diagonalization"],
    },
    # ======================================================================
    # STATISTICS
    # ======================================================================
    {
        "name": "Arithmetic Mean",
        "category": "Statistics",
        "subcategory": "Descriptive",
        "latex": r"$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$",
        "description": "The average value of a dataset, computed as the sum of all values divided by the count.",
        "variables": {"x_bar": "Mean", "n": "Number of observations", "x_i": "Individual observation"},
        "tags": ["mean", "average", "central tendency", "descriptive"],
    },
    {
        "name": "Variance",
        "category": "Statistics",
        "subcategory": "Descriptive",
        "latex": r"$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$",
        "description": "The average of the squared deviations from the mean, measuring data spread.",
        "variables": {"sigma^2": "Variance", "x_i": "Observation", "x_bar": "Mean", "n": "Count"},
        "tags": ["variance", "spread", "dispersion", "descriptive"],
    },
    {
        "name": "Standard Deviation",
        "category": "Statistics",
        "subcategory": "Descriptive",
        "latex": r"$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$",
        "description": "The square root of the variance, in the same units as the data.",
        "variables": {"sigma": "Standard deviation", "x_i": "Observation", "x_bar": "Mean", "n": "Count"},
        "tags": ["standard deviation", "spread", "dispersion"],
    },
    {
        "name": "Normal Distribution PDF",
        "category": "Statistics",
        "subcategory": "Distributions",
        "latex": r"$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$",
        "description": "The probability density function of the Gaussian (normal) distribution.",
        "variables": {"f(x)": "Probability density", "mu": "Mean", "sigma": "Standard deviation", "x": "Random variable"},
        "tags": ["normal", "gaussian", "distribution", "bell curve", "pdf"],
    },
    {
        "name": "Bayes' Theorem",
        "category": "Statistics",
        "subcategory": "Probability",
        "latex": r"$P(A|B) = \frac{P(B|A)\, P(A)}{P(B)}$",
        "description": "Gives the posterior probability of event A given evidence B by combining prior knowledge with new data.",
        "variables": {"P(A|B)": "Posterior probability", "P(B|A)": "Likelihood", "P(A)": "Prior", "P(B)": "Evidence"},
        "tags": ["bayes", "posterior", "prior", "conditional", "probability"],
    },
    {
        "name": "Chi-Square Statistic",
        "category": "Statistics",
        "subcategory": "Inference",
        "latex": r"$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$",
        "description": "Measures how observed frequencies differ from expected frequencies in categorical data.",
        "variables": {"chi^2": "Chi-square statistic", "O_i": "Observed frequency", "E_i": "Expected frequency"},
        "tags": ["chi-square", "test", "goodness of fit", "categorical"],
    },
    {
        "name": "Student's t-Statistic",
        "category": "Statistics",
        "subcategory": "Inference",
        "latex": r"$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$",
        "description": "Test statistic for comparing a sample mean to a hypothesized population mean.",
        "variables": {"t": "t-statistic", "x_bar": "Sample mean", "mu_0": "Hypothesized mean", "s": "Sample std dev", "n": "Sample size"},
        "tags": ["t-test", "student", "hypothesis", "inference", "mean"],
    },
    {
        "name": "Linear Regression (OLS)",
        "category": "Statistics",
        "subcategory": "Regression",
        "latex": r"$\hat{\beta} = (X^T X)^{-1} X^T y$",
        "description": "The ordinary least squares estimator for the coefficient vector in linear regression.",
        "variables": {"beta_hat": "Estimated coefficients", "X": "Design matrix", "y": "Response vector"},
        "tags": ["regression", "OLS", "least squares", "linear", "fit"],
    },
    # ======================================================================
    # ENGINEERING
    # ======================================================================
    {
        "name": "Normal Stress",
        "category": "Engineering",
        "subcategory": "Solid Mechanics",
        "latex": r"$\sigma = \frac{F}{A}$",
        "description": "Stress is the internal force per unit area within a material.",
        "variables": {"sigma": "Stress (Pa)", "F": "Applied force (N)", "A": "Cross-sectional area (m^2)"},
        "tags": ["stress", "force", "area", "mechanics", "materials"],
    },
    {
        "name": "Normal Strain",
        "category": "Engineering",
        "subcategory": "Solid Mechanics",
        "latex": r"$\epsilon = \frac{\Delta L}{L_0}$",
        "description": "Strain is the fractional change in length of a material under load.",
        "variables": {"epsilon": "Strain (dimensionless)", "Delta_L": "Change in length", "L_0": "Original length"},
        "tags": ["strain", "deformation", "length", "materials"],
    },
    {
        "name": "Young's Modulus (Hooke's Law for Materials)",
        "category": "Engineering",
        "subcategory": "Solid Mechanics",
        "latex": r"$\sigma = E\epsilon$",
        "description": "In the elastic region, stress is proportional to strain with Young's modulus as the proportionality constant.",
        "variables": {"sigma": "Stress", "E": "Young's modulus (Pa)", "epsilon": "Strain"},
        "tags": ["young", "modulus", "elastic", "stiffness", "hooke"],
    },
    {
        "name": "Bernoulli's Equation",
        "category": "Engineering",
        "subcategory": "Fluid Mechanics",
        "latex": r"$P + \frac{1}{2}\rho v^2 + \rho g h = \text{const}$",
        "description": "Along a streamline in steady, incompressible, inviscid flow, the sum of pressure, kinetic, and potential energy per unit volume is constant.",
        "variables": {"P": "Pressure", "rho": "Fluid density", "v": "Flow velocity", "g": "Gravitational acceleration", "h": "Elevation"},
        "tags": ["bernoulli", "fluid", "pressure", "velocity", "streamline"],
    },
    {
        "name": "Reynolds Number",
        "category": "Engineering",
        "subcategory": "Fluid Mechanics",
        "latex": r"$Re = \frac{\rho v L}{\mu}$",
        "description": "A dimensionless number predicting whether flow will be laminar or turbulent.",
        "variables": {"Re": "Reynolds number", "rho": "Density", "v": "Velocity", "L": "Characteristic length", "mu": "Dynamic viscosity"},
        "tags": ["reynolds", "laminar", "turbulent", "fluid", "dimensionless"],
    },
    {
        "name": "Navier-Stokes Equation (Incompressible)",
        "category": "Engineering",
        "subcategory": "Fluid Mechanics",
        "latex": r"$\rho\left(\frac{\partial \vec{v}}{\partial t} + \vec{v}\cdot\nabla\vec{v}\right) = -\nabla P + \mu\nabla^2\vec{v} + \rho\vec{g}$",
        "description": "Governs the motion of viscous, incompressible Newtonian fluids.",
        "variables": {"rho": "Density", "v": "Velocity field", "P": "Pressure", "mu": "Viscosity", "g": "Body force"},
        "tags": ["navier", "stokes", "fluid", "momentum", "viscous", "PDE"],
    },
    {
        "name": "Fourier's Law of Heat Conduction",
        "category": "Engineering",
        "subcategory": "Heat Transfer",
        "latex": r"$q = -k \nabla T$",
        "description": "Heat flux is proportional to the negative temperature gradient.",
        "variables": {"q": "Heat flux (W/m^2)", "k": "Thermal conductivity (W/m K)", "T": "Temperature"},
        "tags": ["fourier", "heat", "conduction", "thermal", "gradient"],
    },
    {
        "name": "Newton's Law of Cooling (Convection)",
        "category": "Engineering",
        "subcategory": "Heat Transfer",
        "latex": r"$q = h(T_s - T_\infty)$",
        "description": "Convective heat transfer rate is proportional to the temperature difference between a surface and the surrounding fluid.",
        "variables": {"q": "Heat flux", "h": "Convection coefficient", "T_s": "Surface temperature", "T_inf": "Fluid temperature"},
        "tags": ["convection", "cooling", "heat transfer", "newton"],
    },
    {
        "name": "Thermal Radiation (Stefan-Boltzmann)",
        "category": "Engineering",
        "subcategory": "Heat Transfer",
        "latex": r"$q = \epsilon\sigma T^4$",
        "description": "Radiative heat flux emitted by a surface with emissivity epsilon.",
        "variables": {"q": "Radiative flux", "epsilon": "Emissivity", "sigma": "Stefan-Boltzmann constant", "T": "Surface temperature (K)"},
        "tags": ["radiation", "stefan", "boltzmann", "emissivity", "thermal"],
    },
]


# ---------------------------------------------------------------------------
# Helper: build category tree structure
# ---------------------------------------------------------------------------

def _build_category_tree(formulas):
    """Return OrderedDict: category -> subcategory -> [formula indices]."""
    tree = OrderedDict()
    for idx, f in enumerate(formulas):
        cat = f["category"]
        sub = f["subcategory"]
        tree.setdefault(cat, OrderedDict()).setdefault(sub, []).append(idx)
    return tree


# ---------------------------------------------------------------------------
# LaTeX renderer using matplotlib mathtext
# ---------------------------------------------------------------------------

def render_latex_to_pixmap(latex_str, fontsize=18, dpi=120):
    """Render a LaTeX string to a QPixmap using matplotlib mathtext."""
    lines = [l.strip() for l in latex_str.strip().splitlines() if l.strip()]
    fig = matplotlib.figure.Figure(figsize=(8, 0.6 * max(len(lines), 1)), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")

    y_positions = []
    n = len(lines)
    for i in range(n):
        y_positions.append(1.0 - (i + 0.5) / n)

    for line, yp in zip(lines, y_positions):
        try:
            fig.text(0.5, yp, line, fontsize=fontsize,
                     ha="center", va="center",
                     fontfamily="serif")
        except Exception:
            fig.text(0.5, yp, line, fontsize=fontsize - 4,
                     ha="center", va="center")

    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()

    from PyQt5.QtGui import QImage
    qimg = QImage(bytes(buf), w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# FormulaRefWidget
# ---------------------------------------------------------------------------

class FormulaRefWidget(QWidget):
    """Searchable scientific formula reference encyclopedia."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._formulas = FORMULAS
        self._category_tree = _build_category_tree(self._formulas)
        self._current_formula_idx = None
        self._init_ui()
        self._log("FormulaRefWidget initialised with {} formulas".format(len(self._formulas)))

    # ----- public API -----

    def set_logger(self, fn):
        """Set an external logging callback: fn(message: str)."""
        self._logger = fn

    # ----- internal logging -----

    def _log(self, msg):
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    # ----- UI construction -----

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ---- LEFT PANEL: search + category tree ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        search_label = QLabel("Search formulas:")
        search_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(search_label)

        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("Type to filter (name, tags, variables)...")
        self._search_bar.setClearButtonEnabled(True)
        self._search_bar.textChanged.connect(self._on_search_changed)
        left_layout.addWidget(self._search_bar)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(True)
        self._tree.setIndentation(18)
        self._tree.itemClicked.connect(self._on_tree_item_clicked)
        left_layout.addWidget(self._tree)

        result_count_layout = QHBoxLayout()
        self._result_count_label = QLabel("")
        self._result_count_label.setStyleSheet("color: #666; font-size: 11px;")
        result_count_layout.addWidget(self._result_count_label)
        result_count_layout.addStretch()
        left_layout.addLayout(result_count_layout)

        splitter.addWidget(left_widget)

        # ---- RIGHT PANEL: formula display ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Scroll area for the entire right side
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        self._display_container = QWidget()
        self._display_layout = QVBoxLayout(self._display_container)
        self._display_layout.setContentsMargins(8, 8, 8, 8)
        self._display_layout.setSpacing(12)

        # Title
        self._title_label = QLabel("Select a formula")
        self._title_label.setWordWrap(True)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._display_layout.addWidget(self._title_label)

        # Category badge
        self._category_label = QLabel("")
        self._category_label.setStyleSheet(
            "color: #fff; background: #3b82f6; border-radius: 4px; "
            "padding: 2px 8px; font-size: 11px; font-weight: bold;"
        )
        self._category_label.setFixedHeight(22)
        self._category_label.hide()
        self._display_layout.addWidget(self._category_label)

        # Rendered equation
        eq_group = QGroupBox("Equation")
        eq_group_layout = QVBoxLayout(eq_group)
        self._equation_label = QLabel()
        self._equation_label.setAlignment(Qt.AlignCenter)
        self._equation_label.setMinimumHeight(60)
        self._equation_label.setStyleSheet("background: #fafafa; border: 1px solid #ddd; border-radius: 4px; padding: 12px;")
        eq_group_layout.addWidget(self._equation_label)
        self._display_layout.addWidget(eq_group)

        # LaTeX source + copy button
        latex_row = QHBoxLayout()
        self._latex_source = QLineEdit()
        self._latex_source.setReadOnly(True)
        self._latex_source.setPlaceholderText("LaTeX source")
        self._latex_source.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        latex_row.addWidget(self._latex_source, stretch=1)

        self._copy_btn = QPushButton("Copy LaTeX")
        self._copy_btn.setFixedWidth(100)
        self._copy_btn.clicked.connect(self._copy_latex)
        latex_row.addWidget(self._copy_btn)
        self._display_layout.addLayout(latex_row)

        # Description
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout(desc_group)
        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("font-size: 13px; line-height: 1.5;")
        desc_layout.addWidget(self._desc_label)
        self._display_layout.addWidget(desc_group)

        # Variables table
        var_group = QGroupBox("Variables")
        var_layout = QVBoxLayout(var_group)
        self._var_table = QTableWidget()
        self._var_table.setColumnCount(2)
        self._var_table.setHorizontalHeaderLabels(["Symbol", "Meaning"])
        self._var_table.horizontalHeader().setStretchLastSection(True)
        self._var_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._var_table.verticalHeader().setVisible(False)
        self._var_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._var_table.setSelectionMode(QTableWidget.NoSelection)
        self._var_table.setAlternatingRowColors(True)
        var_layout.addWidget(self._var_table)
        self._display_layout.addWidget(var_group)

        # Tags
        tags_group = QGroupBox("Tags / Keywords")
        tags_layout = QVBoxLayout(tags_group)
        self._tags_label = QLabel()
        self._tags_label.setWordWrap(True)
        self._tags_label.setStyleSheet("font-size: 12px; color: #555;")
        tags_layout.addWidget(self._tags_label)
        self._display_layout.addWidget(tags_group)

        # Related formulas
        related_group = QGroupBox("Related Formulas")
        related_layout = QVBoxLayout(related_group)
        self._related_list = QTextEdit()
        self._related_list.setReadOnly(True)
        self._related_list.setMaximumHeight(120)
        self._related_list.setStyleSheet("font-size: 12px;")
        related_layout.addWidget(self._related_list)
        self._display_layout.addWidget(related_group)

        self._display_layout.addStretch()

        scroll.setWidget(self._display_container)
        right_layout.addWidget(scroll)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([280, 520])

        self._populate_tree()

    # ----- Tree population -----

    def _populate_tree(self, matching_indices=None):
        """Populate the category tree. If matching_indices is given, only show those formulas."""
        self._tree.clear()
        show_all = matching_indices is None
        count = 0

        for cat, subs in self._category_tree.items():
            cat_item = QTreeWidgetItem([cat])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
            cat_font = QFont()
            cat_font.setBold(True)
            cat_item.setFont(0, cat_font)
            cat_has_children = False

            for sub, indices in subs.items():
                sub_item = QTreeWidgetItem([sub])
                sub_item.setFlags(sub_item.flags() & ~Qt.ItemIsSelectable)
                sub_font = QFont()
                sub_font.setItalic(True)
                sub_item.setFont(0, sub_font)
                sub_has_children = False

                for idx in indices:
                    if show_all or idx in matching_indices:
                        f = self._formulas[idx]
                        leaf = QTreeWidgetItem([f["name"]])
                        leaf.setData(0, Qt.UserRole, idx)
                        sub_item.addChild(leaf)
                        sub_has_children = True
                        count += 1

                if sub_has_children:
                    cat_item.addChild(sub_item)
                    cat_has_children = True

            if cat_has_children:
                self._tree.addTopLevelItem(cat_item)

        self._tree.expandAll()
        self._result_count_label.setText("{} formula{}".format(count, "s" if count != 1 else ""))

    # ----- Search -----

    def _on_search_changed(self, text):
        query = text.strip().lower()
        if not query:
            self._populate_tree()
            return

        tokens = query.split()
        matching = set()
        for idx, f in enumerate(self._formulas):
            searchable = " ".join([
                f["name"].lower(),
                f["description"].lower(),
                " ".join(f["tags"]),
                " ".join(f["variables"].keys()).lower(),
                " ".join(f["variables"].values()).lower(),
                f["category"].lower(),
                f["subcategory"].lower(),
            ])
            if all(tok in searchable for tok in tokens):
                matching.add(idx)

        self._populate_tree(matching)
        self._log("Search '{}' matched {} formula(s)".format(text, len(matching)))

    # ----- Tree click -----

    def _on_tree_item_clicked(self, item, column):
        idx = item.data(0, Qt.UserRole)
        if idx is not None:
            self._show_formula(idx)

    # ----- Formula display -----

    def _show_formula(self, idx):
        f = self._formulas[idx]
        self._current_formula_idx = idx
        self._log("Displaying formula: {}".format(f["name"]))

        self._title_label.setText(f["name"])

        self._category_label.setText("{}  >  {}".format(f["category"], f["subcategory"]))
        cat_colors = {
            "Physics": "#3b82f6",
            "Chemistry": "#10b981",
            "Mathematics": "#8b5cf6",
            "Statistics": "#f59e0b",
            "Engineering": "#ef4444",
        }
        bg = cat_colors.get(f["category"], "#6b7280")
        self._category_label.setStyleSheet(
            "color: #fff; background: {}; border-radius: 4px; "
            "padding: 2px 8px; font-size: 11px; font-weight: bold;".format(bg)
        )
        self._category_label.show()

        # Render equation
        try:
            pixmap = render_latex_to_pixmap(f["latex"], fontsize=18, dpi=150)
            self._equation_label.setPixmap(
                pixmap.scaled(
                    pixmap.width(), pixmap.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        except Exception as e:
            self._equation_label.setText("(Render error: {})".format(e))
            self._log("Render error for {}: {}".format(f["name"], e))

        # LaTeX source
        self._latex_source.setText(f["latex"])

        # Description
        self._desc_label.setText(f["description"])

        # Variables table
        variables = f["variables"]
        self._var_table.setRowCount(len(variables))
        for row, (sym, meaning) in enumerate(variables.items()):
            sym_item = QTableWidgetItem(sym)
            sym_item.setFont(QFont("Consolas", 11))
            self._var_table.setItem(row, 0, sym_item)
            self._var_table.setItem(row, 1, QTableWidgetItem(meaning))
        self._var_table.resizeRowsToContents()

        # Tags
        self._tags_label.setText(", ".join(f["tags"]))

        # Related formulas
        related = self._find_related(idx)
        if related:
            lines = []
            for ri in related:
                rf = self._formulas[ri]
                lines.append("- {} ({} / {})".format(rf["name"], rf["category"], rf["subcategory"]))
            self._related_list.setPlainText("\n".join(lines))
        else:
            self._related_list.setPlainText("(none found)")

    def _find_related(self, idx, max_results=6):
        """Find formulas related to the one at idx using tag overlap."""
        f = self._formulas[idx]
        tags_set = set(f["tags"])
        scores = []
        for i, other in enumerate(self._formulas):
            if i == idx:
                continue
            overlap = len(tags_set & set(other["tags"]))
            same_sub = 1 if (other["subcategory"] == f["subcategory"] and other["category"] == f["category"]) else 0
            score = overlap * 2 + same_sub
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        return [i for _, i in scores[:max_results]]

    # ----- Copy LaTeX -----

    def _copy_latex(self):
        if self._current_formula_idx is not None:
            latex = self._formulas[self._current_formula_idx]["latex"]
            clipboard = QApplication.clipboard()
            clipboard.setText(latex)
            self._log("Copied LaTeX to clipboard")


# ---------------------------------------------------------------------------
# Standalone execution for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FormulaRefWidget()
    win.set_logger(print)
    win.setWindowTitle("Axiom Scientific Suite - Formula Reference")
    win.resize(900, 650)
    win.show()
    sys.exit(app.exec_())
