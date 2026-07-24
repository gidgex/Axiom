@echo off
echo ============================================
echo  QuantumRes Scientific Suite - Build Script
echo ============================================
echo.

echo [1/3] Installing dependencies...
pip install PyQt5 numpy scipy matplotlib sympy pandas scikit-learn scikit-image Pillow pyinstaller

echo.
echo [2/3] Testing application launch...
python -c "from app import QuantumResMainWindow; print('Import OK')"
if errorlevel 1 (
    echo ERROR: Application import failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Building executable...
pyinstaller --noconfirm --onedir --windowed ^
    --name "QuantumRes" ^
    --add-data "modules;modules" ^
    --hidden-import numpy ^
    --hidden-import scipy ^
    --hidden-import scipy.signal ^
    --hidden-import scipy.optimize ^
    --hidden-import scipy.sparse ^
    --hidden-import scipy.sparse.linalg ^
    --hidden-import scipy.spatial ^
    --hidden-import scipy.ndimage ^
    --hidden-import scipy.stats ^
    --hidden-import scipy.fft ^
    --hidden-import scipy.interpolate ^
    --hidden-import scipy.integrate ^
    --hidden-import matplotlib ^
    --hidden-import matplotlib.backends.backend_qt5agg ^
    --hidden-import mpl_toolkits.mplot3d ^
    --hidden-import sympy ^
    --hidden-import pandas ^
    --hidden-import sklearn ^
    --hidden-import sklearn.neighbors ^
    --hidden-import sklearn.tree ^
    --hidden-import sklearn.ensemble ^
    --hidden-import sklearn.svm ^
    --hidden-import sklearn.linear_model ^
    --hidden-import sklearn.cluster ^
    --hidden-import sklearn.preprocessing ^
    --hidden-import sklearn.decomposition ^
    --hidden-import sklearn.metrics ^
    --hidden-import sklearn.model_selection ^
    --hidden-import sklearn.neural_network ^
    --hidden-import sklearn.naive_bayes ^
    --hidden-import sklearn.mixture ^
    --hidden-import sklearn.datasets ^
    --hidden-import PIL ^
    --hidden-import skimage ^
    --hidden-import xml.etree.ElementTree ^
    --hidden-import json ^
    --hidden-import csv ^
    --hidden-import io ^
    --hidden-import code ^
    --hidden-import traceback ^
    --hidden-import struct ^
    --collect-submodules numpy ^
    --collect-submodules scipy ^
    --collect-submodules matplotlib ^
    --collect-submodules sklearn ^
    main.py

echo.
if exist "dist\QuantumRes\QuantumRes.exe" (
    echo ============================================
    echo  BUILD SUCCESSFUL!
    echo  Executable: dist\QuantumRes\QuantumRes.exe
    echo ============================================
) else (
    echo BUILD FAILED - check errors above
)
echo.
pause
