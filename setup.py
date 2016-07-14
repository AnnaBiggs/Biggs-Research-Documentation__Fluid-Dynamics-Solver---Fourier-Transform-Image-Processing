#Use setup.py to build Cython file, see Program Description or advection2.pyx comments for instructions 

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("advection2.pyx")
)