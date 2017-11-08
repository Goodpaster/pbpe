from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy

os.environ['CFLAGS'] = '-O3 -Wall -ffast-math -fopenmp'

setup(
    ext_modules = cythonize("src/*.pyx"),
    include_dirs = [numpy.get_include()]
)
