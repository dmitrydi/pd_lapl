from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([
    Extension("cyfuncs", ["cyfuncs.pyx"],
              libraries=["m"])
    ]), include_dirs=[numpy.get_include()]
)