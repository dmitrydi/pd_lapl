from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='integrate dyds nnz nnnn',
      ext_modules=cythonize("integrate_dyd_nnz.pyx"),
      include_dirs=[numpy.get_include()])