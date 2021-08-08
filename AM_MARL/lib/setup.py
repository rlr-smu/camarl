import numpy
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='test',
    ext_modules=cythonize("lib/cy_utils.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)