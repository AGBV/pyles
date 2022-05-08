from setuptools import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("cython/test_for_loop.pyx"))