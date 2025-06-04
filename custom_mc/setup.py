# python setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name="_marching_cubes_lewiner_cy",
        sources=["_marching_cubes_lewiner_cy.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
    )
]

setup(
    name="My MC",
    ext_modules=cythonize(ext_modules),
)
