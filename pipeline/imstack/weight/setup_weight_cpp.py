"""
Build script for C++ weightmap calculation extension.

To build:
    cd gppy/imstack/weight
    python setup_weight_cpp.py build_ext --inplace

Requirements:
    - pybind11
    - C++ compiler with OpenMP support
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11

ext_modules = [
    Pybind11Extension(
        "weight_cpp",  # Module name (will be imported as weight_cpp)
        ["weight_cpp.cpp"],
        cxx_std=14,
        extra_compile_args=["-fopenmp", "-O3", "-march=native"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="weight_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
