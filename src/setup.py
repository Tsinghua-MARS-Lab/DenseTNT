from distutils.core import setup, Extension

import numpy
from Cython.Build import cythonize

setup(ext_modules=cythonize(Extension(
    'utils_cython',
    sources=['utils_cython.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))
