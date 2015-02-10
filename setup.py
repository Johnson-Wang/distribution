from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]

extension = Extension('kappa._kthin',
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-lgomp'],
                      include_dirs=include_dirs_numpy,
                      sources=['c/_kappa_thin_film.c'])

setup(name='distribution',
      version='0.1',
      description='This is a module that can help calculate kappa in thin film and the distribution of one parameter onto another',
      author='Wang Xinjiang',
      author_email='swanxinjiang@gmail.com',
      packages=['kappa'],
      scripts=['script/kappa'],
      ext_modules=[extension])
