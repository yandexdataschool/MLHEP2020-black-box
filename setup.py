from setuptools import find_packages, setup

setup(
  name='fel',
  version='0.1.0',
  description='Study of optimization methods for XFEL accelerator tuning.',
  author='Maxim Borisyak, Alena Zarodnyuk',
  license='MIT',

  packages=find_packages('src'),
  package_dir={'': 'src/'},
)
