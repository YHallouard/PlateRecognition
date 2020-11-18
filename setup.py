from setuptools import find_packages, setup
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()

required_setup = ['pytest-runner',
                  'wheel',
                  'flake8',
                  'Cython']
required_tests = ['pytest',
                  'pytest-cov',
                  'treon']

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

__version__: str
exec(open('LRP/version.py').read())

setup(name='LRP',
      packages=find_packages(),
      version=__version__,
      description=' ',
      author='Yann Hallouard',
      long_description=long_description,
      long_description_content_type="text/markdown",
      setup_requires=required_setup,
      tests_require=required_tests,
      install_requires=all_reqs,
      license='',
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      )
