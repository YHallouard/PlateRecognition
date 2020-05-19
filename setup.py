from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

required = ['Cython']


setup(name='LRP',
      packages=find_packages(),
      version='0.1.0',
      description=' ',
      author='Yann Hallouard',
      long_description=long_description,
      long_description_content_type="text/markdown",
      setup_requires=['pytest-runner', 'wheel'],
      tests_require=required,
      install_requires=required,
      license='',
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      )
