#!/usr/bin/env python

# -*- coding: utf-8 -*-

import os.path

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
    find_packages = lambda *x: None


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


exec(open('db_dumper/version.py').read())

tests_require = [
    'coverage',
    'flake8',
    'pydocstyle',
    'pylint',
    'pytest-pep8',
    'pytest-cov',
    # for pytest-runner to work, it is important that pytest comes last in
    # this list: https://github.com/pytest-dev/pytest-runner/issues/11
    'pytest'
]

version = '0.1.0'

setup(name='db_dumper',
      version=__version__,
      description='A data dumper for the MMTO measurements database.',
      long_description=read('README.rst'),
      author='Scott Swindell',
      author_email='sswindell@mmto.org',
      url='https://github.com/Scott Swindell/db_dumper',
      classifiers=[
          'Development Status :: 2 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      include_package_data=True,
      packages=find_packages(include=['db_dumper*']),
      test_suite='tests',
      setup_requires=['pytest-runner'],
      tests_require=tests_require)
