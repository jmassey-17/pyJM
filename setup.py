#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['igor==0.3',
                'matplotlib>=3.4.3',
                'numpy>=1.15.1',
                'pandas>=1.3.4',
                'Pillow>=8.4.0',
                'pytest>=6.2.4',
                'pyvista>=0.31.0',
                'scikit_image>=0.18.3',
                'scipy>=1.7.1',
                'seaborn>=0.11.2',
                'setuptools>=40.2.0',
                'Stoner==0.10.4',

 ]

test_requirements = ['pytest>=3', ]

setup(
    author="Jamie Massey",
    author_email='jamie435@hotmail.co.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Package for analysis and visualisation of reconstructed three-dimensional magnetic imaging",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyJM',
    name='pyJM',
    packages=find_packages(),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jmassey-17/pyJM',
    version='0.4.0',
    zip_safe=False,
)
