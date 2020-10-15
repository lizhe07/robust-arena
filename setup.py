# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:35:37 2020

@author: Zhe
"""

from roarena import __version__
from setuptools import setup, find_packages

setup(
    name="roarena",
    version=__version__,
    author='Zhe Li',
    python_requires='>=3',
    packages=find_packages(),
    install_requires=['torch', 'jarvis'],
)
