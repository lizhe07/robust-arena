from setuptools import setup, find_packages

with open('roarena/VERSION.txt', 'r') as f:
    VERSION = f.readline().split('"')[1]

setup(
    name="roarena",
    version=VERSION,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=['torch', 'jarvis>=0.5', 'eagerpy', 'numba', 'foolbox'],
)
