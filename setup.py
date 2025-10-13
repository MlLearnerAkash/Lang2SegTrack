from setuptools import setup, find_packages

setup(
    name='segTrack', 
    version='1.0', 
    packages=find_packages(),
    package_dir={'': '.'},  # Look in current directory
    package_data={'': ['*/__init__.py']}  # Include __init__.py files
)