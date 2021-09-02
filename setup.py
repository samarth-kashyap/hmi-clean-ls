# File: setup.py
from setuptools import find_packages, setup

setup(
    use_scm_version=True,
    name="hmiSurfCon",
    packages=find_packages(where="src"),
    package_dir={"", "src"},
)
