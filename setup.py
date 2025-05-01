from setuptools import setup, find_packages

setup(
    name="mcbs",
    version="0.1.0",
    packages=find_packages(include=["mcbs", "mcbs.*"]),
)