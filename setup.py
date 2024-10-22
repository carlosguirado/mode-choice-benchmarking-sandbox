# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcbs",
    version="0.1.1",
    author="Carlos Guirado",
    author_email="guirado@berkeley.edu",
    description="A benchmarking sandbox for mode choice models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carlosguirado/mode-choice-benchmarking-sandbox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "biogeme>=3.2.14",
        "matplotlib>=3.0.0",
    ],
    include_package_data=True,
    package_data={
        "mcbs": ["datasets/*.json", "datasets/*.csv", "datasets/*.csv.gz"],
    }
)