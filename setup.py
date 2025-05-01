from setuptools import setup, find_packages

setup(
    name="mcbs",
    version="0.1.2",  # Incremented version number
    packages=find_packages(include=["mcbs", "mcbs.*"]),
    author="Carlos Guirado",
    author_email="your.email@example.com",
    description="Mode Choice Benchmarking System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/carlosguirado/mode-choice-benchmarking-sandbox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "biogeme",
        "requests",
    ],
    include_package_data=True,
)